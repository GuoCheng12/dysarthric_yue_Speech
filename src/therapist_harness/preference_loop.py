"""MAS audio-preference execution on the existing evidence/reflection RunStore.

Each round's four arms share the current MAS-DPO parent. Only MAS-DPO advances
the next prescription; controls are same-parent local interventions. No test
selection or invented negatives are performed here. SG review explicitly records
whether the Agent receives audio or an independently decoded ASR transcript.
"""

from collections import Counter
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from .agent import (build_therapist_request, execute_therapist_request,
                    parse_therapist_response, serialize_response,
                    build_reflection_request, parse_reflection_response)
from .cli import load_protocol
from .evidence_pack import build_evidence_pack_from_files, load_prediction_rows
from .materialize import materialize_proposal, build_materialization_sidecar
from .memory import SkillMemory, compile_intervention_summary
from .outcome import compile_evaluation_plan, compile_round_result, compile_reflection_pack
from .preference_contract import PreferenceLoopConfig, PreferenceDelivery, PreferenceSampleDelivery
from .preference_data import validate_sampling_inputs, build_preference_pairs, target_text
from .round_schema import (EvaluationPlan, MaterializationSidecar, SynthesisReceipt,
                           SynthesizedSample, TrainingReceipt, ReflectionPack)
from .schema import RunProtocol, EvidencePack, Proposal
from .store import RunStore, sha256_file, sha256_directory, write_immutable_json, write_immutable_bytes
from .text import identity_keys


REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src/therapist_harness"
RECORDS = Path("/data/qwen3-asr/records/therapist_mas_setting_d_v1")
REAL = Path("/data/qwen3-asr/finetune/data_setting_d_v1/setting_d_train.jsonl")
FEEDBACK = Path("/data/qwen3-asr/finetune/data_setting_d_merged_test_v1/setting_d_merged_test.jsonl")
DEFAULT_R = RECORDS / "tts_utility_diagnosis_v1/additive/real_w0/seed_42"
ARMS = ("r_sft", "r_dpo", "mas_sft", "mas_dpo")


def enabled_arms(protocol):
    if protocol.preference_loop.comparison == "synthetic_only":
        return ("mas_dpo",)
    return ("r_dpo", "mas_dpo") if protocol.preference_loop.comparison == "dpo_only" else ARMS


SCOPE = """
This run constructs audio ASR preference data. The 511 real recordings are
development feedback, not an independent sealed test; the 2117 real training
pool and all SG references remain isolated from feedback content and audio.
Your target, contrast, protection and maintenance sentences should provide
useful correction opportunities for the CURRENT ASR. Contrast means valid
alternative practice content, never an intentionally false chosen label.
Write complete natural Cantonese sentences ending in 。！？!? as required by
the Proposal schema. Preserve each target verbatim through SG, review and
chosen construction; punctuation-only ASR differences do not create pairs.
The SG receives each intended text and patient reference speech. A lightweight
audio Agent screens only gross generation failures, not patient intelligibility,
CER, phonemes or strict word fidelity. Passing does not prove a verbatim label.
Four actual current-ASR samples per admitted audio produce distinct erroneous
rejected transcripts. All-correct audio yields no DPO pair; repeated identical
errors yield one pair. Chosen is the admitted intended target; no fabricated
rejections. Transcript and EOS log probabilities use the SAME audio and prompt.
The DPO reference is the frozen initial R-only ASR adapter, a separate concept
from TTS reference speech. R-SFT/R-DPO/R+MAS-SFT/R+MAS-DPO are same-parent
local comparisons. Only the MAS-DPO endpoint advances the next round.
Generation counts and requested-text carrier counts are not DPO training dose.
Read training.preference_delivery and episode_memory.preference_delivery for
review failures, realized pairs, all-correct skips and actual audio exposure.
Reflect on true real-speech recovery AND retained abilities. A missing pair or
unrealized target cannot by itself refute the prescription's learning hypothesis.
Small single-round changes do not establish causality or stable lessons.
No change to optimizer, references or generation budgets is authorized to the MAS.
"""


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text("utf-8").splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    if path.exists():
        if read(path) != json.loads(json.dumps(payload)):
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    write_immutable_json(path, payload)


def jsonl(path, values):
    path = Path(path)
    content = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values).encode()
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable manifest differs: {path}")
        return
    write_immutable_bytes(path, content)


def load(path, cls):
    return cls.model_validate_json(Path(path).read_text("utf-8"))


def round_dir(root, number):
    return Path(root) / f"round_{number:03d}"


def checkpoint(root, number, arm="mas_dpo"):
    p = RunStore(Path(root)).protocol()
    return Path(p.initial_adapter.path) if number == 0 else round_dir(root, number) / arm / "training" / f"checkpoint-{p.lora.optimizer_steps_per_round}"


def predictions(root, number, arm="mas_dpo"):
    p = RunStore(Path(root)).protocol()
    if number and p.preference_loop.execution_version == 'persistent_kl_dpo_v3':
        return round_dir(root, number) / arm / 'residual_evaluation/evaluation/predictions.jsonl'
    return Path(p.initial_adapter_dev_predictions.path) if number == 0 else round_dir(root, number) / arm / "evaluation/predictions.jsonl"


def accepted_checkpoint(root, number):
    p=RunStore(Path(root)).protocol()
    if number == 0 or p.preference_loop.checkpoint_acceptance == "always":
        return checkpoint(root,number)
    return Path(read(round_dir(root,number)/"promotion_decision.json")["accepted_checkpoint"])


def accepted_predictions(root, number):
    p=RunStore(Path(root)).protocol()
    if number == 0 or p.preference_loop.checkpoint_acceptance == "always":
        return predictions(root,number)
    return Path(read(round_dir(root,number)/"promotion_decision.json")["accepted_predictions"])


def run_command(command, log, *, env_overrides=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(env_overrides or {})
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "src"), str(REPO / "examples"), env.get("PYTHONPATH", "")])
    with log.open("x", encoding="utf-8") as handle:
        subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def verify_sources(root):
    for path, digest in read(Path(root) / "sources.json").items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"frozen execution source changed: {path}; prepare a new run")


def split_audit(real, feedback):
    if (len(real), len(feedback)) != (2117, 511):
        raise ValueError("this Setting D protocol requires the full 2117 train / 511 feedback pools")
    for field in ("utt_id", "prompt_id", "prompt_normalized_sha16", "prompt_jyutping_sha16"):
        a, b = ({x[field] for x in data if x.get(field)} for data in (real, feedback))
        if a & b:
            raise ValueError(f"train/feedback overlap: {field}")
    real_audio = {sha256_file(Path(x["audio"])) for x in real}
    feedback_audio = {sha256_file(Path(x["audio"])) for x in feedback}
    if real_audio & feedback_audio:
        raise ValueError("train/feedback audio hash overlap")
    return {"train_rows": len(real), "feedback_rows": len(feedback),
            "audio_overlap": 0, "independent_test": False}


def dpo_instructions(text, review_mode="audio", comparison="four_arms", *, controlled_generation=False, checkpoint_acceptance="always"):
    """Adapt new-run copies only; historical SFT prompts/Skills stay immutable."""
    replacements = {
        "This is ordinary supervised\nrehearsal, not KL distillation or a new loss.":
            "Maintenance supplies valid practice content. It contributes a DPO pair only when the current ASR actually samples an error; all-correct audio yields no pair. Its role name does not create a separate loss.",
        "Ordinary ASR SFT receives only synthetic speech and its transcript. It does not see a\nstimulus role, rationale, pair label, margin, or contrastive objective. Therefore a\n`contrast` stimulus is not contrastive learning and must never be described as such.\nIts weaker purpose is counterbalancing: place carriers for the primary pattern and an\nobserved alternative confusion in controlled co-occurrence so the SFT distribution is\nnot dominated by one prescription.":
            "The DPO update receives same-audio chosen/rejected transcripts, with rejected drawn from actual current-ASR errors. A stimulus role does not determine the negative response or introduce a role-specific loss. Contrast practice counterbalances the primary prescription through valid content and observed alternative carriers; never fabricate an erroneous chosen or rejected label.",
        "Roles are prescription bookkeeping: ordinary ASR SFT sees speech and transcript,\nnot a maintenance/protection label or a special loss.":
            "Roles are prescription bookkeeping. The fixed DPO objective sees audio-conditioned chosen/rejected transcripts, not a role-specific loss. Use actual pair availability and training exposure when interpreting each role.",
        "Ordinary SFT data can incidentally affect both.": "Preference updates can incidentally affect both.",
        "role names do not change the ordinary ASR SFT loss.":
            "role names do not change the fixed DPO loss; only actual sampled errors produce preference pairs.",
        "- `round_result.training`: actual replay, synthetic rows and executed training steps.":
            "- `round_result.training`: actual preference rows and executed steps; `preference_delivery` separates considered audio, generated audio, review passes, pairless skips and trained exposures.",
        "`real_replay_count` is the fixed number of real training examples mixed in by the":
            "`real_replay_count` is the fixed number of real audio candidates sampled by the",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if "Ordinary ASR SFT receives only" in text or "ordinary ASR SFT loss" in text:
        raise ValueError("unadapted SFT-only instruction in preference prompt")
    scope = SCOPE
    if comparison == "dpo_only":
        scope = scope.replace("R-SFT/R-DPO/R+MAS-SFT/R+MAS-DPO are same-parent",
                              "Only R-DPO/R+MAS-DPO are executed as same-parent")
    elif comparison == "synthetic_only":
        scope = scope.replace("R-SFT/R-DPO/R+MAS-SFT/R+MAS-DPO are same-parent\nlocal comparisons. Only the MAS-DPO endpoint advances the next round.",
            "Only MAS-TTS DPO is executed from the R-only starting adapter. No real\naudio is sampled for preferences or replayed in DPO. The 2117 real training\nrecordings are used only for existing evidence and TTS reference selection;\n511 real development recordings are used only for feedback/evaluation.\nOptimize ordinary mean synthetic-audio DPO loss with weight 1. Only the\nMAS-DPO endpoint advances the next round. Do not claim R+S mixed training.")
    if review_mode == "asr_transcript":
        scope = scope.replace(
            "audio Agent screens only gross generation failures, not patient intelligibility,\nCER, phonemes or strict word fidelity. Passing does not prove a verbatim label.",
            "text Agent compares target text, reference transcript and a fixed R-only ASR\ntranscript. It only flags strongly suggested gross generation failures. It does\nnot hear the audio and must not equate ASR mistakes with SG failure. It uses no\nCER threshold, patient-intelligibility filter or strict word-fidelity check.\nPassing does not prove a verbatim label. The review transcriber stays frozen at\nthe initial R-only checkpoint; only preference sampling uses the current ASR.")
    if checkpoint_acceptance != 'always':
        scope=scope.replace('Only the MAS-DPO endpoint advances the next round.',
            'The MAS-DPO candidate advances only after the fixed real-development acceptance rule passes; failed experiment experience remains in memory.')
        scope=scope.replace('Only the\nMAS-DPO endpoint advances the next round.',
            'The candidate advances only after the fixed real-development acceptance rule passes; failed experience remains in memory.')
    if controlled_generation:
        scope=scope.replace('No change to optimizer, references or generation budgets is authorized to the MAS.',
            'A separate MAS generation planner selects specific same-patient references and allowed recipes. You control content and role budgets; keep optimizer and total budget fixed.')
        scope+='\nPrefer everyday natural Cantonese and varied contexts. Avoid tongue-twister-like target packing merely to raise carrier counts. Written phonetic coverage does not prove acoustic realization.\n'
    return text + scope


def prepare(root, *, parent=None, parent_predictions=None, review_model, review_base_url,
            review_key_env="OPENAI_API_KEY", rounds_count=2, synthetic_count=96,
            unique_texts=6, real_count=2117, steps=12, real_per_step=32,
            synthetic_per_step=8, seed=42, learning_rate=1e-5, review_mode="audio", comparison="four_arms", execution_version="serial_v1",
            generation_design_path=None, checkpoint_acceptance="always", include_greedy=False, review_allow_pending=False,
            training_allocation="uniform", retention_design_path=None):
    root = Path(root).resolve()
    real, feedback = rows(REAL), rows(FEEDBACK)
    isolation = split_audit(real, feedback)
    if comparison == "synthetic_only":
        real_count, real_per_step = 0, 0
    if not 0 <= real_count <= len(real) or (real_count == 0 and comparison != "synthetic_only"):
        raise ValueError("real replay selection must come from the 2117 training pool")
    if synthetic_count % unique_texts:
        raise ValueError("use a whole-number copy budget per unique text")
    if (parent is None) != (parent_predictions is None):
        raise ValueError("a custom parent requires its own explicit prediction file")
    parent = Path(parent or DEFAULT_R / "training/checkpoint-50").resolve()
    parent_predictions = Path(parent_predictions or DEFAULT_R / "evaluation/merged_step_50/predictions.jsonl").resolve()
    base = load_protocol(REPO / "configs/therapist_setting_d_v1.yaml").model_dump(mode="json")
    # Retry transient transport/status failures without regenerating experiment data.
    base["agent"]["max_retries"] = 4
    zero = RECORDS / "therapist_m0_b8g4_pilot_v1/merged_test_reassessment_v1/zero_shot/predictions.jsonl"
    base.update(protocol_status="frozen", run_id=root.name, round_count=rounds_count,
                synthetic_count_per_round=synthetic_count, unique_stimulus_count_per_round=unique_texts,
                real_replay_count_per_round=real_count, sealed_test_split=None, skill_update_rounds=[])
    base["initial_adapter"] = {"path": str(parent), "kind": "directory", "sha256": sha256_directory(parent)}
    for field, path, count in (("initial_adapter_dev_predictions", parent_predictions, 511),
                                ("zero_shot_dev_predictions", zero, 511), ("feedback_split", FEEDBACK, 511), ("real_train", REAL, 2117)):
        base[field] = {"path": str(path), "sha256": sha256_file(path), "expected_rows": count}
    base["initial_adapter_prediction_column"] = base["zero_shot_prediction_column"] = "prediction"
    base["lora"].update(seed=seed, learning_rate=learning_rate, optimizer_steps_per_round=steps,
                        scheduler="constant", warmup_ratio=0.0, training_passes_per_round=1,
                        per_device_batch_size=1, gradient_accumulation_steps=1)
    for field in ("max_copies_per_stimulus", "threshold_edge_max_linked_copies"):
        base["proposal_constraints"][field] = min(base["proposal_constraints"][field], synthetic_count)
    base["proposal_constraints"]["threshold_edge_max_unique_stimuli"] = min(2, unique_texts)
    review_options = {"review_mode": review_mode, "include_greedy":include_greedy,
                      "review_allow_pending":review_allow_pending,"checkpoint_acceptance":checkpoint_acceptance,
                      "training_allocation":training_allocation}
    if generation_design_path is not None:
        review_options.update(generation_design_path=str(Path(generation_design_path).resolve()),
                              generation_design_sha256=sha256_file(Path(generation_design_path)))
    if retention_design_path is not None:
        review_options.update(retention_design_path=str(Path(retention_design_path).resolve()),
                              retention_design_sha256=sha256_file(Path(retention_design_path)))
    if review_mode == "asr_transcript":
        review_options.update(review_asr_adapter=str(parent),
            review_asr_adapter_sha256=base["initial_adapter"]["sha256"],
            review_asr_base_model_sha256=sha256_directory(Path(base["base_model_dir"])))
    base["preference_loop"] = PreferenceLoopConfig(comparison=comparison, execution_version=execution_version, real_per_step=real_per_step,
        synthetic_per_step=synthetic_per_step, synthetic_weight=1.0 if comparison == "synthetic_only" else 0.05, audio_review_model=review_model,
        audio_review_base_url=review_base_url, audio_review_api_key_env=review_key_env,
        **review_options).model_dump(mode="json")
    protocol = RunProtocol.model_validate(base)
    if execution_version == 'persistent_kl_dpo_v3':
        from .retention_contract import load_retention_design, validate_retention_rows
        design = load_retention_design(protocol)
        validate_retention_rows(protocol, design)
        if sha256_directory(parent) != design.frozen_reference_asr_sha256:
            raise ValueError('a new core KL/DPO run starts from initial R-only; warm continuation occurs inside its committed rounds')
    RunStore.initialize(protocol, root)
    scope_options=dict(controlled_generation=generation_design_path is not None,checkpoint_acceptance=checkpoint_acceptance)
    def instructions(text):
        result = dpo_instructions(text, review_mode, comparison, **scope_options)
        if execution_version == 'persistent_kl_dpo_v3':
            from .retention_loop import adapt_scope
            result = adapt_scope(result)
        return result
    write_immutable_bytes(root / "initial_skill.md", instructions((PACKAGE / "skills/v000/SKILL.md").read_text()).encode())
    SkillMemory.initialize(root / "skills", root / "initial_skill.md")
    for name in ("therapist.md", "reflection.md"):
        write_immutable_bytes(root / "prompts" / name, instructions((PACKAGE / "prompts" / name).read_text()).encode())
    freeze_sources(root)
    save(root / "split_audit.json", isolation)
    selected = sorted(real, key=lambda x: x["utt_id"])
    random.Random(seed).shuffle(selected)
    jsonl(root / "real_selection.jsonl", [{**x, "source": "real", "audio_id": x["utt_id"]} for x in selected[:real_count]])
    save(root / "execution_policy.json", {"arms": enabled_arms(protocol), "parent_policy": "enabled local arms share current MAS-DPO parent",
        "next_parent": "mas_dpo fixed endpoint" if checkpoint_acceptance=='always' else 'last accepted endpoint under '+checkpoint_acceptance, "dpo_reference": str(parent), "dpo_reference_sha256": protocol.initial_adapter.sha256,
        "synthetic_requested": synthetic_count, "real_audio_selected_for_sampling": real_count,
        "samples_per_audio": protocol.preference_loop.samples_per_audio,
        "training_counts": "pair rows and actual exposures are recorded separately from generated audio",
        "across_objectives": "SFT keeps all admitted audio; DPO omits pairless audio, so effective eligible pools can differ",
        "review": {"mode": review_mode, "rule": "only obvious failures; zero-shot retry once then instruct2 once",
                   "transcriber": str(parent) if review_mode == "asr_transcript" else None,
                   "transcriber_policy": "frozen initial R-only; greedy Cantonese; no target/reference context" if review_mode == "asr_transcript" else None},
        "formal_run_started": False,
        **({'auxiliary_real_KL': design.model_dump(mode='json'),
            'primary_manifest': 'Synthetic preference rows only; auxiliary real-KL rows/exposures are recorded separately in preference_delivery.real_kl_retention.'}
           if execution_version == 'persistent_kl_dpo_v3' else {})})
    return protocol


def freeze_sources(root):
    source_paths = [*PACKAGE.glob("*.py"), *PACKAGE.joinpath("prompts").glob("*.md"),
        REPO / "examples/mas_asr_dpo_loop.py", REPO / "examples/generate_reviewed_sg.py",
        REPO / "examples/sample_asr_preferences.py", REPO / "finetune/scripts/qwen3_asr_audio_dpo.py",
        REPO / "examples/transcribe_sg_review.py", REPO / "finetune/scripts/evaluate_asr_checkpoint.py",
        REPO / "finetune/official/qwen3_asr_sft.py", REPO / "synthesis/cosyvoice3_patient_sft/synthesize_manifest.py",
        PACKAGE / "skills/v000/SKILL.md"]
    if RunStore(Path(root)).protocol().preference_loop.execution_version == 'persistent_kl_dpo_v3':
        from .retention_loop import execution_sources
        source_paths += execution_sources()
    save(Path(root) / "sources.json", {str(x): sha256_file(x) for x in sorted(source_paths)})


def assess(root, number):
    root = Path(root); verify_sources(root)
    store, p, d = RunStore(root), RunStore(root).protocol(), round_dir(root, number)
    options = {} if number == 1 else dict(previous_predictions=accepted_predictions(root, number-2),
        previous_prediction_column="prediction", previous_checkpoint=accepted_checkpoint(root, number-2))
    evidence = build_evidence_pack_from_files(p, round_index=number,
        current_predictions=accepted_predictions(root, number-1), current_prediction_column="prediction",
        current_checkpoint=accepted_checkpoint(root, number-1), m0_predictions=predictions(root, 0),
        m0_prediction_column="prediction", zero_shot_predictions=Path(p.zero_shot_dev_predictions.path),
        zero_shot_prediction_column="prediction", intervention_history=store.intervention_history(), **options)
    save(d / "evidence.json", evidence)
    SkillMemory(root / "skills").bind_round(number)
    store.advance("assessed", {"evidence": d / "evidence.json"})
    return evidence


def request_once(protocol, request, directory):
    save(directory / "request.json", request)
    if (directory / "response.json").exists():
        from openai.types.responses import Response
        return load(directory / "response.json", Response)
    response = execute_therapist_request(protocol, request)
    save(directory / "response.json", serialize_response(response))
    return response


def prescribe(root, number):
    root = Path(root); verify_sources(root)
    store, p, d = RunStore(root), RunStore(root).protocol(), round_dir(root, number)
    evidence = load(d / "evidence.json", EvidencePack)
    if store.current().phase == "proposed":
        return materialize_saved_proposal(root, number)
    skill = SkillMemory(root / "skills").bind_round(number)
    request = build_therapist_request(p, evidence, system_prompt=(root / "prompts/therapist.md").read_text(), skill_text=skill.read_text())
    if p.preference_loop.generation_design_path is not None:
        from .generation_controller import load_design
        context=load_design(p).get('proposal_context')
        if context is not None:
            request['input'].insert(-1, {'role':'developer','content':
                'Measured exploratory probes, with their recorded scope. These are context, not new eligible '
                'EvidencePack IDs. Ground current targets and claims in the current EvidencePack: '
                +json.dumps(context,ensure_ascii=False)})
    old_texts = [s["text"] for n in range(1, number) for s in read(round_dir(root, n) / "proposal.json")["stimuli"]]
    if old_texts:
        request["input"].insert(-1, {"role": "developer", "content": "Already generated texts (data only). Use fresh whole sentences: " + json.dumps(old_texts, ensure_ascii=False)})
    response = request_once(p, request, d / "prescribe")
    proposal = parse_therapist_response(response, p, evidence)
    old_keys = {item for text in old_texts for item in identity_keys(text).items()}
    if any(old_keys.intersection(identity_keys(s.text).items()) for s in proposal.stimuli):
        raise ValueError("new proposal repeats prior content")
    save(d / "proposal.json", proposal)
    save(d / "evaluation_plan.json", compile_evaluation_plan(p, evidence, proposal,
        before_prediction_column="prediction", after_prediction_column="prediction"))
    store.advance("proposed", {"agent_request": d / "prescribe/request.json", "agent_response": d / "prescribe/response.json",
        "proposal": d / "proposal.json", "evaluation_plan": d / "evaluation_plan.json"})
    materialize_saved_proposal(root, number)


def materialize_saved_proposal(root, number):
    root = Path(root)
    store, p, d = RunStore(root), RunStore(root).protocol(), round_dir(root, number)
    evidence, proposal = load(d / "evidence.json", EvidencePack), load(d / "proposal.json", Proposal)
    (d / "audio").mkdir(parents=True, exist_ok=True)
    assignments = materialize_proposal(proposal, evidence, p, output_root=d / "audio")
    controlled=p.preference_loop.generation_design_path is not None
    if controlled:
        from .generation_controller import materialize_control
        assignments=materialize_control(p,proposal,assignments,directory=d,
            history=[x.model_dump(mode='json') for x in store.intervention_history()])
    jsonl(d / "synthesis_manifest.jsonl", assignments)
    save(d / "materialization_sidecar.json", build_materialization_sidecar(proposal, evidence, p, assignments,
        **({'generation_plan_path':d/'generation_plan.json'} if controlled else {})))
    store.advance("materialized", {"synthesis_manifest": d / "synthesis_manifest.jsonl", "materialization_sidecar": d / "materialization_sidecar.json",
        **({'generation_plan':d/'generation_plan.json'} if controlled else {})})


def generate(root, number):
    root = Path(root); verify_sources(root)
    p, d = RunStore(root).protocol(), round_dir(root, number)
    c = p.preference_loop
    generator = 'generate_reviewed_sg_recovery_v2.py' if c.execution_version == 'persistent_kl_dpo_v3' else 'generate_reviewed_sg.py'
    command = [sys.executable, str(REPO / 'examples' / generator), "--manifest", str(d / "synthesis_manifest.jsonl"),
        "--out-dir", str(d / "generation"), "--audio-review-model", c.audio_review_model,
        "--audio-review-base-url", c.audio_review_base_url, "--audio-review-api-key-env", c.audio_review_api_key_env,
        "--review-mode", c.review_mode]
    if c.review_mode == "asr_transcript":
        command += ["--review-asr-model", p.base_model_dir, "--review-asr-adapter", c.review_asr_adapter,
            "--review-asr-adapter-sha256", c.review_asr_adapter_sha256,
            "--review-asr-base-model-sha256", c.review_asr_base_model_sha256]
    if c.review_allow_pending:
        command += ['--review-allow-pending']
    if c.generation_design_path is not None:
        command += ['--generation-plan',str(d/'generation_plan.json')]
    dual = c.execution_version == "persistent_dual_gpu_v2"
    if dual or c.execution_version in {"persistent_synthetic_only_v1", "persistent_kl_dpo_v3"}:
        command += ["--review-asr-worker"]
        if (d / "generation_reuse.json").exists():
            command += ["--reuse-manifest", str(d / "generation_reuse.json")]
    run_command(command, d / "generation.log", env_overrides={"CUDA_VISIBLE_DEVICES": "0"} if dual else None)
    assignments = rows(d / "synthesis_manifest.jsonl")
    import soundfile as sf
    sidecar = load(d / "materialization_sidecar.json", MaterializationSidecar)
    receipt = SynthesisReceipt(schema_version="1", run_id=p.run_id, round_index=number, status="complete",
        assignments_canonical_sha256=sidecar.assignments_canonical_sha256,
        requested=len(assignments), generated=len(assignments), failed=0,
        samples=[SynthesizedSample(sample_id=x["sample_id"], audio_file_sha256=sha256_file(Path(x["output_wav"])),
                                  duration_seconds=float(sf.info(x["output_wav"]).duration)) for x in assignments])
    save(d / "synthesis_receipt.json", receipt)
    jsonl(d / "sampling_inputs.jsonl", rows(root / "real_selection.jsonl") + rows(d / "generation/accepted.jsonl"))


def build_branch_manifests(inputs, pairs):
    """Real order matches within each objective; pairless SFT examples survive."""
    from .preference_data import sft_rows
    sft = sft_rows(inputs)
    return {"r_sft": [r for r in sft if r["source"] == "real"],
            "r_dpo": [r for r in pairs if r["source"] == "real"],
            "mas_sft": sft, "mas_dpo": list(pairs)}


def sample_and_pair(root, number):
    root = Path(root); verify_sources(root)
    p, d = RunStore(root).protocol(), round_dir(root, number)
    c = p.preference_loop
    command = [sys.executable, str(REPO / "examples/sample_asr_preferences.py"), "sample",
        "--inputs", str(d / "sampling_inputs.jsonl"), "--real-train", p.real_train.path,
        "--feedback", p.feedback_split.path, "--reviews", str(d / "generation/reviews.jsonl"),
        "--out-dir", str(d / "sampling"), "--base-model", p.base_model_dir,
        "--current-adapter", str(accepted_checkpoint(root, number-1)), "--expected-parent-sha256", sha256_directory(accepted_checkpoint(root, number-1)),
        "--samples", str(c.samples_per_audio), "--seed", str(p.lora.seed), "--temperature", str(c.temperature),
        "--top-p", str(c.top_p), "--max-new-tokens", str(c.max_new_tokens)]
    if c.include_greedy:
        command += ['--include-greedy']
    if c.execution_version == 'persistent_kl_dpo_v3':
        command[1] = str(REPO / 'examples/sample_residual_asr_preferences_v1.py')
        command += ['--initial-r-only-adapter', p.initial_adapter.path]
    if not (d / "sampling/preference_audit.json").exists():
        if c.execution_version == "persistent_dual_gpu_v2":
            raise ValueError("dual GPU execution requires separately sampled and validated merged shards")
        run_command(command, d / "sampling.log")
    inputs, pairs = rows(d / "sampling/validated_inputs.jsonl"), rows(d / "sampling/preferences.jsonl")
    if c.execution_version == 'persistent_kl_dpo_v3':
        from .retention_loop import training_pairs
        pairs = training_pairs(pairs)
    for arm, data in build_branch_manifests(inputs, pairs).items():
        if arm not in enabled_arms(p):
            continue
        jsonl(d / arm / "train.jsonl", data)
        save(d / arm / "training_index.json", [{"row": i, "source": x["source"], "audio_id": x["audio_id"]} for i, x in enumerate(data)])
    if not pairs:
        save(d / "no_update.json", {"reason": "all actual samples correct; no preference pairs", "dpo_training_executed": False})
        return False
    if c.training_allocation=='role_budget_v1':
        from .training_allocation import compile_role_allocation
        save(d/'training_allocation.json',compile_role_allocation(read(d/'materialization_sidecar.json'),pairs,
            p.lora.optimizer_steps_per_round,c.synthetic_per_step,p.lora.seed))
    elif c.training_allocation == 'mas_audio_budget_v1':
        from .preference_allocation_agent import choose_exposure_allocation
        save(d / 'training_allocation.json', choose_exposure_allocation(p, load(d / 'proposal.json', Proposal),
            read(d / 'materialization_sidecar.json'), pairs, rows(d / 'sampling/greedy.jsonl'),
            directory=d / 'allocation_prescribe', history=[x.model_dump(mode='json') for x in RunStore(root).intervention_history()]))
    RunStore(root).advance("synthesized", {"synthesis_summary": d / "synthesis_receipt.json",
        "train_manifest": d / "mas_dpo/train.jsonl", "training_index": d / "mas_dpo/training_index.json",
        "preference_inputs": d / "sampling/validated_inputs.jsonl", "preference_candidates": d / "sampling/candidates.jsonl",
        "preference_reviews": d / "generation/reviews.jsonl", "preference_sampling_receipt": d / "sampling/sampling_receipt.json",
        "generation_receipt": d / "generation/generation_summary.json", "preference_audit": d / "sampling/preference_audit.json",
        **({'preference_greedy':d/'sampling/greedy.jsonl'} if c.include_greedy else {}),
        **({'preference_allocation':d/'training_allocation.json'} if c.training_allocation in {'role_budget_v1', 'mas_audio_budget_v1'} else {})})
    return True


def validate_registered_preferences(protocol, paths, train_rows, assignments, parent_hash):
    required = {"preference_inputs", "preference_candidates", "preference_reviews", "preference_sampling_receipt"}
    if not required <= paths.keys():
        raise ValueError("preference run lacks reviewed-audio and current-ASR sampling sources")
    reviews = rows(paths["preference_reviews"])
    inputs = validate_sampling_inputs(rows(paths["preference_inputs"]), rows(protocol.real_train.path), rows(protocol.feedback_split.path), reviews)
    receipt = read(paths["preference_sampling_receipt"])
    if receipt.get("status") != "complete" or receipt["current_asr_sha256"] != parent_hash:
        raise ValueError("preference sampling did not complete on the current parent")
    if receipt.get("samples_per_audio") != protocol.preference_loop.samples_per_audio:
        raise ValueError("sampling receipt count differs from the frozen preference protocol")
    if receipt["candidates_sha256"] != sha256_file(paths["preference_candidates"]) or receipt["validated_inputs_sha256"] != sha256_file(paths["preference_inputs"]):
        raise ValueError("preference sampling artifacts changed")
    if receipt["real_train_sha256"] != protocol.real_train.sha256 or receipt["feedback_sha256"] != protocol.feedback_split.sha256:
        raise ValueError("preference sampling data scope changed")
    if protocol.preference_loop.execution_version == "persistent_dual_gpu_v2":
        from .preference_sampling_merge import validate_merged_receipt
        validate_merged_receipt(receipt, protocol, parent_hash, inputs, rows(paths["preference_candidates"]))
    greedy=None
    if protocol.preference_loop.include_greedy:
        if 'preference_greedy' not in paths or receipt.get('greedy_sha256') != sha256_file(paths['preference_greedy']):
            raise ValueError('missing or changed normal decoding receipt')
        greedy=rows(paths['preference_greedy'])
    expected, _ = build_preference_pairs(inputs, rows(paths["preference_candidates"]), parent_hash,
                                         expected_samples=protocol.preference_loop.samples_per_audio,
                                         greedy_candidates=greedy)
    if protocol.preference_loop.execution_version == 'persistent_kl_dpo_v3':
        from .retention_loop import training_pairs, validate_residual_sampling
        expected = training_pairs(expected)
        validate_residual_sampling(protocol, receipt, parent_hash)
    if train_rows != expected:
        raise ValueError("registered training preferences differ from actual deduplicated errors")
    if sum(x["source"] == "real" for x in inputs) != protocol.real_replay_count_per_round:
        raise ValueError("real preference sampling count differs from declared replay selection")
    review_by_hash = {x["candidate_audio_sha256"]: x for x in reviews}
    for item in inputs:
        if item["source"] != "synthetic":
            continue
        embedded = {key: value for key, value in item["review"].items() if key not in {"audio_id", "sample_id"}}
        ledger = {key: value for key, value in review_by_hash.get(item["audio_sha256"], {}).items()
                  if key not in {"audio_id", "sample_id"}}
        if embedded != ledger:
            raise ValueError("sampled synthetic review differs from the registered review ledger")
    expected_pass_ids = set()
    for assignment in assignments:
        review = review_by_hash.get(sha256_file(Path(assignment["output_wav"])))
        verdicts = {"pass", "obvious_failure", "pending"} if protocol.preference_loop.review_allow_pending else {"pass", "obvious_failure"}
        if not review or review.get("target_text") != assignment["target_text"] or review.get("reference_audio_sha256") != assignment["reference_wav_sha256"] or review.get("reference_text") != assignment["reference_text"] or review.get("verdict") not in verdicts:
            raise ValueError("every generated assignment requires a bound coarse review")
        config = protocol.preference_loop
        if review.get("review_mode", "audio") != config.review_mode:
            raise ValueError("review mode differs from the frozen protocol")
        if config.review_mode == "asr_transcript":
            from .transcript_review import validate_transcript_review
            validate_transcript_review(review, expected_adapter_sha256=config.review_asr_adapter_sha256,
                expected_base_model_sha256=config.review_asr_base_model_sha256)
        elif review.get("audio_inspected") is not True:
            raise ValueError("every generated assignment requires a bound actual audio review")
        if review["verdict"] == "pass":
            expected_pass_ids.add(assignment["sample_id"])
    if expected_pass_ids != {x["audio_id"] for x in inputs if x["source"] == "synthetic"}:
        raise ValueError("sampled synthetic inputs differ from coarse-review admissions")


def validate_registered_preference_training(protocol, paths, training):
    """Rebuild dose from the real trainer's protocol/cache/exposure artifacts."""
    if protocol.preference_loop.execution_version == 'persistent_kl_dpo_v3':
        from .retention_loop import validate_training_artifacts
        return validate_training_artifacts(protocol, paths, training)
    import math
    from .dpo_training import SourceBalancedWindows, group_training_rows, preference_key
    required = {"preference_training_protocol", "preference_training_result", "preference_exposures", "reference_log_probs"}
    if not required <= paths.keys():
        raise ValueError("preference training lacks its real execution artifacts")
    run = read(paths["preference_training_protocol"])
    result, cache = read(paths["preference_training_result"]), read(paths["reference_log_probs"])
    config, args = protocol.preference_loop, run["arguments"]
    digest = sha256_file(paths["preference_training_protocol"])
    expected_args = {"objective": "dpo", "model_path": protocol.base_model_dir,
        "reference_adapter_path": protocol.initial_adapter.path, "max_steps": protocol.lora.optimizer_steps_per_round,
        "real_per_step": config.real_per_step, "synthetic_per_step": config.synthetic_per_step,
        "synthetic_weight": config.synthetic_weight, "beta": config.beta, "lr": protocol.lora.learning_rate,
        "seed": protocol.lora.seed}
    if any(args.get(k) != v for k, v in expected_args.items()):
        raise ValueError("executed DPO arguments differ from the frozen protocol")
    if run["reference_asr_sha256"] != protocol.initial_adapter.sha256 or run["policy_parent_sha256"] != training.parent_checkpoint_directory_sha256:
        raise ValueError("executed ASR reference or parent differs from the receipt")
    if run["training_file_sha256"] != training.train_manifest_file_sha256:
        raise ValueError("trainer used a different preference manifest")
    if cache.get("protocol_sha256") != digest or result.get("protocol_sha256") != digest:
        raise ValueError("reference cache/result differs from the executed training protocol")
    if cache.get("reference_unchanged") is not True or result.get("reference_files_unchanged") is not True or result.get("status") != "complete":
        raise ValueError("training/reference execution was not verified complete")
    if Path(result["checkpoint"]).resolve() != paths["checkpoint"].resolve():
        raise ValueError("training result refers to a different checkpoint")
    pairs = rows(paths["train_manifest"])
    groups = group_training_rows(pairs, "dpo")
    allocation=None
    if config.training_allocation=='role_budget_v1':
        from .training_allocation import compile_role_allocation
        allocation=read(paths['preference_allocation'])
        if allocation!=compile_role_allocation(read(paths['materialization_sidecar']),pairs,
                protocol.lora.optimizer_steps_per_round,config.synthetic_per_step,protocol.lora.seed):
            raise ValueError('allocation differs from actual MAS role budgets and pair availability')
        if run.get('synthetic_allocation_sha256')!=sha256_file(paths['preference_allocation']):
            raise ValueError('trainer used a different audio exposure allocation')
    windows = SourceBalancedWindows(groups, config.real_per_step, config.synthetic_per_step, config.synthetic_weight, protocol.lora.seed,
        synthetic_schedule=allocation['steps'] if allocation else None)
    expected = []
    for step in range(1, protocol.lora.optimizer_steps_per_round + 1):
        for x in windows.next():
            expected.append({"step": step, "source": x["source"], "audio_id": x["_audio_group"], "occurrence": x["_occurrence"],
                             "preference_key": preference_key(x), "loss_weight": x["_loss_weight"]})
    actual = rows(paths["preference_exposures"])
    if len(actual) != len(expected) or any(any(a.get(k) != e[k] for k in e if k != "loss_weight") or
            not math.isclose(a.get("loss_weight", -1), e["loss_weight"], rel_tol=1e-7, abs_tol=1e-12) for a, e in zip(actual, expected)):
        raise ValueError("recorded exposure sequence/weights differ from the frozen training streams")
    delivery = training.preference_delivery
    for field, key in (("reviews_sha256", "preference_reviews"), ("candidates_sha256", "preference_candidates"), ("exposure_log_sha256", "preference_exposures")):
        if getattr(delivery, field) != sha256_file(paths[key]):
            raise ValueError("delivery source hash differs from registered execution")
    unique = {(x["step"], x["source"], x["audio_id"], x["occurrence"]) for x in actual}
    exposure_counts = Counter(x[2] for x in unique)
    pair_counts = Counter(x["audio_id"] for x in pairs)
    if delivery.real_training_exposures != sum(x[1] == "real" for x in unique) or delivery.real_pair_audio_count != len(groups["real"]):
        raise ValueError("real preference delivery differs from actual exposure")
    reviews = {x["candidate_audio_sha256"]: x for x in rows(paths["preference_reviews"])}
    assignments = {x["sample_id"]: x for x in rows(paths["synthesis_manifest"])}
    for item in delivery.synthetic_samples:
        review = reviews[sha256_file(Path(assignments[item.sample_id]["output_wav"]))]
        if item.pair_count != pair_counts[item.sample_id] or item.training_exposures != exposure_counts[item.sample_id] or item.review_verdict != review["verdict"]:
            raise ValueError("synthetic preference delivery differs from actual pairs/reviews/exposures")
        if item.role is not None:
            role_by_id={x['sample_id']:x['role'] for x in read(paths['materialization_sidecar'])['samples']}
            if item.role!=role_by_id[item.sample_id]:raise ValueError('delivered role differs from proposal materialization')


def train_branch(root, number, arm):
    root = Path(root); verify_sources(root)
    p, d = RunStore(root).protocol(), round_dir(root, number)
    if arm not in enabled_arms(p):
        raise ValueError("comparison branch is not enabled in the frozen protocol")
    c = p.preference_loop
    if c.execution_version == 'persistent_kl_dpo_v3':
        from .retention_loop import train_branch as train_retention
        return train_retention(root, number, arm)
    manifest = d / arm / "train.jsonl"
    if not rows(manifest):
        save(d / arm / "no_update.json", {"status": "no_preference_pairs", "training_executed": False,
                                            "parent": str(accepted_checkpoint(root, number-1))})
        return False
    run_command([sys.executable, str(REPO / "finetune/scripts/qwen3_asr_audio_dpo.py"),
        "--objective", "dpo" if arm.endswith("dpo") else "sft", "--model_path", p.base_model_dir,
        "--policy_adapter_path", str(accepted_checkpoint(root, number-1)), "--reference_adapter_path", p.initial_adapter.path,
        "--train_file", str(manifest), "--output_dir", str(d / arm / "training"),
        "--max_steps", str(p.lora.optimizer_steps_per_round), "--batch_size", str(p.lora.per_device_batch_size),
        "--real_per_step", str(c.real_per_step), "--synthetic_per_step", str(c.synthetic_per_step),
        "--synthetic_weight", str(c.synthetic_weight), "--beta", str(c.beta), "--lr", str(p.lora.learning_rate),
        "--seed", str(p.lora.seed)] + (['--synthetic_allocation_file',str(d/'training_allocation.json')]
            if c.training_allocation=='role_budget_v1' and arm=='mas_dpo' else []), d / arm / "training.log")
    return True


def evaluate_branch(root, number, arm):
    root = Path(root); verify_sources(root)
    p, d = RunStore(root).protocol(), round_dir(root, number)
    if arm not in enabled_arms(p):
        raise ValueError("comparison branch is not enabled in the frozen protocol")
    if p.preference_loop.execution_version == 'persistent_kl_dpo_v3':
        from .retention_loop import evaluate_branch as evaluate_retention
        return evaluate_retention(root, number, arm)
    endpoint = checkpoint(root, number, arm)
    if (d / arm / "no_update.json").exists():
        endpoint = accepted_checkpoint(root, number-1)
    run_command([sys.executable, str(REPO / "finetune/scripts/evaluate_asr_checkpoint.py"),
        "--base-model-path", p.base_model_dir, "--adapter-path", str(endpoint),
        "--split-jsonl", p.feedback_split.path, "--out-dir", str(d / arm / "evaluation"),
        "--batch-size", str(p.evaluation.batch_size), "--critical-cer-threshold", str(p.evaluation.critical_cer_threshold),
        "--max-new-tokens", "256"], d / arm / "evaluation.log")


def delivery_from_artifacts(root, number):
    root = Path(root)
    p, d = RunStore(root).protocol(), round_dir(root, number)
    if p.preference_loop.execution_version == 'persistent_kl_dpo_v3':
        from .retention_loop import delivery_from_artifacts as retention_delivery
        return retention_delivery(root, number)
    audit = read(d / "sampling/preference_audit.json")
    by_id = {x["audio_id"]: x for x in audit["audio_results"]}
    reviews = {x["candidate_audio_sha256"]: x for x in rows(d / "generation/reviews.jsonl")}
    exposures = rows(d / "mas_dpo/training/training_exposures.jsonl")
    unique_exposures = {(x["step"], x["source"], x["audio_id"], x["occurrence"]) for x in exposures}
    counts = Counter(x[2] for x in unique_exposures)
    real = [x for x in by_id.values() if x["source"] == "real"]
    roles = {x['sample_id']:x['role'] for x in read(d/'materialization_sidecar.json')['samples']} if p.preference_loop.generation_design_path else {}
    generation={x['sample_id']:x for x in read(d/'generation/generation_summary.json')['outputs']} if roles else {}
    return PreferenceDelivery(current_asr_sha256=sha256_directory(accepted_checkpoint(root, number-1)),
        frozen_reference_asr_sha256=p.initial_adapter.sha256, reviews_sha256=sha256_file(d / "generation/reviews.jsonl"),
        candidates_sha256=sha256_file(d / "sampling/candidates.jsonl"), pair_manifest_sha256=sha256_file(d / "mas_dpo/train.jsonl"),
        exposure_log_sha256=sha256_file(d / "mas_dpo/training/training_exposures.jsonl"), samples_per_audio=p.preference_loop.samples_per_audio,
        real_audio_considered=len(real), real_pair_audio_count=sum(x["pair_count"] > 0 for x in real),
        real_pair_count=sum(x["pair_count"] for x in real), real_training_exposures=sum(x[1] == "real" for x in unique_exposures),
        synthetic_samples=[PreferenceSampleDelivery(sample_id=x["sample_id"],
            review_verdict=reviews[sha256_file(Path(x["output_wav"]))]["verdict"],
            pair_count=by_id.get(x["sample_id"], {}).get("pair_count", 0), training_exposures=counts[x["sample_id"]],
            **({'role':roles[x['sample_id']],
                'reference_id':generation[x['sample_id']]['generation_choice']['reference_id'],
                'generation_mode':generation[x['sample_id']]['generation_mode'],
                'instruction_id':generation[x['sample_id']]['effective_instruction_id']} if roles else {}))
            for x in rows(d / "synthesis_manifest.jsonl")])


def reassess(root, number):
    root = Path(root); verify_sources(root)
    store, p, d = RunStore(root), RunStore(root).protocol(), round_dir(root, number)
    endpoint = checkpoint(root, number)
    data = rows(d / "mas_dpo/train.jsonl")
    receipt = TrainingReceipt(schema_version="1", run_id=p.run_id, round_index=number, status="complete",
        parent_checkpoint_directory_sha256=sha256_directory(accepted_checkpoint(root, number-1)),
        new_checkpoint_directory_sha256=sha256_directory(endpoint), train_manifest_file_sha256=sha256_file(d / "mas_dpo/train.jsonl"),
        training_index_file_sha256=sha256_file(d / "mas_dpo/training_index.json"), trainer_state_file_sha256=sha256_file(endpoint / "trainer_state.json"),
        real_rows=sum(x["source"] == "real" for x in data), synthetic_rows=sum(x["source"] == "synthetic" for x in data), total_rows=len(data),
        expected_optimizer_steps=p.lora.optimizer_steps_per_round, actual_optimizer_steps=read(endpoint / "trainer_state.json")["global_step"],
        preference_delivery=delivery_from_artifacts(root, number))
    save(d / "training_receipt.json", receipt)
    if store.current().phase == "synthesized":
        extra_training = {}
        if p.preference_loop.execution_version == 'persistent_kl_dpo_v3':
            from .retention_loop import training_registration
            extra_training = training_registration(p, d)
        store.advance("trained", {"checkpoint": endpoint, "training_receipt": d / "training_receipt.json",
            "preference_training_result": d / "mas_dpo/training/training_result.json",
            "preference_training_protocol": d / "mas_dpo/training/run_protocol.json",
            "preference_exposures": d / "mas_dpo/training/training_exposures.jsonl",
            "reference_log_probs": d / "mas_dpo/training/reference_log_probs.json", **extra_training})
    evidence, proposal = load(d / "evidence.json", EvidencePack), load(d / "proposal.json", Proposal)
    plan, sidecar = load(d / "evaluation_plan.json", EvaluationPlan), load(d / "materialization_sidecar.json", MaterializationSidecar)
    before, after = accepted_predictions(root, number-1), predictions(root, number)
    result = compile_round_result(p, evidence, proposal, plan, sidecar, load(d / "synthesis_receipt.json", SynthesisReceipt), receipt,
        before_rows=load_prediction_rows(before), after_rows=load_prediction_rows(after),
        before_predictions_file_sha256=sha256_file(before), after_predictions_file_sha256=sha256_file(after))
    save(d / "round_result.json", result)
    skill = SkillMemory(root / "skills").bind_round(number)
    pack = compile_reflection_pack(p, evidence, proposal, plan, sidecar, result,
        prescription_prompt_file_sha256=sha256_file(root / "prompts/therapist.md"), reflection_prompt_file_sha256=sha256_file(root / "prompts/reflection.md"),
        skill_version=skill.version, skill_file_sha256=skill.file_sha256)
    save(d / "reflection_pack.json", pack)
    extra={}
    if p.preference_loop.checkpoint_acceptance != 'always':
        from .checkpoint_promotion import make_decision
        save(d/'promotion_decision.json',make_decision(pack,parent=accepted_checkpoint(root,number-1),
            before=before,candidate=endpoint,after=after,reference_sha256=p.initial_adapter.sha256,protocol=p))
        extra['promotion_decision']=d/'promotion_decision.json'
    store.advance("reassessed", {"after_predictions": after, "round_result": d / "round_result.json", "reflection_pack": d / "reflection_pack.json",**extra})


def reflect(root, number):
    root = Path(root); verify_sources(root)
    store, p, d = RunStore(root), RunStore(root).protocol(), round_dir(root, number)
    evidence, proposal, pack = load(d / "evidence.json", EvidencePack), load(d / "proposal.json", Proposal), load(d / "reflection_pack.json", ReflectionPack)
    skill = SkillMemory(root / "skills").bind_round(number)
    request = build_reflection_request(p, evidence, proposal, pack,
        system_prompt=(root / "prompts/reflection.md").read_text(), skill_text=skill.read_text())
    reflection = parse_reflection_response(request_once(p, request, d / "reflect"), pack)
    save(d / "reflection.json", reflection)
    if store.current().phase == "reassessed":
        store.advance("reflected", {"reflection_request": d / "reflect/request.json", "reflection_response": d / "reflect/response.json", "reflection": d / "reflection.json"})
    save(d / "ledger_entry.json", compile_intervention_summary(p, evidence, proposal, pack, reflection))
    store.advance("committed", {"ledger_entry": d / "ledger_entry.json"})
    store.verify_integrity()
