"""Two real sequential Therapist rounds and a matched real-only shadow trajectory.

Each round consumes 232 shared real + 24 new synthetic examples (shadow: 256 real),
eight new optimizer steps, and the same merged 511-utterance feedback set.
No gate, rollback, best-step selection, automatic retry, or invented reflection.
"""

# ruff: noqa: E402 -- runnable entry point adds the repository import roots.

import argparse
import getpass
import json
import os
from pathlib import Path
import random
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "finetune/scripts"), str(REPO / "examples")]

from therapist_first_real_round import PACKAGE, request_once, save
from therapist_harness.agent import (build_therapist_request, parse_therapist_response,
    build_reflection_request, parse_reflection_response)
from therapist_harness.evidence_pack import build_evidence_pack_from_files, load_prediction_rows
from therapist_harness.materialize import materialize_proposal, build_materialization_sidecar
from therapist_harness.memory import SkillMemory, compile_intervention_summary
from therapist_harness.outcome import compile_evaluation_plan, compile_round_result, compile_reflection_pack
from therapist_harness.round_schema import (EvaluationPlan, MaterializationSidecar, SynthesisReceipt,
    SynthesizedSample, TrainingReceipt, ReflectionPack)
from therapist_harness.schema import RunProtocol, EvidencePack, Proposal
from therapist_harness.store import RunStore, sha256_file, sha256_directory, write_immutable_bytes
from therapist_harness.text import identity_keys

RECORDS = Path("/data/qwen3-asr/records/therapist_mas_setting_d_v1")
PILOT = RECORDS / "therapist_m0_b8g4_pilot_v1"
SMALL = RECORDS / "m0_small_dose_pair_b8g4_lr2e5_step8_v1"
TEST = Path("/data/qwen3-asr/finetune/data_setting_d_merged_test_v1/setting_d_merged_test.jsonl")
SCOPE_NOTE = """
Run-specific measurement scope: all snapshots, evidence and outcomes use the SAME
511 real utterances (former dev 258 + test 253, 12 content groups). This run has
NO independent sealed test. This feedback set is also the reported evaluation set;
do not describe it as an untouched final test or claim independent validation.
Legacy names containing dev refer to this merged feedback scope. Training uses
232 real replay utterances sampled without replacement from the 2117-row training
pool plus exactly 24 current-round synthetic utterances, with 6 distinct texts.
Real replay rotates across rounds. Each round applies 8 new optimizer steps to its
parent adapter; optimization settings are fixed. Small or mixed responses do not
justify changing these budgets. The full 2117 real rows are NOT replayed each round.
Skill v000 stays frozen in these two rounds; actual Reflection episode memory is
passed into the next EvidencePack. This is not a Skill-Optimizer experiment.
"""


def read(path):
    return json.loads(path.read_text())


def jsonl(path, rows):
    write_immutable_bytes(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())


def load(path, cls):
    return cls.model_validate_json(path.read_text())


def directory(root, number):
    return root / f"round_{number:03d}"


def checkpoint(root, number, arm="mas"):
    if number == 0:
        return Path(RunStore(root).protocol().initial_adapter.path)
    return directory(root, number) / arm / "training/checkpoint-8"


def predictions(root, number):
    if number == 0:
        return Path(RunStore(root).protocol().initial_adapter_dev_predictions.path)
    return directory(root, number) / "mas/evaluation/merged_step_8/predictions.jsonl"


def prepare(root):
    payload = read(PILOT / "protocol.json")
    payload.update(run_id=root.name, round_count=2, real_replay_count_per_round=232,
                   sealed_test_split=None, skill_update_rounds=[])
    for field, model in (("initial_adapter_dev_predictions", "m0_step50"),
                         ("zero_shot_dev_predictions", "zero_shot")):
        path = PILOT / f"merged_test_reassessment_v1/{model}/predictions.jsonl"
        payload[field] = {"path": str(path), "sha256": sha256_file(path), "expected_rows": 511}
    payload["initial_adapter_prediction_column"] = payload["zero_shot_prediction_column"] = "prediction"
    payload["feedback_split"] = {"path": str(TEST), "sha256": sha256_file(TEST), "expected_rows": 511}
    payload["lora"].update(learning_rate=2e-5, optimizer_steps_per_round=8, scheduler="constant", warmup_ratio=0.)
    protocol = RunProtocol.model_validate(payload)
    assert (protocol.synthetic_count_per_round, protocol.unique_stimulus_count_per_round) == (24, 6)
    RunStore.initialize(protocol, root)
    SkillMemory.initialize(root / "skills", PACKAGE / "skills/v000/SKILL.md")
    for name in ("therapist.md", "reflection.md"):
        write_immutable_bytes(root / "prompts" / name,
                              ((PACKAGE / "prompts" / name).read_text() + SCOPE_NOTE).encode())
    source_paths = [Path(__file__), REPO / "examples/therapist_first_real_round.py",
                    *sorted(PACKAGE.glob("*.py")),
                    *sorted((REPO / "finetune/scripts").glob("*.py")),
                    REPO / "finetune/official/qwen3_asr_sft.py",
                    REPO / "synthesis/cosyvoice3_patient_sft/synthesize_manifest.py"]
    save(root / "sources.json", {str(p): sha256_file(p) for p in source_paths})
    save(root / "execution_policy.json", {"rounds": 2, "replay_seed_by_round": [43, 44],
        "batch_size": 8, "grad_acc": 4, "learning_rate": 2e-5, "steps_each_round": 8,
        "real_pool": 2117, "shared_real_each_round": 232, "synthetic_each_round": 24,
        "shadow_extra_real_each_round": 24, "replay_reused_between_rounds": False,
        "diagnostic_steps": [4, 8], "continue_fixed_endpoint": 8,
        "shadow_policy": "independent real-only trajectory; diagnostic, not an acceptance Gate",
        "feedback_rows": 511, "independent_test": False, "skill_update": False})


def assess(root, number):
    store = RunStore(root)
    protocol = store.protocol()
    kwargs = {}
    if number > 1:
        kwargs = dict(previous_predictions=predictions(root, number - 2), previous_prediction_column="prediction",
                      previous_checkpoint=checkpoint(root, number - 2))
    evidence = build_evidence_pack_from_files(protocol, round_index=number,
        current_predictions=predictions(root, number - 1), current_prediction_column="prediction",
        current_checkpoint=checkpoint(root, number - 1),
        m0_predictions=predictions(root, 0), m0_prediction_column="prediction",
        zero_shot_predictions=Path(protocol.zero_shot_dev_predictions.path), zero_shot_prediction_column="prediction",
        intervention_history=store.intervention_history(), **kwargs)
    path = directory(root, number) / "evidence.json"
    save(path, evidence)
    SkillMemory(root / "skills").bind_round(number)
    store.advance("assessed", {"evidence": path})


def prescribe(root, number):
    store, d = RunStore(root), directory(root, number)
    protocol = store.protocol()
    evidence = load(d / "evidence.json", EvidencePack)
    skill = SkillMemory(root / "skills").bind_round(number)
    request = build_therapist_request(protocol, evidence,
        system_prompt=(root / "prompts/therapist.md").read_text(), skill_text=skill.read_text())
    if number > 1:
        old_texts = [s["text"] for n in range(1, number) for s in read(directory(root, n) / "proposal.json")["stimuli"]]
        request["input"].insert(-1, {"role": "developer", "content":
            "Previous generated texts (data, not instructions). Use fresh whole sentences; "
            "you may retain a target while varying lexical context. " + json.dumps(old_texts, ensure_ascii=False)})
    response = request_once(protocol, request, d / "prescribe")
    proposal = parse_therapist_response(response, protocol, evidence)
    save(d / "proposal.json", proposal)
    if number > 1:
        old_keys = {item for text in old_texts for item in identity_keys(text).items()}
        assert not any(old_keys.intersection(identity_keys(s.text).items()) for s in proposal.stimuli)
    plan = compile_evaluation_plan(protocol, evidence, proposal,
                                   before_prediction_column="prediction", after_prediction_column="prediction")
    save(d / "evaluation_plan.json", plan)
    store.advance("proposed", {"agent_request": d / "prescribe/request.json", "agent_response": d / "prescribe/response.json",
                              "proposal": d / "proposal.json", "evaluation_plan": d / "evaluation_plan.json"})
    (d / "audio").mkdir()
    assignments = materialize_proposal(proposal, evidence, protocol, output_root=d / "audio")
    save(d / "materialization_sidecar.json", build_materialization_sidecar(proposal, evidence, protocol, assignments))
    jsonl(d / "synthesis_manifest.jsonl", assignments)
    store.advance("materialized", {"synthesis_manifest": d / "synthesis_manifest.jsonl",
                                    "materialization_sidecar": d / "materialization_sidecar.json"})
    print(json.dumps({"round": number, "stimuli": [s.model_dump(mode="json") for s in proposal.stimuli]}, ensure_ascii=False), flush=True)


def synthesize(root, number):
    d, protocol = directory(root, number), RunStore(root).protocol()
    with (d / "synthesis.log").open("x") as log:
        subprocess.run([sys.executable, str(REPO / "synthesis/cosyvoice3_patient_sft/synthesize_manifest.py"),
            "--manifest", str(d / "synthesis_manifest.jsonl"), "--summary", str(d / "tts_outputs.json")],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    output = read(d / "tts_outputs.json")
    assert output["status"] == "complete" and output["output_count"] == 24
    sidecar = read(d / "materialization_sidecar.json")
    receipt = SynthesisReceipt(schema_version="1", run_id=protocol.run_id, round_index=number, status="complete",
        assignments_canonical_sha256=sidecar["assignments_canonical_sha256"], requested=24, generated=24, failed=0,
        samples=[SynthesizedSample(sample_id=r["sample_id"], audio_file_sha256=r["output_wav_sha256"],
                                  duration_seconds=r["duration_seconds"]) for r in output["outputs"]])
    save(d / "synthesis_receipt.json", receipt)


def pair(root, number):
    from train_m0_small_dose_pair import stratified_sample, stats

    d, protocol = directory(root, number), RunStore(root).protocol()
    real = load_prediction_rows(Path(protocol.real_train.path))
    used = {r["utt_id"] for n in range(1, number)
            for rows in read(directory(root, n) / "real_selection.json").values() for r in rows}
    available = [r for r in real if r["utt_id"] not in used]
    rng = random.Random(42 + number)
    shared = stratified_sample(available, 232, rng)
    ids = {r["utt_id"] for r in shared}
    extra = stratified_sample([r for r in available if r["utt_id"] not in ids], 24, rng)
    synth = read(d / "tts_outputs.json")["outputs"].copy()
    for rows in (shared, extra, synth):
        rng.shuffle(rows)
    arms, indices, paired = {"real_only": [], "mas": []}, {"real_only": [], "mas": []}, []
    def append(arm, row, source):
        i = len(arms[arm])
        if source == "real":
            arms[arm].append({k: row[k] for k in ("audio", "text", "prompt")})
            indices[arm].append({"row": i, "source": source, "utt_id": row["utt_id"], "audio": row["audio"]})
        else:
            arms[arm].append({"audio": row["output_wav"], "text": "language Cantonese<asr_text>" + row["target_text"], "prompt": ""})
            indices[arm].append({"row": i, "source": source, "sample_id": row["sample_id"],
                "audio": row["output_wav"], "audio_sha256": row["output_wav_sha256"]})
    for step in range(8):
        replacements = set(rng.sample(range(32), 3))
        common, additional, generated = iter(shared[step*29:(step+1)*29]), iter(extra[step*3:(step+1)*3]), iter(synth[step*3:(step+1)*3])
        for position in range(32):
            if position in replacements:
                append("real_only", next(additional), "real")
                append("mas", next(generated), "synthetic")
            else:
                row = next(common)
                for arm in arms:
                    append(arm, row, "real")
            paired.append({"row": len(paired), "step": step+1, "shared": position not in replacements})
    test_rows = load_prediction_rows(TEST)
    for field in ("utt_id", "prompt_id"):
        assert not {r[field] for r in shared+extra} & {r[field] for r in test_rows}
    assert len({r["utt_id"] for r in shared+extra}) == 256
    assert sum(a != b for a, b in zip(arms["mas"], arms["real_only"])) == 24
    save(d / "real_selection.json", {"shared": shared, "extra": extra})
    save(d / "paired_index.json", paired)
    template = read(SMALL / "run_config.json")["commands"]["plus24"]
    commands, parents = {}, {}
    for arm, rows in arms.items():
        jsonl(d / arm / "train.jsonl", rows)
        save(d / arm / "training_index.json", indices[arm])
        parent = checkpoint(root, number-1, arm)
        parents[arm] = {"path": str(parent), "sha256": sha256_directory(parent)}
        command = template.copy()
        for key, value in {"--train_file": str(d / arm / "train.jsonl"), "--output_dir": str(d / arm / "training"),
                           "--lora_init_adapter_path": str(parent)}.items():
            command[command.index(key)+1] = value
        commands[arm] = command
    save(d / "pair_config.json", {"parents": parents, "commands": commands, "shared_real": stats(shared),
        "extra_real": stats(extra), "seed": 42+number, "train_sha256": {a: sha256_file(d/a/"train.jsonl") for a in arms},
        "audio_sha256": {r["output_wav"]: r["output_wav_sha256"] for r in synth}})
    RunStore(root).advance("synthesized", {"synthesis_summary": d / "synthesis_receipt.json",
        "train_manifest": d / "mas/train.jsonl", "training_index": d / "mas/training_index.json",
        "paired_execution": d / "pair_config.json", "replay_selection": d / "real_selection.json",
        "paired_order": d / "paired_index.json"})


def train(root, number, arm):
    import qwen3_asr_lora_sft as training
    from torch.utils.data import SequentialSampler
    from train_m0_small_dose_pair import Monitor

    d = directory(root, number)
    config = read(d / "pair_config.json")
    for path, digest in read(root / "sources.json").items():
        assert sha256_file(Path(path)) == digest, path
    assert sha256_directory(Path(config["parents"][arm]["path"])) == config["parents"][arm]["sha256"]
    assert sha256_file(d / arm / "train.jsonl") == config["train_sha256"][arm]
    for path, digest in config["audio_sha256"].items():
        assert sha256_file(Path(path)) == digest
    monitor = Monitor(d / arm)
    monitor.publish()
    class PairedTrainer(training.QwenASRLoRATrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(monitor)

        def _get_train_sampler(self, train_dataset=None):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            assert len(dataset) == 256
            return SequentialSampler(dataset)
    training.QwenASRLoRATrainer = PairedTrainer
    sys.argv = config["commands"][arm][1:]
    training.main()
    endpoint = d / arm / "training/checkpoint-8"
    state = read(endpoint / "trainer_state.json")
    assert state["global_step"] == 8 and state["epoch"] == 1.
    assert [r["step"] for r in monitor.history] == [4, 8]
    assert all(abs(r["learning_rate"]-2e-5) < 1e-12 for r in state["log_history"] if "learning_rate" in r)
    save(d / arm / "run_summary.json", {"status": "complete", "history": monitor.history,
        "endpoint_sha256": sha256_directory(endpoint), "train_log": state["log_history"]})
    monitor.state.update(status="complete")
    monitor.publish()


def reassess(root, number):
    store, d = RunStore(root), directory(root, number)
    protocol = store.protocol()
    parent, endpoint = checkpoint(root, number-1), checkpoint(root, number)
    receipt = TrainingReceipt(schema_version="1", run_id=protocol.run_id, round_index=number, status="complete",
        parent_checkpoint_directory_sha256=sha256_directory(parent), new_checkpoint_directory_sha256=sha256_directory(endpoint),
        train_manifest_file_sha256=sha256_file(d / "mas/train.jsonl"), training_index_file_sha256=sha256_file(d / "mas/training_index.json"),
        trainer_state_file_sha256=sha256_file(endpoint / "trainer_state.json"), real_rows=232, synthetic_rows=24, total_rows=256,
        expected_optimizer_steps=8, actual_optimizer_steps=read(endpoint / "trainer_state.json")["global_step"])
    save(d / "training_receipt.json", receipt)
    store.advance("trained", {"checkpoint": endpoint, "training_receipt": d / "training_receipt.json",
        "real_only_checkpoint": checkpoint(root, number, "real_only"), "real_only_summary": d / "real_only/run_summary.json"})
    evidence, proposal = load(d / "evidence.json", EvidencePack), load(d / "proposal.json", Proposal)
    plan, sidecar = load(d / "evaluation_plan.json", EvaluationPlan), load(d / "materialization_sidecar.json", MaterializationSidecar)
    before, after = predictions(root, number-1), predictions(root, number)
    result = compile_round_result(protocol, evidence, proposal, plan, sidecar,
        load(d / "synthesis_receipt.json", SynthesisReceipt), receipt,
        before_rows=load_prediction_rows(before), after_rows=load_prediction_rows(after),
        before_predictions_file_sha256=sha256_file(before), after_predictions_file_sha256=sha256_file(after))
    save(d / "round_result.json", result)
    skill = SkillMemory(root / "skills").bind_round(number)
    pack = compile_reflection_pack(protocol, evidence, proposal, plan, sidecar, result,
        prescription_prompt_file_sha256=sha256_file(root / "prompts/therapist.md"),
        reflection_prompt_file_sha256=sha256_file(root / "prompts/reflection.md"),
        skill_version=skill.version, skill_file_sha256=skill.file_sha256)
    save(d / "reflection_pack.json", pack)
    store.advance("reassessed", {"after_predictions": after, "round_result": d / "round_result.json",
                                 "reflection_pack": d / "reflection_pack.json"})


def reflect(root, number):
    store, d = RunStore(root), directory(root, number)
    protocol = store.protocol()
    evidence, proposal = load(d / "evidence.json", EvidencePack), load(d / "proposal.json", Proposal)
    pack = load(d / "reflection_pack.json", ReflectionPack)
    skill = SkillMemory(root / "skills").bind_round(number)
    request = build_reflection_request(protocol, evidence, proposal, pack,
        system_prompt=(root / "prompts/reflection.md").read_text(), skill_text=skill.read_text())
    reflection = parse_reflection_response(request_once(protocol, request, d / "reflect"), pack)
    save(d / "reflection.json", reflection)
    store.advance("reflected", {"reflection_request": d / "reflect/request.json", "reflection_response": d / "reflect/response.json",
                                "reflection": d / "reflection.json"})
    save(d / "ledger_entry.json", compile_intervention_summary(protocol, evidence, proposal, pack, reflection))
    store.advance("committed", {"ledger_entry": d / "ledger_entry.json"})
    store.verify_integrity()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stage", choices=("run", "prepare", "assess", "prescribe", "synthesize", "pair", "train", "reassess", "reflect"), default="run")
    parser.add_argument("--round", type=int, choices=(1, 2), default=1)
    parser.add_argument("--arm", choices=("mas", "real_only"))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage in ("run", "prescribe", "reflect") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Gateway API key (not saved): ")
    if args.stage == "prepare":
        prepare(root)
    elif args.stage == "train":
        assert args.arm is not None
        train(root, args.round, args.arm)
    elif args.stage != "run":
        globals()[args.stage](root, args.round)
    else:
        prepare(root)
        for number in (1, 2):
            for stage in (assess, prescribe, synthesize, pair):
                print(f"[loop] round={number} phase={stage.__name__}", flush=True)
                stage(root, number)
            for arm in ("real_only", "mas"):
                print(f"[loop] round={number} training={arm}", flush=True)
                with (directory(root, number) / f"{arm}.log").open("x") as log:
                    subprocess.run([sys.executable, str(Path(__file__)), str(root), "--stage", "train",
                        "--round", str(number), "--arm", arm], stdout=log, stderr=subprocess.STDOUT, check=True)
            for stage in (reassess, reflect):
                print(f"[loop] round={number} phase={stage.__name__}", flush=True)
                stage(root, number)
        save(root / "run_summary.json", {"status": "two_rounds_committed", "rounds": [
            {"round": n, "arms": {a: read(directory(root,n)/a/"run_summary.json") for a in ("real_only", "mas")},
             "ledger": read(directory(root,n)/"ledger_entry.json")} for n in (1,2)]})
        print("[loop] both rounds committed with actual Reflection", flush=True)


if __name__ == "__main__":
    main()
