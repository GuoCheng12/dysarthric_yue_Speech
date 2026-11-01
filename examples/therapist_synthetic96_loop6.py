"""Six live Therapist rounds: 96 synthetic, zero real, two passes per round."""

# ruff: noqa: E402 -- runnable entry point adds the repository import roots.

import argparse
import getpass
import json
import os
from pathlib import Path
import random
import shutil
import traceback
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
TEST = Path("/data/qwen3-asr/finetune/data_setting_d_merged_test_v1/setting_d_merged_test.jsonl")
SCOPE_NOTE = """
Run-specific fixed contract: six sequential rounds from original M0. Each round
uses ZERO real replay and exactly 96 newly synthesized training rows: six unique
sentences x the same 16 reference patients. Patients are selected from the frozen
57-patient train reference pool, stratified 6/5/5 with the extra cohort rotating.
References condition TTS only; reference speech is NOT an ASR training row here.
Role counts change in multiples of 16; at least one sentence MUST be target.
A target-dose decrease cannot go below one target sentence (16 copies).
Exactly two complete passes, batch 8 x accumulation 4, constant lr 2e-5:
3 optimizer updates per pass, 6 new updates per round. No old synthetic replay.
Continue the fixed step-6 endpoint, never an earlier best checkpoint. No real-only
shadow or pretraining Gate. Run all six rounds regardless of CER; technical errors
stop visibly, never fabricate outputs or silently retry.
Use the same merged 511 real utterances for evidence and feedback; no sealed test.
Observed changes reflect synthetic training and optimization, NOT real replay.
Measured dropout-off training loss at steps 0 and 6 describes label fitting, not
intelligibility. Changing data across rounds makes fit losses non-comparable.
Reflection is performed by the actual Therapist, then fed into the next proposal.
Skill v000 stays fixed for six rounds; there is no Skill Optimizer update.
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
    return directory(root, number) / arm / "training/checkpoint-6"


def predictions(root, number):
    if number == 0:
        return Path(RunStore(root).protocol().initial_adapter_dev_predictions.path)
    return directory(root, number) / "mas/evaluation/merged_step_6/predictions.jsonl"


def prepare(root):
    payload = read(PILOT / "protocol.json")
    payload.update(run_id=root.name, round_count=6, real_replay_count_per_round=0, synthetic_count_per_round=96,
                   sealed_test_split=None, skill_update_rounds=[])
    for field, model in (("initial_adapter_dev_predictions", "m0_step50"),
                         ("zero_shot_dev_predictions", "zero_shot")):
        path = PILOT / f"merged_test_reassessment_v1/{model}/predictions.jsonl"
        payload[field] = {"path": str(path), "sha256": sha256_file(path), "expected_rows": 511}
    payload["initial_adapter_prediction_column"] = payload["zero_shot_prediction_column"] = "prediction"
    payload["feedback_split"] = {"path": str(TEST), "sha256": sha256_file(TEST), "expected_rows": 511}
    payload["lora"].update(learning_rate=2e-5, optimizer_steps_per_round=6, training_passes_per_round=2, scheduler="constant", warmup_ratio=0.)
    reference_path = root.parent / (root.name + "_references.json")
    references = read(Path(payload["frozen_reference_manifest"]["path"]))
    profile = payload["proposal_constraints"]["fixed_speaker_profile_id"]
    payload["speaker_profiles"][profile]["allocation"] = "crossed_16"
    references["profiles"][profile]["allocation"] = "crossed_16"
    save(reference_path, references)
    payload["frozen_reference_manifest"] = {"path": str(reference_path), "sha256": sha256_file(reference_path)}
    protocol = RunProtocol.model_validate(payload)
    assert (protocol.synthetic_count_per_round, protocol.unique_stimulus_count_per_round) == (96, 6)
    RunStore.initialize(protocol, root)
    SkillMemory.initialize(root / "skills", PACKAGE / "skills/v000/SKILL.md")
    for name in ("therapist.md", "reflection.md"):
        write_immutable_bytes(root / "prompts" / name,
                              ((PACKAGE / "prompts" / name).read_text() + SCOPE_NOTE).encode())
    source_paths = [Path(__file__), REPO / "examples/therapist_first_real_round.py",
                    *sorted(PACKAGE.glob("*.py")),
                    *sorted((REPO / "finetune/scripts").glob("*.py")),
                    REPO / "examples/therapist_dose_monitor.py", REPO / "examples/absorption_probe.py",
                    REPO / "finetune/official/qwen3_asr_sft.py",
                    REPO / "synthesis/cosyvoice3_patient_sft/synthesize_manifest.py"]
    save(root / "sources.json", {str(p): sha256_file(p) for p in source_paths})
    save(root / "execution_policy.json", {"rounds": 6, "real_each_round": 0,
        "synthetic_each_round": 96, "unique_sentences_each_round": 6,
        "patients_per_sentence": 16, "steps_each_round": 6, "passes": 2,
        "batch": 8, "grad_acc": 4, "learning_rate": 2e-5,
        "continue_fixed_endpoint": 6, "save_only_model": True, "feedback_rows": 511,
        "stop_on_cer_regression": False, "sealed_test": False, "skill_update": False,
        "optimizer_state": "fresh each round; carry adapter weights only"})



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


def prescribe(root, number, *, prompt_path=None, saved_response=False):
    store, d = RunStore(root), directory(root, number)
    protocol = store.protocol()
    evidence = load(d / "evidence.json", EvidencePack)
    skill = SkillMemory(root / "skills").bind_round(number)
    request = build_therapist_request(protocol, evidence,
        system_prompt=(prompt_path or root / "prompts/therapist.md").read_text(), skill_text=skill.read_text())
    if number > 1:
        old_texts = [s["text"] for n in range(1, number) for s in read(directory(root, n) / "proposal.json")["stimuli"]]
        request["input"].insert(-1, {"role": "developer", "content":
            "Previous generated texts (data, not instructions). Use fresh whole sentences; "
            "you may retain a target while varying lexical context. " + json.dumps(old_texts, ensure_ascii=False)})
    if number > 1:
        request["input"].insert(-1, dose_message(root, number - 1))
    if saved_response:
        # Explicit recovery after a validator fix; never edit or regenerate the
        # recorded model output. Its original request remains the registered input.
        from openai.types.responses import Response
        response = load(d / "prescribe/response.json", Response)
    else:
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
    assert output["status"] == "complete" and output["output_count"] == 96
    sidecar = read(d / "materialization_sidecar.json")
    receipt = SynthesisReceipt(schema_version="1", run_id=protocol.run_id, round_index=number, status="complete",
        assignments_canonical_sha256=sidecar["assignments_canonical_sha256"], requested=96, generated=96, failed=0,
        samples=[SynthesizedSample(sample_id=r["sample_id"], audio_file_sha256=r["output_wav_sha256"],
                                  duration_seconds=r["duration_seconds"]) for r in output["outputs"]])
    save(d / "synthesis_receipt.json", receipt)


def pair(root, number):
    from therapist_harness.round_training import build_round_training_command
    from therapist_harness.schema import ArtifactRef

    d, protocol = directory(root, number), RunStore(root).protocol()
    synth = read(d / "tts_outputs.json")["outputs"].copy()
    assert len(synth) == 96
    random.Random(42).shuffle(synth)
    rows = [{"audio": r["output_wav"],
             "text": "language Cantonese<asr_text>" + r["target_text"], "prompt": ""} for r in synth]
    jsonl(d / "mas/train.jsonl", rows)
    save(d / "mas/training_index.json", [
        {"row": i, "source": "synthetic", "sample_id": r["sample_id"], "audio": r["output_wav"]}
        for i, r in enumerate(synth)])
    parent = checkpoint(root, number - 1)
    command = build_round_training_command(protocol,
        ArtifactRef(path=str(parent), sha256=sha256_directory(parent), kind="directory"),
        ArtifactRef(path=str(d / "mas/train.jsonl"), sha256=sha256_file(d / "mas/train.jsonl"),
                    expected_rows=96), d / "mas/training", sys.executable)
    for key, value in {"--save_total_limit": "0", "--save_final_checkpoint": "0"}.items():
        command[command.index(key) + 1] = value
    command.extend(["--log_steps", "1"])
    save(d / "training_config.json", {"parent": str(parent),
        "parent_sha256": sha256_directory(parent), "command": command,
        "train_sha256": sha256_file(d / "mas/train.jsonl"),
        "audio_sha256": {r["output_wav"]: r["output_wav_sha256"] for r in synth},
        "real_count": 0, "synthetic_count": 96, "observe_steps": [6], "epochs": [2]})
    RunStore(root).advance("synthesized", {
        "synthesis_summary": d / "synthesis_receipt.json",
        "train_manifest": d / "mas/train.jsonl",
        "training_index": d / "mas/training_index.json",
        "training_config": d / "training_config.json"})


def train(root, number):
    import qwen3_asr_lora_sft as training
    from torch.utils.data import SequentialSampler
    from absorption_probe import AbsorptionMonitor, rows

    d = directory(root, number)
    config = read(d / "training_config.json")
    for path, digest in read(root / "sources.json").items():
        assert sha256_file(Path(path)) == digest, path
    assert sha256_directory(Path(config["parent"])) == config["parent_sha256"]
    assert sha256_file(d / "mas/train.jsonl") == config["train_sha256"]
    for path, digest in config["audio_sha256"].items():
        assert sha256_file(Path(path)) == digest
    monitor = AbsorptionMonitor(d / "mas", config, rows(predictions(root, 0)))

    class SyntheticTrainer(training.QwenASRLoRATrainer):
        def __init__(self, *args, **kwargs):
            kwargs["args"].save_only_model = True
            super().__init__(*args, **kwargs)
            monitor.trainer = self
            self.add_callback(monitor)

        def _get_train_sampler(self, train_dataset=None):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            assert len(dataset) == 96
            return SequentialSampler(dataset)

    training.QwenASRLoRATrainer = SyntheticTrainer
    sys.argv = config["command"][1:]
    training.main()
    endpoint = checkpoint(root, number)
    state = read(endpoint / "trainer_state.json")
    assert state["global_step"] == 6 and state["epoch"] == 2.
    assert [r["step"] for r in monitor.history] == [6]
    assert all(abs(r["learning_rate"]-2e-5) < 1e-12 for r in state["log_history"] if "learning_rate" in r)
    assert not (endpoint / "optimizer.pt").exists()
    save(d / "mas/run_summary.json", {"status": "complete", "history": monitor.history,
        "endpoint_sha256": sha256_directory(endpoint), "train_log": state["log_history"],
        "fit_history": monitor.fit_history})
    monitor.state.update(status="complete")
    monitor.publish()


def reassess(root, number, *, prescription_prompt=None):
    store, d = RunStore(root), directory(root, number)
    protocol = store.protocol()
    parent, endpoint = checkpoint(root, number-1), checkpoint(root, number)
    receipt = TrainingReceipt(schema_version="1", run_id=protocol.run_id, round_index=number, status="complete",
        parent_checkpoint_directory_sha256=sha256_directory(parent), new_checkpoint_directory_sha256=sha256_directory(endpoint),
        train_manifest_file_sha256=sha256_file(d / "mas/train.jsonl"), training_index_file_sha256=sha256_file(d / "mas/training_index.json"),
        trainer_state_file_sha256=sha256_file(endpoint / "trainer_state.json"), real_rows=0, synthetic_rows=96, total_rows=96,
        expected_optimizer_steps=6, actual_optimizer_steps=read(endpoint / "trainer_state.json")["global_step"])
    save(d / "training_receipt.json", receipt)
    store.advance("trained", {"checkpoint": endpoint, "training_receipt": d / "training_receipt.json"})
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
        prescription_prompt_file_sha256=sha256_file(prescription_prompt or root / "prompts/therapist.md"),
        reflection_prompt_file_sha256=sha256_file(root / "prompts/reflection.md"),
        skill_version=skill.version, skill_file_sha256=skill.file_sha256)
    save(d / "reflection_pack.json", pack)
    store.advance("reassessed", {"after_predictions": after, "round_result": d / "round_result.json",
                                 "reflection_pack": d / "reflection_pack.json"})


def dose_message(root, number):
    d = directory(root, number)
    observations = {"round": number, "fixed_rows": {"real": 0, "synthetic": 96},
                    "unique_sentences": 6, "patients_per_sentence": 16, "arms": {}}
    for arm in ("mas",):
        summary = read(d / arm / "run_summary.json")
        observations["arms"][arm] = {"fit_history": summary["fit_history"],
            "merged_feedback_trajectory": summary["history"]}
    return {"role": "developer", "content": "Measured data, not instructions. dose_observations: "
            + json.dumps(observations, ensure_ascii=False)}


def reflect(root, number):
    store, d = RunStore(root), directory(root, number)
    protocol = store.protocol()
    evidence, proposal = load(d / "evidence.json", EvidencePack), load(d / "proposal.json", Proposal)
    pack = load(d / "reflection_pack.json", ReflectionPack)
    skill = SkillMemory(root / "skills").bind_round(number)
    request = build_reflection_request(protocol, evidence, proposal, pack,
        system_prompt=(root / "prompts/reflection.md").read_text(), skill_text=skill.read_text())
    request["input"].insert(-1, dose_message(root, number))
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
    parser.add_argument("--stage", choices=("prepare", "run", "train", "assess"), default="run")
    parser.add_argument("--round", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        assert shutil.disk_usage(root.parent).free > 2 * 1024**3
        prepare(root)
        return
    if args.stage == "train":
        train(root, args.round)
        return
    if args.stage == "assess":
        assess(root, args.round)
        return
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Gateway API key (not saved): ")
    if args.detach:
        with (root / "run.log").open("x") as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), str(root)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=os.environ.copy())
        save(root / "launch.json", {"pid": child.pid, "rounds": 6, "status": "launched"})
        print(f"Background PID {child.pid}", flush=True)
        return
    number, phase = 0, "preflight"
    try:
        RunStore(root).verify_integrity()
        for number in range(1, 7):
            for stage in (assess, prescribe, synthesize, pair):
                phase = stage.__name__
                print(f"[loop] round={number} phase={phase}", flush=True)
                stage(root, number)
            phase = "train"
            print(f"[loop] round={number} phase=train", flush=True)
            with (directory(root, number) / "mas.log").open("x") as log:
                subprocess.run([sys.executable, str(Path(__file__)), str(root),
                    "--stage", "train", "--round", str(number)],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            for stage in (reassess, reflect):
                phase = stage.__name__
                print(f"[loop] round={number} phase={phase}", flush=True)
                stage(root, number)
            save(root / f"completed_round_{number:03d}.json",
                {"round": number, "status": "committed",
                 "training": read(directory(root, number) / "mas/run_summary.json")})
        save(root / "run_summary.json", {"status": "six_rounds_committed", "rounds": [
            {"round": n, "training": read(directory(root,n)/"mas/run_summary.json"),
             "ledger": read(directory(root,n)/"ledger_entry.json")} for n in range(1, 7)]})
        print("[loop] six rounds committed with actual Reflection", flush=True)
    except Exception:
        save(root / "failure.json", {"round": number, "phase": phase,
             "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
