"""Prepare or explicitly launch the three-arm persistent real curriculum experiment."""

# ruff: noqa: E402
import argparse
from collections import Counter
import csv
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "finetune/scripts")]

from openai.lib._pydantic import to_strict_json_schema
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.curriculum_v2 import (ENDPOINTS, CourseProposal, CourseReflection,
    allocation_actions, compact_catalog, compile_block, learning_catalog, make_coverage,
    outcome_pack, patient_progress, register_proposal, summarize_pair, validate_reflection)
from therapist_harness.real_curriculum import catalog
from therapist_harness.schema import AgentConfig
from therapist_harness.store import sha256_file, write_immutable_bytes, write_immutable_json as save

RECORDS = Path("/data/qwen3-asr/records/therapist_mas_setting_d_v1")
ROOT = RECORDS / "real_curriculum_v2"
TRAIN = Path("/data/qwen3-asr/finetune/data_setting_d_v1/setting_d_train.jsonl")
FEEDBACK = Path("/data/qwen3-asr/finetune/data_setting_d_merged_test_v1/setting_d_merged_test.jsonl")
ZERO = Path("/data/qwen3-asr/inference/outputs/setting_d_seen_patient_unseen_content_v1_qc_frozen")
BASE = Path("/data/qwen3-asr/models/Qwen3-ASR-1.7B")
PROMPTS = REPO / "src/therapist_harness/prompts"
ARMS = ("static", "random", "dynamic")


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save_rows(path, data):
    write_immutable_bytes(path, "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in data).encode())


def source_files():
    return [Path(__file__), REPO / "src/therapist_harness/curriculum_v2.py",
            REPO / "src/therapist_harness/curriculum_v2_training.py",
            *[REPO / "src/therapist_harness" / f"{name}.py" for name in
              ("agent", "schema", "real_curriculum", "phonology", "text", "evidence", "store")],
            *[PROMPTS / f"curriculum_v2_{name}.md" for name in ("prescribe", "reflect")],
            REPO / "finetune/scripts/qwen3_asr_lora_sft.py",
            REPO / "finetune/official/qwen3_asr_sft.py",
            REPO / "finetune/scripts/evaluate_asr_checkpoint.py",
            REPO / "docs/real_curriculum_v2.zh.md"]


def anonymize(data, local_map, feedback=False):
    patients = {r["speaker_id"]: r["patient"] for r in local_map.values()}
    recording_ids = {r["utt_id"]: key for key, r in local_map.items()}
    result = {}
    for n, row in enumerate(sorted(data, key=lambda r: r["utt_id"]), 1):
        key = f"F{n:04d}" if feedback else recording_ids[row["utt_id"]]
        patient = patients[row["speaker_id"]]
        result[key] = {"utt_id": key, "speaker_id": patient, "patient": patient,
                       "clean_gt": row["clean_gt"], "prediction": row["prediction"],
                       "zero_shot_bucket": row["zero_shot_bucket"]}
    return result


def prepare(root, *, content_centered=False):
    assert not root.exists(), "use a new experiment directory; old artifacts are immutable"
    train, feedback = rows(TRAIN), rows(FEEDBACK)
    assert sha256_file(TRAIN) == "6d232f763e065f2bacea234a483948ec790a16354cfb5d76732453c1bfd31be8"
    assert sha256_file(FEEDBACK) == "189d749e7e65f14a0440d7f6707a0e1dcfa1a5a62c6c0e5f362b08c1c53e2f98"
    assert len(train) == 2117 and len(feedback) == 511
    assert len({r["speaker_id"] for r in train}) == 66
    assert len({r["clean_gt"] for r in train}) == 48
    for field in ("utt_id", "prompt_id", "prompt_normalized_sha16", "prompt_jyutping_sha16", "audio"):
        assert not {r[field] for r in train} & {r[field] for r in feedback}, field
    assert {r["speaker_id"] for r in feedback} <= {r["speaker_id"] for r in train}
    zero = {}
    for split in ("train", "dev", "test"):
        with (ZERO / f"setting_d_{split}.csv").open() as f:
            for r in csv.DictReader(f):
                assert r["utt_id"] not in zero
                zero[r["utt_id"]] = r
    assert set(zero) == {r["utt_id"] for r in train+feedback}
    snapshots = {}
    for name, data in (("train", train), ("feedback", feedback)):
        for r in data:
            assert zero[r["utt_id"]]["clean_gt"] == r["clean_gt"]
        snapshots[name] = [{**r, "prediction": zero[r["utt_id"]]["raw_pred"]} for r in data]
    cat, mapping = catalog(snapshots["train"], {r["utt_id"]: zero[r["utt_id"]] for r in train})
    entries = {r["id"]: r for r in cat["recordings"]}
    local_map = {key: {**r, "patient": entries[key]["patient"], "content": entries[key]["content"]}
                 for key, r in mapping.items()}
    coverage = make_coverage(list(local_map))
    assert set(i for block in coverage.values() for i in block) == set(local_map)
    # No base-model or audio copies: only manifests, hashes, predictions and adapters.
    model_files = [p for p in BASE.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors", ".txt")]
    assert any(p.suffix == ".safetensors" for p in model_files)
    protocol = {"version": 2, "setting": "D", "train_rows": 2117, "feedback_rows": 511,
                "train_patients": 66, "train_contents": 48, "synthetic": 0,
                "initialization": "original Qwen3-ASR, fresh LoRA; never load M0",
                "base_model": str(BASE), "endpoints": list(ENDPOINTS), "max_steps": 100,
                "microbatch": 8, "gradient_accumulation": 4, "peak_lr": 2e-4,
                "warmup_steps": 2, "scheduler": "linear over 100 global steps",
                "rank": 16, "alpha": 4, "dropout": .05, "seed": 42,
                "coverage_per_update_after20": 20, "course_per_update_after20": 12,
                "max_exposures_per_recording_per_block": 2,
                "agent": read(RECORDS / "real_curriculum_pilot_v1/protocol.json")["agent"],
                "arms_execution_order": list(ARMS), "api_retries": 0,
                "evaluation_policy": "every arm evaluates its own checkpoint at every endpoint; no cross-arm state/loss/prediction reuse",
                "feedback_scope": "existing merged 511, exploratory; never used as training rows",
                "selection": "report all shared endpoints, fixed step100, and lowest mean-utterance CER on the same endpoints per arm",
                "failure_policy": "stop on error; no automatic repair, retry, extra seed or training extension",
                "source_sha256": {str(p): sha256_file(p) for p in source_files()},
                "data_sha256": {str(p): sha256_file(p) for p in [TRAIN, FEEDBACK, *[ZERO / f"setting_d_{s}.csv" for s in ("train", "dev", "test")]]},
                "model_sha256": {str(p): sha256_file(p) for p in model_files},
                "audio_sha256": {r["audio"]: sha256_file(Path(r["audio"])) for r in train+feedback},
                "free_bytes_preparation": shutil.disk_usage(root.parent).free}
    if content_centered:
        content_sources = [REPO / "examples/content_curriculum_run.py",
            REPO / "examples/prepare_content_curriculum.py",
            REPO / "src/therapist_harness/content_curriculum.py",
            *[PROMPTS / f"content_curriculum_{kind}.md" for kind in ("prescribe", "reflect")]]
        protocol.update(course_design="content_centered", arms_execution_order=["dynamic"],
            first_prescription="new diagnosis from this run's own step20; never reuse smoke prescription")
        protocol["source_sha256"].update({str(p): sha256_file(p) for p in content_sources})
    root.mkdir()
    save(root / "local_map.json", local_map)
    save(root / "coverage.json", coverage)
    save(root / "zero_train.json", {r["utt_id"]: zero[r["utt_id"]] for r in train})
    for name, snapshot in snapshots.items():
        save_rows(root / f"zero_{name}_predictions.jsonl", snapshot)
    save_rows(root / "pool.jsonl", [local_map[i] for i in sorted(local_map)])
    protocol["prepared_artifact_sha256"] = {str(root / name): sha256_file(root / name) for name in
        ("local_map.json", "coverage.json", "zero_train.json", "zero_train_predictions.jsonl", "zero_feedback_predictions.jsonl", "pool.jsonl")}
    save(root / "protocol.json", protocol)
    save(root / "prepared.json", {"status": "prepared_not_started", "training_started": False,
         "shared_coverage_exposures": 2240, "unique_coverage": len(set(i for b in coverage.values() for i in b)),
         "per_arm_total_exposures": 3200, "formal_agent_calls": 0})
    print(json.dumps(read(root / "prepared.json")), flush=True)


def verify_frozen(root):
    cfg = read(root / "protocol.json")
    for group in ("source_sha256", "data_sha256", "model_sha256", "audio_sha256", "prepared_artifact_sha256"):
        for name, digest in cfg[group].items():
            assert sha256_file(Path(name)) == digest, name
    return cfg


def build_request(config, payload, schema, kind):
    def display_precision(value):
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, dict):
            return {k: display_precision(v) for k, v in value.items()}
        if isinstance(value, list):
            return [display_precision(v) for v in value]
        return value
    content = json.dumps(display_precision(payload), ensure_ascii=False, separators=(",", ":"))
    assert "/data/" not in content and "vlink" not in content.lower() and '"speaker_id"' not in content
    return {"model": config["model"], "reasoning": {"effort": config["reasoning_effort"]},
            "store": False, "stream": True, "max_output_tokens": config["max_output_tokens"],
            "input": [{"role": "system", "content": (PROMPTS / f"curriculum_v2_{kind}.md").read_text()},
                      {"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": schema.__name__, "strict": True,
                                "schema": to_strict_json_schema(schema)}}}


def call_agent(root, destination, payload, schema, kind):
    config = AgentConfig.model_validate(read(root / "protocol.json")["agent"])
    request = build_request(config.model_dump(mode="json"), payload, schema, kind)
    save(destination / "request.json", request)
    started = time.monotonic()
    response = execute_therapist_request(SimpleNamespace(agent=config), request)
    save(destination / "response.json", serialize_response(response))
    save(destination / "timing.json", {"seconds": time.monotonic()-started,
         "usage": response.usage.model_dump(mode="json") if response.usage is not None else None})
    assert response.status == "completed", response.status
    parsed = schema.model_validate_json(response.output_text)
    save(destination / "parsed.json", parsed.model_dump(mode="json"))
    return parsed


def decode(checkpoint, source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(REPO / "finetune/scripts/evaluate_asr_checkpoint.py"),
               "--base-model-path", str(BASE), "--adapter-path", str(checkpoint),
               "--split-jsonl", str(source), "--out-dir", str(destination), "--batch-size", "8",
               "--critical-cer-threshold", ".5", "--max-new-tokens", "256"]
    with destination.with_suffix(".log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return rows(destination / "predictions.jsonl")


def training_arguments(root, arm):
    return ["curriculum-v2", "--model_path", str(BASE), "--train_file", str(root / "pool.jsonl"),
            "--output_dir", str(root / arm / "training"), "--eval_file", "",
            "--batch_size", "8", "--grad_acc", "4", "--lr", "2e-4", "--max_steps", "100",
            "--lr_scheduler_type", "linear", "--warmup_ratio", ".02", "--seed", "42",
            "--lora_rank", "16", "--lora_alpha", "4", "--lora_dropout", ".05",
            "--lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            "--num_workers", "0", "--persistent_workers", "0", "--log_steps", "1",
            "--save_steps", "20", "--save_total_limit", "0", "--save_final_checkpoint", "0"]


def train_arm(root, arm):
    import torch
    from peft import get_peft_model_state_dict
    from transformers import TrainerCallback
    from finetune.scripts import qwen3_asr_lora_sft as training
    from therapist_harness.curriculum_v2_training import (BlockDataset, ContinuousCourseTrainer,
        measure_losses, preserve_rng, state_digest)

    cfg = verify_frozen(root)
    destination = root / arm
    destination.mkdir(exist_ok=False)
    save(destination / "launch.json", {"argv": training_arguments(root, arm), "mode": "formal_training"})
    mapping, zero = read(root / "local_map.json"), read(root / "zero_train.json")
    coverage = {int(k): v for k, v in read(root / "coverage.json").items()}
    base_entries = catalog(rows(root / "zero_train_predictions.jsonl"), zero)[0]["recordings"]

    class Monitor(TrainerCallback):
        def __init__(self):
            self.manifests, self.episodes, self.metric_history = [], [], []
            self.registry, self.current_proposal = {}, None
            self.previous_rows = rows(root / "zero_train_predictions.jsonl")
            self.previous_feedback = rows(root / "zero_feedback_predictions.jsonl")
            self.previous_losses = {}
            self.content_course = None
            if cfg.get("course_design") == "content_centered":
                from content_curriculum_run import ContentCourse
                assert arm == "dynamic"
                self.content_course = ContentCourse(root, destination, mapping, coverage)

        def publish(self, status, step, **extra):
            # Mutable UI status only. Scientific receipts remain immutable.
            path = destination / "status.json"
            path.write_text(json.dumps({"status": status, "step": step, **extra}, ensure_ascii=False)+"\n")
            print(f"[curriculum-v2] {arm} step={step} {status}", flush=True)

        def install(self, end, entries):
            if self.content_course is not None and end > 20:
                manifest = self.content_course.compile(end, entries, self.manifests)
            else:
                manifest = compile_block(end, coverage[end], entries, self.registry,
                        None if arm == "random" or end == 20 else self.current_proposal["allocations"], self.manifests)
            save(destination / f"manifest_{end}.json", manifest)
            self.manifests.append(manifest)
            self.trainer.train_dataset.install(manifest)

        def on_log(self, args, state, control, logs=None, **kwargs):
            self.publish("training", state.global_step, metrics=logs)

        def on_save(self, args, state, control, **kwargs):
            step = state.global_step
            assert step in ENDPOINTS
            assert self.trainer.train_dataset.read_positions == list(range(step*32)), "lookahead/duplicate training read"
            trainer = self.trainer
            def training_state():
                return {"adapter": state_digest(get_peft_model_state_dict(trainer.model)),
                        "optimizer": state_digest(trainer.optimizer.state_dict()),
                        "scheduler": state_digest(trainer.lr_scheduler.state_dict()),
                        "cpu_rng": state_digest(torch.random.get_rng_state()),
                        "cuda_rng": state_digest(torch.cuda.get_rng_state_all())}
            before_barrier = training_state()
            with preserve_rng():
                self.publish("evaluating", step)
                checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
                torch.cuda.empty_cache()
                current = decode(checkpoint, TRAIN, destination / f"evaluation/train_{step}")
                feedback = decode(checkpoint, FEEDBACK, destination / f"evaluation/feedback_{step}")
                self.publish("measuring_unique_train_losses", step)
                losses = measure_losses(trainer, trainer.train_dataset.features)
                save(destination / f"losses_{step}.json", losses)
                cat, _ = learning_catalog(current, self.previous_rows, zero, self.manifests,
                                           self.previous_losses, losses)
                save(destination / f"catalog_{step}.json", cat)
                fb_before = anonymize(self.previous_feedback, mapping, True)
                fb_after = anonymize(feedback, mapping, True)
                feedback_metrics = {bucket: summarize_pair(fb_before, fb_after,
                                   [i for i, r in fb_after.items() if bucket == "overall" or r["zero_shot_bucket"] == bucket])
                                    for bucket in ("overall", "easy", "medium", "hard")}
                self.metric_history.append({"step": step, "checkpoint": str(checkpoint),
                    "evaluation_policy": "own_checkpoint", "feedback": feedback_metrics})
                save(destination / f"metrics_{step}.json", self.metric_history[-1])
                if self.content_course is not None:
                    self.content_course.advance(step, cat, self.manifests[-1],
                        anonymize(self.previous_rows, mapping), anonymize(current, mapping),
                        fb_before, fb_after, feedback_metrics, self.publish)
                if self.current_proposal is not None:
                    outcome = outcome_pack(step, self.current_proposal, self.registry, self.manifests[-1],
                              anonymize(self.previous_rows, mapping), anonymize(current, mapping),
                              fb_before, fb_after, mapping, self.previous_losses, losses)
                    save(destination / f"outcome_{step}.json", outcome)
                    episode = {"outcome": outcome}
                    if arm == "dynamic":
                        self.publish("reflecting", step)
                        reflection = call_agent(root, destination / f"reflect_{step}",
                            {"task": "Reflect on the completed block; do not prescribe yet.", "outcome": outcome,
                             "registry": self.registry, "history": self.episodes}, CourseReflection, "reflect")
                        validate_reflection(reflection, outcome)
                        episode["reflection"] = reflection.model_dump(mode="json")
                    self.episodes.append(episode)
                    save(destination / f"episode_{step}.json", episode)
                if step < 100 and arm != "random" and self.content_course is None:
                    if arm == "static" and step == 20 or arm == "dynamic" and step > 20:
                        self.publish("prescribing", step)
                        background_blocks = {str(end): coverage[end] for end in ENDPOINTS if end > step and (step == 20 or end == step+20)}
                        payload = {"task": "Allocate the NEXT block's 240 real course exposures.", "checkpoint_step": step,
                                   **compact_catalog(cat), "feedback": feedback_metrics,
                                   "patient_progress": patient_progress(cat, background_blocks),
                                   "registry": self.registry, "history": self.episodes,
                                   "background_blocks": background_blocks}
                        proposal = call_agent(root, destination / f"prescribe_{step}", payload, CourseProposal, "prescribe")
                        self.registry = register_proposal(proposal, self.registry, cat["recordings"], background_blocks.values())
                        new = proposal.model_dump(mode="json")
                        save(destination / f"allocation_changes_{step}.json", allocation_actions(
                             [] if self.current_proposal is None else self.current_proposal["allocations"], new["allocations"]))
                        self.current_proposal = new
                        if step == 20:
                            save(root / "first_course.json", {"proposal": new, "registry": self.registry})
                    elif arm == "dynamic" and step == 20:
                        first = read(root / "first_course.json")
                        self.registry, self.current_proposal = first["registry"], first["proposal"]
                        register_proposal(CourseProposal.model_validate({**self.current_proposal, "new_units": []}),
                                          self.registry, cat["recordings"], [coverage[40]])
                    save(destination / f"active_course_{step}.json", {"registry": self.registry, "proposal": self.current_proposal})
                self.previous_rows, self.previous_feedback, self.previous_losses = current, feedback, losses
                if step < 100:
                    self.install(step+20, cat["recordings"])
            assert training_state() == before_barrier, "assessment changed model/optimizer/scheduler/RNG"
            save(destination / f"state_{step}.json", before_barrier)
            self.publish("complete" if step == 100 else "training", step)
            return control

    monitor = Monitor()

    class ExperimentTrainer(ContinuousCourseTrainer):
        def __init__(self, *args, **kwargs):
            feature_rows = kwargs["train_dataset"]
            assert len(feature_rows) == len(mapping)
            features = dict(zip(sorted(mapping), feature_rows))
            kwargs["train_dataset"] = BlockDataset(features)
            kwargs["args"].save_only_model = True
            super().__init__(*args, **kwargs)
            monitor.trainer = self
            self.add_callback(monitor)
            monitor.install(20, base_entries)

    training.QwenASRLoRATrainer = ExperimentTrainer
    sys.argv = training_arguments(root, arm)
    try:
        training.main()
        assert monitor.trainer.state.global_step == 100
        assert monitor.trainer.train_dataset.read_positions == list(range(3200))
        save(destination / "result.json", {"status": "complete", "history": monitor.metric_history,
             "unique_train_recordings": len({r["id"] for m in monitor.manifests for r in m}),
             "exposures": 3200, "optimizer_resets_at_barriers": 0})
    except Exception as error:
        monitor.publish("failed", monitor.trainer.state.global_step, error=f"{type(error).__name__}: {error}")
        raise


def report(root):
    result = {arm: read(root / arm / "result.json") for arm in ARMS}
    for arm, r in result.items():
        assert [h["step"] for h in r["history"]] == list(ENDPOINTS)
        assert r["unique_train_recordings"] == 2117 and r["exposures"] == 3200
        r["best"] = min(r["history"], key=lambda h: h["feedback"]["overall"]["after"]["mean_utterance_cer"])
    # All common positions match; fixed prescription and member pools never change.
    for step in ENDPOINTS:
        manifests = {arm: read(root / arm / f"manifest_{step}.json") for arm in ARMS}
        shared = [[(r["position"], r["id"]) for r in manifests[a] if r["stream"] == "coverage"] for a in ARMS]
        assert shared[0] == shared[1] == shared[2]
        if step <= 40:
            assert manifests["static"] == manifests["dynamic"]
    first = read(root / "first_course.json")
    assert all(read(root / "static" / f"active_course_{s}.json") == first for s in (20, 40, 60, 80))
    save(root / "results.json", result)
    lines = ["# Real curriculum v2 results", "", "Same 511 exploratory feedback recordings; lower CER is better.",
             "Each arm/endpoint is independently evaluated with its own checkpoint; no cross-arm prediction/loss reuse.", "",
             "| Arm | Step | Overall | Easy | Medium | Hard | Patient macro |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for h in result[arm]["history"]:
            f = h["feedback"]
            values = [f[k]["after"]["mean_utterance_cer"] for k in ("overall", "easy", "medium", "hard")]
            values.append(f["overall"]["after"]["patient_macro_cer"])
            lines.append(f"| {arm} | {h['step']} | " + " | ".join(f"{100*x:.4f}%" for x in values) + " |")
    write_immutable_bytes(root / "report.md", ("\n".join(lines)+"\n").encode())


def run(root):
    verify_frozen(root)
    ready = read(root / "readiness.json")
    assert ready["offline_checks_passed"] and ready["real_audio_backward_passed"]
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Gateway API key (memory only): ")
    assert os.environ["OPENAI_API_KEY"], "API key is required before training begins"
    save(root / "started.json", {"status": "authorized_run_started", "time_utc_unix": time.time()})
    for arm in ARMS:
        with (root / f"{arm}.log").open("x") as log:
            subprocess.run([sys.executable, str(Path(__file__)), "train-arm", "--root", str(root), "--arm", arm],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    report(root)


def offline_check(root):
    """Compile all slots against actual data; fixtures never become formal proposals."""
    cfg = verify_frozen(root)
    from therapist_harness.evidence import evaluate_snapshot
    destination = root / "validation"
    destination.mkdir(exist_ok=True)
    before = rows(root / "zero_train_predictions.jsonl")
    cat, _ = learning_catalog(before, before, read(root / "zero_train.json"), [], {}, {})
    coverage = {int(k): v for k, v in read(root / "coverage.json").items()}
    patients = sorted({r["patient"] for r in cat["recordings"]})
    definitions = []
    for n in range(4):
        group = patients[n::4]
        citation = next(r["id"] for r in cat["recordings"] if r["patient"] in group and r["SDI"][1] > 0)
        definitions.append({"unit_id": f"U{n+1:03d}", "patients": group, "contents": [], "target": "deletion",
            "evidence_ids": [citation], "rationale": "OFFLINE COMPILER FIXTURE, NOT AGENT OUTPUT", "expected_change": "No scientific prediction"})
    fixture = CourseProposal(diagnosis="OFFLINE FIXTURE", hypothesis="No training/no intervention", risk="Not an Agent result",
        new_units=definitions, allocations=[{"unit_id": d["unit_id"], "exposures": 60, "reason": "fixture quota"} for d in definitions])
    registry = register_proposal(fixture, {}, cat["recordings"], [coverage[e] for e in ENDPOINTS[1:]])
    save(destination / "compiler_fixture.json", {"is_agent_output": False, "proposal": fixture.model_dump(mode="json")})
    audits, compiled = {}, {}
    mapping = read(root / "local_map.json")
    for arm in ARMS:
        manifests = []
        for end in ENDPOINTS:
            allocations = fixture.model_dump()["allocations"]
            if arm == "dynamic" and end > 40:
                allocations = [{**a, "exposures": n} for a, n in zip(allocations, (90, 70, 50, 30))]
            manifest = compile_block(end, coverage[end], cat["recordings"], registry,
                                    None if arm == "random" or end == 20 else allocations, manifests)
            manifests.append(manifest)
        compiled[arm] = [r for m in manifests for r in m]
        save(destination / f"fixture_{arm}_manifest.json", compiled[arm])
        counts = Counter(r["id"] for r in compiled[arm])
        assert sum(counts.values()) == 3200 and len(counts) == 2117
        audits[arm] = {"exposures": 3200, "unique_recordings": len(counts),
            "max_block_repeat": max(max(Counter(r["id"] for r in m).values()) for m in manifests),
            "audio_hours_with_repetition": sum(float(mapping[i]["duration"])*n for i, n in counts.items())/3600}
    assert compiled["static"][:1280] == compiled["dynamic"][:1280]
    assert compiled["static"][1280:] != compiled["dynamic"][1280:]
    shared = [[r for r in compiled[arm] if r["stream"] == "coverage"] for arm in ARMS]
    assert shared[0] == shared[1] == shared[2]
    payload = {"task": "PREPARATION SHAPE CHECK ONLY: no Agent call, no v2 checkpoint exists yet.",
               "checkpoint_step": 0, **compact_catalog(cat), "history": [], "registry": {},
               "patient_progress": patient_progress(cat, {str(e): coverage[e] for e in ENDPOINTS[1:]}),
               "background_blocks": {str(e): coverage[e] for e in ENDPOINTS[1:]}}
    request = build_request(cfg["agent"], payload, CourseProposal, "prescribe")
    save(destination / "request_shape_only.json", request)
    metric_conformance = {}
    for name, path in (("zero", root / "zero_feedback_predictions.jsonl"),
                       ("historical_m0", RECORDS / "therapist_m0_b8g4_pilot_v1/merged_test_reassessment_v1/m0_step50/predictions.jsonl")):
        snapshot = rows(path)
        keyed = {r["utt_id"]: r for r in snapshot}
        observed = summarize_pair(keyed, keyed, sorted(keyed))["after"]
        expected = evaluate_snapshot(snapshot, prediction_column="prediction", critical_cer_threshold=.5).overall.model_dump(mode="json")
        assert all(abs(value-expected[key]) < 1e-12 for key, value in observed.items())
        metric_conformance[name] = observed
    suites = ["tests/test_curriculum_v2.py", "tests/test_curriculum_v2_training.py", "tests/test_curriculum_v2_runner.py",
              "tests/test_real_curriculum.py", "tests/test_qwen3_asr_lora_gradient_accumulation.py", "tests/test_absorption_probe.py"]
    with (destination / "pytest.log").open("x") as log:
        subprocess.run([sys.executable, "-m", "pytest", "-q", *suites, "--junitxml", str(destination / "pytest.xml")],
                       cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    import tiktoken
    encoding = tiktoken.get_encoding("o200k_base")
    approximate_tokens = sum(len(encoding.encode(m["content"])) for m in request["input"])
    approximate_tokens += len(encoding.encode(json.dumps(request["text"]["format"]["schema"], ensure_ascii=False)))
    receipt = {"offline_checks_passed": True, "formal_training_started": False, "agent_calls": 0,
               "fixture_audits": audits, "request_characters": len(json.dumps(request, ensure_ascii=False)),
               "request_tokens_o200k_estimate": approximate_tokens,
               "metric_conformance_with_maintained_evaluator": metric_conformance,
               "suite_paths": suites, "pytest_xml_sha256": sha256_file(destination / "pytest.xml")}
    save(destination / "offline_checks.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def model_smoke(root):
    """Actual 8x4 real-audio backward check. No optimizer, updates or checkpoint."""
    import torch
    from peft import get_peft_model_state_dict
    from finetune.scripts import qwen3_asr_lora_sft as training
    from therapist_harness.curriculum_v2_training import state_digest

    verify_frozen(root)
    assert read(root / "validation/offline_checks.json")["offline_checks_passed"]
    destination = root / "validation/model_smoke"
    assert not destination.exists()
    started = time.monotonic()
    record = {}
    ids = sorted(read(root / "local_map.json"))
    selected = read(root / "coverage.json")["20"][:32]

    class BackwardOnly(training.QwenASRLoRATrainer):
        def train(self, *args, **kwargs):
            self.current_gradient_accumulation_steps = 4
            before = state_digest(get_peft_model_state_dict(self.model))
            torch.cuda.reset_peak_memory_stats()
            losses = []
            for start in range(0, 32, 8):
                features = [self.train_dataset[ids.index(i)] for i in selected[start:start+8]]
                batch = self.data_collator(features)
                assert batch["labels"].shape[0] == 8
                losses.append(float(self.training_step(self.model, batch, num_items_in_batch=32)))
            gradients = [p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert any(torch.count_nonzero(g) for g in gradients)
            assert all(p.grad is None for p in self.model.parameters() if not p.requires_grad)
            assert before == state_digest(get_peft_model_state_dict(self.model))
            assert self.optimizer is None and self.state.global_step == 0
            record.update(real_audio_backward_passed=True, optimizer_steps=0, parameter_updates=0,
                          checkpoint_saved=False, real_recordings=32, microbatch=8, gradient_accumulation=4,
                          accumulation_window_loss=sum(losses), finite_gradient_tensors=len(gradients),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          elapsed_seconds=time.monotonic()-started)
            self.model.zero_grad(set_to_none=True)

    training.QwenASRLoRATrainer = BackwardOnly
    argv = training_arguments(root, "unused")
    argv[argv.index("--output_dir")+1] = str(destination)
    argv.extend(["--save_strategy", "no"])
    sys.argv = argv
    training.main()
    save(root / "validation/model_smoke.json", record)
    save(root / "readiness.json", {"status": "ready_awaiting_user_start", "offline_checks_passed": True,
         "real_audio_backward_passed": True, "formal_training_started": False, "formal_agent_calls": 0,
         "online_agent_smoke": "not run; runtime credential required, first formal prescription follows actual step20",
         "protocol_sha256": sha256_file(root / "protocol.json"), "model_smoke": record})
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check", "model-smoke", "run", "train-arm", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.stage == "train-arm":
        assert args.arm is not None
        train_arm(args.root, args.arm)
    else:
        {"prepare": prepare, "check": offline_check, "model-smoke": model_smoke, "run": run, "report": report}[args.stage](args.root)


if __name__ == "__main__":
    main()
