"""Matched eight-update M0 calibration; no Agent, TTS or proposal gate."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from torch.utils.data import SequentialSampler

import qwen3_asr_lora_sft as training
from train_therapist_round import RoundMonitor
from train_setting_d_m0 import decode
from train_m0_real_replay_control import PILOT, REAL, TEST, read_jsonl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from therapist_harness.evidence import evaluate_snapshot  # noqa: E402
from therapist_harness.store import sha256_directory, sha256_file, write_immutable_bytes, write_immutable_json  # noqa: E402


def stratified_sample(rows, count, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["speaker_id"], row["zero_shot_bucket"])].append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    allocation = {k: len(groups[k]) * count // len(rows) for k in keys}
    order = sorted(keys, key=lambda k: len(groups[k]) * count % len(rows), reverse=True)
    for key in order[:count - sum(allocation.values())]:
        allocation[key] += 1
    selected = []
    for key in keys:
        rng.shuffle(groups[key])
        selected.extend(groups[key][:allocation[key]])
    return selected


def stats(rows):
    return {"rows": len(rows), "speakers": len({r["speaker_id"] for r in rows}),
            "contents": len({r["prompt_id"] for r in rows}),
            "buckets": dict(Counter(r["zero_shot_bucket"] for r in rows))}


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((PILOT / "protocol.json").read_text())
    execution = json.loads((PILOT / "training_execution.json").read_text())
    parent = Path(protocol["initial_adapter"]["path"])
    assert sha256_directory(parent) == protocol["initial_adapter"]["sha256"]
    assert sha256_file(REAL) == protocol["real_train"]["sha256"]
    assert sha256_file(TEST) == "189d749e7e65f14a0440d7f6707a0e1dcfa1a5a62c6c0e5f362b08c1c53e2f98"
    for path in (Path(training.__file__), REPO / "finetune/official/qwen3_asr_sft.py"):
        assert sha256_file(path) == execution["source_sha256"][str(path)]
    real, previous = read_jsonl(REAL), read_jsonl(PILOT / "train.jsonl")
    def project(row):
        return {k: row[k] for k in ("audio", "text", "prompt")}
    assert [project(r) for r in real] == previous[:2117]
    synthetic = previous[2117:]
    assert len(real) == 2117 and len(synthetic) == 24
    rng = random.Random(42)
    shared = stratified_sample(real, 232, rng)
    ids = {r["utt_id"] for r in shared}
    extra = stratified_sample([r for r in real if r["utt_id"] not in ids], 24, rng)
    rng.shuffle(shared)
    rng.shuffle(extra)
    rng.shuffle(synthetic)
    arms = {"real_only": [], "plus24": []}
    index = []
    for step in range(8):
        positions = list(range(32))
        rng.shuffle(positions)
        replacements = set(positions[:3])
        r = iter(shared[step * 29:(step + 1) * 29])
        e = iter(extra[step * 3:(step + 1) * 3])
        s = iter(synthetic[step * 3:(step + 1) * 3])
        for position in range(32):
            if position in replacements:
                old, new = next(e), next(s)
                arms["real_only"].append(project(old))
                arms["plus24"].append(new)
                index.append({"row": len(index), "step": step + 1, "role": "replacement",
                              "real_utt_id": old["utt_id"], "synthetic_audio": new["audio"]})
            else:
                row = next(r)
                for arm in arms.values():
                    arm.append(project(row))
                index.append({"row": len(index), "step": step + 1, "role": "shared", "real_utt_id": row["utt_id"]})
    test = read_jsonl(TEST)
    for field in ("utt_id", "prompt_id"):
        assert not {r[field] for r in shared + extra} & {r[field] for r in test}
    assert len(ids) == 232 and len({r["utt_id"] for r in shared + extra}) == 256
    assert sum(a != b for a, b in zip(arms["real_only"], arms["plus24"])) == 24
    commands = {}
    for name, rows in arms.items():
        directory = root / name
        directory.mkdir()
        write_immutable_bytes(directory / "train.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())
        command = execution["command"].copy()
        changes = {"--train_file": str(directory / "train.jsonl"), "--output_dir": str(directory / "training"),
                   "--lr": "2e-5", "--max_steps": "8", "--lr_scheduler_type": "constant",
                   "--warmup_ratio": "0", "--save_steps": "4"}
        for key, value in changes.items():
            command[command.index(key) + 1] = value
        command.extend(["--log_steps", "1"])
        commands[name] = command
    parsed = [vars(training.parse_args(c[2:])) for c in commands.values()]
    assert {k for k in parsed[0] if parsed[0][k] != parsed[1][k]} == {"train_file", "output_dir"}
    m0 = next(r for r in json.loads((PILOT / "merged_test_reassessment_v1/results.json").read_text()) if r["model"] == "m0_step50")
    config = {"parent": str(parent), "parent_sha256": sha256_directory(parent), "seed": 42,
              "shared_real": stats(shared), "extra_real": stats(extra), "control_real": stats(shared + extra),
              "synthetic_rows": 24, "synthetic_unique_texts": len({r["text"] for r in synthetic}),
              "sampler": "SequentialSampler over frozen seeded paired order; 29 shared + 3 replacements per update",
              "stratification": "speaker_id x zero_shot_bucket, proportional largest remainder, without replacement",
              "commands": commands, "m0_metrics": m0["metrics"], "test_path": str(TEST), "test_sha256": sha256_file(TEST),
              "source_sha256": {str(p): sha256_file(p) for p in (Path(__file__), Path(training.__file__),
                  REPO / "finetune/official/qwen3_asr_sft.py", REPO / "finetune/scripts/train_setting_d_m0.py",
                  REPO / "finetune/scripts/train_therapist_round.py", REPO / "finetune/scripts/train_m0_real_replay_control.py")},
              "train_sha256": {name: sha256_file(root / name / "train.jsonl") for name in arms},
              "synthetic_audio_sha256": {r["audio"]: sha256_file(Path(r["audio"])) for r in synthetic}}
    write_immutable_json(root / "paired_index.json", index)
    write_immutable_json(root / "real_selection.json", {"shared": shared, "extra": extra})
    write_immutable_json(root / "run_config.json", config)
    return config


class Monitor(RoundMonitor):
    def __init__(self, root):
        super().__init__(root, 8)

    def on_step_end(self, args, state, control, **kwargs):
        return control

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        self.state.update(status="evaluating_merged_test", step=step)
        self.publish()
        torch.cuda.empty_cache()
        started = time.monotonic()
        predictions = {}
        for shard in ("dev", "test"):
            destination = self.root / f"evaluation/step_{step}_{shard}_shard"
            decode(Path(args.output_dir) / f"checkpoint-{step}", shard, destination, destination.with_suffix(".log"))
            with (destination / "predictions.csv").open() as handle:
                for row in csv.DictReader(handle):
                    assert row["utt_id"] not in predictions
                    predictions[row["utt_id"]] = row
        scope = read_jsonl(TEST)
        assert set(predictions) == {r["utt_id"] for r in scope}
        for row in scope:
            prediction = predictions[row["utt_id"]]
            for key in ("speaker_id", "prompt_id", "clean_gt", "zero_shot_bucket"):
                assert row[key] == prediction[key]
            row["prediction"] = prediction["prediction"]
        metrics = evaluate_snapshot(scope, prediction_column="prediction", critical_cer_threshold=.5).model_dump(mode="json")
        destination = self.root / f"evaluation/merged_step_{step}"
        write_immutable_bytes(destination / "predictions.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scope).encode())
        write_immutable_json(destination / "summary.json", {"step": step, "metrics": metrics, "test_sha256": sha256_file(TEST)})
        self.history.append({"step": step, "metrics": metrics, "decode_seconds": time.monotonic() - started})
        self.state.update(status="training")
        self.publish()
        print("[small-dose] " + json.dumps(self.history[-1]), flush=True)
        return control


def train_arm(root, name):
    config = json.loads((root / "run_config.json").read_text())
    for path, digest in config["source_sha256"].items():
        assert sha256_file(Path(path)) == digest
    assert sha256_directory(Path(config["parent"])) == config["parent_sha256"]
    assert sha256_file(root / name / "train.jsonl") == config["train_sha256"][name]
    monitor = Monitor(root / name)
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
    sys.argv = config["commands"][name][1:]
    try:
        training.main()
        state = json.loads((root / name / "training/checkpoint-8/trainer_state.json").read_text())
        assert state["global_step"] == 8 and state["epoch"] == 1.0
        assert [r["step"] for r in monitor.history] == [4, 8]
        assert all(abs(r["learning_rate"] - 2e-5) < 1e-12 for r in state["log_history"] if "learning_rate" in r)
        write_immutable_json(root / name / "run_summary.json", {"status": "complete", "history": monitor.history,
            "endpoint_sha256": sha256_directory(root / name / "training/checkpoint-8"), "train_log": state["log_history"]})
        monitor.state.update(status="complete")
        monitor.publish()
    except Exception as error:
        monitor.state.update(status="failed", error=f"{type(error).__name__}: {error}")
        monitor.publish()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("real_only", "plus24"))
    args = parser.parse_args()
    root = args.output.resolve()
    if args.arm:
        train_arm(root, args.arm)
        return
    config = prepare(root)
    results = {"m0": config["m0_metrics"]}
    for name in ("real_only", "plus24"):
        with (root / f"{name}.log").open("x") as handle:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(root), "--arm", name],
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        results[name] = json.loads((root / name / "run_summary.json").read_text())
    for path, digest in config["source_sha256"].items():
        assert sha256_file(Path(path)) == digest
    assert sha256_directory(Path(config["parent"])) == config["parent_sha256"]
    write_immutable_json(root / "results.json", results)
    print(json.dumps({"status": "complete", "output": str(root)}), flush=True)


if __name__ == "__main__":
    main()
