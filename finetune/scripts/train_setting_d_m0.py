"""Supervised Setting D real-only M0 run: dev-selected, one final test decode."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import TrainerCallback

import qwen3_asr_lora_sft as training

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from therapist_harness.evidence import evaluate_snapshot  # noqa: E402

DATA = Path("/data/qwen3-asr/finetune/data_setting_d_v1")
BASE = "/data/qwen3-asr/models/Qwen3-ASR-1.7B"
OLD = Path("/data/qwen3-asr/finetune/setting_d_gradacc_fixed_v1/lora_current_default_r16_scale025_step140_seed42_v1")
EXPECTED = {
    "train": (2117, "6d232f763e065f2bacea234a483948ec790a16354cfb5d76732453c1bfd31be8"),
    "dev": (258, "978c9537352f3725d0019a0329f5533d5900e9dfc74af472ee05032eab38ec4d"),
    "test": (253, "c150f7f7d8380901727a45dfa5481137d2a2ead823c01e08f11b3389e9aee543"),
}


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def audit_splits():
    splits, audit = {}, {}
    for name, (count, digest) in EXPECTED.items():
        path = DATA / f"setting_d_{name}.jsonl"
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == digest
        rows = [json.loads(line) for line in content.splitlines()]
        assert len(rows) == count
        speakers = {row["speaker_id"] for row in rows}
        assert not speakers.intersection({"vlink201", "vlink203", "vlink225", "vlink243", "vlink_Kaho"})
        assert all(row["utt_id"] != "vlink211/7290411755587742893228" for row in rows)
        splits[name] = rows
        audit[name] = {"path": str(path), "sha256": digest, "rows": count,
                       "speakers": len(speakers), "contents": len({r["prompt_id"] for r in rows})}
    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        for field in ("utt_id", "prompt_id", "prompt_normalized_sha16", "prompt_jyutping_sha16"):
            assert not {r[field] for r in splits[first]}.intersection(r[field] for r in splits[second])
    assert {r["speaker_id"] for r in splits["dev"] + splits["test"]} <= {r["speaker_id"] for r in splits["train"]}
    return audit


def decode(checkpoint, split, destination, log):
    command = [sys.executable, str(REPO / "finetune/scripts/evaluate_asr_checkpoint.py"),
               "--base-model-path", BASE, "--adapter-path", str(checkpoint),
               "--split-jsonl", str(DATA / f"setting_d_{split}.jsonl"),
               "--out-dir", str(destination), "--batch-size", "8",
               "--critical-cer-threshold", "0.5", "--max-new-tokens", "256"]
    with log.open("x") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return json.loads((destination / "summary.json").read_text())["metrics"]


def is_sustained_regression(history, margin=0.02, patience=3):
    if len(history) < patience:
        return False
    best = min(row["metrics"]["overall"]["mean_utterance_cer"] for row in history)
    return all(row["metrics"]["overall"]["mean_utterance_cer"] >= best + margin for row in history[-patience:])


class DevMonitor(TrainerCallback):
    def __init__(self, root):
        self.root = root
        self.history = []
        self.best = None
        self.state = {"status": "initializing", "step": 0}
        (root / "evaluation").mkdir()

    def publish(self):
        write_json(self.root / "monitor.json", {**self.state, "best": self.best, "history": self.history})

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.state.update(status="training", step=state.global_step, last_log=logs)
        self.publish()

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
        self.state.update(status="evaluating_dev", step=step)
        self.publish()
        print(f"[dev-monitor] decoding checkpoint-{step}", flush=True)
        torch.cuda.empty_cache()
        started = time.perf_counter()
        metrics = decode(checkpoint, "dev", self.root / "evaluation" / f"dev_step_{step}",
                         self.root / "evaluation" / f"dev_step_{step}.log")
        row = {"step": step, "checkpoint": str(checkpoint), "metrics": metrics,
               "decode_seconds": time.perf_counter() - started}
        self.history.append(row)
        if self.best is None or metrics["overall"]["mean_utterance_cer"] < self.best["metrics"]["overall"]["mean_utterance_cer"]:
            self.best = row
        stop = step >= 40 and is_sustained_regression(self.history)
        self.state.update(status="stopping_overfitting" if stop else "training", stop_for_overfitting=stop)
        control.should_training_stop = control.should_training_stop or stop
        self.publish()
        print("[dev-monitor] " + json.dumps({"step": step, "mean_cer": metrics["overall"]["mean_utterance_cer"],
              "pooled_cer": metrics["overall"]["pooled_cer"], "best_step": self.best["step"], "stop": stop}), flush=True)
        return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    root = cli.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    audit = audit_splits()
    old_rows = list(csv.DictReader((OLD / "evaluation/dev_step_50/predictions.csv").open()))
    comparisons = {name: evaluate_snapshot(old_rows, prediction_column=column, critical_cer_threshold=0.5).model_dump(mode="json")
                   for name, column in (("old_m0", "fine_tuned_pred"), ("zero_shot", "raw_pred"))}
    config = {"setting": "D", "data": audit, "initialization": "base_model_new_lora", "base_model": BASE,
              "synthetic_count": 0, "microbatch": 8, "gradient_accumulation": 4, "effective_batch": 32,
              "loss": "utterance_mean_per_accumulation_window", "rank": 16, "alpha": 4, "dropout": 0.05,
              "lr": 2e-4, "scheduler": "linear", "warmup_ratio": 0.02, "seed": 42,
              "max_steps": 140, "dev_decode_every_steps": 10, "selection_metric": "dev.overall.mean_utterance_cer",
              "stop_rule": "step>=40 and last 3 dev CERs >= best + 0.02", "test_policy": "once_after_selection",
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                  (Path(__file__), Path(training.__file__), REPO / "finetune/official/qwen3_asr_sft.py")},
              "dev_comparisons_same_evaluator": comparisons}
    write_json(root / "run_config.json", config)
    monitor = DevMonitor(root)
    monitor.publish()

    class SupervisedTrainer(training.QwenASRLoRATrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(monitor)

    training.QwenASRLoRATrainer = SupervisedTrainer
    sys.argv = ["m0", "--model_path", BASE, "--train_file", str(DATA / "setting_d_train.jsonl"),
                "--eval_file", str(DATA / "setting_d_dev.jsonl"), "--output_dir", str(root / "training"),
                "--batch_size", "8", "--grad_acc", "4", "--max_steps", "140", "--log_steps", "5",
                "--save_steps", "10", "--save_total_limit", "0", "--save_final_checkpoint", "0"]
    try:
        training.main()
        assert monitor.best is not None
        selection = {"checkpoint": monitor.best["checkpoint"], "step": monitor.best["step"],
                     "selection_split": "dev", "selection_metric": config["selection_metric"],
                     "dev_metrics": monitor.best["metrics"], "training_stopped_step": monitor.state["step"],
                     "stop_for_overfitting": monitor.state.get("stop_for_overfitting", False)}
        adapter = Path(selection["checkpoint"]) / "adapter_model.safetensors"
        selection["adapter_sha256"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
        write_json(root / "M0_selection.json", selection)
        monitor.state["status"] = "evaluating_selected_m0_test"
        monitor.publish()
        torch.cuda.empty_cache()
        test = decode(Path(selection["checkpoint"]), "test", root / "evaluation/test_selected_m0", root / "evaluation/test_selected_m0.log")
        write_json(root / "run_summary.json", {**selection, "test_metrics": test, "dev_comparisons": comparisons})
        monitor.state["status"] = "complete"
        monitor.publish()
    except Exception as error:
        monitor.state.update(status="failed", error=f"{type(error).__name__}: {error}")
        monitor.publish()
        raise


if __name__ == "__main__":
    main()
