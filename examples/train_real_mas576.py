"""Fresh Qwen3-ASR LoRA on 2117 real + the completed six-round 576 synthetic rows."""

# ruff: noqa: E402 -- standalone experiment entry point.
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "finetune/scripts")]
import qwen3_asr_lora_sft as training
from train_m0_small_dose_pair import Monitor
from train_setting_d_m0 import is_sustained_regression
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.store import sha256_file, write_immutable_json, write_immutable_bytes
from therapist_harness.text import identity_keys

RECORDS = Path("/data/qwen3-asr/records/therapist_mas_setting_d_v1")
SOURCE = RECORDS / "therapist_synthetic96_loop6_v1"
M0 = RECORDS / "real_only_m0_b8g4_utterance_mean_v1"
ROOT = RECORDS / "real2117_mas576_fresh_lora_b8g4_v1"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def command(root):
    return [sys.executable, str(Path(training.__file__)),
        "--model_path", "/data/qwen3-asr/models/Qwen3-ASR-1.7B",
        "--train_file", str(root / "train.jsonl"), "--eval_file", "",
        "--output_dir", str(root / "training"), "--batch_size", "8", "--grad_acc", "4",
        "--lr", "2e-4", "--lr_scheduler_type", "linear", "--warmup_ratio", "0.02",
        "--lora_rank", "16", "--lora_alpha", "4", "--lora_dropout", "0.05",
        "--lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--seed", "42", "--max_steps", "140", "--log_steps", "1",
        "--num_workers", "2", "--prefetch_factor", "2", "--save_steps", "10",
        "--save_total_limit", "0", "--save_final_checkpoint", "0", "--report_to", "none"]


def prepare(root):
    assert not root.exists()
    protocol = read(SOURCE / "protocol.json")
    assert read(SOURCE / "run_summary.json")["status"] == "six_rounds_committed"
    baseline = read(M0 / "run_config.json")
    for p in (Path(training.__file__), REPO / "finetune/official/qwen3_asr_sft.py"):
        assert sha256_file(p) == baseline["source_sha256"][str(p)]
    real_path = Path(protocol["real_train"]["path"])
    test_path = Path(protocol["feedback_split"]["path"])
    assert sha256_file(real_path) == protocol["real_train"]["sha256"]
    assert sha256_file(test_path) == protocol["feedback_split"]["sha256"]
    real, test = rows(real_path), rows(test_path)
    assert len(real) == 2117 and len(test) == 511
    references = {r["audio"]: r for r in real}
    blocked = {kv for r in real + test for kv in identity_keys(r["clean_gt"]).items()}
    synthetic, index, clips = [], [], []
    for n in range(1, 7):
        d = SOURCE / f"round_{n:03d}"
        receipt = read(d / "tts_outputs.json")
        assert receipt["status"] == "complete" and receipt["output_count"] == 96
        for s in receipt["outputs"]:
            assert sha256_file(Path(s["output_wav"])) == s["output_wav_sha256"]
            ref = references[s["reference_wav"]]
            assert sha256_file(Path(s["reference_wav"])) == s["reference_wav_sha256"]
            assert not blocked.intersection(identity_keys(s["target_text"]).items())
            synthetic.append({"audio": s["output_wav"],
                "text": "language Cantonese<asr_text>" + s["target_text"], "prompt": ""})
            index.append({"row": 2117 + len(index), "source": "synthetic", "round": n,
                "sample_id": s["sample_id"], "speaker_id": ref["speaker_id"],
                "reference_audio": s["reference_wav"], "audio": s["output_wav"],
                "text": s["target_text"], "duration_seconds": s["duration_seconds"]})
            clips.append(s)
    assert len(synthetic) == 576 and len({s["audio"] for s in synthetic}) == 576
    assert len({s["text"] for s in synthetic}) == 36
    assert set(Counter(s["text"] for s in synthetic).values()) == {16}
    training_rows = [{k: r[k] for k in ("audio", "text", "prompt")} for r in real] + synthetic
    root.mkdir(parents=True)
    write_immutable_bytes(root / "train.jsonl", "".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in training_rows).encode())
    write_immutable_json(root / "training_index.json", [
        {"row": i, "source": "real", "utt_id": r["utt_id"], "speaker_id": r["speaker_id"],
         "audio": r["audio"]} for i, r in enumerate(real)] + index)
    comparisons = {}
    for name, field in (("M0_step50", "initial_adapter_dev_predictions"),
                        ("zero_shot", "zero_shot_dev_predictions")):
        path = Path(protocol[field]["path"])
        assert sha256_file(path) == protocol[field]["sha256"]
        comparisons[name] = evaluate_snapshot(rows(path), prediction_column="prediction",
            critical_cer_threshold=.5).model_dump(mode="json")
    comparisons["sequential_round6"] = read(SOURCE / "round_006/mas/run_summary.json")["history"][-1]["metrics"]
    sources = [Path(__file__), Path(training.__file__), REPO / "finetune/official/qwen3_asr_sft.py",
        *sorted((REPO / "src/therapist_harness").glob("*.py")),
        *sorted((REPO / "finetune/scripts").glob("train_*.py"))]
    cfg = {"initialization": "original_base_model_new_lora_no_adapter", "real_rows": 2117,
        "synthetic_rows": 576, "total_rows": 2693, "synthetic_unique_texts": 36,
        "real_patients": len({r["speaker_id"] for r in real}),
        "synthetic_patients": len({r["speaker_id"] for r in index}),
        "real_seconds": sum(float(r["duration"]) for r in real),
        "synthetic_seconds": sum(r["duration_seconds"] for r in index),
        "synthetic_fraction": 576 / 2693, "sampler": "Trainer seeded random sampler",
        "feedback_rows": 511, "feedback_sha256": sha256_file(test_path),
        "selection_metric": "merged_511.overall.mean_utterance_cer",
        "stop_rule": "step>=50 and last 3 merged CERs >= best + 0.02; cap 140",
        "step50_comparison_required": True, "save_only_model": True,
        "source_run": str(SOURCE), "real_manifest": str(real_path),
        "real_sha256": sha256_file(real_path), "train_sha256": sha256_file(root / "train.jsonl"),
        "audio_sha256": {r["output_wav"]: r["output_wav_sha256"] for r in clips},
        "source_sha256": {str(p): sha256_file(p) for p in sources},
        "command": command(root), "comparisons": comparisons}
    write_immutable_json(root / "run_config.json", cfg)
    print(json.dumps({k: v for k, v in cfg.items() if k not in
        ("source_sha256", "audio_sha256", "comparisons", "command")}, ensure_ascii=False), flush=True)


class AugmentationMonitor(Monitor):
    def __init__(self, root, comparisons):
        super().__init__(root)
        self.end_step = 140
        self.state["endpoint_step"] = 140
        self.comparisons = comparisons

    def on_save(self, args, state, control, **kwargs):
        super().on_save(args, state, control, **kwargs)
        best = min(self.history, key=lambda r: r["metrics"]["overall"]["mean_utterance_cer"])
        stop = state.global_step >= 50 and is_sustained_regression(self.history)
        control.should_training_stop = control.should_training_stop or stop
        self.state.update(best_step=best["step"], best_metrics=best["metrics"], stop_for_overfitting=stop)
        self.publish()
        if state.global_step == 50:
            write_immutable_json(self.root / "step50_comparison.json", {
                **self.comparisons, "real_plus_MAS576_step50": self.history[-1]["metrics"]})
        return control


def run(root):
    cfg = read(root / "run_config.json")
    for path, digest in cfg["source_sha256"].items():
        assert sha256_file(Path(path)) == digest, path
    assert sha256_file(root / "train.jsonl") == cfg["train_sha256"]
    for path, digest in cfg["audio_sha256"].items():
        assert sha256_file(Path(path)) == digest
    assert not training.parse_args(cfg["command"][2:]).lora_init_adapter_path
    monitor = AugmentationMonitor(root, cfg["comparisons"])
    monitor.publish()

    class FreshTrainer(training.QwenASRLoRATrainer):
        def __init__(self, *args, **kwargs):
            kwargs["args"].save_only_model = True
            super().__init__(*args, **kwargs)
            self.add_callback(monitor)

    training.QwenASRLoRATrainer = FreshTrainer
    sys.argv = cfg["command"][1:]
    try:
        training.main()
        assert any(r["step"] == 50 for r in monitor.history)
        best = min(monitor.history, key=lambda r: r["metrics"]["overall"]["mean_utterance_cer"])
        write_immutable_json(root / "run_summary.json", {"status": "complete", "best": best,
            "history": monitor.history, "comparisons": cfg["comparisons"],
            "stopped_for_overfitting": monitor.state["stop_for_overfitting"],
            "best_checkpoint": str(root / f'training/checkpoint-{best["step"]}')})
        monitor.state["status"] = "complete"
        monitor.publish()
    except Exception as error:
        monitor.state.update(status="failed", error=f"{type(error).__name__}: {error}")
        monitor.publish()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root)
    elif args.detach:
        with (args.root / "train.log").open("x") as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "--root", str(args.root)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_immutable_json(args.root / "launch.json", {"pid": child.pid})
        print(f"Background PID {child.pid}", flush=True)
    else:
        run(args.root)
