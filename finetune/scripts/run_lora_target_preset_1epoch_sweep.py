#!/usr/bin/env python3
"""Run a one-epoch LoRA target-module preset sweep and evaluate on test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PRESETS = (
    "audio_attn_qkv",
    "audio_attn_qkvo",
    "audio_projector",
    "audio_adapter_convout_proj",
    "audio_encoder_all",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--dev-jsonl", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--language", default="Cantonese")
    p.add_argument("--presets", default=",".join(DEFAULT_PRESETS))
    p.add_argument("--reference-name", default="current_default_scale025_step68")
    p.add_argument("--reference-summary", default="")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-scale", type=float, default=0.25)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-acc", type=int, default=32)
    p.add_argument("--log-steps", type=int, default=5)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--skip-existing", type=int, default=1)
    return p.parse_args()


def count_jsonl(path: str | Path) -> int:
    with Path(path).open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def read_summary(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["group"]: row for row in rows}


def metric(summary: dict[str, dict[str, str]], group: str, key: str) -> str:
    return summary.get(group, {}).get(key, "")


def build_summary_row(method: str, preset: str, checkpoint_step: str, summary_path: Path) -> dict[str, str]:
    summary = read_summary(summary_path)
    return {
        "method": method,
        "target_preset": preset,
        "checkpoint_step": checkpoint_step,
        "sample_count": metric(summary, "overall", "sample_count"),
        "zero_shot_cer": metric(summary, "overall", "zero_shot_cer"),
        "test_cer": metric(summary, "overall", "fine_tuned_cer"),
        "delta_cer": metric(summary, "overall", "delta_cer"),
        "zero_shot_critical_count": metric(summary, "overall", "zero_shot_critical_count"),
        "test_critical_count": metric(summary, "overall", "fine_tuned_critical_count"),
        "test_critical_rate": metric(summary, "overall", "fine_tuned_critical_rate"),
        "hard_cer": metric(summary, "zero_shot_bucket=hard", "fine_tuned_cer"),
        "hard_critical_count": metric(summary, "zero_shot_bucket=hard", "fine_tuned_critical_count"),
        "medium_cer": metric(summary, "zero_shot_bucket=medium", "fine_tuned_cer"),
        "medium_critical_count": metric(summary, "zero_shot_bucket=medium", "fine_tuned_critical_count"),
        "easy_cer": metric(summary, "zero_shot_bucket=easy", "fine_tuned_cer"),
        "easy_critical_count": metric(summary, "zero_shot_bucket=easy", "fine_tuned_critical_count"),
        "summary_path": str(summary_path),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "method",
        "target_preset",
        "checkpoint_step",
        "sample_count",
        "zero_shot_cer",
        "test_cer",
        "delta_cer",
        "zero_shot_critical_count",
        "test_critical_count",
        "test_critical_rate",
        "hard_cer",
        "hard_critical_count",
        "medium_cer",
        "medium_critical_count",
        "easy_cer",
        "easy_critical_count",
        "summary_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    presets = [item.strip() for item in args.presets.split(",") if item.strip()]
    train_count = count_jsonl(args.train_jsonl)
    steps_per_epoch = math.ceil(train_count / (args.batch_size * args.grad_acc))
    total_steps = math.ceil(steps_per_epoch * args.epochs)

    config = {
        "split": "prompt_disjoint_v1",
        "train_count": train_count,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "presets": presets,
        "lora_rank": args.lora_rank,
        "lora_scale": args.lora_scale,
        "lora_alpha": args.lora_rank * args.lora_scale,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_acc": args.grad_acc,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, str]] = []
    if args.reference_summary:
        rows.append(
            build_summary_row(
                args.reference_name,
                "current_default",
                "68",
                Path(args.reference_summary),
            )
        )

    for preset in presets:
        run_dir = output_dir / preset
        eval_dir = run_dir / f"eval_test_checkpoint_{total_steps}"
        summary_path = eval_dir / "summary_by_group.csv"
        if args.skip_existing == 1 and summary_path.exists():
            print(f"[skip] {preset}: found {summary_path}", flush=True)
            rows.append(build_summary_row(preset, preset, str(total_steps), summary_path))
            write_csv(output_dir / "target_preset_test_summary.csv", rows)
            continue

        checkpoint_dir = run_dir / f"checkpoint-{total_steps}"
        if not checkpoint_dir.exists():
            train_cmd = [
                sys.executable,
                str(REPO_ROOT / "finetune/scripts/qwen3_asr_lora_sft.py"),
                "--model_path",
                args.model_path,
                "--train_file",
                args.train_jsonl,
                "--eval_file",
                args.dev_jsonl,
                "--output_dir",
                str(run_dir),
                "--batch_size",
                str(args.batch_size),
                "--grad_acc",
                str(args.grad_acc),
                "--lr",
                str(args.lr),
                "--epochs",
                str(args.epochs),
                "--log_steps",
                str(args.log_steps),
                "--save_strategy",
                "steps",
                "--save_steps",
                str(total_steps),
                "--save_total_limit",
                "1",
                "--save_final_checkpoint",
                "1",
                "--num_workers",
                "2",
                "--pin_memory",
                "1",
                "--persistent_workers",
                "1",
                "--prefetch_factor",
                "2",
                "--lora_rank",
                str(args.lora_rank),
                "--lora_scale",
                str(args.lora_scale),
                "--lora_dropout",
                str(args.lora_dropout),
                "--lora_target_preset",
                preset,
                "--report_to",
                "none",
                "--run_name",
                f"target_preset_{preset}_1epoch",
            ]
            print(f"[train] {preset} -> {run_dir}", flush=True)
            run_cmd(train_cmd, run_dir / "train.log")
        else:
            print(f"[skip-train] {preset}: found {checkpoint_dir}", flush=True)

        eval_cmd = [
            sys.executable,
            str(REPO_ROOT / "finetune/scripts/evaluate_e2_lora_checkpoint.py"),
            "--base-model-path",
            args.model_path,
            "--adapter-path",
            str(checkpoint_dir),
            "--split-csv",
            args.test_csv,
            "--out-dir",
            str(eval_dir),
            "--language",
            args.language,
            "--batch-size",
            str(args.eval_batch_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--merge-adapter",
            "1",
        ]
        print(f"[eval-test] {preset} -> {eval_dir}", flush=True)
        run_cmd(eval_cmd, run_dir / f"eval_test_checkpoint_{total_steps}.log")
        rows.append(build_summary_row(preset, preset, str(total_steps), summary_path))
        write_csv(output_dir / "target_preset_test_summary.csv", rows)


if __name__ == "__main__":
    main()
