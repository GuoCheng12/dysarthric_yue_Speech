#!/usr/bin/env python3
"""Run LoRA SFT once, then log dev CER for saved checkpoints to W&B.

The training process logs Trainer loss/eval_loss to W&B during training. After
training exits, this runner evaluates saved checkpoints on the dev split and
logs CER/critical metrics at the corresponding global steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--dev-jsonl", required=True)
    p.add_argument("--dev-csv", required=True)
    p.add_argument("--language", default="Cantonese")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-scale", type=float, default=0.25)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-acc", type=int, default=32)
    p.add_argument("--eval-every-steps", type=int, default=50)
    p.add_argument("--log-steps", type=int, default=5)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--wandb-run-id", default="")
    return p.parse_args()


def count_jsonl(path: str | Path) -> int:
    with Path(path).open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_summary(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["group"]: row for row in rows}


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else math.nan


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(value) if value != "" else 0


def run_cmd(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    args = parse_args()
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency `wandb`; install it in the training environment.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_count = count_jsonl(args.train_jsonl)
    steps_per_epoch = math.ceil(train_count / (args.batch_size * args.grad_acc))
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    checkpoint_steps = list(range(args.eval_every_steps, total_steps + 1, args.eval_every_steps))
    if total_steps not in checkpoint_steps:
        checkpoint_steps.append(total_steps)

    run_id = args.wandb_run_id or uuid.uuid4().hex[:8]
    env = os.environ.copy()
    env.update(
        {
            "WANDB_PROJECT": args.project,
            "WANDB_RUN_ID": run_id,
            "WANDB_RESUME": "allow",
            "WANDB_NAME": args.run_name,
        }
    )

    config = {
        "split": "prompt_disjoint_v1",
        "train_count": train_count,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "checkpoint_steps": checkpoint_steps,
        "lora_rank": args.lora_rank,
        "lora_scale": args.lora_scale,
        "lora_alpha": args.lora_rank * args.lora_scale,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_acc": args.grad_acc,
        "eval_every_steps": args.eval_every_steps,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

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
        str(output_dir),
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
        str(args.eval_every_steps),
        "--save_total_limit",
        "10",
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
        "--report_to",
        "wandb",
        "--run_name",
        args.run_name,
    ]
    run_cmd(train_cmd, output_dir / "train.log", env)

    run = wandb.init(project=args.project, id=run_id, resume="allow", name=args.run_name, config=config)
    eval_rows: list[dict[str, object]] = []
    for step in checkpoint_steps:
        checkpoint = output_dir / f"checkpoint-{step}"
        if not checkpoint.exists():
            print(f"[warn] missing checkpoint: {checkpoint}", flush=True)
            continue
        eval_dir = output_dir / f"eval_dev_checkpoint_{step}"
        eval_cmd = [
            sys.executable,
            str(REPO_ROOT / "finetune/scripts/evaluate_e2_lora_checkpoint.py"),
            "--base-model-path",
            args.model_path,
            "--adapter-path",
            str(checkpoint),
            "--split-csv",
            args.dev_csv,
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
        run_cmd(eval_cmd, output_dir / f"eval_dev_checkpoint_{step}.log", env)

        summary = read_summary(eval_dir / "summary_by_group.csv")
        overall = summary["overall"]
        hard = summary.get("zero_shot_bucket=hard", {})
        medium = summary.get("zero_shot_bucket=medium", {})
        easy = summary.get("zero_shot_bucket=easy", {})
        metrics = {
            "dev/cer": as_float(overall, "fine_tuned_cer"),
            "dev/critical_rate": as_float(overall, "fine_tuned_critical_rate"),
            "dev/critical_count": as_int(overall, "fine_tuned_critical_count"),
            "dev/delta_cer_vs_zero": as_float(overall, "delta_cer"),
            "dev/hard_cer": as_float(hard, "fine_tuned_cer"),
            "dev/hard_critical_count": as_int(hard, "fine_tuned_critical_count"),
            "dev/medium_cer": as_float(medium, "fine_tuned_cer"),
            "dev/medium_critical_count": as_int(medium, "fine_tuned_critical_count"),
            "dev/easy_cer": as_float(easy, "fine_tuned_cer"),
            "dev/easy_critical_count": as_int(easy, "fine_tuned_critical_count"),
            "checkpoint/global_step": step,
        }
        run.log(metrics, step=step)
        eval_rows.append({"step": step, **metrics})

        artifact = wandb.Artifact(f"{args.run_name}-dev-summary-step-{step}", type="dev_eval")
        artifact.add_file(str(eval_dir / "summary_by_group.csv"))
        run.log_artifact(artifact)

    if eval_rows:
        best_row = min(eval_rows, key=lambda row: row["dev/cer"])
        best_step = int(best_row["step"])
        retention = {
            "best_metric": "dev/cer",
            "best_step": best_step,
            "best_checkpoint": str(output_dir / f"checkpoint-{best_step}"),
            "deleted_checkpoints": [],
            "retained_checkpoints": [str(output_dir / f"checkpoint-{best_step}")],
        }
        fields = list(eval_rows[0].keys())
        with (output_dir / "dev_cer_by_checkpoint.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(eval_rows)
        run.save(str(output_dir / "dev_cer_by_checkpoint.csv"), base_path=str(output_dir))
        run.summary["best/dev_cer"] = best_row["dev/cer"]
        run.summary["best/dev_critical_count"] = best_row["dev/critical_count"]
        run.summary["best/dev_hard_cer"] = best_row["dev/hard_cer"]
        run.summary["best/global_step"] = best_step

        for checkpoint in sorted(output_dir.glob("checkpoint-*")):
            if checkpoint.name == f"checkpoint-{best_step}":
                continue
            shutil.rmtree(checkpoint)
            retention["deleted_checkpoints"].append(str(checkpoint))
        (output_dir / "checkpoint_retention.json").write_text(
            json.dumps(retention, indent=2) + "\n",
            encoding="utf-8",
        )
        run.save(str(output_dir / "checkpoint_retention.json"), base_path=str(output_dir))
    run.finish()


if __name__ == "__main__":
    main()
