"""Build, but never execute, the frozen weights-only per-round ASR command."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import ArtifactRef, RunProtocol
from .store import verify_artifact


TRAINING_SCRIPT = (
    Path(__file__).resolve().parents[2] / "finetune/scripts/qwen3_asr_lora_sft.py"
)


def build_round_training_command(
    protocol: RunProtocol,
    parent_adapter: ArtifactRef,
    train_manifest: ArtifactRef,
    output_dir: Path,
    python_executable: str,
) -> list[str]:
    """Load parent weights and execute one *new* round's optimizer-step budget.

    No Trainer checkpoint state is resumed: optimizer, scheduler and global_step
    start fresh. A parent named checkpoint-50 plus max_steps=82 means 82 new
    updates, not stopping at cumulative step 82 after only 32 updates.
    """

    if protocol.protocol_status != "frozen":
        raise ValueError("only a frozen protocol can build a round training command")
    if parent_adapter.kind != "directory" or train_manifest.kind != "file":
        raise ValueError("round training requires an adapter directory and manifest file")
    verify_artifact(parent_adapter)
    verify_artifact(train_manifest)
    verify_artifact(protocol.base_model_fingerprint)
    row_count = 0
    with Path(train_manifest.path).open("r", encoding="utf-8") as handle:
        for row_count, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict) or any(
                not isinstance(row.get(field), str) or not row[field].strip()
                for field in ("audio", "text")
            ):
                raise ValueError(
                    f"training row {row_count} requires audio and text strings"
                )
    expected_rows = (
        protocol.real_replay_count_per_round + protocol.synthetic_count_per_round
    )
    if row_count != expected_rows:
        raise ValueError(
            f"round training row count differs from protocol: {row_count} != {expected_rows}"
        )
    if not output_dir.is_absolute():
        raise ValueError("round output_dir must be absolute")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"round output_dir must be new or empty: {output_dir}")
    executable = Path(python_executable)
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("python_executable must identify an existing absolute file")

    lora = protocol.lora
    options = {
        "--model_path": protocol.base_model_dir,
        "--train_file": train_manifest.path,
        "--eval_file": "",
        "--output_dir": str(output_dir),
        "--lora_init_adapter_path": parent_adapter.path,
        "--lora_rank": lora.rank,
        "--lora_alpha": lora.alpha,
        "--lora_dropout": lora.dropout,
        "--lora_target_modules": ",".join(lora.target_modules),
        "--batch_size": lora.per_device_batch_size,
        "--grad_acc": lora.gradient_accumulation_steps,
        "--lr": lora.learning_rate,
        "--epochs": lora.training_passes_per_round,
        "--max_steps": lora.optimizer_steps_per_round,
        "--lr_scheduler_type": lora.scheduler,
        "--warmup_ratio": lora.warmup_ratio,
        "--seed": lora.seed,
        "--save_strategy": "steps",
        "--save_steps": lora.optimizer_steps_per_round,
        "--save_total_limit": 1,
        "--save_final_checkpoint": 1,
        "--report_to": "none",
    }
    return [python_executable, str(TRAINING_SCRIPT)] + [
        item for key, value in options.items() for item in (key, str(value))
    ]


__all__ = ["build_round_training_command"]
