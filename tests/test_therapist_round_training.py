from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from test_therapist_outcome import _protocol
from therapist_harness.round_training import build_round_training_command
from therapist_harness.schema import ArtifactRef, LoRAConfig
from therapist_harness.store import sha256_directory, sha256_file


def _inputs(tmp_path: Path, *, rows: int = 8):
    protocol = _protocol(tmp_path)
    parent = tmp_path / "checkpoint-50"
    parent.mkdir()
    (parent / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (parent / "trainer_state.json").write_text('{"global_step":50}', encoding="utf-8")
    (parent / "scheduler.pt").write_bytes(b"old scheduler, deliberately not loaded")
    manifest = tmp_path / "round_train.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"audio": f"/private/utterance-{index}.wav", "text": "我想飲水"})
            + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )
    return (
        protocol,
        ArtifactRef(path=str(parent), sha256=sha256_directory(parent), kind="directory"),
        ArtifactRef(path=str(manifest), sha256=sha256_file(manifest), expected_rows=rows),
        tmp_path / "round_001_output",
        sys.executable,
    )


def test_round_command_loads_only_weights_and_adds_fresh_steps(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    protocol, parent, manifest, output, executable = inputs
    command = build_round_training_command(*inputs)
    options = dict(zip(command[2::2], command[3::2]))
    assert command[:2] == [
        executable,
        str(Path(__file__).resolve().parents[1] / "finetune/scripts/qwen3_asr_lora_sft.py"),
    ]
    assert options["--lora_init_adapter_path"] == parent.path
    assert "--resume_from" not in command and "--resume" not in command
    # Even with old global_step=50 and scheduler state present, this toy round
    # requests two fresh updates. Production's same rule requests 82, not 32.
    assert options["--max_steps"] == "2"
    assert options["--epochs"] == "1"
    assert options["--batch_size"] == "1"
    assert options["--grad_acc"] == "4"
    assert options["--lora_rank"] == str(protocol.lora.rank)
    assert options["--lora_alpha"] == str(protocol.lora.alpha)
    assert options["--lora_dropout"] == str(protocol.lora.dropout)
    assert options["--lora_target_modules"] == ",".join(protocol.lora.target_modules)
    assert options["--lr"] == str(protocol.lora.learning_rate)
    assert options["--warmup_ratio"] == "0.02"
    assert options["--lr_scheduler_type"] == protocol.lora.scheduler
    assert options["--save_steps"] == options["--max_steps"]
    assert options["--save_final_checkpoint"] == "1"
    assert options["--train_file"] == manifest.path
    assert options["--eval_file"] == ""
    assert protocol.feedback_split.path not in command
    assert protocol.sealed_test_split.path not in command
    assert not output.exists()  # Building a command neither trains nor writes.
    assert (Path(parent.path) / "trainer_state.json").read_text() == '{"global_step":50}'


def test_generated_command_matches_the_maintained_training_parser(tmp_path: Path) -> None:
    from finetune.scripts.qwen3_asr_lora_sft import parse_args

    inputs = _inputs(tmp_path)
    args = parse_args(build_round_training_command(*inputs)[2:])
    assert args.max_steps == inputs[0].lora.optimizer_steps_per_round
    assert args.lora_init_adapter_path == inputs[1].path
    assert args.resume_from == "" and args.resume == 0
    assert args.eval_file == ""


def test_round_command_rejects_signed_but_wrong_row_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="row count differs from protocol: 7 != 8"):
        build_round_training_command(*_inputs(tmp_path, rows=7))


def test_round_command_verifies_parent_weights_hash(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    (Path(inputs[1].path) / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        build_round_training_command(*inputs)


def test_round_command_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = inputs[3]
    output.mkdir()
    existing = output / "trainer_state.json"
    existing.write_text("previous experiment", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be new or empty"):
        build_round_training_command(*inputs)
    assert existing.read_text() == "previous experiment"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initialization", "trainer_resume"),
        ("reset_optimizer_each_round", False),
        ("reset_scheduler_each_round", False),
    ],
)
def test_protocol_rejects_mixed_continuation_semantics(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _protocol(tmp_path).lora.model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValidationError):
        LoRAConfig.model_validate(payload)


@pytest.mark.parametrize("resume", [["--resume_from", "/checkpoint-50"], ["--resume", "1"]])
def test_training_parser_rejects_weights_only_with_trainer_resume(resume) -> None:
    from finetune.scripts.qwen3_asr_lora_sft import parse_args

    with pytest.raises(SystemExit) as error:
        parse_args(["--lora_init_adapter_path", "/checkpoint-50", *resume])
    assert error.value.code == 2


def test_training_parser_accepts_explicitly_disabled_resume() -> None:
    from finetune.scripts.qwen3_asr_lora_sft import parse_args

    args = parse_args(["--lora_init_adapter_path", "/checkpoint-50", "--resume", "0"])
    assert args.lora_init_adapter_path == "/checkpoint-50"
    assert args.resume == 0
