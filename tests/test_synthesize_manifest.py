from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from synthesis.cosyvoice3_patient_sft.synthesize_manifest import (
    PROMPT_INSTRUCTION,
    load_frozen_manifest,
    materialize_manifest,
    sha256_file,
)


class MockSpeechGenerator:
    sample_rate = 24_000

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def inference_zero_shot(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        yield {"tts_speech": torch.full((1, 12_000), 0.1)}


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def row(tmp_path: Path, sample_id: str, reference: Path) -> dict:
    return {
        "sample_id": sample_id,
        "target_text": f"目標句子{sample_id}",
        "reference_text": f"參考句子{sample_id}",
        "reference_wav": str(reference),
        "reference_wav_sha256": sha256_file(reference),
        "output_wav": str(tmp_path / f"{sample_id}.wav"),
        "seed": len(sample_id),
    }


def test_invalid_manifest_fails_before_runtime_is_loaded(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")
    manifest = tmp_path / "manifest.jsonl"
    first = row(tmp_path, "sample-a", reference)
    second = row(tmp_path, "sample-b", reference)
    second["output_wav"] = first["output_wav"]
    write_manifest(manifest, [first, second])
    runtime_loads = 0

    def runtime_factory():
        nonlocal runtime_loads
        runtime_loads += 1
        return MockSpeechGenerator(), {}

    with pytest.raises(ValueError, match="duplicate output_wav"):
        materialize_manifest(
            manifest.resolve(),
            (tmp_path / "summary.json").resolve(),
            runtime_factory=runtime_factory,
        )
    assert runtime_loads == 0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("reference_wav", "relative.wav", "absolute"),
        ("output_wav", "relative.wav", "absolute"),
        ("seed", True, "signed int64"),
        ("target_text", "", "non-empty"),
    ],
)
def test_manifest_rejects_invalid_required_values(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")
    manifest = tmp_path / "manifest.jsonl"
    payload = row(tmp_path, "sample-a", reference)
    payload[field] = value
    write_manifest(manifest, [payload])

    with pytest.raises((ValueError, FileNotFoundError), match=error):
        load_frozen_manifest(manifest.resolve())


def test_each_row_makes_exactly_one_fixed_inference_call(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"fixed reference bytes")
    manifest = tmp_path / "manifest.jsonl"
    rows = [row(tmp_path, "sample-a", reference), row(tmp_path, "sample-b", reference)]
    write_manifest(manifest, rows)
    model = MockSpeechGenerator()
    summary_path = (tmp_path / "summary.json").resolve()

    summary = materialize_manifest(
        manifest.resolve(),
        summary_path,
        runtime_factory=lambda: (model, {"model": "mock-speech-generator"}),
    )

    assert len(model.calls) == len(rows) == 2
    for call, expected in zip(model.calls, rows, strict=True):
        args, kwargs = call
        assert args == (
            expected["target_text"],
            PROMPT_INSTRUCTION + expected["reference_text"],
            expected["reference_wav"],
        )
        assert kwargs == {"stream": False, "text_frontend": False}
    assert summary["status"] == "complete"
    assert summary["output_count"] == 2
    assert all(item["inference_call_count"] == 1 for item in summary["outputs"])
    assert all(Path(item["output_wav"]).is_file() for item in summary["outputs"])
    assert all(len(item["output_wav_sha256"]) == 64 for item in summary["outputs"])
    assert all(
        item["duration_seconds"] == pytest.approx(0.5) for item in summary["outputs"]
    )
    assert (
        json.loads(summary_path.read_text(encoding="utf-8"))["outputs"]
        == summary["outputs"]
    )


def test_failed_batch_publishes_no_partial_outputs(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"fixed reference bytes")
    manifest = tmp_path / "manifest.jsonl"
    rows = [row(tmp_path, "sample-a", reference), row(tmp_path, "sample-b", reference)]
    write_manifest(manifest, rows)
    model = MockSpeechGenerator()
    calls = 0

    def inference(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic failure")
        yield {"tts_speech": torch.full((1, 12_000), 0.1)}

    model.inference_zero_shot = inference
    summary_path = (tmp_path / "summary.json").resolve()
    with pytest.raises(RuntimeError, match="synthetic failure"):
        materialize_manifest(
            manifest.resolve(),
            summary_path,
            runtime_factory=lambda: (model, {"model": "mock"}),
        )

    assert not summary_path.exists()
    assert all(not Path(item["output_wav"]).exists() for item in rows)
