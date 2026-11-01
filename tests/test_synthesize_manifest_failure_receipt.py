from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_synthesize_manifest import MockSpeechGenerator, row, write_manifest
from synthesis.cosyvoice3_patient_sft.synthesize_manifest import (
    materialize_manifest,
    sha256_file,
)


def test_failure_receipt_retains_valid_audio_and_stops_without_retry(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"fixed reference")
    manifest = tmp_path / "manifest.jsonl"
    rows = [row(tmp_path, f"sample-{i}", reference) for i in range(3)]
    write_manifest(manifest, rows)
    model = MockSpeechGenerator()
    original_inference = model.inference_zero_shot
    attempted: list[str] = []

    def inference(text, *args, **kwargs):
        attempted.append(text)
        if len(attempted) == 2:
            raise RuntimeError("fixture generation failure")
        yield from original_inference(text, *args, **kwargs)

    model.inference_zero_shot = inference
    summary_path = tmp_path / "summary.json"
    with pytest.raises(RuntimeError, match="fixture generation failure"):
        materialize_manifest(
            manifest,
            summary_path,
            runtime_factory=lambda: (model, {"model": "mock"}),
        )

    assert attempted == [rows[0]["target_text"], rows[1]["target_text"]]
    assert not summary_path.exists()
    receipt = json.loads((tmp_path / "summary.failure.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["stage"] == "synthesis"
    assert receipt["failed_sample_id"] == "sample-1"
    assert receipt["completed_sample_ids"] == ["sample-0"]
    assert receipt["output_count"] == 1
    assert receipt["error"] == {
        "type": "RuntimeError",
        "message": "fixture generation failure",
    }
    output = Path(receipt["outputs"][0]["output_wav"])
    assert output.is_file()
    assert sha256_file(output) == receipt["outputs"][0]["output_wav_sha256"]
    assert not Path(rows[1]["output_wav"]).exists()
    assert not Path(rows[2]["output_wav"]).exists()
    assert not list(tmp_path.glob(".*.tmp.*"))

    # No automatic resume or overwrite is introduced by keeping partial work.
    with pytest.raises(FileExistsError, match="output_wav already exists"):
        materialize_manifest(
            manifest,
            summary_path,
            runtime_factory=lambda: pytest.fail("must not reload or retry"),
        )
    assert attempted == [rows[0]["target_text"], rows[1]["target_text"]]


def test_runtime_failure_is_recorded_without_a_failed_sample(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"fixed reference")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [row(tmp_path, "sample-0", reference)])

    def broken_runtime():
        raise RuntimeError("fixture model load failure")

    with pytest.raises(RuntimeError, match="fixture model load failure"):
        materialize_manifest(
            manifest, tmp_path / "summary.json", runtime_factory=broken_runtime
        )
    receipt = json.loads((tmp_path / "summary.failure.json").read_text())
    assert receipt["stage"] == "runtime_load"
    assert receipt["failed_sample_id"] is None
    assert receipt["completed_sample_ids"] == []
    assert receipt["output_count"] == 0
    assert not (tmp_path / "summary.json").exists()
