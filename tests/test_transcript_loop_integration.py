"""CPU contract tests only; fixture receipts are not live ASR or Agent evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_preference_loop_runner import (
    PARENT, _validate, _write_json, _write_rows, registered as registered,
)
from therapist_harness.preference_contract import PreferenceLoopConfig
from therapist_harness.preference_data import build_preference_pairs, validate_sampling_inputs
from therapist_harness.preference_loop import PACKAGE, dpo_instructions
from therapist_harness.store import sha256_file


@pytest.fixture
def transcript_registered(registered, tmp_path):
    item = registered
    item.protocol.preference_loop = PreferenceLoopConfig(
        audio_review_model="text-agent-fixture", audio_review_base_url="https://example.invalid",
        review_mode="asr_transcript", review_asr_adapter="/fixture/initial-r-only",
        review_asr_adapter_sha256=PARENT, review_asr_base_model_sha256="b" * 64,
    )
    for assignment, review in zip(item.assignments, item.reviews, strict=True):
        path = tmp_path / f"{assignment['sample_id']}.transcription.json"
        payload = {
            "status": "complete", "actual_transcription": True,
            "audio": assignment["output_wav"], "audio_sha256": review["candidate_audio_sha256"],
            "prediction": "存在辨識錯誤", "raw_output": "language Cantonese<asr_text>存在辨識錯誤",
            "model": "/fixture/base", "base_model_sha256": "b" * 64, "adapter_sha256": PARENT,
            "decode_config": {"language": "Cantonese", "context": "", "do_sample": False, "num_beams": 1},
            "ended_with_eos": True, "reached_max_new_tokens": False,
        }
        _write_json(path, payload)
        review.update(
            review_mode="asr_transcript", audio_inspected=False, transcript_inspected=True,
            asr_transcription=payload["prediction"], asr_raw_output=payload["raw_output"],
            transcription_model=payload["model"], transcription_base_model_sha256="b" * 64,
            transcription_adapter_sha256=PARENT, transcription_receipt_path=str(path),
            transcription_receipt_sha256=sha256_file(path), prompt_sha256="c" * 64,
            response_id="explicit-counterfactual-review-fixture",
            sample_id=assignment["sample_id"], audio_id=assignment["sample_id"],
        )
    raw_inputs = [{key: value for key, value in row.items() if key != "review"} for row in item.inputs]
    item.inputs = validate_sampling_inputs(raw_inputs, item.real, item.feedback, item.reviews)
    # The SG worker puts sample IDs on the ledger, outside its inline review.
    for row in item.inputs:
        if row["source"] == "synthetic":
            row["review"] = {key: value for key, value in row["review"].items()
                             if key not in {"sample_id", "audio_id"}}
    _write_rows(item.paths["preference_reviews"], item.reviews)
    _persist_inputs(item)
    return item


def _persist_inputs(item):
    _write_rows(item.paths["preference_inputs"], item.inputs)
    item.receipt["validated_inputs_sha256"] = sha256_file(item.paths["preference_inputs"])
    _write_json(item.paths["preference_sampling_receipt"], item.receipt)
    item.pairs, item.audit = build_preference_pairs(item.inputs, item.candidates, PARENT)


def test_transcript_registry_admits_bound_mode_with_ledger_only_ids(transcript_registered):
    _validate(transcript_registered)
    pair = next(row for row in transcript_registered.pairs if row["source"] == "synthetic")
    assert pair["metadata"]["review"]["audio_inspected"] is False
    assert pair["metadata"]["review"]["transcription_adapter_sha256"] == PARENT


@pytest.mark.parametrize("field", ["review_asr_adapter_sha256", "review_asr_base_model_sha256"])
def test_registry_checks_frozen_asr_against_protocol(transcript_registered, field):
    item = transcript_registered
    item.protocol.preference_loop = item.protocol.preference_loop.model_copy(update={field: "d" * 64})
    with pytest.raises(ValueError, match="frozen review ASR"):
        _validate(item)


def test_audio_mode_ledger_cannot_enter_transcript_protocol(transcript_registered):
    item = transcript_registered
    failed = item.reviews[-1]
    assert failed["verdict"] == "obvious_failure"
    failed.update(review_mode="audio", audio_inspected=True)
    _write_rows(item.paths["preference_reviews"], item.reviews)
    with pytest.raises(ValueError, match="review mode differs"):
        _validate(item)


@pytest.mark.parametrize("mutation", ["audio_mode", "other_asr"])
def test_inline_review_cannot_disagree_with_authoritative_ledger(transcript_registered, tmp_path, mutation):
    item = transcript_registered
    review = next(row for row in item.inputs if row["source"] == "synthetic")["review"]
    if mutation == "audio_mode":
        review.update(review_mode="audio", audio_inspected=True)
    else:
        receipt = json.loads(Path(review["transcription_receipt_path"]).read_text())
        receipt["adapter_sha256"] = "d" * 64
        path = tmp_path / "different_asr.transcription.json"
        _write_json(path, receipt)
        review.update(transcription_adapter_sha256="d" * 64,
                      transcription_receipt_path=str(path), transcription_receipt_sha256=sha256_file(path))
    _persist_inputs(item)
    with pytest.raises(ValueError, match="registered review ledger"):
        _validate(item)


@pytest.mark.parametrize("field", ["target_text", "reference_text", "context"])
def test_review_asr_input_manifest_rejects_text_hints(registered, tmp_path, field):
    from examples.transcribe_sg_review import audio_inputs

    path = tmp_path / "audio_only.jsonl"
    _write_rows(path, [{"audio": registered.real[0]["audio"], field: "唔可以畀辨識器睇嘅提示"}])
    with pytest.raises(ValueError, match="no target or reference text"):
        audio_inputs(SimpleNamespace(audio=None, manifest=path))


def test_sg_transcription_subprocess_only_receives_audio_and_fixed_asr(tmp_path, monkeypatch):
    from examples import generate_reviewed_sg as worker

    commands = []
    monkeypatch.setattr(worker.subprocess, "run", lambda command, **kwargs: commands.append(command))
    args = SimpleNamespace(review_asr_model=Path("/fixture/base"),
        review_asr_adapter=Path("/fixture/fixed-initial-r-only"), review_asr_device="cuda:0",
        target_text="目標文本不可傳入ASR", reference_text="參考文本不可傳入ASR")
    output = tmp_path / "receipt.json"
    worker.transcribe_for_review("/fixture/audio.wav", output, args)
    assert len(commands) == 1
    command = commands[0]
    assert command[2:] == ["--audio", "/fixture/audio.wav", "--output", str(output),
                          "--base-model", "/fixture/base", "--adapter", "/fixture/fixed-initial-r-only",
                          "--device", "cuda:0"]
    assert not output.exists()  # The mocked command produced no actual transcription.


@pytest.mark.parametrize("path", [PACKAGE / "prompts/therapist.md", PACKAGE / "prompts/reflection.md", PACKAGE / "skills/v000/SKILL.md"])
def test_mas_and_reflection_are_told_the_agent_only_reads_transcription(path):
    adapted = dpo_instructions(path.read_text(), "asr_transcript")
    assert "not hear the audio" in adapted
    assert "only preference sampling uses the current ASR" in adapted
    assert "no\nCER threshold" in adapted
    assert "audio Agent screens only gross generation failures" not in adapted
