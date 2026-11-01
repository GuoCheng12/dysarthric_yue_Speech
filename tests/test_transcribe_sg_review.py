import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "transcribe_sg_review_under_test",
    Path(__file__).resolve().parents[1] / "examples/transcribe_sg_review.py",
)
transcription = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transcription)


@pytest.fixture
def args(tmp_path):
    audio = tmp_path / "candidate.wav"
    # Only decoder-independent receipt tests use this opaque payload; never live audio.
    audio.write_bytes(b"unit-test-not-a-live-waveform")
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir(); adapter.mkdir()
    (base / "config.json").write_text("{}")
    (adapter / "adapter_config.json").write_text("{}")
    return transcription.parse_args(["--audio", str(audio), "--output", str(tmp_path / "receipt.json"),
                                     "--base-model", str(base), "--adapter", str(adapter), "--device", "cpu"])


@pytest.mark.parametrize("extra", [{"target_text": "禁止提示答案"}, {"reference_text": "參考內容"}, {"prompt": "答案"}])
def test_manifest_rejects_target_reference_or_other_text_context(args, extra):
    args.manifest = args.audio.parent / "manifest.jsonl"
    args.manifest.write_text(json.dumps({"audio": str(args.audio), **extra}))
    args.audio = None
    with pytest.raises(ValueError, match="no target or reference"):
        transcription.audio_inputs(args)


@pytest.mark.parametrize("tokens,error", [([], "asr_missing_eos"), ([1, 2], "asr_missing_eos"),
    ([1] * 512, "asr_token_limit"), ([1] * 511 + [151645], "asr_token_limit")])
def test_cap_or_missing_stop_is_a_technical_error_not_sg_failure(tokens, error):
    receipt = transcription.transcription_record({}, {"generated_token_ids": tokens,
        "prediction": "含糊發音", "raw_output": "含糊發音"}, {})
    assert receipt["status"] == "technical_error"
    assert receipt["technical_error"] == error and not receipt["review_usable"]
    assert not receipt["actual_transcription"]
    assert "verdict" not in receipt


def test_actual_empty_asr_transcript_with_eos_is_not_a_generation_failure():
    receipt = transcription.transcription_record({}, {"generated_token_ids": [151645], "prediction": "", "raw_output": ""}, {})
    assert receipt["status"] == "complete" and receipt["prediction"] == ""
    assert not receipt["actual_transcription"] and not receipt["review_usable"]


def test_test_decoder_cannot_claim_live_transcription_and_output_is_bound(args):
    calls = []
    class Decoder:
        def __init__(self, _args):
            calls.append("loaded")
        def transcribe(self, audio):
            calls.append(audio)
            return {"generated_token_ids": [12, 151645], "prediction": "測試", "raw_output": "測試"}
    receipt = transcription.run(args, decoder_factory=Decoder)
    assert calls == ["loaded", str(args.audio)]
    assert receipt["status"] == "complete"
    assert receipt["actual_transcription"] is False and receipt["review_usable"] is False
    assert receipt["audio_sha256"] == transcription.sha256_file(args.audio)
    assert receipt["adapter_sha256"] == transcription.sha256_directory(args.adapter)
    assert receipt["base_model_sha256"] == transcription.sha256_directory(args.base_model)
    assert receipt["decode_config"]["context"] == ""
    assert receipt["decode_config"]["max_new_tokens"] == 512
    assert receipt["decode_config"]["do_sample"] is False
    with pytest.raises(FileExistsError, match="immutable"):
        transcription.run(args, decoder_factory=Decoder)


def test_batch_loads_once_and_persists_failures_as_technical(args):
    manifest = args.audio.parent / "batch.jsonl"
    manifest.write_text("\n".join(json.dumps({"audio": str(args.audio), "audio_id": str(i)}) for i in range(2)))
    args.manifest, args.audio = manifest, None
    loads = []
    class Decoder:
        def __init__(self, _args):
            loads.append(1)
        def transcribe(self, audio):
            raise RuntimeError("decoder failure")
    receipt = transcription.run(args, decoder_factory=Decoder)
    assert loads == [1]
    assert receipt["status"] == "technical_error"
    assert len(receipt["records"]) == 2
    assert all(r["technical_error"] == "RuntimeError" and not r["actual_transcription"] for r in receipt["records"])


def test_adapter_mutation_invalidates_completed_decode(args):
    class Decoder:
        def __init__(self, _args):
            pass
        def transcribe(self, audio):
            (args.adapter / "adapter_config.json").write_text('{"changed":true}')
            return {"generated_token_ids": [151645], "prediction": "", "raw_output": ""}
    receipt = transcription.run(args, decoder_factory=Decoder)
    assert receipt["status"] == "technical_error"
    assert receipt["technical_error"] == "adapter_changed"
    assert not receipt["adapter_files_unchanged"]
