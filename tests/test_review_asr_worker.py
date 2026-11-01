"""Worker protocol unit fixtures; no mocked result is evidence of actual audio review."""

import io
import json
from pathlib import Path

import pytest

from examples import transcribe_sg_review as module


@pytest.fixture
def setup(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir(); adapter.mkdir()
    (base / "config.json").write_text("{}"); (adapter / "adapter.json").write_text("{}")
    audio = tmp_path / "fixture.wav"; audio.write_bytes(b"not-live-audio")
    args = module.parse_args(["--worker", "--output", str(tmp_path / "session.json"),
        "--base-model", str(base), "--adapter", str(adapter), "--device", "cpu"])
    return args, audio


def test_one_load_for_multiple_requests_and_models_verified_on_close(setup):
    args, audio = setup; calls = []
    class Decoder:
        def __init__(self, _args): calls.append("load")
        def transcribe(self, path):
            calls.append(path)
            return {"generated_token_ids": [5, 151645], "prediction": "測試", "raw_output": "測試"}
    requests = [{"audio": str(audio), "output": str(audio.parent / f"r{i}.json")} for i in range(3)]
    stream = io.StringIO(''.join(json.dumps(x) + '\n' for x in [*requests, {"command": "close"}]))
    output = io.StringIO()
    final = module.worker(args, decoder_factory=Decoder, requests=stream, responses=output)
    assert calls == ["load", str(audio), str(audio), str(audio)]
    assert final["status"] == "complete" and final["base_files_unchanged"] and final["adapter_files_unchanged"]
    assert len(final["receipts"]) == 3
    for request in requests:
        record = json.loads(Path(request["output"]).read_text())
        assert record["actual_transcription"] is False
        assert record["worker_session_sha256"] == module.sha256_file(args.output)
        assert record["decode_config"] == module.DECODE_CONFIG


@pytest.mark.parametrize("field", ["target_text", "reference_text", "context", "prompt", "audio_id"])
def test_worker_queue_rejects_text_or_other_payload_fields(setup, field):
    args, audio = setup
    class Decoder:
        def __init__(self, _args): pass
        def transcribe(self, _audio): pytest.fail("invalid request must never reach ASR")
    request = {"audio": str(audio), "output": str(audio.parent / "r.json"), field: "not allowed"}
    with pytest.raises(ValueError, match="audio/output only"):
        module.worker(args, decoder_factory=Decoder, requests=io.StringIO(json.dumps(request)+'\n'), responses=io.StringIO())
    assert not (audio.parent / "r.json").exists()
    assert json.loads(args.output.with_suffix('.closed.json').read_text())["status"] == "technical_error"


@pytest.mark.parametrize("part", ["base_model", "adapter"])
def test_changed_model_files_fail_session_finalization(setup, part):
    args, audio = setup
    class Decoder:
        def __init__(self, _args): pass
        def transcribe(self, _audio):
            (getattr(args, part) / "new_file").write_text("changed")
            return {"generated_token_ids": [151645], "prediction": "", "raw_output": ""}
    request = {"audio": str(audio), "output": str(audio.parent / 'r.json')}
    final = module.worker(args, decoder_factory=Decoder,
        requests=io.StringIO(json.dumps(request)+'\n'+json.dumps({"command":"close"})+'\n'), responses=io.StringIO())
    assert final['status'] == 'technical_error'
    assert not final['base_files_unchanged' if part == 'base_model' else 'adapter_files_unchanged']
