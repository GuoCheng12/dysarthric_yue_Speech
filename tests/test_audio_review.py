from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from therapist_harness.audio_review import (
    AudioReviewer,
    ReviewJudgment,
    next_generation_action,
)


def response(payload):
    return SimpleNamespace(
        id="mock-response",
        choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload))
        )],
    )


class MockAudioClient:
    def __init__(self, *, broken_probe=False, unavailable=False):
        self.requests = []
        self.broken_probe = broken_probe
        self.unavailable = unavailable
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        content = kwargs["messages"][1]["content"]
        if len(content) == 1:
            raw = base64.b64decode(content[0]["input_audio"]["data"])
            samples, _ = sf.read(io.BytesIO(raw))
            label = "audible_speech" if np.any(samples) else "silence"
            if self.broken_probe:
                label = "audible_speech"
            return response({"label": label, "reason": "Mock, not a live inspection."})
        if self.unavailable:
            return response({"error": "audio_unavailable"})
        return response({"verdict": "pass", "reason": "模糊發音，未聽到明顯生成失敗。"})


@pytest.fixture
def clips(tmp_path):
    rate = 16000
    candidate = tmp_path / "candidate.wav"
    reference = tmp_path / "reference.wav"
    t = np.arange(rate) / rate
    sf.write(candidate, 0.15 * np.sin(2 * np.pi * 300 * t), rate)
    sf.write(reference, 0.10 * np.sin(2 * np.pi * 600 * t), rate)
    return candidate, reference


def reviewer(client):
    return AudioReviewer(model="mock-audio", base_url="https://example.invalid/v1", client=client)


def test_both_actual_audio_inputs_and_bound_receipt(clips):
    candidate, reference = clips
    client = MockAudioClient()
    agent = reviewer(client)
    evidence = agent.verify_audio_capability(reference)
    result = agent.review(candidate, "今日去街市", reference, "食咗飯未")
    content = client.requests[-1]["messages"][1]["content"]
    audio_parts = [part for part in content if part["type"] == "input_audio"]
    assert len(audio_parts) == 2
    assert base64.b64decode(audio_parts[0]["input_audio"]["data"]) == candidate.read_bytes()
    assert base64.b64decode(audio_parts[1]["input_audio"]["data"]) == reference.read_bytes()
    assert result.candidate_audio_sha256 == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert result.reference_audio_sha256 == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert result.target_text == "今日去街市"
    assert result.reference_text == "食咗飯未"
    assert result.verdict == "pass"
    assert result.fidelity_claim == "coarse_only_not_word_verified"
    assert result.prompt_sha256 == agent.prompt_sha256
    assert evidence["passed"]
    assert not evidence["live_api"]
    assert not result.audio_inspected  # Mock success cannot admit real preference data.
    assert str(candidate) not in json.dumps(content)


def test_capability_probe_has_identical_text_and_distinct_actual_sound(clips):
    candidate, _ = clips
    client = MockAudioClient()
    agent = reviewer(client)
    evidence = agent.verify_audio_capability(candidate)
    assert client.requests[0]["messages"][0] == client.requests[1]["messages"][0]
    assert evidence["checks"][0]["audio_sha256"] != evidence["checks"][1]["audio_sha256"]
    assert [x["observed"]["label"] for x in evidence["checks"]] == ["silence", "audible_speech"]


def test_review_requires_successful_audio_capability_probe(clips):
    candidate, reference = clips
    client = MockAudioClient(broken_probe=True)
    agent = reviewer(client)
    with pytest.raises(RuntimeError, match="verify_audio_capability"):
        agent.review(candidate, "今日去街市", reference)
    assert not client.requests
    with pytest.raises(RuntimeError, match="failed the audio-only"):
        agent.verify_audio_capability(reference)
    with pytest.raises(RuntimeError, match="verify_audio_capability"):
        agent.review(candidate, "今日去街市", reference)
    assert len(client.requests) == 2


def test_inaccessible_attachment_is_technical_error_not_pass(clips):
    candidate, reference = clips
    agent = reviewer(MockAudioClient(unavailable=True))
    agent.verify_audio_capability(reference)
    with pytest.raises(RuntimeError, match="could not access/hear"):
        agent.review(candidate, "今日去街市", reference)


def test_simple_rulebook_is_bounded_and_only_obvious_failure_retries():
    passed = ReviewJudgment(verdict="pass", reason="Speech is unclear; no obvious failure heard.")
    failed = ReviewJudgment(verdict="obvious_failure", reason="A short fragment loops mechanically for the whole clip.")
    assert [next_generation_action(passed, i) for i in range(3)] == ["keep"] * 3
    assert [next_generation_action(failed, i) for i in range(3)] == [
        "retry_zero_shot", "retry_instruct2", "exclude"
    ]
    with pytest.raises(ValueError, match="attempt"):
        next_generation_action(failed, 3)
    with pytest.raises(ValidationError):
        ReviewJudgment(verdict="uncertain", reason="unclear patient speech")
    with pytest.raises(ValidationError):
        ReviewJudgment(verdict="pass", reason="ok", cer=0.9)


def test_missing_credential_does_not_fall_back_to_transcription(monkeypatch):
    monkeypatch.delenv("AUDIO_REVIEW_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AUDIO_REVIEW_TEST_KEY"):
        AudioReviewer(model="configured-audio-model", base_url="https://example.invalid/v1",
                      api_key_env="AUDIO_REVIEW_TEST_KEY")
