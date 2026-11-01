"""Rulebook tests are pure routing: they never mint an audio-inspected receipt."""
from __future__ import annotations

import pytest

from examples.generate_reviewed_sg import INSTRUCTION, InstructProxy, run_attempts
from therapist_harness.audio_review import ReviewJudgment


@pytest.mark.parametrize("verdicts,expected_modes,action", [
    (["pass"], ["zero_shot"], "keep"),
    (["obvious_failure", "pass"], ["zero_shot", "zero_shot"], "keep"),
    (["obvious_failure", "obvious_failure", "pass"], ["zero_shot", "zero_shot", "instruct2"], "keep"),
    (["obvious_failure"] * 3, ["zero_shot", "zero_shot", "instruct2"], "exclude"),
])
def test_finite_rulebook_keeps_seed_reference_target_controlled(verdicts, expected_modes, action):
    calls = []
    def generate(mode, seed, attempt):
        calls.append((mode, seed, attempt))
        return {"fixture": True, "mode": mode}
    def review(_record, attempt):
        return ReviewJudgment(verdict=verdicts[attempt], reason="routing fixture, not an audio inspection")
    results, actual = run_attempts(42, generate, review)
    assert actual == action
    assert [c[0] for c in calls] == expected_modes
    assert [c[1] for c in calls] == list(range(42, 42 + len(verdicts)))
    assert all("audio_inspected" not in r["review"] for r in results)


@pytest.mark.parametrize("stage", ["generate", "review"])
def test_technical_error_stops_without_quality_retry(stage):
    calls = []
    def generate(mode, seed, attempt):
        calls.append(attempt)
        if stage == "generate": raise RuntimeError("technical generator error")
        return {"fixture": True}
    def review(_record, attempt):
        raise RuntimeError("audio input inaccessible")
    with pytest.raises(RuntimeError):
        run_attempts(42, generate, review)
    assert calls == [0]


def test_unclear_patient_speech_pass_does_not_trigger_regeneration():
    calls = []
    def generate(mode, seed, attempt):
        calls.append(mode)
        return {"fixture": True}
    results, action = run_attempts(42, generate, lambda *_: ReviewJudgment(
        verdict="pass", reason="患者語音含糊，不能確定逐字內容，但無非常明顯的生成失敗。"))
    assert action == "keep" and calls == ["zero_shot"]


def test_instruct_proxy_keeps_acoustic_reference_and_target():
    calls = []
    class Model:
        sample_rate = 24000
        def inference_instruct2(self, target, instruction, reference, **kwargs):
            calls.append((target, instruction, reference, kwargs))
            return iter([])
    proxy = InstructProxy(Model())
    proxy.inference_zero_shot("目標", "舊的參考文字提示", "/real/ref.wav", stream=False, text_frontend=False)
    assert calls == [("目標", INSTRUCTION, "/real/ref.wav", {"stream": False, "text_frontend": False})]
