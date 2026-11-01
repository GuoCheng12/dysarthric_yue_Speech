from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from resume_tts_reference_policy import install_empty_token_guard


def backend(original, completed=True):
    return SimpleNamespace(token2wav=original, lock=threading.Lock(),
        tts_speech_token_dict={"current": [], "other": [1]},
        llm_end_dict={"current": completed, "other": True},
        hift_cache_dict={"current": None, "other": None})


def test_empty_completed_sequence_is_retriable_and_cleans_only_its_session():
    def original(**kwargs):
        pytest.fail("Empty tokens must not reach flow/vocoder")
    model = backend(original)
    install_empty_token_guard(model)
    with pytest.raises(RuntimeError, match="^Speech Generator returned no speech tokens"):
        model.token2wav(token=SimpleNamespace(shape=(1, 0)), uuid="current")
    assert model.tts_speech_token_dict == {"other": [1]}
    assert model.llm_end_dict == {"other": True}
    assert model.hift_cache_dict == {"other": None}


def test_nonempty_tokens_are_forwarded_without_changing_arguments():
    calls = []
    def original(**kwargs):
        calls.append(kwargs)
        return "unchanged waveform"
    model = backend(original)
    install_empty_token_guard(model)
    token = SimpleNamespace(shape=(1, 23))
    assert model.token2wav(token=token, uuid="current", speed=1.) == "unchanged waveform"
    assert calls == [{"token": token, "uuid": "current", "speed": 1.}]
    assert "current" in model.llm_end_dict


def test_unfinished_llm_and_cuda_errors_are_not_reclassified():
    def original(**kwargs):
        raise RuntimeError("CUDA out of memory")
    model = backend(original, completed=False)
    install_empty_token_guard(model)
    with pytest.raises(RuntimeError, match="^CUDA out of memory$"):
        model.token2wav(token=SimpleNamespace(shape=(1, 0)), uuid="current")
    assert "current" in model.llm_end_dict
