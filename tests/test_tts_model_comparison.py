import sys
import pytest
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tts_model_comparison import ROOT, PREVIOUS, read, SPECS, MIN_FREE
from report_tts_model_comparison import VARIANTS, NEW, error_retention


def test_frozen_paired_subset():
    selected = read(ROOT / "selection.json")
    source = read(PREVIOUS / "selection.json")
    expected = [x for bucket in ("easy", "medium", "hard")
                for x in [v for v in source if v["row"]["zero_shot_bucket"] == bucket][::2]]
    assert selected == expected
    assert len(selected) == len({x["pair"]["sample_id"] for x in selected}) == 24
    assert Counter(x["row"]["zero_shot_bucket"] for x in selected) == {"easy": 8, "medium": 8, "hard": 8}
    assert all(x["pair"]["reference_wav"] != x["audio"]["real"] for x in selected)


def test_conditions_and_storage_scope():
    assert len(SPECS) == 3
    assert len(NEW) == 6
    assert len(VARIANTS) == len(set(VARIANTS)) == 11
    assert MIN_FREE == 20 * 1024**3
    assert "no new ASR or TTS training" in read(ROOT / "protocol.json")["scope"]


@pytest.mark.parametrize("model", SPECS)
def test_full_wavs_and_paired_inputs(model):
    import soundfile as sf
    from therapist_harness.store import sha256_file
    for item in read(ROOT / "selection.json"):
        for condition in ("tts", "reconstruction"):
            receipt = read(ROOT / "audio" / model / item["pair"]["sample_id"] / f"{condition}.json")
            audio = Path(receipt["audio"])
            info = sf.info(audio)
            assert info.frames == receipt["samples"]
            assert info.samplerate == receipt["sample_rate"]
            assert info.subtype == "FLOAT"
            assert sha256_file(audio) == receipt["sha256"]
            assert receipt["seed"] == item["seed"]
            assert receipt["input_reference_sha256"] == item["pair"]["reference_wav_sha256"]


def test_error_retention_separates_preserved_and_new_errors():
    pairs = {"u": {"real": {"clean_gt": "甲乙丙", "prediction": "甲丁丙"},
                   "same": {"prediction": "甲丁丙"}, "normalized": {"prediction": "甲乙丙"},
                   "different": {"prediction": "甲戊丙"}}}
    result = error_retention(pairs, ("same", "normalized", "different"))
    assert result["rows"] == result["real_SD_errors"] == 1
    assert result["counts"] == {"same": {"retained": 1, "new": 0},
                                "normalized": {"retained": 0, "new": 0},
                                "different": {"retained": 0, "new": 1}}
