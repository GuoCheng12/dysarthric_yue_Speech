import sys
from pathlib import Path
from collections import Counter

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from absorption_probe import ARMS, interleave, small_synthetic, retention  # noqa: E402


@pytest.mark.parametrize("s,r", ARMS)
def test_complete_updates_and_no_resampling(s, r):
    real = [("real", i) for i in range(r)]
    synth = [("synthetic", i) for i in range(s)]
    mixed = interleave(real, synth)
    assert len(mixed) % 32 == 0
    assert len(set(mixed)) == r + s
    assert [x for x in mixed if x[0] == "real"] == real
    assert [x for x in mixed if x[0] == "synthetic"] == synth
    windows = [sum(x[0] == "synthetic" for x in mixed[i:i + 32]) for i in range(0, len(mixed), 32)]
    assert max(windows) - min(windows) <= 1


def test_nested_synthetic_keeps_all_patients_and_texts():
    large = [{"reference_wav": f"p{p:02d}", "target_text": f"t{t}", "id": (p, t)}
             for p in range(16) for t in range(6)]
    small = small_synthetic(large)
    assert len(small) == 32
    assert set(x["id"] for x in small) <= set(x["id"] for x in large)
    assert set(Counter(x["reference_wav"] for x in small).values()) == {2}
    assert sorted(Counter(x["target_text"] for x in small).values()) == [5, 5, 5, 5, 6, 6]


def test_retention_separates_repair_from_damage():
    row = {"utt_id": "x", "clean_gt": "甲乙丙", "zero_shot_bucket": "easy"}
    result = retention([{**row, "prediction": "甲丁丙"}], [{**row, "prediction": "甲乙戊"}])
    assert result["easy"] == {"correct_to_correct": 1, "error_to_correct": 1, "correct_to_error": 1}
    assert result["overall"] == result["easy"]
