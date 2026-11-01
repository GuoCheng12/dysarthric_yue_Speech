import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tts_reconstruction_probe import SOURCE, read, select48


def test_selection_is_fixed_balanced_and_without_replacement():
    q = read(SOURCE / "selection.json")["Q"]
    selected = select48(q)
    assert selected == select48(q)
    assert len({x["utt_id"] for x in selected}) == 48
    assert Counter(x["zero_shot_bucket"] for x in selected) == {"easy":16,"medium":16,"hard":16}
    assert all(x in q for x in selected)
    for bucket in ("easy", "medium", "hard"):
        subset = [x for x in selected if x["zero_shot_bucket"] == bucket]
        available = {x["speaker_id"] for x in q if x["zero_shot_bucket"] == bucket}
        assert len({x["speaker_id"] for x in subset}) == min(16,len(available))
