import pytest

from therapist_harness.real_curriculum import (CurriculumProposal, catalog,
    compile_proposal, rule_selection, validate_selection)


def candidates():
    return [{"id": f"R{i:04d}", "patient": f"P{i%40:03d}", "content": f"C{i%30:02d}",
             "bucket": ("easy", "medium", "hard")[i%3], "cer": i / 300} for i in range(300)]


@pytest.mark.parametrize("method", ["random", "difficulty"])
def test_rules_unique_capped_reproducible(method):
    entries = candidates()
    background = [r["id"] for r in entries[:32]]
    ids = rule_selection(entries, background, method, 42)
    assert ids == rule_selection(entries, background, method, 42)
    assert len(set(ids)) == 64
    assert set(ids).isdisjoint(background)
    if method == "difficulty":
        assert min(int(i[1:]) for i in ids) >= 200


def test_compiler_cites_known_recordings_and_rejects_unknowns():
    entries = candidates()
    background = [r["id"] for r in entries[:32]]
    ids = rule_selection(entries, background, "random", 42)
    proposal = CurriculumProposal(diagnosis="measured", hypothesis="practice", risk="regression",
        groups=[dict(role="repair", rationale="current errors", evidence_ids=[ids[0]], sample_ids=ids,
                     expected_change="lower CER")])
    assert compile_proposal(proposal, entries, background) == ids
    proposal.groups[0].evidence_ids = ["fake"]
    with pytest.raises(AssertionError, match="citation"):
        compile_proposal(proposal, entries, background)


@pytest.mark.parametrize("bad", ["duplicate", "unknown", "background"])
def test_bad_selection_fails(bad):
    entries = candidates()
    background = [r["id"] for r in entries[:32]]
    ids = rule_selection(entries, background, "random", 42)
    ids[0] = {"duplicate": ids[1], "unknown": "fake", "background": background[0]}[bad]
    with pytest.raises(AssertionError):
        validate_selection(ids, entries, background)


def test_catalog_current_errors_and_anonymization():
    common = dict(utt_id="local_patient/1", speaker_id="local_patient", clean_gt="今日食飯",
                  duration="2", zero_shot_bucket="easy")
    cat, mapping = catalog([{**common, "prediction": "今日食"}],
                           {common["utt_id"]: {**common, "raw_pred": "今日食飯"}})
    r = cat["recordings"][0]
    assert r["cer"] == .25 and r["zero_cer"] == 0
    assert r["SDI"] == [0, 1, 0] and r["regressed_chars"] == 1
    assert r["patient"] == "P001" and r["id"] == "R0001"
    assert "local_patient" not in str(cat)
    assert mapping["R0001"]["utt_id"] == common["utt_id"]


def test_patient_and_content_caps_apply_to_background_too():
    entries = candidates()
    for r in entries:
        r["patient"] = "P001"
    with pytest.raises(AssertionError, match="patient"):
        validate_selection([r["id"] for r in entries[32:96]], entries, [r["id"] for r in entries[:32]])
