from therapist_harness.material_selection_offline import build_index, prepare_candidates, digest


def test_known_zero_exposures_are_distinct_from_unavailable_history():
    rows = [dict(utt_id=str(i), audio=f"/{i}.wav", speaker_id=f"p{i}", clean_gt=text,
                 duration=2, split="train") for i,text in enumerate(["北", "木", "媽"])]
    index = build_index(rows, [], {})
    brief = dict(producer="offline_fixture", round_index=1, source_evidence_sha256="a"*64,
        summary="练习韵尾", focuses=[dict(focus_id="k", purpose="target", level="coda",
        target_unit="k", reason="对照", evidence_ids=["example"])])
    unknown = prepare_candidates(index, brief)
    zero = prepare_candidates(index, brief, exposures={})
    partial = prepare_candidates(index, brief, exposures={"0":3})
    assert unknown["exposure_available"] is False and unknown["exposure_sha256"] is None
    assert all(r["prior_exposures"] is None for r in unknown["focuses"][0]["candidates"])
    assert zero["exposure_available"] is True and zero["exposure_sha256"] == digest({})
    assert all(r["prior_exposures"] == 0 for r in zero["focuses"][0]["candidates"])
    counts = {r["recording_id"]:r["prior_exposures"] for r in partial["focuses"][0]["candidates"]}
    assert counts == {"0":3,"1":0}
    assert partial["exposure_available"] and partial["exposure_sha256"] == digest({"0":3})
    assert [r["recording_id"] for r in zero["focuses"][0]["candidates"]] == [r["recording_id"] for r in unknown["focuses"][0]["candidates"]]
