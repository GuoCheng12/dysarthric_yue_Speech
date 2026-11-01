from collections import Counter
import json

import pytest

from therapist_harness.curriculum_v2 import (ENDPOINTS, CourseProposal, CourseReflection,
    allocation_actions, compile_block, learning_catalog, make_coverage, outcome_pack,
    patient_progress, register_proposal, summarize_pair, validate_reflection)


def entries():
    return [{"id": f"R{i+1:04d}", "patient": f"P{i//33+1:03d}", "content": f"C{i%33+1:02d}",
             "SDI": [1, 1, 0], "cer": .2, "recent_regressed_chars": 0, "regressed_chars": 0}
            for i in range(2117)]


def proposal(data):
    units = []
    for i in range(4):
        patients = [f"P{p:03d}" for p in range(i*4+1, i*4+5)]
        units.append({"unit_id": f"U{i+1:03d}", "patients": patients, "contents": [],
                      "target": "deletion", "evidence_ids": [next(r["id"] for r in data if r["patient"] == patients[0])],
                      "rationale": "fixture evidence", "expected_change": "test-only expectation"})
    return CourseProposal(diagnosis="test fixture", hypothesis="not an Agent result", risk="test fixture",
                          new_units=units, allocations=[{"unit_id": u["unit_id"], "exposures": 60, "reason": "fixture"} for u in units])


def test_full_three_arm_budget_and_frozen_course():
    data = entries()
    coverage = make_coverage([r["id"] for r in data])
    assert coverage == make_coverage([r["id"] for r in reversed(data)])
    assert sum(map(len, coverage.values())) == 2240
    assert len(set(i for block in coverage.values() for i in block)) == 2117
    p = proposal(data)
    registry = register_proposal(p, {}, data, [coverage[e] for e in ENDPOINTS[1:]])
    all_manifests = {}
    for arm in ("random", "static", "dynamic"):
        manifests = []
        for end in ENDPOINTS:
            allocations = None if arm == "random" or end == 20 else p.model_dump()["allocations"]
            manifest = compile_block(end, coverage[end], data, registry, allocations, manifests)
            assert max(Counter(r["id"] for r in manifest).values()) <= 2
            if end > 20:
                assert Counter(r["stream"] for r in manifest) == {"coverage": 400, "course": 240}
                for i in range(0, 640, 32):
                    assert Counter(r["stream"] for r in manifest[i:i+32]) == {"coverage": 20, "course": 12}
            manifests.append(manifest)
        flattened = [r for block in manifests for r in block]
        assert [r["position"] for r in flattened] == list(range(3200))
        assert Counter(r["step"] for r in flattened) == {s: 32 for s in range(1, 101)}
        assert len({r["id"] for r in flattened}) == 2117
        all_manifests[arm] = flattened
    assert all_manifests["static"] == all_manifests["dynamic"]
    assert [r for r in all_manifests["static"] if r["stream"] == "coverage"] == [r for r in all_manifests["random"] if r["stream"] == "coverage"]


@pytest.mark.parametrize("fault", ["unknown", "overlap", "budget", "evidence", "overwrite", "one_content", "capacity"])
def test_invalid_proposals_stop_without_repair(fault):
    data = entries()
    coverage = make_coverage([r["id"] for r in data])
    p = proposal(data)
    registry = {}
    if fault == "unknown":
        p.allocations[0].unit_id = "U999"
    elif fault == "overlap":
        p.new_units[1].patients = p.new_units[0].patients
        p.new_units[1].evidence_ids = p.new_units[0].evidence_ids
    elif fault == "budget":
        p.allocations[0].exposures = 59
    elif fault == "evidence":
        p.new_units[0].target = "maintenance"
    elif fault == "overwrite":
        registry = register_proposal(p, {}, data, [coverage[40]])
    elif fault == "one_content":
        p.new_units[0].contents = ["C01"]
    else:
        p.new_units[0].patients = ["P001"]
        p.allocations[0].exposures = 180
        for allocation in p.allocations[1:]:
            allocation.exposures = 20
    with pytest.raises(AssertionError):
        register_proposal(p, registry, data, [coverage[e] for e in ENDPOINTS[1:]])


def test_paused_units_remain_and_actions_are_computed():
    old = [{"unit_id": "U001", "exposures": 100}, {"unit_id": "U002", "exposures": 140}]
    new = [{"unit_id": "U001", "exposures": 120}, {"unit_id": "U003", "exposures": 120}]
    actions = allocation_actions(old, new)
    assert actions["U001"]["action"] == "increase"
    assert actions["U002"] == {"before": 140, "after": 0, "action": "pause"}
    assert actions["U003"]["action"] == "activate"


def sample(prediction, speaker="P001", gt="今日食飯", bucket="hard"):
    return dict(prediction=prediction, speaker_id=speaker, patient=speaker, clean_gt=gt, zero_shot_bucket=bucket)


def test_equal_critical_counts_do_not_mean_same_members():
    before = {"F0001": sample(""), "F0002": sample("今日食飯"), "F0003": sample("")}
    after = {"F0001": sample("今日食飯"), "F0002": sample(""), "F0003": sample("")}
    value = summarize_pair(before, after, sorted(before))
    transition = value["critical_transitions"]
    assert transition["exited"] == {"count": 1, "ids": ["F0001"]}
    assert transition["entered"] == {"count": 1, "ids": ["F0002"]}
    assert transition["remained"] == {"count": 1, "ids": ["F0003"]}
    assert summarize_pair(before, after, [])["after"] is None


def test_patient_macro_pools_characters_within_patient_before_averaging():
    data = {"R1": sample("", "P001", "飯"),
            "R2": sample("今日食飯", "P001", "今日食飯"),
            "R3": sample("今日", "P002", "今日食飯")}
    metric = summarize_pair(data, data, sorted(data))["after"]
    assert metric["mean_utterance_cer"] == .5
    assert metric["pooled_cer"] == 3/9
    assert metric["patient_macro_cer"] == (.2+.5)/2


def test_recent_changes_are_not_base_changes_and_exposures_include_coverage():
    common = dict(utt_id="private/1", speaker_id="private", clean_gt="今日食飯", zero_shot_bucket="easy", duration=2)
    previous = [{**common, "prediction": "今日食"}]
    current = [{**common, "prediction": "今日食飯"}]
    zero = {common["utt_id"]: {**common, "raw_pred": "今日食飯"}}
    manifests = [[{"id": "R0001", "stream": "coverage"}], [{"id": "R0001", "stream": "coverage"}, {"id": "R0001", "stream": "course"}]]
    cat, _ = learning_catalog(current, previous, zero, manifests, {"R0001": .4}, {"R0001": .3})
    r = cat["recordings"][0]
    assert r["recent_recovered_chars"] == 1 and r["recovered_chars"] == 0
    assert r["previous_cer"] == .25 and r["cer"] == 0
    assert r["exposures_total"] == 3 and r["exposures_recent"] == 2
    assert r["loss_previous"] == .4 and r["loss_current"] == .3
    assert "private" not in json.dumps(cat)
    summary = patient_progress(cat, {40: ["R0001"]})["P001"]
    assert summary["recent_recovered_chars"] == 1 and summary["exposures_total"] == 3
    assert summary["all_content_capacity_by_endpoint"] == {"40": 1}


def test_transfer_excludes_coverage_and_is_other_content_per_patient():
    mapping = {"R0001": {"patient": "P001", "content": "C01"},
               "R0002": {"patient": "P001", "content": "C02"},
               "R0003": {"patient": "P001", "content": "C03"},
               "R0004": {"patient": "P002", "content": "C03"},
               "R0005": {"patient": "P002", "content": "C01"}}
    before = {i: sample("", r["patient"]) for i, r in mapping.items()}
    after = {i: sample("今日食飯", r["patient"]) for i, r in mapping.items()}
    manifest = [{"id": "R0001", "unit_id": "U001", "stream": "course"},
                {"id": "R0002", "unit_id": None, "stream": "coverage"},
                {"id": "R0004", "unit_id": None, "stream": "coverage"}]
    p = {"allocations": [{"unit_id": "U001", "exposures": 1}]}
    o = outcome_pack(40, p, {"U001": {"patients": ["P001", "P002"]}}, manifest,
                     before, after, {}, {}, mapping, {"R0001": .4}, {"R0001": .2})
    transfer = o["observations"]["S40_U001_unpracticed_other_content"]
    assert transfer["rows"] == 2
    assert transfer["critical_transitions"]["exited"]["ids"] == ["R0003", "R0005"]
    assert o["observations"]["S40_U001_heldout_content"]["rows"] == 0
    assert o["global_observation_ids"] == [f"S40_feedback_{b}" for b in ("overall", "easy", "medium", "hard")]
    reflection = CourseReflection(observed_changes="fixture", next_guidance="fixture", units=[{
        "unit_id": "U001", "observation_ids": ["S40_U001_practiced"], "conclusion": "mixed",
        "explanation": "fixture", "next_action": "continue"}])
    validate_reflection(reflection, o)
    reflection.units[0].observation_ids = ["invented"]
    with pytest.raises(AssertionError):
        validate_reflection(reflection, o)


@pytest.mark.parametrize("citations,valid", [
    (["S40_U004_practiced"], True),
    (["S40_U004_practiced", "S40_feedback_easy"], True),
    (["S40_U004_practiced", "S40_feedback_overall", "S40_feedback_medium", "S40_feedback_hard"], True),
    (["S40_feedback_easy"], False),
    (["S40_U004_practiced", "S40_U001_practiced"], False),
    (["S40_U004_practiced", "S20_feedback_easy"], False),
    (["S40_U004_practiced", "invented"], False),
])
def test_reflection_can_cite_own_unit_and_current_global_feedback(citations, valid):
    global_ids = [f"S40_feedback_{b}" for b in ("overall", "easy", "medium", "hard")]
    own = ["S40_U004_practiced", "S40_U004_unpracticed_other_content", "S40_U004_heldout_content"]
    outcome = {"unit_observation_ids": {"U004": own}, "global_observation_ids": global_ids,
               "observations": {i: {} for i in own + global_ids + ["S40_U001_practiced"]}}
    reflection = CourseReflection(observed_changes="fixture", next_guidance="fixture", units=[{
        "unit_id": "U004", "observation_ids": citations, "conclusion": "mixed",
        "explanation": "Local maintenance improved; global easy regressed.", "next_action": "replace"}])
    if valid:
        validate_reflection(reflection, outcome)
    else:
        with pytest.raises(AssertionError):
            validate_reflection(reflection, outcome)


def test_reflection_cannot_cite_declared_but_missing_measurement():
    outcome = {"unit_observation_ids": {"U004": ["S40_U004_practiced"]},
               "global_observation_ids": ["S40_feedback_easy"],
               "observations": {"S40_U004_practiced": {}}}
    reflection = CourseReflection(observed_changes="fixture", next_guidance="fixture", units=[{
        "unit_id": "U004", "observation_ids": ["S40_U004_practiced", "S40_feedback_easy"],
        "conclusion": "mixed", "explanation": "fixture", "next_action": "replace"}])
    with pytest.raises(AssertionError, match="unknown observation"):
        validate_reflection(reflection, outcome)
