from collections import Counter
import importlib.util
import json
from pathlib import Path

import pytest

from therapist_harness.content_curriculum import (ContentPrescription, ContentReflection,
    character_states, compile_lesson_block, delivery_report, enrich_catalog,
    evidence_pack, lesson_outcome, region_bounds, validate_lesson_reflection,
    validate_prescription)


def fixture_catalog():
    contents = {f"C{i:02d}": f"今日食飯{i}" for i in range(1, 17)}
    data = []
    for ci, text in contents.items():
        for n in range(40):
            data.append({"id": f"R{len(data)+1:04d}", "patient": f"P{n+1:03d}",
                "content": ci, "bucket": "easy" if n % 2 else "hard", "seconds": 2,
                "cer": 0 if n % 2 else .2, "previous_cer": 0,
                "prediction": text if n % 2 else text.replace("飯", ""),
                "previous_prediction": text, "recent_regressed_chars": 0 if n % 2 else 1})
    return {"contents": contents, "recordings": data}


def fixture_proposal():
    lessons = []
    for index in range(2):
        practice = []
        for n, role in enumerate(("target", "target", "related", "protection")):
            cid = f"C{index*4+n+1:02d}"
            practice.append({"content_id": cid, "role": role, "region": "whole",
                "selection": "regression" if role == "target" else "correct" if role == "protection" else "any",
                "buckets": ["easy", "medium"] if role == "protection" else ["easy", "medium", "hard"],
                "exposures": 30, "reason": "unit-test fixture, not an intervention"})
        lessons.append({"lesson_id": f"L{index+1:03d}", "title": "fixture", "recognition_goal": "fixture",
            "evidence_ids": [f"E_{d['content_id']}" for d in practice[:2]], "hypothesis": "fixture",
            "practice": practice, "expected_transfer": "fixture", "revision_condition": "fixture"})
    return ContentPrescription(diagnosis="fixture", lessons=lessons, history_used="empty", risk="fixture")


def fixture_inputs():
    cat = fixture_catalog()
    entries = enrich_catalog(cat)
    background = [r["id"] for r in entries[320:]] + [r["id"] for r in entries[320:400]]
    return cat, entries, background, fixture_proposal()


def test_selectors_and_regions_are_measured_not_labels():
    assert region_bounds("今日食飯呀", "first_half") == (0, 2)
    assert region_bounds("今日食飯呀", "second_half") == (2, 5)
    assert character_states("今日食飯", "今日食") == ["correct", "correct", "correct", "deletion"]
    assert character_states("哈哈", "哈") == ["uncertain", "uncertain"]
    cat = {"contents": {"C01": "今日食飯"}, "recordings": [
        {"id": "R0001", "content": "C01", "prediction": "今日食", "previous_prediction": "今日食飯"},
        {"id": "R0002", "content": "C01", "prediction": "今日食飯呀", "previous_prediction": "今日食飯"}]}
    a, b = enrich_catalog(cat)
    assert a["region_hits"]["last_two"]["regression"] == 1
    assert a["region_hits"]["first_half"]["error"] == 0
    assert b["region_hits"]["whole"]["error"] == 0  # insertion-only is excluded
    assert b["region_hits"]["whole"]["correct"] == 0  # no partial-correct protection


def test_exact_content_quotas_balanced_patients_and_training_positions():
    cat, entries, background, proposal = fixture_inputs()
    manifest = compile_lesson_block(proposal, entries, background, 40)
    assert manifest == compile_lesson_block(proposal, list(reversed(entries)), background, 40)
    assert [r["position"] for r in manifest] == list(range(640, 1280))
    assert Counter(r["step"] for r in manifest) == {s: 32 for s in range(21, 41)}
    report = delivery_report(proposal, entries, background, manifest)
    for lesson in report.values():
        for item in lesson:
            assert item["actual_exposures"] == item["selector_match_exposures"] == 30
            assert max(item["patient_exposures"].values()) <= 2
            if item["role"] == "target":
                assert item["recent_regression_exposures"] == 30
    pack, _ = evidence_pack(cat, background, 20, {}, [{"reflection": "prior experience fixture"}])
    first = pack["contents"][0]
    capacity = next(x for x in first["selector_capacity"] if x[:3] == ["whole", "regression", "hard"])
    assert capacity[3:] == [20, 20, 40]
    assert pack["history"] == [{"reflection": "prior experience fixture"}]
    assert "/data/" not in json.dumps(pack)


def test_large_patient_pool_does_not_take_over_content_quota():
    _, entries, background, proposal = fixture_inputs()
    source = entries[0]
    entries += [{**source, "id": f"X{i:04d}"} for i in range(100)]
    manifest = compile_lesson_block(proposal, entries, background, 40)
    delivery = delivery_report(proposal, entries, background, manifest)
    counts = delivery["L001"][0]["patient_exposures"]
    assert counts["P001"] <= 2
    assert len(counts) == 20  # all eligible patients participate before recycling


@pytest.mark.parametrize("fault", ["unknown_content", "overlap", "budget", "unsupported_target",
    "wrong_evidence", "capacity", "missing_protection", "protection_hard", "wrong_coverage", "duplicate_bucket"])
def test_invalid_prescriptions_stop_without_fallback(fault):
    _, entries, background, p = fixture_inputs()
    d = p.lessons[0].practice[0]
    if fault == "unknown_content":
        d.content_id = "C99"
    elif fault == "overlap":
        p.lessons[1].practice[0].content_id = d.content_id
    elif fault == "budget":
        d.exposures = 29
    elif fault == "unsupported_target":
        d.selection = "any"
    elif fault == "wrong_evidence":
        p.lessons[0].evidence_ids = ["E_C99", "E_C98"]
    elif fault == "capacity":
        d.buckets = ["easy"]  # only hard rows have the target regression
    elif fault == "missing_protection":
        for lesson in p.lessons:
            lesson.practice[-1].role = "related"
    elif fault == "protection_hard":
        p.lessons[0].practice[-1].buckets = ["hard"]
    elif fault == "wrong_coverage":
        background[0] = "F0001"
    else:
        d.buckets = ["hard", "hard"]
    with pytest.raises(AssertionError):
        validate_prescription(p, entries, background)


def test_delivery_rejects_wrong_role_and_nonmatching_recording():
    _, entries, background, proposal = fixture_inputs()
    plan = compile_lesson_block(proposal, entries, background, 40)
    row = next(r for r in plan if r.get("role") == "target")
    row["role"] = "related"
    with pytest.raises(AssertionError):
        delivery_report(proposal, entries, background, plan)


def test_reassessment_distinguishes_content_transfer_and_excludes_coverage():
    p = fixture_proposal()
    mapping = {"R1": {"patient": "P1", "content": "C01"},
        "R2": {"patient": "P2", "content": "C01"},
        "R3": {"patient": "P3", "content": "C01"},
        "R4": {"patient": "P1", "content": "C10"},
        "R5": {"patient": "P1", "content": "C11"},
        "R6": {"patient": "P4", "content": "C05"}}
    def data(pred):
        return {i: {"speaker_id": r["patient"], "patient": r["patient"], "clean_gt": "今日食飯",
                    "prediction": pred, "zero_shot_bucket": "easy"} for i, r in mapping.items()}
    before, after = data("今日食"), data("今日食飯")
    manifest = [{"id": "R1", "unit_id": "L001", "role": "target"},
        {"id": "R3", "unit_id": None}, {"id": "R5", "unit_id": None},
        {"id": "R6", "unit_id": "L002", "role": "target"}]
    out = lesson_outcome(40, p, {"L001": [{"actual_exposures": 1}]}, manifest, mapping, before, after, {}, {})
    assert out["observations"]["S40_L001_same_content_unpracticed_patients"]["critical_transitions"]["neither"]["ids"] == ["R2"]
    assert out["observations"]["S40_L001_same_patients_other_content"]["critical_transitions"]["neither"]["ids"] == ["R4"]
    assert out["observations"]["S40_L001_heldout_content"]["after"] is None
    region = out["observations"]["S40_L001_C01_region"]
    assert region["before"]["deletion"] == 1 and region["after"]["deletion"] == 0
    assert region["recovered_positions"] == 1 and region["reference_positions"] == 4
    assert out["delivery"]["L001"][0]["actual_exposures"] == 1
    reflection = ContentReflection(observed_changes="unit-test fixture", next_guidance="fixture", lessons=[{
        "lesson_id": lid, "observation_ids": [f"S40_{lid}_practiced_target"],
        "delivery_interpretation": "fixture", "conclusion": "mixed", "next_action": "reduce",
        "explanation": "fixture", "reusable_lesson": "fixture, not real learning"} for lid in ("L001", "L002")])
    validate_lesson_reflection(reflection, out)
    reflection.lessons[0].observation_ids = ["S40_L002_practiced_target"]
    with pytest.raises(AssertionError):
        validate_lesson_reflection(reflection, out)


def test_requests_are_standalone_structured_and_keep_history():
    path = Path(__file__).resolve().parents[1]/"examples/prepare_content_curriculum.py"
    spec = importlib.util.spec_from_file_location("prepare_content", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"model": "gpt-5.6-sol", "reasoning_effort": "xhigh", "max_output_tokens": 32768}
    history = [{"outcome": {"delivery": {}}, "reflection": {"reusable_lesson": "fixture"}}]
    for kind, schema in (("prescribe", ContentPrescription), ("reflect", ContentReflection)):
        req = module.build_request(config, {"task": "unit test", "history": history}, schema, kind)
        assert req["reasoning"] == {"effort": "xhigh"} and req["store"] is False
        assert json.loads(req["input"][1]["content"])["history"] == history
        shape = req["text"]["format"]["schema"]
        for obj in [shape, *shape["$defs"].values()]:
            if obj.get("type") == "object":
                assert obj["additionalProperties"] is False
                assert set(obj["required"]) == set(obj["properties"])
        assert "TTS" not in req["input"][1]["content"]
