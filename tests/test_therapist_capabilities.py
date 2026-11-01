from __future__ import annotations

import pytest

from therapist_harness.capabilities import (
    count_capability_carriers,
    evaluate_capability_anchors,
    extract_capability_anchors,
)


def row(utt_id, patient, content, gt, prediction=None, difficulty="easy"):
    return {
        "utt_id": utt_id,
        "speaker_id": patient,
        "prompt_id": content,
        "zero_shot_bucket": difficulty,
        "clean_gt": gt,
        "prediction": gt if prediction is None else prediction,
    }


def anchors(current, m0=None, previous=None, max_anchors=12):
    return extract_capability_anchors(
        current,
        current if m0 is None else m0,
        current_prediction_column="prediction",
        m0_prediction_column="prediction",
        previous_rows=previous,
        previous_prediction_column="prediction" if previous is not None else None,
        max_anchors=max_anchors,
    )


def k_anchor(rows, **kwargs):
    return next(
        item
        for item in anchors(rows, **kwargs)
        if item.level == "coda" and item.reference_unit == "k"
    )


def evaluate(anchor, before, after):
    return evaluate_capability_anchors(
        [anchor],
        before,
        after,
        before_prediction_column="prediction",
        after_prediction_column="prediction",
    )[0]


def test_error_free_capability_is_eligible_and_has_no_private_rows():
    rows = [row("u1", "p1", "c1", "食飯"), row("u2", "p2", "c2", "返屋企")]
    anchor = k_anchor(rows)
    assert anchor.current_correct_count == anchor.opportunity_count == 2
    assert anchor.m0_lost_count == anchor.newly_acquired_count == 0
    assert (
        anchor.current_correct_patient_count
        == anchor.current_correct_content_count
        == 2
    )
    assert anchor.previous_correct_count is None
    serialized = anchor.model_dump_json()
    assert not any(
        private in serialized for private in ("食飯", "返屋企", "p1", "c1", "u1")
    )
    assert count_capability_carriers("我食粥先返屋企。", anchor) == 3
    assert evaluate(anchor, rows, rows).response == "retained"


def test_hard_only_or_single_content_does_not_create_protection_evidence():
    hard = [
        row("u1", "p1", "c1", "食飯", difficulty="hard"),
        row("u2", "p2", "c2", "返屋企", difficulty="hard"),
    ]
    assert anchors(hard) == []
    one_content = [row("u1", "p1", "c1", "食飯"), row("u2", "p2", "c1", "食飯")]
    assert anchors(one_content) == []


def test_bounded_selection_balances_difficulty_and_component_levels():
    rows = [
        row(f"{difficulty}-1", "p1", "c1", "我食飯再返屋企", difficulty=difficulty)
        for difficulty in ("easy", "medium")
    ] + [
        row(f"{difficulty}-2", "p2", "c2", "你返屋企先食飯", difficulty=difficulty)
        for difficulty in ("easy", "medium")
    ]
    selected = anchors(rows)
    assert len(selected) == 12
    assert sum(item.difficulty == "easy" for item in selected) == 6
    assert {item.level for item in selected} == {"onset", "nucleus", "coda", "tone"}
    assert selected == anchors(list(reversed(rows)))


def test_wrong_to_different_wrong_is_not_an_acquired_capability():
    m0 = [
        row("u1", "p1", "c1", "食飯"),
        row("u2", "p2", "c2", "返屋企"),
        row("u3", "p3", "c3", "食", "色"),
    ]
    current = [*m0[:2], {**m0[2], "prediction": "息"}]
    anchor = k_anchor(current, m0=m0)
    assert anchor.current_correct_count == 2
    assert anchor.newly_acquired_count == 0
    assert evaluate(k_anchor(m0), m0, current).newly_correct_count == 0


def test_fixed_correct_cohort_loss_is_not_cancelled_by_new_success():
    before = [
        row("u1", "p1", "c1", "食飯"),
        row("u2", "p2", "c2", "返屋企"),
        row("u3", "p3", "c3", "食", "色"),
    ]
    after = [
        {**before[0], "prediction": "色飯"},
        before[1],
        {**before[2], "prediction": "食"},
    ]
    change = evaluate(k_anchor(before), before, after)
    assert change.before_correct_count == change.after_correct_count == 2
    assert change.retained_count == change.lost_count == change.newly_correct_count == 1
    assert change.lost_to_error_count == 1
    assert change.response == "regressed"
    assert change.citation_id == f"result.capability.{change.evidence_id}"


def test_ambiguous_alignment_is_unresolved_not_retained():
    before = [row("u1", "p1", "c1", "食食"), row("u2", "p2", "c2", "返屋企")]
    after = [{**before[0], "prediction": "食"}, before[1]]
    change = evaluate(k_anchor(before), before, after)
    assert change.lost_count == change.lost_to_unresolved_count == 2
    assert change.after_unresolved_count == 2
    assert change.retained_count == 1
    assert change.response == "inconclusive"


def test_insertions_do_not_hide_behind_retained_carrier_characters():
    before = [row("u1", "p1", "c1", "食飯"), row("u2", "p2", "c2", "返屋企")]
    after = [{**before[0], "prediction": "我食飯"}, before[1]]
    change = evaluate(k_anchor(before), before, after)
    assert change.lost_count == 0
    assert change.before_cohort_character_insertions == 0
    assert change.after_cohort_character_insertions == 1
    assert change.response == "inconclusive"


def test_m0_and_previous_retention_use_their_own_fixed_correct_cohorts():
    m0 = [
        row("u1", "p1", "c1", "食飯"),
        row("u2", "p2", "c2", "返屋企"),
        row("u3", "p3", "c3", "食", "色"),
        row("u4", "p4", "c4", "黑", "克"),
    ]
    previous = [
        m0[0],
        {**m0[1], "prediction": "返企"},
        {**m0[2], "prediction": "食"},
        m0[3],
    ]
    current = [m0[0], m0[1], m0[2], {**m0[3], "prediction": "黑"}]
    anchor = k_anchor(current, m0=m0, previous=previous)
    assert (
        anchor.m0_correct_count,
        anchor.m0_preserved_count,
        anchor.m0_lost_count,
    ) == (2, 2, 0)
    assert anchor.newly_acquired_count == 1
    assert (
        anchor.previous_correct_count,
        anchor.previous_preserved_count,
        anchor.previous_lost_count,
        anchor.previous_acquired_count,
    ) == (2, 1, 1, 2)


def test_evaluation_rejects_different_pre_round_correct_cohort_counts():
    before = [row("u1", "p1", "c1", "食飯"), row("u2", "p2", "c2", "返屋企")]
    anchor = k_anchor(before)
    changed_before = [{**before[0], "prediction": "色飯"}, before[1]]
    with pytest.raises(ValueError, match="pre-round capability cohort differs"):
        evaluate(anchor, changed_before, changed_before)


def test_pairing_metadata_and_previous_snapshot_requirements_are_not_ignored():
    rows = [row("u1", "p1", "c1", "食飯"), row("u2", "p2", "c2", "返屋企")]
    with pytest.raises(ValueError, match="metadata mismatch"):
        anchors(rows, m0=[{**rows[0], "speaker_id": "p9"}, rows[1]])
    with pytest.raises(ValueError, match="supplied together"):
        extract_capability_anchors(
            rows,
            rows,
            current_prediction_column="prediction",
            m0_prediction_column="prediction",
            previous_rows=rows,
        )
