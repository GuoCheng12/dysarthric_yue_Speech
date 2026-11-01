from __future__ import annotations

import math

import pytest

from therapist_harness.evidence import extract_evidence
from therapist_harness.schema import EvidenceTargetDescriptor


def row(
    utt_id: str,
    speaker_id: str,
    content_id: str,
    bucket: str,
    gt: str,
    prediction: str,
) -> dict[str, str]:
    return {
        "utt_id": utt_id,
        "speaker_id": speaker_id,
        "prompt_id": content_id,
        "zero_shot_bucket": bucket,
        "clean_gt": gt,
        "prediction": prediction,
    }


def extract(
    current: list[dict[str, str]],
    m0: list[dict[str, str]],
    zero_shot: list[dict[str, str]],
    *,
    previous: list[dict[str, str]] | None = None,
):
    return extract_evidence(
        current,
        m0,
        zero_shot,
        current_prediction_column="prediction",
        m0_prediction_column="prediction",
        zero_shot_prediction_column="prediction",
        previous_rows=previous,
        previous_prediction_column="prediction" if previous is not None else None,
    )


def test_metrics_denominators_transitions_and_union_character_clusters() -> None:
    zero_shot = [
        row("u1", "p1", "c1", "easy", "abcd", "abcd"),
        row("u2", "p1", "c2", "easy", "z", ""),
        row("u3", "p2", "c3", "hard", "xy", "xy"),
    ]
    current = [
        row("u1", "p1", "c1", "easy", "abcd", "abxd"),
        row("u2", "p1", "c2", "easy", "z", ""),
        row("u3", "p2", "c3", "hard", "xy", "xyq"),
    ]

    result = extract(current, zero_shot, zero_shot)
    metrics = result.current_metrics
    assert metrics.overall.sample_count == 3
    assert metrics.overall.total_edits == 3
    assert metrics.overall.total_reference_characters == 7
    assert math.isclose(metrics.overall.mean_utterance_cer, 7 / 12)
    assert math.isclose(metrics.overall.pooled_cer, 3 / 7)
    assert math.isclose(metrics.overall.patient_macro_cer, (2 / 5 + 1 / 2) / 2)
    assert metrics.overall.critical_count == 2
    assert metrics.overall.exact_count == 0
    assert metrics.easy.total_edits == 2
    assert metrics.hard.pooled_cer == 0.5

    transitions = result.comparisons.current_vs_zero_shot.transitions.overall
    assert transitions.exact.model_dump() == {
        "recovered": 0,
        "newly_wrong": 2,
        "preserved_correct": 0,
        "persistent_wrong": 1,
    }
    assert transitions.utterance_cer.model_dump() == {
        "improved": 0,
        "worsened": 2,
        "tied": 1,
    }
    assert result.comparisons.current_vs_previous is None

    clusters = {
        (cluster.operation, cluster.reference_unit, cluster.hypothesis_unit): cluster
        for cluster in result.error_clusters
        if cluster.level == "character"
    }
    substitution = clusters[("substitution", "c", "x")]
    deletion = clusters[("deletion", "z", "")]
    insertion = clusters[("insertion", "", "q")]
    assert substitution.observations.current.opportunity_count == 1
    assert substitution.observations.current.patient_count == 1
    assert substitution.observations.zero_shot.rate == 0.0
    assert deletion.observations.current.rate == 1.0
    assert deletion.observations.zero_shot.rate == 1.0
    assert insertion.observations.current.opportunity_count == 10
    assert insertion.observations.current.rate == 0.1
    assert insertion.targetable is False
    assert insertion.tts_actionable is False
    assert insertion.tts_actionability_reasons == [
        "character_insertion_has_no_gt_side_tts_carrier"
    ]
    assert "pure_insertion_is_not_a_primary_training_target" in (
        insertion.selection_reasons
    )


def test_previous_snapshot_is_paired_and_reported() -> None:
    m0 = [row("u1", "p1", "c1", "medium", "甲乙", "甲")]
    previous = [row("u1", "p1", "c1", "medium", "甲乙", "甲乙")]
    current = [row("u1", "p1", "c1", "medium", "甲乙", "甲")]

    result = extract(current, m0, m0, previous=previous)

    assert result.previous_metrics is not None
    comparison = result.comparisons.current_vs_previous
    assert comparison is not None
    assert comparison.transitions.overall.exact.newly_wrong == 1
    deletion = next(
        cluster
        for cluster in result.error_clusters
        if cluster.level == "character" and cluster.operation == "deletion"
    )
    assert deletion.observations.m0.rate == 1.0
    assert deletion.observations.previous is not None
    assert deletion.observations.previous.rate == 0.0


def test_cluster_union_keeps_an_error_that_current_fixed() -> None:
    m0 = [row("u1", "p1", "c1", "easy", "甲乙", "甲")]
    current = [row("u1", "p1", "c1", "easy", "甲乙", "甲乙")]

    result = extract(current, m0, m0)
    deletion = next(
        cluster
        for cluster in result.error_clusters
        if cluster.level == "character" and cluster.operation == "deletion"
    )
    assert deletion.observations.current.error_count == 0
    assert deletion.observations.m0.error_count == 1
    assert deletion.targetable is False


def test_pairing_set_mismatch_fails_fast() -> None:
    current = [row("u1", "p1", "c1", "easy", "甲", "甲")]
    m0 = [row("u2", "p1", "c1", "easy", "甲", "甲")]

    with pytest.raises(ValueError, match="utterance sets differ"):
        extract(current, m0, m0)


def test_prompt_id_is_required_and_never_inferred() -> None:
    missing_content = {
        "utt_id": "u1",
        "speaker_id": "p1",
        "zero_shot_bucket": "easy",
        "clean_gt": "甲",
        "prediction": "甲",
    }
    with pytest.raises(ValueError, match="prompt_id"):
        extract_evidence(
            [missing_content],
            [missing_content],
            [missing_content],
            current_prediction_column="prediction",
            m0_prediction_column="prediction",
            zero_shot_prediction_column="prediction",
        )


def test_previous_rows_and_prediction_column_are_atomic() -> None:
    rows = [row("u1", "p1", "c1", "easy", "甲", "甲")]
    with pytest.raises(ValueError, match="must be supplied together"):
        extract_evidence(
            rows,
            rows,
            rows,
            current_prediction_column="prediction",
            m0_prediction_column="prediction",
            zero_shot_prediction_column="prediction",
            previous_rows=rows,
        )


def test_phone_clusters_aggregate_across_syllable_contexts_and_report_rates() -> None:
    rows = [
        row("u1", "p1", "c1", "easy", "怕", "爸"),
        row("u2", "p2", "c2", "hard", "平", "並"),
        row("u3", "p3", "c3", "easy", "拍", "拍"),
        row("u4", "p4", "c4", "hard", "噴", "噴"),
    ]

    result = extract(rows, rows, rows)
    onset = [
        cluster
        for cluster in result.error_clusters
        if cluster.level == "onset"
        and cluster.operation == "substitution"
        and cluster.reference_unit == "p"
        and cluster.hypothesis_unit == "b"
    ]

    assert len(onset) == 1
    cluster = onset[0]
    assert cluster.left_context is None
    assert cluster.right_context is None
    assert cluster.targetable is True
    assert cluster.observations.current.error_count == 2
    assert cluster.observations.current.opportunity_count == 4
    assert cluster.observations.current.rate == 0.5
    assert cluster.observations.current.by_difficulty["easy"].model_dump() == {
        "error_count": 1,
        "opportunity_count": 2,
        "rate": 0.5,
    }
    assert cluster.observations.current.by_difficulty["medium"].model_dump() == {
        "error_count": 0,
        "opportunity_count": 0,
        "rate": 0.0,
    }
    assert cluster.observations.current.by_difficulty["hard"].model_dump() == {
        "error_count": 1,
        "opportunity_count": 2,
        "rate": 0.5,
    }


def test_phone_component_insertion_can_be_targetable() -> None:
    rows = [
        row("u1", "p1", "c1", "easy", "呀", "巴"),
        row("u2", "p2", "c2", "hard", "呀", "巴"),
    ]

    result = extract(rows, rows, rows)
    onset_insertion = next(
        cluster
        for cluster in result.error_clusters
        if cluster.level == "onset"
        and cluster.operation == "insertion"
        and cluster.reference_unit == "∅"
        and cluster.hypothesis_unit == "b"
    )

    assert onset_insertion.targetable is True
    assert onset_insertion.tts_actionable is True
    assert onset_insertion.tts_actionability_reasons == [
        "phone_component_has_pronunciation_facing_reference_carrier"
    ]
    assert onset_insertion.observations.current.opportunity_count == 2
    assert "pure_insertion_is_not_a_primary_training_target" not in (
        onset_insertion.selection_reasons
    )


def test_homophonic_character_substitutions_are_not_tts_actionable() -> None:
    rows = [
        row("u1", "p1", "c1", "easy", "褂", "掛"),
        row("u2", "p2", "c2", "hard", "柺", "枴"),
    ]

    result = extract(rows, rows, rows)
    clusters = {
        (cluster.reference_unit, cluster.hypothesis_unit): cluster
        for cluster in result.error_clusters
        if cluster.level == "character" and cluster.operation == "substitution"
    }

    for pair in (("褂", "掛"), ("柺", "枴")):
        assert clusters[pair].tts_actionable is False
        assert clusters[pair].tts_actionability_reasons == [
            "character_substitution_is_homophonic_in_isolated_jyutping"
        ]


def test_nonhomophonic_character_substitution_is_tts_actionable() -> None:
    rows = [row("u1", "p1", "c1", "easy", "怕", "爸")]

    result = extract(rows, rows, rows)
    character_cluster = next(
        cluster
        for cluster in result.error_clusters
        if cluster.level == "character" and cluster.operation == "substitution"
    )

    assert character_cluster.tts_actionable is True
    assert character_cluster.tts_actionability_reasons == [
        "character_substitution_has_distinct_isolated_jyutping"
    ]
    assert any(
        cluster.level != "character" and cluster.tts_actionable
        for cluster in result.error_clusters
    )


def test_character_deletion_requires_resolved_isolated_jyutping() -> None:
    rows = [
        row("u1", "p1", "c1", "easy", "甲", ""),
        row("u2", "p2", "c2", "hard", "𠮷", ""),
    ]

    result = extract(rows, rows, rows)
    deletions = {
        cluster.reference_unit: cluster
        for cluster in result.error_clusters
        if cluster.level == "character" and cluster.operation == "deletion"
    }

    assert deletions["甲"].tts_actionable is True
    assert deletions["甲"].tts_actionability_reasons == [
        "pronounceable_reference_character_can_be_rehearsed"
    ]
    assert deletions["𠮷"].tts_actionable is False
    assert deletions["𠮷"].tts_actionability_reasons == [
        "surface_carrier_projection_or_pronunciation_is_not_unique"
    ]


def test_phone_attribution_uses_surface_reading_without_changing_cer() -> None:
    rows = [row("u1", "p1", "c1", "hard", "沒", "木")]

    result = extract(rows, rows, rows)

    assert result.current_metrics.overall.total_edits == 1
    assert result.current_metrics.overall.total_reference_characters == 1
    phone_clusters = {
        (
            cluster.level,
            cluster.operation,
            cluster.reference_unit,
            cluster.hypothesis_unit,
        )
        for cluster in result.error_clusters
        if cluster.level != "character"
    }
    assert ("coda", "substitution", "t", "k") in phone_clusters
    assert ("nucleus", "substitution", "o", "u") not in phone_clusters
    assert ("coda", "substitution", "u", "k") not in phone_clusters
    coverage = result.current_phonology_coverage.overall
    assert coverage.normalization_projection_unresolved_forced_substitutions == 0
    assert coverage.not_cer_forced_phonologically_different_substitutions == 0
    assert coverage.phone_attributed_forced_substitutions == 1


def test_cer_equivalent_spelling_is_not_added_as_a_phone_error() -> None:
    rows = [row("u1", "p1", "c1", "hard", "冇怕", "無爸")]

    result = extract(rows, rows, rows)

    assert result.current_metrics.overall.total_edits == 1
    phone_clusters = {
        (
            cluster.level,
            cluster.operation,
            cluster.reference_unit,
            cluster.hypothesis_unit,
        )
        for cluster in result.error_clusters
        if cluster.level != "character"
    }
    assert ("onset", "substitution", "p", "b") in phone_clusters
    assert ("tone", "substitution", "5", "4") not in phone_clusters
    coverage = result.current_phonology_coverage.overall
    assert coverage.normalization_projection_unresolved_forced_substitutions == 0
    assert coverage.not_cer_forced_phonologically_different_substitutions == 1
    assert coverage.phone_attributed_forced_substitutions == 1


def test_context_sensitive_substitutions_are_excluded_from_phone_clusters() -> None:
    rows = [
        row("u1", "p1", "c1", "easy", "銀行", "銀巷"),
        row("u2", "p2", "c2", "hard", "銀行", "銀巷"),
    ]

    result = extract(rows, rows, rows)

    assert all(cluster.level == "character" for cluster in result.error_clusters)
    assert (
        result.current_phonology_coverage.overall.context_sensitive_forced_substitutions
        == 2
    )
    assert (
        result.current_phonology_coverage.overall.phone_attributed_forced_substitutions
        == 0
    )


def test_top_k_keeps_targetable_clusters_before_larger_unsupported_clusters() -> None:
    rows = [
        row("u1", "p1", "c10", "easy", "乙", ""),
        row("u2", "p1", "c11", "easy", "乙", ""),
        row("u3", "p1", "c12", "easy", "乙", ""),
        row("u4", "p2", "c1", "hard", "甲", ""),
        row("u5", "p3", "c2", "hard", "甲", ""),
    ]

    result = extract_evidence(
        rows,
        rows,
        rows,
        current_prediction_column="prediction",
        m0_prediction_column="prediction",
        zero_shot_prediction_column="prediction",
        max_clusters_per_level=1,
    )

    assert len(result.error_clusters) == 1
    selected = result.error_clusters[0]
    assert selected.level == "character"
    assert selected.reference_unit == "甲"
    assert selected.targetable is True


def test_historical_target_is_retained_outside_new_cluster_limit() -> None:
    m0 = [
        row("u1", "p1", "c1", "easy", "甲乙", "乙"),
        row("u2", "p2", "c2", "hard", "甲乙", "乙"),
    ]
    current = [
        row("u1", "p1", "c1", "easy", "甲乙", "甲"),
        row("u2", "p2", "c2", "hard", "甲乙", "甲"),
    ]
    first = extract(m0, m0, m0)
    historical = next(
        cluster
        for cluster in first.error_clusters
        if cluster.level == "character" and cluster.reference_unit == "甲"
    )
    descriptor = EvidenceTargetDescriptor(
        evidence_id=historical.evidence_id,
        operation=historical.operation,
        level=historical.level,
        reference_unit=historical.reference_unit,
        hypothesis_unit=historical.hypothesis_unit,
        left_context=historical.left_context,
        right_context=historical.right_context,
    )

    result = extract_evidence(
        current,
        m0,
        m0,
        current_prediction_column="prediction",
        m0_prediction_column="prediction",
        zero_shot_prediction_column="prediction",
        max_clusters_per_level=1,
        prior_target_ids=[historical.evidence_id],
        prior_targets=[descriptor],
    )

    character_units = [
        cluster.reference_unit
        for cluster in result.error_clusters
        if cluster.level == "character"
    ]
    assert character_units == ["甲", "乙"]
