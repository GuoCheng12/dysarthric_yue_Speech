from __future__ import annotations

import math

import pytest

from therapist_harness.evidence import extract_evidence


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


def test_metrics_denominators_transitions_and_character_clusters() -> None:
    initial = [
        row("u1", "p1", "c1", "easy", "abcd", "abcd"),
        row("u2", "p1", "c2", "easy", "z", ""),
        row("u3", "p2", "c3", "hard", "xy", "xy"),
    ]
    current = [
        row("u1", "p1", "c1", "easy", "abcd", "abxd"),
        row("u2", "p1", "c2", "easy", "z", ""),
        row("u3", "p2", "c3", "hard", "xy", "xyq"),
    ]

    result = extract_evidence(
        current,
        initial,
        current_prediction_column="prediction",
        initial_prediction_column="prediction",
    )

    metrics = result.current_metrics
    assert result.analysis_level == "character"
    assert metrics.overall.sample_count == 3
    assert metrics.overall.total_edits == 3
    assert metrics.overall.total_reference_characters == 7
    assert math.isclose(metrics.overall.mean_utterance_cer, 7 / 12)
    assert math.isclose(metrics.overall.pooled_cer, 3 / 7)
    # p1: 2 edits / 5 chars; p2: 1 edit / 2 chars.
    assert math.isclose(metrics.overall.patient_macro_cer, (2 / 5 + 1 / 2) / 2)
    assert metrics.overall.critical_count == 2
    assert metrics.overall.exact_count == 0
    assert metrics.easy.total_edits == 2
    assert metrics.easy.total_reference_characters == 5
    assert metrics.medium.sample_count == 0
    assert metrics.hard.pooled_cer == 0.5

    assert result.transitions_vs_initial.model_dump() == {
        "recovered": 0,
        "newly_wrong": 2,
        "preserved_correct": 0,
        "persistent_wrong": 1,
    }
    assert result.delta_vs_previous is None
    assert result.transitions_vs_previous is None

    clusters = {
        (cluster.operation, cluster.reference_unit, cluster.hypothesis_unit): cluster
        for cluster in result.error_clusters
    }
    substitution = clusters[("substitution", "c", "x")]
    deletion = clusters[("deletion", "z", "")]
    insertion = clusters[("insertion", "", "q")]
    assert substitution.opportunity_count == 1
    assert substitution.patient_count == 1
    assert substitution.content_count == 1
    assert substitution.bounded_examples[0].startswith("utt-")
    assert "u1" not in substitution.bounded_examples
    assert substitution.initial_rate == 0.0
    assert deletion.opportunity_count == 1
    assert deletion.initial_rate == 1.0
    # Character boundaries: (4 + 1) + (1 + 1) + (2 + 1).
    assert insertion.opportunity_count == 10
    assert insertion.error_rate == 0.1
    assert all(cluster.level == "character" for cluster in result.error_clusters)


def test_previous_snapshot_is_paired_and_reported() -> None:
    initial = [row("u1", "p1", "c1", "medium", "甲乙", "甲")]
    previous = [row("u1", "p1", "c1", "medium", "甲乙", "甲乙")]
    current = [row("u1", "p1", "c1", "medium", "甲乙", "甲")]

    result = extract_evidence(
        current,
        initial,
        current_prediction_column="prediction",
        initial_prediction_column="prediction",
        previous_rows=previous,
        previous_prediction_column="prediction",
    )

    assert result.previous_metrics is not None
    assert result.transitions_vs_previous is not None
    assert result.transitions_vs_previous.newly_wrong == 1
    deletion = result.error_clusters[0]
    assert deletion.operation == "deletion"
    assert deletion.initial_rate == 1.0
    assert deletion.previous_rate == 0.0
    assert deletion.observed_snapshot_count == 2


def test_pairing_set_mismatch_fails_fast() -> None:
    current = [row("u1", "p1", "c1", "easy", "甲", "甲")]
    initial = [row("u2", "p1", "c1", "easy", "甲", "甲")]

    with pytest.raises(ValueError, match="utterance sets differ"):
        extract_evidence(
            current,
            initial,
            current_prediction_column="prediction",
            initial_prediction_column="prediction",
        )


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
            current_prediction_column="prediction",
            initial_prediction_column="prediction",
        )


def test_previous_rows_and_prediction_column_are_atomic() -> None:
    rows = [row("u1", "p1", "c1", "easy", "甲", "甲")]
    with pytest.raises(ValueError, match="must be supplied together"):
        extract_evidence(
            rows,
            rows,
            current_prediction_column="prediction",
            initial_prediction_column="prediction",
            previous_rows=rows,
        )
