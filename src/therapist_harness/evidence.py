"""Strict, experiment-agnostic evidence extraction from paired ASR outputs.

This module deliberately stays at the character level.  It describes errors in
the ASR output relative to the transcript; it does not infer phones, patient
articulation rules, or clinical impairment patterns.

The caller must supply the prediction column for every snapshot. The extractor
requires the Setting D metadata columns and always uses the canonical project
normalizer; no prediction column, content identity, or normalization policy is
guessed from the input.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

from .schema import ErrorCluster, MetricDelta, MetricGroup, MetricSet, TransitionCounts
from .text import text_normalize


Difficulty = Literal["easy", "medium", "hard"]
Operation = Literal["deletion", "substitution", "insertion"]
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")
REQUIRED_METADATA = (
    "utt_id",
    "speaker_id",
    "prompt_id",
    "zero_shot_bucket",
    "clean_gt",
)


@dataclass(frozen=True)
class EvidenceExtraction:
    """The deterministic evidence components needed to build an EvidencePack.

    ``transitions_*`` classify exact-transcription status.  In particular,
    ``recovered`` means wrong in the comparison snapshot and exact in the
    current snapshot.  Critical counts remain available in each MetricSet.
    """

    current_metrics: MetricSet
    initial_metrics: MetricSet
    previous_metrics: MetricSet | None
    delta_vs_initial: MetricDelta
    delta_vs_previous: MetricDelta | None
    transitions_vs_initial: TransitionCounts
    transitions_vs_previous: TransitionCounts | None
    error_clusters: tuple[ErrorCluster, ...]
    analysis_level: Literal["character"] = "character"


@dataclass(frozen=True)
class _EditEvent:
    operation: Operation
    reference_unit: str
    hypothesis_unit: str


@dataclass(frozen=True)
class _Utterance:
    utt_id: str
    speaker_id: str
    content_id: str
    difficulty: Difficulty
    clean_gt: str
    reference: str
    hypothesis: str
    edit_count: int
    exact: bool
    critical: bool
    events: tuple[_EditEvent, ...]
    alignment_ambiguous: bool

    @property
    def reference_length(self) -> int:
        return len(self.reference)

    @property
    def cer(self) -> float:
        return self.edit_count / self.reference_length


@dataclass
class _ClusterAccumulator:
    error_count: int = 0
    patients: set[str] = field(default_factory=set)
    contents: set[str] = field(default_factory=set)
    difficulty: Counter[Difficulty] = field(default_factory=Counter)
    examples: set[str] = field(default_factory=set)
    ambiguous_count: int = 0


def _require_string(
    row: Mapping[str, object], field: str, *, role: str, row_index: int
) -> str:
    if field not in row:
        raise ValueError(f"{role} row {row_index} is missing required column {field!r}")
    value = row[field]
    if not isinstance(value, str):
        raise TypeError(
            f"{role} row {row_index} column {field!r} must be a string, "
            f"got {type(value).__name__}"
        )
    return value


def _align(reference: str, hypothesis: str) -> tuple[int, tuple[_EditEvent, ...], bool]:
    """Return one deterministic optimal alignment and whether the optimum is unique."""

    n = len(reference)
    m = len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    # Counts are capped at two: only unique versus ambiguous matters here.
    ways = [[0] * (m + 1) for _ in range(n + 1)]
    ways[0][0] = 1
    for i in range(1, n + 1):
        costs[i][0] = i
        ways[i][0] = 1
    for j in range(1, m + 1):
        costs[0][j] = j
        ways[0][j] = 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
            deletion = costs[i - 1][j] + 1
            insertion = costs[i][j - 1] + 1
            best = min(diagonal, deletion, insertion)
            costs[i][j] = best
            path_count = 0
            if diagonal == best:
                path_count += ways[i - 1][j - 1]
            if deletion == best:
                path_count += ways[i - 1][j]
            if insertion == best:
                path_count += ways[i][j - 1]
            ways[i][j] = min(2, path_count)

    events: list[_EditEvent] = []
    i, j = n, m
    while i or j:
        if i and j:
            substitution_cost = reference[i - 1] != hypothesis[j - 1]
            if costs[i][j] == costs[i - 1][j - 1] + substitution_cost:
                if substitution_cost:
                    events.append(
                        _EditEvent(
                            operation="substitution",
                            reference_unit=reference[i - 1],
                            hypothesis_unit=hypothesis[j - 1],
                        )
                    )
                i -= 1
                j -= 1
                continue
        if i and costs[i][j] == costs[i - 1][j] + 1:
            events.append(
                _EditEvent(
                    operation="deletion",
                    reference_unit=reference[i - 1],
                    hypothesis_unit="",
                )
            )
            i -= 1
            continue
        if j and costs[i][j] == costs[i][j - 1] + 1:
            events.append(
                _EditEvent(
                    operation="insertion",
                    reference_unit="",
                    hypothesis_unit=hypothesis[j - 1],
                )
            )
            j -= 1
            continue
        raise RuntimeError("Levenshtein traceback is inconsistent with its cost table")

    events.reverse()
    if len(events) != costs[n][m]:
        raise RuntimeError("Levenshtein event count does not equal edit distance")
    return costs[n][m], tuple(events), ways[n][m] > 1


def _read_snapshot(
    rows: Sequence[Mapping[str, object]],
    *,
    role: str,
    prediction_column: str,
    critical_cer_threshold: float,
) -> tuple[_Utterance, ...]:
    if not rows:
        raise ValueError(f"{role} prediction rows are empty")
    if not prediction_column:
        raise ValueError(f"{role} prediction column must be explicit and non-empty")
    if prediction_column in REQUIRED_METADATA:
        raise ValueError(
            f"{role} prediction column {prediction_column!r} collides with metadata"
        )

    utterances: list[_Utterance] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        values = {
            field: _require_string(row, field, role=role, row_index=index)
            for field in REQUIRED_METADATA
        }
        prediction = _require_string(row, prediction_column, role=role, row_index=index)
        utt_id = values["utt_id"].strip()
        speaker_id = values["speaker_id"].strip()
        content_id = values["prompt_id"].strip()
        if not utt_id:
            raise ValueError(f"{role} row {index} has an empty utt_id")
        if utt_id in seen_ids:
            raise ValueError(f"duplicate utt_id in {role} rows: {utt_id}")
        if not speaker_id:
            raise ValueError(f"{role} {utt_id} has an empty speaker_id")
        if not content_id:
            raise ValueError(f"{role} {utt_id} has an empty prompt_id")
        bucket = values["zero_shot_bucket"].strip().lower()
        if bucket not in DIFFICULTIES:
            raise ValueError(
                f"{role} {utt_id} has unknown zero_shot_bucket {bucket!r}; "
                f"expected one of {DIFFICULTIES}"
            )
        clean_gt = values["clean_gt"]
        reference = text_normalize(clean_gt)
        if not reference:
            raise ValueError(f"{role} {utt_id} has an empty normalized clean_gt")
        hypothesis = text_normalize(prediction)
        edit_count, events, ambiguous = _align(reference, hypothesis)
        cer = edit_count / len(reference)
        utterances.append(
            _Utterance(
                utt_id=utt_id,
                speaker_id=speaker_id,
                content_id=content_id,
                difficulty=bucket,
                clean_gt=clean_gt,
                reference=reference,
                hypothesis=hypothesis,
                edit_count=edit_count,
                exact=edit_count == 0,
                critical=not hypothesis or cer >= critical_cer_threshold,
                events=events,
                alignment_ambiguous=ambiguous,
            )
        )
        seen_ids.add(utt_id)
    return tuple(utterances)


def _strictly_pair(
    current: Sequence[_Utterance], comparison: Sequence[_Utterance], *, role: str
) -> None:
    current_by_id = {row.utt_id: row for row in current}
    comparison_by_id = {row.utt_id: row for row in comparison}
    current_ids = set(current_by_id)
    comparison_ids = set(comparison_by_id)
    if current_ids != comparison_ids:
        missing = sorted(current_ids - comparison_ids)
        extra = sorted(comparison_ids - current_ids)
        raise ValueError(
            f"utterance sets differ between current and {role}: "
            f"missing_in_{role}={missing[:5]} (count={len(missing)}), "
            f"extra_in_{role}={extra[:5]} (count={len(extra)})"
        )

    for utt_id in sorted(current_ids):
        left = current_by_id[utt_id]
        right = comparison_by_id[utt_id]
        for metadata_field in (
            "speaker_id",
            "content_id",
            "difficulty",
            "clean_gt",
            "reference",
        ):
            if getattr(left, metadata_field) != getattr(right, metadata_field):
                raise ValueError(
                    f"paired metadata mismatch for {utt_id}, field={metadata_field}: "
                    f"current={getattr(left, metadata_field)!r}, "
                    f"{role}={getattr(right, metadata_field)!r}"
                )


def _metric_group(rows: Sequence[_Utterance]) -> MetricGroup:
    if not rows:
        return MetricGroup(
            sample_count=0,
            speaker_count=0,
            total_edits=0,
            total_reference_characters=0,
            mean_utterance_cer=0.0,
            pooled_cer=0.0,
            patient_macro_cer=0.0,
            critical_count=0,
            exact_count=0,
        )

    total_edits = sum(row.edit_count for row in rows)
    total_reference = sum(row.reference_length for row in rows)
    patient_rows: dict[str, list[_Utterance]] = defaultdict(list)
    for row in rows:
        patient_rows[row.speaker_id].append(row)
    patient_cers = [
        sum(row.edit_count for row in group)
        / sum(row.reference_length for row in group)
        for group in patient_rows.values()
    ]
    return MetricGroup(
        sample_count=len(rows),
        speaker_count=len(patient_rows),
        total_edits=total_edits,
        total_reference_characters=total_reference,
        mean_utterance_cer=sum(row.cer for row in rows) / len(rows),
        pooled_cer=total_edits / total_reference,
        patient_macro_cer=sum(patient_cers) / len(patient_cers),
        critical_count=sum(row.critical for row in rows),
        exact_count=sum(row.exact for row in rows),
    )


def _metric_set(rows: Sequence[_Utterance]) -> MetricSet:
    return MetricSet(
        overall=_metric_group(rows),
        easy=_metric_group([row for row in rows if row.difficulty == "easy"]),
        medium=_metric_group([row for row in rows if row.difficulty == "medium"]),
        hard=_metric_group([row for row in rows if row.difficulty == "hard"]),
    )


def _metric_delta(current: MetricSet, comparison: MetricSet) -> MetricDelta:
    return MetricDelta(
        overall_pooled_cer=current.overall.pooled_cer - comparison.overall.pooled_cer,
        easy_pooled_cer=current.easy.pooled_cer - comparison.easy.pooled_cer,
        medium_pooled_cer=current.medium.pooled_cer - comparison.medium.pooled_cer,
        hard_pooled_cer=current.hard.pooled_cer - comparison.hard.pooled_cer,
        patient_macro_cer=current.overall.patient_macro_cer
        - comparison.overall.patient_macro_cer,
    )


def _transitions(
    current: Sequence[_Utterance], comparison: Sequence[_Utterance]
) -> TransitionCounts:
    current_by_id = {row.utt_id: row for row in current}
    recovered = 0
    newly_wrong = 0
    preserved_correct = 0
    persistent_wrong = 0
    for old in comparison:
        new = current_by_id[old.utt_id]
        if old.exact and new.exact:
            preserved_correct += 1
        elif old.exact and not new.exact:
            newly_wrong += 1
        elif not old.exact and new.exact:
            recovered += 1
        else:
            persistent_wrong += 1
    return TransitionCounts(
        recovered=recovered,
        newly_wrong=newly_wrong,
        preserved_correct=preserved_correct,
        persistent_wrong=persistent_wrong,
    )


def _event_counts(rows: Sequence[_Utterance]) -> Counter[tuple[Operation, str, str]]:
    return Counter(
        (event.operation, event.reference_unit, event.hypothesis_unit)
        for row in rows
        for event in row.events
    )


def _cluster_id(key: tuple[Operation, str, str]) -> str:
    encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"char-{key[0][:3]}-{digest}"


def _anonymous_utterance_id(utt_id: str) -> str:
    digest = hashlib.sha256(utt_id.encode("utf-8")).hexdigest()[:12]
    return f"utt-{digest}"


def _error_clusters(
    current: Sequence[_Utterance],
    initial: Sequence[_Utterance],
    previous: Sequence[_Utterance] | None,
) -> tuple[ErrorCluster, ...]:
    initial_counts = _event_counts(initial)
    previous_counts = _event_counts(previous) if previous is not None else None
    reference_units = Counter(char for row in current for char in row.reference)
    insertion_opportunities = sum(row.reference_length + 1 for row in current)

    accumulators: dict[tuple[Operation, str, str], _ClusterAccumulator] = {}
    for row in current:
        for event in row.events:
            key = (event.operation, event.reference_unit, event.hypothesis_unit)
            accumulator = accumulators.setdefault(key, _ClusterAccumulator())
            accumulator.error_count += 1
            accumulator.patients.add(row.speaker_id)
            accumulator.contents.add(row.content_id)
            accumulator.difficulty[row.difficulty] += 1
            accumulator.examples.add(_anonymous_utterance_id(row.utt_id))
            if row.alignment_ambiguous:
                accumulator.ambiguous_count += 1

    clusters: list[ErrorCluster] = []
    for key, accumulator in accumulators.items():
        operation, reference_unit, hypothesis_unit = key
        error_count = accumulator.error_count
        opportunity_count = (
            insertion_opportunities
            if operation == "insertion"
            else reference_units[reference_unit]
        )
        if opportunity_count <= 0:
            raise RuntimeError(f"non-positive opportunity count for cluster {key!r}")
        initial_count = initial_counts[key]
        previous_count = previous_counts[key] if previous_counts is not None else None
        observed_snapshot_count = 1 + int(initial_count > 0)
        if previous_count is not None:
            observed_snapshot_count += int(previous_count > 0)
        clusters.append(
            ErrorCluster(
                evidence_id=_cluster_id(key),
                operation=operation,
                level="character",
                reference_unit=reference_unit,
                hypothesis_unit=hypothesis_unit,
                left_context=None,
                right_context=None,
                error_count=error_count,
                opportunity_count=opportunity_count,
                error_rate=error_count / opportunity_count,
                patient_count=len(accumulator.patients),
                content_count=len(accumulator.contents),
                difficulty_distribution={
                    difficulty_name: int(accumulator.difficulty[difficulty_name])
                    for difficulty_name in DIFFICULTIES
                },
                previous_rate=(
                    previous_count / opportunity_count
                    if previous_count is not None
                    else None
                ),
                initial_rate=initial_count / opportunity_count,
                observed_snapshot_count=observed_snapshot_count,
                bounded_examples=sorted(accumulator.examples)[:5],
                alignment_uncertainty=accumulator.ambiguous_count / error_count,
            )
        )
    return tuple(
        sorted(
            clusters,
            key=lambda item: (
                -item.error_count,
                item.operation,
                item.reference_unit,
                item.hypothesis_unit,
            ),
        )
    )


def extract_evidence(
    current_rows: Sequence[Mapping[str, object]],
    initial_rows: Sequence[Mapping[str, object]],
    *,
    current_prediction_column: str,
    initial_prediction_column: str,
    previous_rows: Sequence[Mapping[str, object]] | None = None,
    previous_prediction_column: str | None = None,
    critical_cer_threshold: float = 0.5,
) -> EvidenceExtraction:
    """Extract metrics, paired transitions, and current character-error clusters.

    Patient-macro CER is the unweighted mean of per-patient pooled CERs.  For
    deletion and substitution clusters, the denominator is the number of
    occurrences of the reference character.  For insertion clusters, it is
    the number of character boundaries (``len(reference) + 1`` per utterance).

    All snapshots use the project's canonical CER normalization. A run cannot
    inject a different normalization policy through this interface.
    """

    if not math.isfinite(critical_cer_threshold) or critical_cer_threshold <= 0:
        raise ValueError("critical_cer_threshold must be finite and positive")
    if (previous_rows is None) != (previous_prediction_column is None):
        raise ValueError(
            "previous_rows and previous_prediction_column must be supplied together"
        )
    current = _read_snapshot(
        current_rows,
        role="current",
        prediction_column=current_prediction_column,
        critical_cer_threshold=critical_cer_threshold,
    )
    initial = _read_snapshot(
        initial_rows,
        role="initial",
        prediction_column=initial_prediction_column,
        critical_cer_threshold=critical_cer_threshold,
    )
    previous = (
        _read_snapshot(
            previous_rows,
            role="previous",
            prediction_column=previous_prediction_column,
            critical_cer_threshold=critical_cer_threshold,
        )
        if previous_rows is not None and previous_prediction_column is not None
        else None
    )
    _strictly_pair(current, initial, role="initial")
    if previous is not None:
        _strictly_pair(current, previous, role="previous")

    current_metrics = _metric_set(current)
    initial_metrics = _metric_set(initial)
    previous_metrics = _metric_set(previous) if previous is not None else None
    return EvidenceExtraction(
        current_metrics=current_metrics,
        initial_metrics=initial_metrics,
        previous_metrics=previous_metrics,
        delta_vs_initial=_metric_delta(current_metrics, initial_metrics),
        delta_vs_previous=(
            _metric_delta(current_metrics, previous_metrics)
            if previous_metrics is not None
            else None
        ),
        transitions_vs_initial=_transitions(current, initial),
        transitions_vs_previous=(
            _transitions(current, previous) if previous is not None else None
        ),
        error_clusters=_error_clusters(current, initial, previous),
    )


def evaluate_snapshot(
    rows: Sequence[Mapping[str, object]],
    *,
    prediction_column: str,
    critical_cer_threshold: float,
) -> MetricSet:
    """Evaluate one prediction snapshot with the canonical evidence semantics."""

    if not math.isfinite(critical_cer_threshold) or critical_cer_threshold <= 0:
        raise ValueError("critical_cer_threshold must be finite and positive")
    snapshot = _read_snapshot(
        rows,
        role="snapshot",
        prediction_column=prediction_column,
        critical_cer_threshold=critical_cer_threshold,
    )
    return _metric_set(snapshot)


__all__ = ["EvidenceExtraction", "evaluate_snapshot", "extract_evidence"]
