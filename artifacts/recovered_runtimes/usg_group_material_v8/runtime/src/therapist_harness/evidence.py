"""Deterministic aggregate evidence from paired Setting D ASR snapshots.

The Agent never receives the source rows.  This module pairs every snapshot on
``utt_id``, computes character CER and paired changes, and emits bounded error
clusters.  Jyutping component attribution is deliberately conservative: only
forced character substitutions with high-confidence contextual G2P are used.
These are ASR transcript errors at phonological positions, not acoustic or
clinical measurements of a patient's articulation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal, Mapping, Sequence

from .phonology import (
    ComponentEvent,
    PhonologyAnalysis,
    align_characters,
    analyze_phonological_errors,
    map_text_to_syllables,
)
from .schema import (
    ClusterObservations,
    ComparisonSet,
    CriticalTransitionCounts,
    DifficultyErrorRate,
    DirectionCounts,
    ErrorCluster,
    ErrorObservation,
    EvidenceTargetDescriptor,
    MetricDelta,
    MetricGroup,
    MetricSet,
    PatientDirectionCounts,
    PhonologyCoverageGroup,
    PhonologyCoverageSet,
    SnapshotComparison,
    TransitionCounts,
    TransitionGroup,
    TransitionSet,
)
from .text import CerNormalizationProjection, cer_normalization_projection
from .round_schema import (
    ClusterChange,
    ClusterDescriptor,
    MetricChange,
    PatientResponse,
    TargetPositionCounts,
    TargetPositionResponse,
    TargetPositionTransition,
    target_position_direction,
)


Difficulty = Literal["easy", "medium", "hard"]
Operation = Literal["deletion", "substitution", "insertion"]
Level = Literal["character", "onset", "nucleus", "coda", "tone"]
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")
LEVELS: tuple[Level, ...] = ("character", "onset", "nucleus", "coda", "tone")
MIN_TARGET_ERROR_COUNT = 2
MIN_TARGET_PATIENT_COUNT = 2
MIN_TARGET_CONTENT_COUNT = 2
MAX_TARGET_ALIGNMENT_UNCERTAINTY = 0.5
REQUIRED_METADATA = (
    "utt_id",
    "speaker_id",
    "prompt_id",
    "zero_shot_bucket",
    "clean_gt",
)


ClusterKey = tuple[Operation, Level, str, str, str | None, str | None]
OpportunityKey = tuple[Level, str, str | None, str | None]


@dataclass(frozen=True)
class EvidenceExtraction:
    current_metrics: MetricSet
    m0_metrics: MetricSet
    zero_shot_metrics: MetricSet
    previous_metrics: MetricSet | None
    comparisons: ComparisonSet
    error_clusters: tuple[ErrorCluster, ...]
    current_phonology_coverage: PhonologyCoverageSet
    m0_phonology_coverage: PhonologyCoverageSet
    zero_shot_phonology_coverage: PhonologyCoverageSet
    previous_phonology_coverage: PhonologyCoverageSet | None
    candidate_cluster_count: int
    sample_count: int
    speaker_count: int
    content_count: int


@dataclass(frozen=True)
class PairedSnapshotEvaluation:
    """Canonical, paired measurements used by the post-round compiler."""

    before_metrics: MetricSet
    after_metrics: MetricSet
    paired_delta: MetricDelta
    transitions: TransitionSet
    patient_response: PatientResponse
    metric_changes: tuple[MetricChange, ...]
    monitored_cluster_changes: tuple[ClusterChange, ...]
    off_target_changes: tuple[ClusterChange, ...]
    phonology_coverage_before: PhonologyCoverageSet
    phonology_coverage_after: PhonologyCoverageSet


@dataclass(frozen=True)
class _EditEvent:
    operation: Operation
    reference_unit: str
    hypothesis_unit: str
    reference_index: int | None
    hypothesis_index: int | None


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
    phonology: PhonologyAnalysis
    phone_events: tuple[ComponentEvent, ...]
    reference_projection: CerNormalizationProjection
    hypothesis_projection: CerNormalizationProjection
    reference_outcomes: tuple[tuple[int | None, ...], ...]
    normalization_projection_unresolved_pairs: int
    not_cer_forced_phonologically_different_pairs: int

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
    uncertain_count: int = 0


@dataclass(frozen=True)
class _SnapshotClusters:
    accumulators: Mapping[ClusterKey, _ClusterAccumulator]


def _require_string(
    row: Mapping[str, object], field_name: str, *, role: str, row_index: int
) -> str:
    if field_name not in row:
        raise ValueError(
            f"{role} row {row_index} is missing required column {field_name!r}"
        )
    value = row[field_name]
    if not isinstance(value, str):
        raise TypeError(
            f"{role} row {row_index} column {field_name!r} must be a string, "
            f"got {type(value).__name__}"
        )
    return value


def _stable_align(
    reference: str, hypothesis: str
) -> tuple[int, tuple[_EditEvent, ...], bool]:
    """Return a stable optimal traceback and whether multiple optima exist."""

    n = len(reference)
    m = len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
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
                            reference_index=i - 1,
                            hypothesis_index=j - 1,
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
                    reference_index=i - 1,
                    hypothesis_index=None,
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
                    reference_index=None,
                    hypothesis_index=j - 1,
                )
            )
            j -= 1
            continue
        raise RuntimeError("Levenshtein traceback is inconsistent")

    events.reverse()
    if len(events) != costs[n][m]:
        raise RuntimeError("Levenshtein event count differs from edit distance")
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
            name: _require_string(row, name, role=role, row_index=index)
            for name in REQUIRED_METADATA
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
        reference_projection = cer_normalization_projection(values["clean_gt"])
        hypothesis_projection = cer_normalization_projection(prediction)
        reference = reference_projection.normalized
        if not reference:
            raise ValueError(f"{role} {utt_id} has an empty normalized clean_gt")
        hypothesis = hypothesis_projection.normalized
        edit_count, events, ambiguous = _stable_align(reference, hypothesis)
        phonology = analyze_phonological_errors(
            reference_projection.surface,
            hypothesis_projection.surface,
        )
        cer_alignment = align_characters(reference, hypothesis)
        if cer_alignment.edit_distance != edit_count:
            raise RuntimeError("CER alignments disagree on edit distance")
        (
            phone_events,
            projection_unresolved_pairs,
            not_cer_forced_pairs,
        ) = _cer_supported_phone_events(
            phonology,
            reference_projection,
            hypothesis_projection,
            {
                (substitution.reference_index, substitution.hypothesis_index)
                for substitution in cer_alignment.forced_substitutions
            },
        )
        cer = edit_count / len(reference)
        utterances.append(
            _Utterance(
                utt_id=utt_id,
                speaker_id=speaker_id,
                content_id=content_id,
                difficulty=bucket,
                clean_gt=values["clean_gt"],
                reference=reference,
                hypothesis=hypothesis,
                edit_count=edit_count,
                exact=edit_count == 0,
                critical=not hypothesis or cer >= critical_cer_threshold,
                events=events,
                alignment_ambiguous=ambiguous,
                phonology=phonology,
                phone_events=phone_events,
                reference_projection=reference_projection,
                hypothesis_projection=hypothesis_projection,
                reference_outcomes=cer_alignment.reference_outcomes,
                normalization_projection_unresolved_pairs=(projection_unresolved_pairs),
                not_cer_forced_phonologically_different_pairs=(not_cer_forced_pairs),
            )
        )
        seen_ids.add(utt_id)
    return tuple(utterances)


def _cer_supported_phone_events(
    phonology: PhonologyAnalysis,
    reference_projection: CerNormalizationProjection,
    hypothesis_projection: CerNormalizationProjection,
    cer_forced_substitutions: set[tuple[int, int]],
) -> tuple[tuple[ComponentEvent, ...], int, int]:
    """Keep raw-phone differences that are also forced CER substitutions.

    Raw surface readings prevent pronunciation-changing CER rewrites from
    corrupting component labels.  Requiring a unique projection onto a forced
    CER substitution prevents CER-equivalent spelling variants from becoming
    new phone errors.  Anything ambiguous is omitted as part of the documented
    lower-bound attribution policy.
    """

    supported: list[ComponentEvent] = []
    projection_unresolved: set[tuple[int, int]] = set()
    not_cer_forced: set[tuple[int, int]] = set()
    for event in phonology.component_events:
        raw_pair = (event.reference_index, event.hypothesis_index)
        reference_index = reference_projection.surface_to_normalized[
            event.reference_index
        ]
        hypothesis_index = hypothesis_projection.surface_to_normalized[
            event.hypothesis_index
        ]
        if reference_index is None or hypothesis_index is None:
            projection_unresolved.add(raw_pair)
            continue
        if (reference_index, hypothesis_index) in cer_forced_substitutions:
            supported.append(event)
        else:
            not_cer_forced.add(raw_pair)
    return tuple(supported), len(projection_unresolved), len(not_cer_forced)


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
        for field_name in (
            "speaker_id",
            "content_id",
            "difficulty",
            "clean_gt",
            "reference",
        ):
            if getattr(left, field_name) != getattr(right, field_name):
                raise ValueError(
                    f"paired metadata mismatch for {utt_id}, field={field_name}: "
                    f"current={getattr(left, field_name)!r}, "
                    f"{role}={getattr(right, field_name)!r}"
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
    return MetricGroup(
        sample_count=len(rows),
        speaker_count=len(patient_rows),
        total_edits=total_edits,
        total_reference_characters=total_reference,
        mean_utterance_cer=sum(row.cer for row in rows) / len(rows),
        pooled_cer=total_edits / total_reference,
        patient_macro_cer=float(_patient_macro_fraction(rows)),
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
        patient_macro_cer=(
            current.overall.patient_macro_cer - comparison.overall.patient_macro_cer
        ),
    )


def _transition_group(
    current: Sequence[_Utterance], comparison: Sequence[_Utterance]
) -> TransitionGroup:
    comparison_by_id = {row.utt_id: row for row in comparison}
    exact = Counter[str]()
    direction = Counter[str]()
    critical = Counter[str]()
    for new in current:
        old = comparison_by_id[new.utt_id]
        if old.exact and new.exact:
            exact["preserved_correct"] += 1
        elif old.exact:
            exact["newly_wrong"] += 1
        elif new.exact:
            exact["recovered"] += 1
        else:
            exact["persistent_wrong"] += 1

        if new.edit_count < old.edit_count:
            direction["improved"] += 1
        elif new.edit_count > old.edit_count:
            direction["worsened"] += 1
        else:
            direction["tied"] += 1

        if old.critical and new.critical:
            critical["persistent_critical"] += 1
        elif old.critical:
            critical["recovered"] += 1
        elif new.critical:
            critical["newly_critical"] += 1
        else:
            critical["preserved_noncritical"] += 1

    return TransitionGroup(
        sample_count=len(current),
        exact=TransitionCounts(
            recovered=exact["recovered"],
            newly_wrong=exact["newly_wrong"],
            preserved_correct=exact["preserved_correct"],
            persistent_wrong=exact["persistent_wrong"],
        ),
        utterance_cer=DirectionCounts(
            improved=direction["improved"],
            worsened=direction["worsened"],
            tied=direction["tied"],
        ),
        critical=CriticalTransitionCounts(
            recovered=critical["recovered"],
            newly_critical=critical["newly_critical"],
            preserved_noncritical=critical["preserved_noncritical"],
            persistent_critical=critical["persistent_critical"],
        ),
    )


def _transition_set(
    current: Sequence[_Utterance], comparison: Sequence[_Utterance]
) -> TransitionSet:
    comparison_by_id = {row.utt_id: row for row in comparison}

    def group(difficulty: Difficulty | None) -> TransitionGroup:
        selected = [
            row for row in current if difficulty is None or row.difficulty == difficulty
        ]
        paired = [comparison_by_id[row.utt_id] for row in selected]
        return _transition_group(selected, paired)

    return TransitionSet(
        overall=group(None),
        easy=group("easy"),
        medium=group("medium"),
        hard=group("hard"),
    )


def _patient_directions(
    current: Sequence[_Utterance], comparison: Sequence[_Utterance]
) -> PatientDirectionCounts:
    current_groups: dict[str, list[_Utterance]] = defaultdict(list)
    comparison_groups: dict[str, list[_Utterance]] = defaultdict(list)
    for row in current:
        current_groups[row.speaker_id].append(row)
    for row in comparison:
        comparison_groups[row.speaker_id].append(row)
    if set(current_groups) != set(comparison_groups):
        raise ValueError("patient sets differ between paired snapshots")
    counts = Counter[str]()
    for speaker_id in current_groups:
        new_rows = current_groups[speaker_id]
        old_rows = comparison_groups[speaker_id]
        new_cer = sum(row.edit_count for row in new_rows) / sum(
            row.reference_length for row in new_rows
        )
        old_cer = sum(row.edit_count for row in old_rows) / sum(
            row.reference_length for row in old_rows
        )
        if new_cer < old_cer:
            counts["improved"] += 1
        elif new_cer > old_cer:
            counts["worsened"] += 1
        else:
            counts["tied"] += 1
    return PatientDirectionCounts(
        patient_count=len(current_groups),
        improved=counts["improved"],
        worsened=counts["worsened"],
        tied=counts["tied"],
    )


def _comparison(
    current: Sequence[_Utterance],
    reference: Sequence[_Utterance],
    *,
    evaluated_snapshot: Literal["current", "m0"],
    reference_snapshot: Literal["zero_shot", "m0", "previous"],
) -> SnapshotComparison:
    return SnapshotComparison(
        evaluated_snapshot=evaluated_snapshot,
        reference_snapshot=reference_snapshot,
        metric_delta=_metric_delta(_metric_set(current), _metric_set(reference)),
        transitions=_transition_set(current, reference),
        patient_directions=_patient_directions(current, reference),
    )


def _encode_unit(value: str) -> str:
    return value if value else "∅"


def _opportunities(
    rows: Sequence[_Utterance],
) -> tuple[Counter[OpportunityKey], dict[Difficulty, Counter[OpportunityKey]]]:
    total: Counter[OpportunityKey] = Counter()
    by_difficulty = {difficulty: Counter() for difficulty in DIFFICULTIES}
    for row in rows:
        for character in row.reference:
            key: OpportunityKey = ("character", character, None, None)
            total[key] += 1
            by_difficulty[row.difficulty][key] += 1
        insertion_key: OpportunityKey = ("character", "", None, None)
        boundaries = row.reference_length + 1
        total[insertion_key] += boundaries
        by_difficulty[row.difficulty][insertion_key] += boundaries

        for position in row.phonology.reference.positions:
            if not position.resolved:
                continue
            for level in ("onset", "nucleus", "coda", "tone"):
                reference_unit = _encode_unit(getattr(position, level) or "")
                phone_key: OpportunityKey = (
                    level,
                    reference_unit,
                    None,
                    None,
                )
                total[phone_key] += 1
                by_difficulty[row.difficulty][phone_key] += 1
    return total, by_difficulty


def _snapshot_clusters(rows: Sequence[_Utterance]) -> _SnapshotClusters:
    accumulators: dict[ClusterKey, _ClusterAccumulator] = {}
    for row in rows:
        for event in row.events:
            key: ClusterKey = (
                event.operation,
                "character",
                event.reference_unit,
                event.hypothesis_unit,
                None,
                None,
            )
            accumulator = accumulators.setdefault(key, _ClusterAccumulator())
            accumulator.error_count += 1
            accumulator.patients.add(row.speaker_id)
            accumulator.contents.add(row.content_id)
            accumulator.difficulty[row.difficulty] += 1
            if row.alignment_ambiguous:
                accumulator.uncertain_count += 1

        for event in row.phone_events:
            # PyCantonese has no pronunciation confidence.  The strict Agent-facing
            # layer only keeps events whose contextual and per-character readings
            # agree on both sides; all exclusions remain visible in coverage.
            if event.context_sensitive is not False:
                continue
            phone_key: ClusterKey = (
                event.operation,
                event.level,
                _encode_unit(event.reference_unit),
                _encode_unit(event.hypothesis_unit),
                None,
                None,
            )
            accumulator = accumulators.setdefault(phone_key, _ClusterAccumulator())
            accumulator.error_count += 1
            accumulator.patients.add(row.speaker_id)
            accumulator.contents.add(row.content_id)
            accumulator.difficulty[row.difficulty] += 1
    return _SnapshotClusters(accumulators=accumulators)


def _cluster_id(key: ClusterKey) -> str:
    encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    operation, level, *_ = key
    return f"{level}-{operation[:3]}-{digest}"


def _observation(
    key: ClusterKey,
    snapshot: _SnapshotClusters,
    opportunities: Counter[OpportunityKey],
    difficulty_opportunities: Mapping[Difficulty, Counter[OpportunityKey]],
) -> ErrorObservation:
    operation, level, reference_unit, _, left_context, right_context = key
    opportunity_unit = (
        "" if level == "character" and operation == "insertion" else reference_unit
    )
    opportunity_key: OpportunityKey = (
        level,
        opportunity_unit,
        left_context,
        right_context,
    )
    denominator = opportunities[opportunity_key]
    if denominator <= 0:
        raise RuntimeError(f"non-positive opportunity count for cluster {key!r}")
    accumulator = snapshot.accumulators.get(key, _ClusterAccumulator())
    by_difficulty: dict[Difficulty, DifficultyErrorRate] = {}
    for difficulty in DIFFICULTIES:
        difficulty_denominator = difficulty_opportunities[difficulty][opportunity_key]
        difficulty_errors = accumulator.difficulty[difficulty]
        by_difficulty[difficulty] = DifficultyErrorRate(
            error_count=difficulty_errors,
            opportunity_count=difficulty_denominator,
            rate=(
                difficulty_errors / difficulty_denominator
                if difficulty_denominator
                else 0.0
            ),
        )
    return ErrorObservation(
        error_count=accumulator.error_count,
        opportunity_count=denominator,
        rate=accumulator.error_count / denominator,
        patient_count=len(accumulator.patients),
        content_count=len(accumulator.contents),
        by_difficulty=by_difficulty,
        alignment_uncertainty=(
            accumulator.uncertain_count / accumulator.error_count
            if accumulator.error_count
            else 0.0
        ),
    )


def _selection_reasons(
    observation: ErrorObservation, operation: Operation, level: Level
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if level == "character" and operation == "insertion":
        failures.append("pure_insertion_is_not_a_primary_training_target")
    if observation.error_count < MIN_TARGET_ERROR_COUNT:
        failures.append("current_error_count_below_2")
    if observation.patient_count < MIN_TARGET_PATIENT_COUNT:
        failures.append("current_patient_coverage_below_2")
    if observation.content_count < MIN_TARGET_CONTENT_COUNT:
        failures.append("current_content_coverage_below_2")
    if observation.alignment_uncertainty > MAX_TARGET_ALIGNMENT_UNCERTAINTY:
        failures.append("character_alignment_uncertainty_above_0.5")
    if failures:
        return False, failures
    return True, ["meets_minimum_cross_patient_and_cross_content_support"]


def _isolated_jyutping(unit: str) -> str | None:
    """Return a single character's isolated Jyutping when it is resolved."""

    if len(unit) != 1:
        return None
    mapping = map_text_to_syllables(unit)
    if len(mapping.positions) != 1 or mapping.positions[0].character != unit:
        raise RuntimeError("isolated G2P did not preserve its input character")
    return mapping.positions[0].jyutping


def _tts_actionability(
    operation: Operation,
    level: Level,
    reference_unit: str,
    hypothesis_unit: str,
) -> tuple[bool, list[str]]:
    """Classify whether a cluster can ground a TTS pronunciation intervention."""

    if level != "character":
        return True, ["phone_component_has_pronunciation_facing_reference_carrier"]
    if operation == "insertion":
        return False, ["character_insertion_has_no_gt_side_tts_carrier"]
    if operation == "deletion":
        if len(reference_unit) != 1 or _isolated_jyutping(reference_unit) is None:
            return False, ["reference_character_is_not_single_resolved_jyutping"]
        return True, ["pronounceable_reference_character_can_be_rehearsed"]

    unresolved: list[str] = []
    reference_jyutping = _isolated_jyutping(reference_unit)
    hypothesis_jyutping = _isolated_jyutping(hypothesis_unit)
    if len(reference_unit) != 1 or reference_jyutping is None:
        unresolved.append("reference_character_is_not_single_resolved_jyutping")
    if len(hypothesis_unit) != 1 or hypothesis_jyutping is None:
        unresolved.append("hypothesis_character_is_not_single_resolved_jyutping")
    if unresolved:
        return False, unresolved
    if reference_jyutping == hypothesis_jyutping:
        return False, ["character_substitution_is_homophonic_in_isolated_jyutping"]
    return True, ["character_substitution_has_distinct_isolated_jyutping"]


def _character_surface_units(
    key: ClusterKey, rows: Sequence[_Utterance]
) -> tuple[list[str], list[str], bool]:
    """Attest pronunciation carriers at the actual normalized edit positions."""
    reference_units: set[str] = set()
    hypothesis_units: set[str] = set()
    resolved = True
    for row in rows:
        for event in row.events:
            if (
                event.operation,
                "character",
                event.reference_unit,
                event.hypothesis_unit,
                None,
                None,
            ) != key:
                continue
            for index, projection, mapping, units in (
                (
                    event.reference_index,
                    row.reference_projection,
                    row.phonology.reference,
                    reference_units,
                ),
                (
                    event.hypothesis_index,
                    row.hypothesis_projection,
                    row.phonology.hypothesis,
                    hypothesis_units,
                ),
            ):
                if index is None:
                    continue
                surface_index = projection.normalized_to_surface[index]
                if surface_index is None:
                    resolved = False
                    continue
                position = mapping.positions[surface_index]
                units.add(position.character)
                if not position.resolved or position.context_sensitive is not False:
                    resolved = False
    return sorted(reference_units), sorted(hypothesis_units), resolved


def _error_clusters(
    current: Sequence[_Utterance],
    m0: Sequence[_Utterance],
    zero_shot: Sequence[_Utterance],
    previous: Sequence[_Utterance] | None,
    *,
    max_clusters_per_level: int,
    prior_target_ids: set[str],
    prior_targets: Sequence[EvidenceTargetDescriptor],
) -> tuple[tuple[ErrorCluster, ...], int]:
    opportunities, difficulty_opportunities = _opportunities(current)
    snapshots = {
        "current": _snapshot_clusters(current),
        "m0": _snapshot_clusters(m0),
        "zero_shot": _snapshot_clusters(zero_shot),
    }
    previous_snapshot = _snapshot_clusters(previous) if previous is not None else None
    keys: set[ClusterKey] = set()
    for snapshot in snapshots.values():
        keys.update(snapshot.accumulators)
    if previous_snapshot is not None:
        keys.update(previous_snapshot.accumulators)
    for target in prior_targets:
        key: ClusterKey = (
            target.operation,
            target.level,
            target.reference_unit,
            target.hypothesis_unit,
            target.left_context,
            target.right_context,
        )
        if _cluster_id(key) != target.evidence_id:
            raise ValueError(
                f"historical target descriptor does not match {target.evidence_id}"
            )
        keys.add(key)

    observations: dict[ClusterKey, ClusterObservations] = {}
    for key in keys:
        observations[key] = ClusterObservations(
            current=_observation(
                key, snapshots["current"], opportunities, difficulty_opportunities
            ),
            m0=_observation(
                key, snapshots["m0"], opportunities, difficulty_opportunities
            ),
            zero_shot=_observation(
                key, snapshots["zero_shot"], opportunities, difficulty_opportunities
            ),
            previous=(
                _observation(
                    key,
                    previous_snapshot,
                    opportunities,
                    difficulty_opportunities,
                )
                if previous_snapshot is not None
                else None
            ),
        )

    selected: list[ClusterKey] = []
    for level in LEVELS:
        level_keys = [key for key in keys if key[1] == level]

        def rank(key: ClusterKey) -> tuple[object, ...]:
            item = observations[key]
            operation, key_level, *_ = key
            targetable, _ = _selection_reasons(item.current, operation, key_level)
            anchor_burden = max(
                item.m0.error_count,
                item.zero_shot.error_count,
                item.previous.error_count if item.previous is not None else 0,
            )
            return (
                not targetable,
                -item.current.error_count,
                -item.current.patient_count,
                -item.current.content_count,
                -anchor_burden,
                key,
            )

        historical = sorted(
            key for key in level_keys if _cluster_id(key) in prior_target_ids
        )
        new_candidates = [
            key for key in level_keys if _cluster_id(key) not in prior_target_ids
        ]
        selected.extend(historical)
        selected.extend(sorted(new_candidates, key=rank)[:max_clusters_per_level])

    result: list[ErrorCluster] = []
    for key in selected:
        (
            operation,
            level,
            reference_unit,
            hypothesis_unit,
            left_context,
            right_context,
        ) = key
        current_observation = observations[key].current
        targetable, reasons = _selection_reasons(current_observation, operation, level)
        reference_surface_units = hypothesis_surface_units = None
        if level == "character":
            source_rows = (
                current
                if observations[key].current.error_count
                else (*m0, *zero_shot, *(previous or ()))
            )
            reference_surface_units, hypothesis_surface_units, resolved = (
                _character_surface_units(key, source_rows)
            )
            if operation == "insertion":
                tts_actionable = False
                tts_actionability_reasons = [
                    "character_insertion_has_no_gt_side_tts_carrier"
                ]
            elif (
                not resolved
                or len(reference_surface_units) != 1
                or (operation == "substitution" and len(hypothesis_surface_units) != 1)
            ):
                tts_actionable = False
                tts_actionability_reasons = [
                    "surface_carrier_projection_or_pronunciation_is_not_unique"
                ]
            else:
                tts_actionable, tts_actionability_reasons = _tts_actionability(
                    operation,
                    level,
                    reference_surface_units[0],
                    hypothesis_surface_units[0] if hypothesis_surface_units else "",
                )
        else:
            tts_actionable, tts_actionability_reasons = _tts_actionability(
                operation,
                level,
                reference_unit,
                hypothesis_unit,
            )
        result.append(
            ErrorCluster(
                evidence_id=_cluster_id(key),
                operation=operation,
                level=level,
                reference_unit=reference_unit,
                hypothesis_unit=hypothesis_unit,
                reference_surface_units=reference_surface_units,
                hypothesis_surface_units=hypothesis_surface_units,
                left_context=left_context,
                right_context=right_context,
                rate_semantics=(
                    "deterministic_character_alignment_rate"
                    if level == "character"
                    else "confirmed_substitution_lower_bound_rate"
                ),
                targetable=targetable,
                selection_reasons=reasons,
                tts_actionable=tts_actionable,
                tts_actionability_reasons=tts_actionability_reasons,
                observations=observations[key],
            )
        )
    return tuple(result), len(keys)


def _coverage_group(rows: Sequence[_Utterance]) -> PhonologyCoverageGroup:
    attributed_pairs = 0
    for row in rows:
        attributed_pairs += len(
            {
                (event.reference_index, event.hypothesis_index)
                for event in row.phone_events
                if event.context_sensitive is False
            }
        )
    return PhonologyCoverageGroup(
        sample_count=len(rows),
        reference_characters=sum(
            row.phonology.coverage.reference_character_count for row in rows
        ),
        hypothesis_characters=sum(
            row.phonology.coverage.hypothesis_character_count for row in rows
        ),
        reference_g2p_resolved_characters=sum(
            row.phonology.reference.coverage.resolved_count for row in rows
        ),
        hypothesis_g2p_resolved_characters=sum(
            row.phonology.hypothesis.coverage.resolved_count for row in rows
        ),
        multi_optimal_alignment_utterances=sum(
            row.phonology.alignment.optimal_path_count > 1 for row in rows
        ),
        character_substitutions=sum(
            row.phonology.coverage.character_substitution_count for row in rows
        ),
        character_deletions=sum(
            row.phonology.coverage.character_deletion_count for row in rows
        ),
        character_insertions=sum(
            row.phonology.coverage.character_insertion_count for row in rows
        ),
        forced_substitutions=sum(
            row.phonology.coverage.forced_substitution_count for row in rows
        ),
        g2p_resolved_forced_substitutions=sum(
            row.phonology.coverage.forced_substitution_g2p_resolved_count
            for row in rows
        ),
        g2p_unresolved_forced_substitutions=sum(
            row.phonology.coverage.forced_substitution_g2p_unresolved_count
            for row in rows
        ),
        context_sensitive_forced_substitutions=sum(
            row.phonology.coverage.forced_substitution_context_sensitive_count
            for row in rows
        ),
        context_uncheckable_forced_substitutions=sum(
            row.phonology.coverage.forced_substitution_context_uncheckable_count
            for row in rows
        ),
        phonologically_equivalent_forced_substitutions=sum(
            row.phonology.coverage.phonologically_equivalent_forced_substitution_count
            for row in rows
        ),
        phonologically_different_forced_substitutions=sum(
            row.phonology.coverage.phonologically_different_forced_substitution_count
            for row in rows
        ),
        normalization_projection_unresolved_forced_substitutions=sum(
            row.normalization_projection_unresolved_pairs for row in rows
        ),
        not_cer_forced_phonologically_different_substitutions=sum(
            row.not_cer_forced_phonologically_different_pairs for row in rows
        ),
        phone_attributed_forced_substitutions=attributed_pairs,
    )


def _coverage_set(rows: Sequence[_Utterance]) -> PhonologyCoverageSet:
    return PhonologyCoverageSet(
        overall=_coverage_group(rows),
        easy=_coverage_group([row for row in rows if row.difficulty == "easy"]),
        medium=_coverage_group([row for row in rows if row.difficulty == "medium"]),
        hard=_coverage_group([row for row in rows if row.difficulty == "hard"]),
    )


def extract_evidence(
    current_rows: Sequence[Mapping[str, object]],
    m0_rows: Sequence[Mapping[str, object]],
    zero_shot_rows: Sequence[Mapping[str, object]],
    *,
    current_prediction_column: str,
    m0_prediction_column: str,
    zero_shot_prediction_column: str,
    previous_rows: Sequence[Mapping[str, object]] | None = None,
    previous_prediction_column: str | None = None,
    critical_cer_threshold: float = 0.5,
    max_clusters_per_level: int = 10,
    prior_target_ids: Sequence[str] = (),
    prior_targets: Sequence[EvidenceTargetDescriptor] = (),
) -> EvidenceExtraction:
    """Build the deterministic evidence components for one proposal round.

    ``current`` is M(r-1), ``m0`` is the permanent Real-only loop anchor, and
    ``zero_shot`` is a diagnostic anchor used to expose adaptation regressions.
    For round 2 and later, ``previous`` is M(r-2).  Every delta is evaluated
    snapshot minus reference snapshot, so a negative CER delta is improvement.
    """

    if not math.isfinite(critical_cer_threshold) or critical_cer_threshold <= 0:
        raise ValueError("critical_cer_threshold must be finite and positive")
    if max_clusters_per_level <= 0:
        raise ValueError("max_clusters_per_level must be positive")
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
    m0 = _read_snapshot(
        m0_rows,
        role="m0",
        prediction_column=m0_prediction_column,
        critical_cer_threshold=critical_cer_threshold,
    )
    zero_shot = _read_snapshot(
        zero_shot_rows,
        role="zero_shot",
        prediction_column=zero_shot_prediction_column,
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
    _strictly_pair(current, m0, role="m0")
    _strictly_pair(current, zero_shot, role="zero_shot")
    if previous is not None:
        _strictly_pair(current, previous, role="previous")

    current_metrics = _metric_set(current)
    m0_metrics = _metric_set(m0)
    zero_shot_metrics = _metric_set(zero_shot)
    previous_metrics = _metric_set(previous) if previous is not None else None
    clusters, candidate_count = _error_clusters(
        current,
        m0,
        zero_shot,
        previous,
        max_clusters_per_level=max_clusters_per_level,
        prior_target_ids=set(prior_target_ids),
        prior_targets=prior_targets,
    )
    comparisons = ComparisonSet(
        m0_vs_zero_shot=_comparison(
            m0,
            zero_shot,
            evaluated_snapshot="m0",
            reference_snapshot="zero_shot",
        ),
        current_vs_zero_shot=_comparison(
            current,
            zero_shot,
            evaluated_snapshot="current",
            reference_snapshot="zero_shot",
        ),
        current_vs_m0=_comparison(
            current,
            m0,
            evaluated_snapshot="current",
            reference_snapshot="m0",
        ),
        current_vs_previous=(
            _comparison(
                current,
                previous,
                evaluated_snapshot="current",
                reference_snapshot="previous",
            )
            if previous is not None
            else None
        ),
    )
    return EvidenceExtraction(
        current_metrics=current_metrics,
        m0_metrics=m0_metrics,
        zero_shot_metrics=zero_shot_metrics,
        previous_metrics=previous_metrics,
        comparisons=comparisons,
        error_clusters=clusters,
        current_phonology_coverage=_coverage_set(current),
        m0_phonology_coverage=_coverage_set(m0),
        zero_shot_phonology_coverage=_coverage_set(zero_shot),
        previous_phonology_coverage=(
            _coverage_set(previous) if previous is not None else None
        ),
        candidate_cluster_count=candidate_count,
        sample_count=len(current),
        speaker_count=len({row.speaker_id for row in current}),
        content_count=len({row.content_id for row in current}),
    )


def evaluate_snapshot(
    rows: Sequence[Mapping[str, object]],
    *,
    prediction_column: str,
    critical_cer_threshold: float,
) -> MetricSet:
    """Evaluate one prediction snapshot with the canonical CER semantics."""

    if not math.isfinite(critical_cer_threshold) or critical_cer_threshold <= 0:
        raise ValueError("critical_cer_threshold must be finite and positive")
    snapshot = _read_snapshot(
        rows,
        role="snapshot",
        prediction_column=prediction_column,
        critical_cer_threshold=critical_cer_threshold,
    )
    return _metric_set(snapshot)


def _descriptor_key(descriptor: ClusterDescriptor) -> ClusterKey:
    key: ClusterKey = (
        descriptor.operation,
        descriptor.level,
        descriptor.reference_unit,
        descriptor.hypothesis_unit,
        descriptor.left_context,
        descriptor.right_context,
    )
    if _cluster_id(key) != descriptor.evidence_id:
        raise ValueError(
            f"cluster descriptor does not match evidence ID {descriptor.evidence_id}"
        )
    return key


def _key_descriptor(key: ClusterKey) -> ClusterDescriptor:
    operation, level, reference, hypothesis, left, right = key
    return ClusterDescriptor(
        evidence_id=_cluster_id(key),
        operation=operation,
        level=level,
        reference_unit=reference,
        hypothesis_unit=hypothesis,
        left_context=left,
        right_context=right,
        rate_semantics=(
            "deterministic_character_alignment_rate"
            if level == "character"
            else "confirmed_substitution_lower_bound_rate"
        ),
    )


def _change_response(
    delta: int | Fraction,
) -> Literal["improved", "unchanged", "worsened"]:
    if delta < 0:
        return "improved"
    if delta > 0:
        return "worsened"
    return "unchanged"


def _patient_macro_fraction(rows: Sequence[_Utterance]) -> Fraction:
    by_patient: dict[str, list[_Utterance]] = defaultdict(list)
    for row in rows:
        by_patient[row.speaker_id].append(row)
    values = [
        Fraction(
            sum(row.edit_count for row in patient_rows),
            sum(row.reference_length for row in patient_rows),
        )
        for patient_rows in by_patient.values()
    ]
    return sum(values, start=Fraction()) / len(values)


def _patient_pattern(
    counts: PatientDirectionCounts,
) -> Literal[
    "all_improved",
    "improvement_with_ties",
    "bidirectional",
    "worsening_with_ties",
    "all_worsened",
    "all_tied",
]:
    if counts.improved and counts.worsened:
        return "bidirectional"
    if counts.improved:
        return "all_improved" if not counts.tied else "improvement_with_ties"
    if counts.worsened:
        return "all_worsened" if not counts.tied else "worsening_with_ties"
    return "all_tied"


def _target_position_response(
    before: Sequence[_Utterance], after: Sequence[_Utterance], key: ClusterKey
) -> TargetPositionResponse:
    """Follow the same GT carrier positions, even when an error changes form.

    Correct means CER-equivalent *character recognition*, not acoustic recovery
    or a diagnosis. Whole-character deletions never disappear from this cohort.
    Multiple optimal mappings and unprojectable surface positions stay unknown.
    """
    operation, level, reference, hypothesis, left, right = key
    if (
        left is not None
        or right is not None
        or (level == "character" and operation == "insertion")
    ):
        raise ValueError("paired target recovery requires a context-free GT carrier")
    after_by_id = {row.utt_id: row for row in after}
    records: list[tuple[str, str, str, str, str]] = []
    before_insertions = after_insertions = 0

    def state(row: _Utterance, index: int | None) -> str:
        if index is None:
            return "unresolved"
        outcomes = row.reference_outcomes[index]
        if len(outcomes) != 1:
            return "unresolved"
        hyp_index = outcomes[0]
        if hyp_index is None:
            if level == "character" and operation == "deletion":
                return "signature_error"
            return "whole_character_deletion"
        if row.reference[index] == row.hypothesis[hyp_index]:
            return "correct"
        if level == "character":
            is_signature = (
                operation == "substitution" and row.hypothesis[hyp_index] == hypothesis
            )
        else:
            is_signature = any(
                event.context_sensitive is False
                and event.level == level
                and event.operation == operation
                and _encode_unit(event.reference_unit) == reference
                and _encode_unit(event.hypothesis_unit) == hypothesis
                and row.reference_projection.surface_to_normalized[
                    event.reference_index
                ]
                == index
                for event in row.phone_events
            )
        return "signature_error" if is_signature else "other_substitution"

    for old in before:
        new = after_by_id[old.utt_id]
        indices = (
            [
                index
                for index, character in enumerate(old.reference)
                if character == reference
            ]
            if level == "character"
            else [
                old.reference_projection.surface_to_normalized[position.index]
                for position in old.phonology.reference.positions
                if position.resolved
                and _encode_unit(getattr(position, level) or "") == reference
            ]
        )
        if indices:
            # Insertions have no GT position. Keep their burden on the target's
            # utterance cohort so substitution -> correct+extra word is visible.
            before_insertions += sum(
                event.operation == "insertion" for event in old.events
            )
            after_insertions += sum(
                event.operation == "insertion" for event in new.events
            )
        for index in indices:
            records.append(
                (
                    state(old, index),
                    state(new, index),
                    old.difficulty,
                    old.speaker_id,
                    old.content_id,
                )
            )

    def summarize(rows) -> TargetPositionCounts:
        grouped = defaultdict(list)
        for item in rows:
            grouped[item[:2]].append(item)
        transitions = [
            TargetPositionTransition(
                before=old,
                after=new,
                count=len(items),
                patient_count=len({item[3] for item in items}),
                content_count=len({item[4] for item in items}),
            )
            for (old, new), items in sorted(grouped.items())
        ]
        errors = {"signature_error", "other_substitution", "whole_character_deletion"}

        def count(predicate):
            return sum(predicate(old, new) for old, new, *_ in rows)

        return TargetPositionCounts(
            opportunity_count=len(rows),
            before_error_count=count(lambda a, b: a != "correct"),
            after_error_count=count(lambda a, b: b != "correct"),
            before_unresolved_count=count(lambda a, b: a == "unresolved"),
            after_unresolved_count=count(lambda a, b: b == "unresolved"),
            recovered_count=count(lambda a, b: a in errors and b == "correct"),
            regressed_count=count(lambda a, b: a == "correct" and b in errors),
            signature_recovered_count=count(
                lambda a, b: a == "signature_error" and b == "correct"
            ),
            signature_to_other_error_count=count(
                lambda a, b: a == "signature_error" and b == "other_substitution"
            ),
            signature_to_deletion_count=count(
                lambda a, b: a == "signature_error" and b == "whole_character_deletion"
            ),
            signature_unresolved_count=count(
                lambda a, b: a == "signature_error" and b == "unresolved"
            ),
            transitions=transitions,
        )

    overall = summarize(records)
    return TargetPositionResponse(
        **overall.model_dump(),
        semantics="paired_gt_carrier_character_recovery_v1",
        response=target_position_direction(
            overall, insertion_growth=after_insertions > before_insertions
        ),
        before_cohort_character_insertions=before_insertions,
        after_cohort_character_insertions=after_insertions,
        by_difficulty={
            group: summarize([item for item in records if item[2] == group])
            for group in DIFFICULTIES
        },
    )


def compare_prediction_snapshots(
    before_rows: Sequence[Mapping[str, object]],
    after_rows: Sequence[Mapping[str, object]],
    *,
    before_prediction_column: str,
    after_prediction_column: str,
    critical_cer_threshold: float,
    monitored_clusters: Sequence[ClusterDescriptor],
    off_target_max_per_level_per_direction: int,
) -> PairedSnapshotEvaluation:
    """Recompute all paired post-round measurements from prediction strings.

    Stored CER columns are intentionally ignored.  Difficulty, GT, content and
    speaker metadata must pair exactly, and cluster opportunities must remain
    identical because both snapshots use the same frozen real-speech split.
    """

    if off_target_max_per_level_per_direction <= 0:
        raise ValueError("off-target reporting limit must be positive")
    before = _read_snapshot(
        before_rows,
        role="before",
        prediction_column=before_prediction_column,
        critical_cer_threshold=critical_cer_threshold,
    )
    after = _read_snapshot(
        after_rows,
        role="after",
        prediction_column=after_prediction_column,
        critical_cer_threshold=critical_cer_threshold,
    )
    _strictly_pair(after, before, role="before")

    before_metrics = _metric_set(before)
    after_metrics = _metric_set(after)

    def pooled_delta(before_group: MetricGroup, after_group: MetricGroup) -> Fraction:
        denominator = before_group.total_reference_characters
        if denominator != after_group.total_reference_characters:
            raise RuntimeError("paired pooled-CER denominators differ")
        if denominator == 0:
            return Fraction()
        return Fraction(after_group.total_edits - before_group.total_edits, denominator)

    metric_specs: tuple[tuple[str, MetricGroup, MetricGroup, Fraction], ...] = (
        (
            "overall_pooled_cer",
            before_metrics.overall,
            after_metrics.overall,
            pooled_delta(before_metrics.overall, after_metrics.overall),
        ),
        (
            "easy_pooled_cer",
            before_metrics.easy,
            after_metrics.easy,
            pooled_delta(before_metrics.easy, after_metrics.easy),
        ),
        (
            "medium_pooled_cer",
            before_metrics.medium,
            after_metrics.medium,
            pooled_delta(before_metrics.medium, after_metrics.medium),
        ),
        (
            "hard_pooled_cer",
            before_metrics.hard,
            after_metrics.hard,
            pooled_delta(before_metrics.hard, after_metrics.hard),
        ),
        (
            "patient_macro_cer",
            before_metrics.overall,
            after_metrics.overall,
            _patient_macro_fraction(after) - _patient_macro_fraction(before),
        ),
    )
    metric_changes: list[MetricChange] = []
    for measure, before_group, after_group, exact_delta in metric_specs:
        before_value = (
            before_group.patient_macro_cer
            if measure == "patient_macro_cer"
            else before_group.pooled_cer
        )
        after_value = (
            after_group.patient_macro_cer
            if measure == "patient_macro_cer"
            else after_group.pooled_cer
        )
        metric_changes.append(
            MetricChange(
                measure=measure,
                before_value=before_value,
                after_value=after_value,
                delta=float(exact_delta),
                response=_change_response(exact_delta),
                citation_id=f"result.metric.{measure}",
            )
        )

    before_opportunities, before_difficulty_opportunities = _opportunities(before)
    after_opportunities, after_difficulty_opportunities = _opportunities(after)
    if (
        before_opportunities != after_opportunities
        or before_difficulty_opportunities != after_difficulty_opportunities
    ):
        raise RuntimeError("paired cluster opportunity maps differ")
    before_clusters = _snapshot_clusters(before)
    after_clusters = _snapshot_clusters(after)

    monitored_keys = {_descriptor_key(item): item for item in monitored_clusters}

    def change(
        key: ClusterKey, descriptor: ClusterDescriptor, *, off_target: bool
    ) -> ClusterChange:
        before_observation = _observation(
            key,
            before_clusters,
            before_opportunities,
            before_difficulty_opportunities,
        )
        after_observation = _observation(
            key,
            after_clusters,
            before_opportunities,
            before_difficulty_opportunities,
        )
        delta = after_observation.error_count - before_observation.error_count
        namespace = "off_target" if off_target else "cluster"
        return ClusterChange(
            descriptor=descriptor,
            before=before_observation,
            after=after_observation,
            error_count_delta=delta,
            rate_delta=after_observation.rate - before_observation.rate,
            response=_change_response(delta),
            target_positions=(
                None if off_target else _target_position_response(before, after, key)
            ),
            citation_id=f"result.{namespace}.{descriptor.evidence_id}",
        )

    monitored_changes = tuple(
        change(key, descriptor, off_target=False)
        for key, descriptor in monitored_keys.items()
    )

    union_keys = set(before_clusters.accumulators) | set(after_clusters.accumulators)
    candidates: list[ClusterChange] = []
    for key in union_keys - set(monitored_keys):
        item = change(key, _key_descriptor(key), off_target=True)
        if item.error_count_delta:
            candidates.append(item)

    selected: list[ClusterChange] = []
    for level in LEVELS:
        for response in ("worsened", "improved"):
            group = [
                item
                for item in candidates
                if item.descriptor.level == level and item.response == response
            ]
            group.sort(
                key=lambda item: (
                    -abs(item.error_count_delta),
                    -max(item.before.patient_count, item.after.patient_count),
                    -max(item.before.content_count, item.after.content_count),
                    item.descriptor.evidence_id,
                )
            )
            selected.extend(group[:off_target_max_per_level_per_direction])

    patient_counts = _patient_directions(after, before)
    return PairedSnapshotEvaluation(
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        paired_delta=_metric_delta(after_metrics, before_metrics),
        transitions=_transition_set(after, before),
        patient_response=PatientResponse(
            counts=patient_counts,
            pattern=_patient_pattern(patient_counts),
            citation_id="result.patient_directions",
        ),
        metric_changes=tuple(metric_changes),
        monitored_cluster_changes=monitored_changes,
        off_target_changes=tuple(selected),
        phonology_coverage_before=_coverage_set(before),
        phonology_coverage_after=_coverage_set(after),
    )


__all__ = [
    "MAX_TARGET_ALIGNMENT_UNCERTAINTY",
    "MIN_TARGET_CONTENT_COUNT",
    "MIN_TARGET_ERROR_COUNT",
    "MIN_TARGET_PATIENT_COUNT",
    "EvidenceExtraction",
    "PairedSnapshotEvaluation",
    "compare_prediction_snapshots",
    "evaluate_snapshot",
    "extract_evidence",
]
