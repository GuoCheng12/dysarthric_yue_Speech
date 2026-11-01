"""Positive ASR-retention evidence on fixed easy/medium GT carrier positions.

A carrier is a contextual GT Jyutping component. Correctness is *whole-character*
CER-equivalent recognition with a unique optimal alignment, not matching a phone
inside an otherwise wrong word and not an acoustic articulation measurement.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .phonology import map_text_to_syllables
from .text import tts_phonology_text


Count = Annotated[int, Field(ge=0)]
Level = Literal["onset", "nucleus", "coda", "tone"]
Difficulty = Literal["easy", "medium"]
State = Literal["correct", "incorrect", "unresolved"]
LEVELS: tuple[Level, ...] = ("onset", "nucleus", "coda", "tone")
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium")


class CapabilityAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    evidence_id: str = Field(pattern=r"^ability\.[a-z0-9._-]+$")
    semantics: Literal["whole_character_recognition_at_gt_phone_carrier_v1"] = (
        "whole_character_recognition_at_gt_phone_carrier_v1"
    )
    level: Level
    reference_unit: str = Field(min_length=1)
    difficulty: Difficulty
    opportunity_count: Count
    patient_count: Count
    content_count: Count
    m0_correct_count: Count
    current_correct_count: Count
    previous_correct_count: Count | None
    m0_preserved_count: Count
    m0_lost_count: Count
    newly_acquired_count: Count
    previous_preserved_count: Count | None
    previous_lost_count: Count | None
    previous_acquired_count: Count | None
    current_correct_patient_count: Count
    current_correct_content_count: Count
    m0_unresolved_count: Count
    current_unresolved_count: Count
    previous_unresolved_count: Count | None

    @model_validator(mode="after")
    def validate_counts(self) -> "CapabilityAnchor":
        if self.m0_preserved_count + self.m0_lost_count != self.m0_correct_count:
            raise ValueError("M0 correct cohort is not partitioned by preserved/lost")
        if (
            self.m0_preserved_count + self.newly_acquired_count
            != self.current_correct_count
        ):
            raise ValueError("current correct count is not preserved plus acquired")
        previous = (
            self.previous_correct_count,
            self.previous_preserved_count,
            self.previous_lost_count,
            self.previous_acquired_count,
            self.previous_unresolved_count,
        )
        if any(value is None for value in previous):
            if not all(value is None for value in previous):
                raise ValueError(
                    "previous capability counts must all be present or absent"
                )
        elif (
            self.previous_preserved_count + self.previous_lost_count
            != self.previous_correct_count
            or self.previous_preserved_count + self.previous_acquired_count
            != self.current_correct_count
        ):
            raise ValueError("previous capability transition counts do not partition")
        for correct, unresolved in (
            (self.m0_correct_count, self.m0_unresolved_count),
            (self.current_correct_count, self.current_unresolved_count),
            (self.previous_correct_count, self.previous_unresolved_count),
        ):
            if correct is not None and correct + unresolved > self.opportunity_count:
                raise ValueError("correct and unresolved exceed carrier opportunities")
        if not (
            2 <= self.current_correct_patient_count <= self.patient_count
            and 2 <= self.current_correct_content_count <= self.content_count
            and self.current_correct_count >= 2
        ):
            raise ValueError(
                "capability anchor needs correct support across two patients and contents"
            )
        return self


class CapabilityChange(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    evidence_id: str
    level: Level
    reference_unit: str
    difficulty: Difficulty
    opportunity_count: Count
    before_correct_count: Count
    after_correct_count: Count
    retained_count: Count
    lost_count: Count
    newly_correct_count: Count
    before_unresolved_count: Count
    after_unresolved_count: Count
    lost_to_error_count: Count
    lost_to_unresolved_count: Count
    before_cohort_character_insertions: Count
    after_cohort_character_insertions: Count
    response: Literal["retained", "regressed", "inconclusive"]
    citation_id: str

    @model_validator(mode="after")
    def validate_counts(self) -> "CapabilityChange":
        if self.retained_count + self.lost_count != self.before_correct_count:
            raise ValueError(
                "fixed pre-round correct cohort must partition into retained/lost"
            )
        if self.retained_count + self.newly_correct_count != self.after_correct_count:
            raise ValueError(
                "after correct count must partition into retained/newly correct"
            )
        if self.lost_to_error_count + self.lost_to_unresolved_count != self.lost_count:
            raise ValueError("capability losses must partition into error/unresolved")
        expected = (
            "regressed"
            if self.lost_to_error_count
            else "inconclusive"
            if self.lost_to_unresolved_count
            or self.after_cohort_character_insertions
            > self.before_cohort_character_insertions
            else "retained"
        )
        if self.response != expected:
            raise ValueError(
                "capability response differs from fixed-cohort retention facts"
            )
        if self.citation_id != f"result.capability.{self.evidence_id}":
            raise ValueError("capability citation must identify its evidence anchor")
        return self


def _read(rows, *, role: str, prediction_column: str):
    # Delayed import: schema.py embeds these independent capability models.
    from .evidence import _read_snapshot

    return _read_snapshot(
        rows, role=role, prediction_column=prediction_column, critical_cer_threshold=0.5
    )


def _state(row, raw_index: int) -> State:
    index = row.reference_projection.surface_to_normalized[raw_index]
    if index is None:
        return "unresolved"
    outcomes = row.reference_outcomes[index]
    if len(outcomes) != 1:
        return "unresolved"
    hyp_index = outcomes[0]
    if hyp_index is None:
        return "incorrect"
    return (
        "correct" if row.reference[index] == row.hypothesis[hyp_index] else "incorrect"
    )


def _carrier_positions(row, level: Level, reference_unit: str):
    return (
        position.index
        for position in row.phonology.reference.positions
        if position.resolved and (getattr(position, level) or "∅") == reference_unit
    )


def extract_capability_anchors(
    current_rows: Sequence[Mapping[str, object]],
    m0_rows: Sequence[Mapping[str, object]],
    *,
    current_prediction_column: str,
    m0_prediction_column: str,
    previous_rows: Sequence[Mapping[str, object]] | None = None,
    previous_prediction_column: str | None = None,
    max_anchors: int = 12,
) -> list[CapabilityAnchor]:
    """Return bounded positive evidence without requiring any current errors.

    Acquired means not confirmed correct in the comparison snapshot, now
    confirmed correct; unresolved counts remain visible rather than being
    interpreted as clinical recovery. Selection round-robins difficulty and
    component level so high-frequency tones do not consume the entire pack.
    """
    from .evidence import _strictly_pair

    if max_anchors < 1:
        raise ValueError("max_anchors must be positive")
    if (previous_rows is None) != (previous_prediction_column is None):
        raise ValueError(
            "previous rows and prediction column must be supplied together"
        )
    current = _read(
        current_rows, role="current", prediction_column=current_prediction_column
    )
    m0 = _read(m0_rows, role="m0", prediction_column=m0_prediction_column)
    _strictly_pair(current, m0, role="m0")
    m0_by_id = {row.utt_id: row for row in m0}
    previous_by_id = None
    if previous_rows is not None:
        previous = _read(
            previous_rows, role="previous", prediction_column=previous_prediction_column
        )
        _strictly_pair(current, previous, role="previous")
        previous_by_id = {row.utt_id: row for row in previous}

    grouped = defaultdict(list)
    for row in current:
        if row.difficulty not in DIFFICULTIES:
            continue
        for position in row.phonology.reference.positions:
            if not position.resolved:
                continue
            current_state = _state(row, position.index)
            m0_state = _state(m0_by_id[row.utt_id], position.index)
            previous_state = (
                _state(previous_by_id[row.utt_id], position.index)
                if previous_by_id is not None
                else None
            )
            for level in LEVELS:
                key = (level, getattr(position, level) or "∅", row.difficulty)
                grouped[key].append(
                    (
                        row.speaker_id,
                        row.content_id,
                        current_state,
                        m0_state,
                        previous_state,
                    )
                )

    candidates = defaultdict(list)
    for (level, unit, difficulty), positions in grouped.items():
        correct = [item for item in positions if item[2] == "correct"]
        correct_patients = len({item[0] for item in correct})
        correct_contents = len({item[1] for item in correct})
        if len(correct) < 2 or correct_patients < 2 or correct_contents < 2:
            continue

        def count(index, state="correct"):
            return sum(item[index] == state for item in positions)

        def transition(index, before_correct, after_correct):
            return sum(
                (item[index] == "correct") == before_correct
                and (item[2] == "correct") == after_correct
                for item in positions
            )

        anchor = CapabilityAnchor(
            evidence_id=f"ability.{level}.{unit if unit != '∅' else 'empty'}.{difficulty}",
            level=level,
            reference_unit=unit,
            difficulty=difficulty,
            opportunity_count=len(positions),
            patient_count=len({item[0] for item in positions}),
            content_count=len({item[1] for item in positions}),
            m0_correct_count=count(3),
            current_correct_count=len(correct),
            previous_correct_count=count(4) if previous_by_id is not None else None,
            m0_preserved_count=transition(3, True, True),
            m0_lost_count=transition(3, True, False),
            newly_acquired_count=transition(3, False, True),
            previous_preserved_count=transition(4, True, True)
            if previous_by_id is not None
            else None,
            previous_lost_count=transition(4, True, False)
            if previous_by_id is not None
            else None,
            previous_acquired_count=transition(4, False, True)
            if previous_by_id is not None
            else None,
            current_correct_patient_count=correct_patients,
            current_correct_content_count=correct_contents,
            m0_unresolved_count=count(3, "unresolved"),
            current_unresolved_count=count(2, "unresolved"),
            previous_unresolved_count=count(4, "unresolved")
            if previous_by_id is not None
            else None,
        )
        candidates[(difficulty, level)].append(anchor)
    for items in candidates.values():
        items.sort(
            key=lambda item: (
                -max(item.m0_lost_count, item.previous_lost_count or 0),
                -item.current_correct_content_count,
                -item.current_correct_patient_count,
                -item.current_correct_count,
                item.evidence_id,
            )
        )
    selected = []
    while len(selected) < max_anchors:
        added = False
        for level in LEVELS:
            for difficulty in DIFFICULTIES:
                items = candidates[(difficulty, level)]
                if items and len(selected) < max_anchors:
                    selected.append(items.pop(0))
                    added = True
        if not added:
            break
    return selected


def evaluate_capability_anchors(
    anchors: Sequence[CapabilityAnchor],
    before_rows: Sequence[Mapping[str, object]],
    after_rows: Sequence[Mapping[str, object]],
    *,
    before_prediction_column: str,
    after_prediction_column: str,
) -> list[CapabilityChange]:
    """Follow the pre-round correct positions; new successes cannot offset losses."""
    from .evidence import _strictly_pair

    before = _read(
        before_rows, role="before", prediction_column=before_prediction_column
    )
    after = _read(after_rows, role="after", prediction_column=after_prediction_column)
    _strictly_pair(before, after, role="after")
    after_by_id = {row.utt_id: row for row in after}
    changes = []
    for anchor in anchors:
        transitions = []
        before_insertions = after_insertions = 0
        for old in before:
            if old.difficulty != anchor.difficulty:
                continue
            new = after_by_id[old.utt_id]
            row_transitions = [
                (_state(old, index), _state(new, index))
                for index in _carrier_positions(
                    old, anchor.level, anchor.reference_unit
                )
            ]
            transitions.extend(row_transitions)
            if any(old_state == "correct" for old_state, _ in row_transitions):
                before_insertions += sum(
                    event.operation == "insertion" for event in old.events
                )
                after_insertions += sum(
                    event.operation == "insertion" for event in new.events
                )

        def count(predicate):
            return sum(predicate(old, new) for old, new in transitions)

        before_correct = count(lambda old, new: old == "correct")
        before_unresolved = count(lambda old, new: old == "unresolved")
        if (len(transitions), before_correct, before_unresolved) != (
            anchor.opportunity_count,
            anchor.current_correct_count,
            anchor.current_unresolved_count,
        ):
            raise ValueError(
                f"{anchor.evidence_id}: pre-round capability cohort differs from evidence"
            )
        lost_to_error = count(lambda old, new: old == "correct" and new == "incorrect")
        lost_to_unresolved = count(
            lambda old, new: old == "correct" and new == "unresolved"
        )
        changes.append(
            CapabilityChange(
                evidence_id=anchor.evidence_id,
                level=anchor.level,
                reference_unit=anchor.reference_unit,
                difficulty=anchor.difficulty,
                opportunity_count=len(transitions),
                before_correct_count=before_correct,
                after_correct_count=count(lambda old, new: new == "correct"),
                retained_count=count(lambda old, new: old == new == "correct"),
                lost_count=lost_to_error + lost_to_unresolved,
                newly_correct_count=count(
                    lambda old, new: old != "correct" and new == "correct"
                ),
                before_unresolved_count=before_unresolved,
                after_unresolved_count=count(lambda old, new: new == "unresolved"),
                lost_to_error_count=lost_to_error,
                lost_to_unresolved_count=lost_to_unresolved,
                before_cohort_character_insertions=before_insertions,
                after_cohort_character_insertions=after_insertions,
                response=(
                    "regressed"
                    if lost_to_error
                    else "inconclusive"
                    if lost_to_unresolved or after_insertions > before_insertions
                    else "retained"
                ),
                citation_id=f"result.capability.{anchor.evidence_id}",
            )
        )
    return changes


def count_capability_carriers(text: str, anchor: CapabilityAnchor) -> int:
    """Count pronunciation-facing text carriers, not retained acoustic realizations."""
    return sum(
        position.resolved
        and (getattr(position, anchor.level) or "∅") == anchor.reference_unit
        for position in map_text_to_syllables(tts_phonology_text(text)).positions
    )
