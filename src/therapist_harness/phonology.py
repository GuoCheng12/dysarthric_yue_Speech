"""Conservative phonological attribution for Cantonese ASR text errors.

The public functions in this module expect the pronunciation-facing surface text:
punctuation and whitespace removed, but no CER equivalence rewrites applied.  They
deliberately do not infer acoustic or clinical articulation errors.  Instead, they
describe phonological differences between a reference transcript and an ASR
transcript.

Only character substitutions that occur in *every* optimal character-level
Levenshtein alignment are decomposed into Jyutping components.  Whole-character
insertions and deletions are counted for coverage, but are never relabelled as
onset, nucleus, coda, or tone errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pycantonese


PYCANTONESE_VERSION = "5.0.0"
if getattr(pycantonese, "__version__", None) != PYCANTONESE_VERSION:
    raise RuntimeError(
        f"therapist phonology requires PyCantonese {PYCANTONESE_VERSION} exactly"
    )


ComponentLevel = Literal["onset", "nucleus", "coda", "tone"]
ComponentOperation = Literal["deletion", "substitution", "insertion"]
UnresolvedReason = Literal[
    "missing_jyutping", "invalid_jyutping", "syllable_count_mismatch"
]


@dataclass(frozen=True)
class G2PPosition:
    """One input character and its context-selected Jyutping analysis."""

    index: int
    character: str
    jyutping: str | None
    onset: str | None
    nucleus: str | None
    coda: str | None
    tone: str | None
    context_sensitive: bool | None
    unresolved_reason: UnresolvedReason | None

    @property
    def resolved(self) -> bool:
        return self.jyutping is not None


@dataclass(frozen=True)
class G2PCoverage:
    """Raw counts needed to report G2P coverage without hiding exclusions."""

    character_count: int
    resolved_count: int
    unresolved_count: int
    missing_jyutping_count: int
    invalid_jyutping_count: int
    syllable_count_mismatch_count: int
    context_comparable_count: int
    context_sensitive_count: int
    context_uncheckable_count: int


@dataclass(frozen=True)
class G2PMapping:
    text: str
    positions: tuple[G2PPosition, ...]
    coverage: G2PCoverage


@dataclass(frozen=True)
class ForcedSubstitution:
    """A diagonal substitution edge shared by every optimal alignment."""

    reference_index: int
    hypothesis_index: int
    reference_character: str
    hypothesis_character: str


@dataclass(frozen=True)
class CharacterAlignment:
    """Character edit counts from one stable traceback plus forced edges.

    The stable traceback uses diagonal, deletion, then insertion priority.  Its
    operation counts are diagnostic only.  ``forced_substitutions`` is invariant
    across traceback choices because each listed edge occurs in every optimal
    path.
    """

    edit_distance: int
    optimal_path_count: int
    substitution_count: int
    deletion_count: int
    insertion_count: int
    forced_substitutions: tuple[ForcedSubstitution, ...]
    # For each GT character, every hypothesis index reachable on an optimal
    # alignment; None means deletion. Multiple alternatives are unresolved.
    reference_outcomes: tuple[tuple[int | None, ...], ...]


@dataclass(frozen=True)
class ComponentEvent:
    """One component difference from a forced, G2P-resolved substitution."""

    level: ComponentLevel
    operation: ComponentOperation
    reference_unit: str
    hypothesis_unit: str
    reference_index: int
    hypothesis_index: int
    reference_character: str
    hypothesis_character: str
    reference_jyutping: str
    hypothesis_jyutping: str
    context_sensitive: bool | None


@dataclass(frozen=True)
class PhonologyCoverage:
    """Raw alignment and attribution counts for one utterance pair."""

    reference_character_count: int
    hypothesis_character_count: int
    character_edit_distance: int
    optimal_alignment_count: int
    character_substitution_count: int
    character_deletion_count: int
    character_insertion_count: int
    forced_substitution_count: int
    forced_substitution_g2p_resolved_count: int
    forced_substitution_g2p_unresolved_count: int
    forced_substitution_context_sensitive_count: int
    forced_substitution_context_uncheckable_count: int
    phonologically_equivalent_forced_substitution_count: int
    phonologically_different_forced_substitution_count: int
    component_event_count: int


@dataclass(frozen=True)
class PhonologyAnalysis:
    reference: G2PMapping
    hypothesis: G2PMapping
    alignment: CharacterAlignment
    component_events: tuple[ComponentEvent, ...]
    coverage: PhonologyCoverage


def _parse_word(surface: str, jyutping: str | None) -> tuple[object, ...] | None:
    """Return parsed syllables only when the word has an exact 1:1 mapping."""

    if jyutping is None:
        return None
    try:
        parsed = tuple(pycantonese.parse_jyutping(jyutping))
    except (TypeError, ValueError):
        return None
    if len(parsed) != len(surface):
        return None
    return parsed


def _context_free_readings(text: str) -> tuple[str | None, ...]:
    """Return per-character dictionary defaults for a sensitivity check."""

    if not text:
        return ()
    pieces = pycantonese.characters_to_jyutping(list(text))
    if len(pieces) != len(text):
        raise RuntimeError(
            "PyCantonese did not preserve the requested per-character segmentation"
        )
    readings: list[str | None] = []
    for expected, piece in zip(text, pieces):
        if not isinstance(piece, tuple) or len(piece) != 2:
            raise RuntimeError("PyCantonese returned an invalid per-character item")
        surface, jyutping = piece
        if surface != expected:
            raise RuntimeError(
                "PyCantonese changed a character in per-character segmentation"
            )
        parsed = _parse_word(surface, jyutping)
        readings.append(str(parsed[0]) if parsed is not None else None)
    return tuple(readings)


def map_text_to_syllables(text: str) -> G2PMapping:
    """Map a complete string to at most one Jyutping syllable per character.

    Contextual word segmentation is used for the primary reading.  A word with
    missing, invalid, or non-1:1 Jyutping is represented by unresolved positions;
    this function never falls back to a guessed pronunciation.  A separately
    requested per-character dictionary reading is used only to mark whether the
    contextual reading is segmentation-sensitive.
    """

    if not isinstance(text, str):
        raise TypeError("map_text_to_syllables requires a string")
    if not text:
        empty_coverage = G2PCoverage(0, 0, 0, 0, 0, 0, 0, 0, 0)
        return G2PMapping(text="", positions=(), coverage=empty_coverage)

    pieces = pycantonese.characters_to_jyutping(text)
    if not isinstance(pieces, list):
        raise RuntimeError("PyCantonese returned a non-list contextual mapping")

    joined_surfaces: list[str] = []
    contextual: list[
        tuple[str, str | None, object | None, UnresolvedReason | None]
    ] = []
    for piece in pieces:
        if not isinstance(piece, tuple) or len(piece) != 2:
            raise RuntimeError("PyCantonese returned an invalid contextual item")
        surface, jyutping = piece
        if not isinstance(surface, str) or not surface:
            raise RuntimeError("PyCantonese returned an empty contextual surface")
        joined_surfaces.append(surface)

        if jyutping is None:
            contextual.extend(
                (character, None, None, "missing_jyutping") for character in surface
            )
            continue
        try:
            parsed = tuple(pycantonese.parse_jyutping(jyutping))
        except (TypeError, ValueError):
            contextual.extend(
                (character, None, None, "invalid_jyutping") for character in surface
            )
            continue
        if len(parsed) != len(surface):
            contextual.extend(
                (character, None, None, "syllable_count_mismatch")
                for character in surface
            )
            continue
        contextual.extend(
            (character, str(syllable), syllable, None)
            for character, syllable in zip(surface, parsed)
        )

    if "".join(joined_surfaces) != text:
        raise RuntimeError("PyCantonese contextual surfaces do not reconstruct input")
    if len(contextual) != len(text):
        raise RuntimeError("internal G2P character mapping length mismatch")

    defaults = _context_free_readings(text)
    positions: list[G2PPosition] = []
    for index, ((character, jyutping, syllable, reason), default) in enumerate(
        zip(contextual, defaults)
    ):
        if syllable is None:
            positions.append(
                G2PPosition(
                    index=index,
                    character=character,
                    jyutping=None,
                    onset=None,
                    nucleus=None,
                    coda=None,
                    tone=None,
                    context_sensitive=None,
                    unresolved_reason=reason,
                )
            )
            continue
        positions.append(
            G2PPosition(
                index=index,
                character=character,
                jyutping=jyutping,
                onset=syllable.onset,
                nucleus=syllable.nucleus,
                coda=syllable.coda,
                tone=syllable.tone,
                context_sensitive=(
                    jyutping != default if default is not None else None
                ),
                unresolved_reason=None,
            )
        )

    reason_counts = {
        reason: sum(position.unresolved_reason == reason for position in positions)
        for reason in (
            "missing_jyutping",
            "invalid_jyutping",
            "syllable_count_mismatch",
        )
    }
    resolved_count = sum(position.resolved for position in positions)
    context_comparable_count = sum(
        position.resolved and position.context_sensitive is not None
        for position in positions
    )
    context_sensitive_count = sum(
        position.context_sensitive is True for position in positions
    )
    coverage = G2PCoverage(
        character_count=len(text),
        resolved_count=resolved_count,
        unresolved_count=len(text) - resolved_count,
        missing_jyutping_count=reason_counts["missing_jyutping"],
        invalid_jyutping_count=reason_counts["invalid_jyutping"],
        syllable_count_mismatch_count=reason_counts["syllable_count_mismatch"],
        context_comparable_count=context_comparable_count,
        context_sensitive_count=context_sensitive_count,
        context_uncheckable_count=resolved_count - context_comparable_count,
    )
    return G2PMapping(text=text, positions=tuple(positions), coverage=coverage)


def align_characters(reference: str, hypothesis: str) -> CharacterAlignment:
    """Align characters and identify substitutions forced in every optimum."""

    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("align_characters requires two strings")
    n = len(reference)
    m = len(hypothesis)

    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0] = i
    for j in range(1, m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i][j] = min(
                costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
            )

    forward = [[0] * (m + 1) for _ in range(n + 1)]
    forward[0][0] = 1
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            if (
                i
                and j
                and costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
                == costs[i][j]
            ):
                forward[i][j] += forward[i - 1][j - 1]
            if i and costs[i - 1][j] + 1 == costs[i][j]:
                forward[i][j] += forward[i - 1][j]
            if j and costs[i][j - 1] + 1 == costs[i][j]:
                forward[i][j] += forward[i][j - 1]

    backward = [[0] * (m + 1) for _ in range(n + 1)]
    backward[n][m] = 1
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i == n and j == m:
                continue
            if (
                i < n
                and j < m
                and costs[i][j] + (reference[i] != hypothesis[j]) == costs[i + 1][j + 1]
            ):
                backward[i][j] += backward[i + 1][j + 1]
            if i < n and costs[i][j] + 1 == costs[i + 1][j]:
                backward[i][j] += backward[i + 1][j]
            if j < m and costs[i][j] + 1 == costs[i][j + 1]:
                backward[i][j] += backward[i][j + 1]

    optimal_path_count = forward[n][m]
    reference_outcomes: list[tuple[int | None, ...]] = []
    for ref_index in range(n):
        outcomes: list[int | None] = []
        for hyp_index in range(m + 1):
            if not forward[ref_index][hyp_index]:
                continue
            if (
                hyp_index < m
                and costs[ref_index][hyp_index]
                + (reference[ref_index] != hypothesis[hyp_index])
                == costs[ref_index + 1][hyp_index + 1]
                and backward[ref_index + 1][hyp_index + 1]
            ):
                outcomes.append(hyp_index)
            if (
                None not in outcomes
                and costs[ref_index][hyp_index] + 1 == costs[ref_index + 1][hyp_index]
                and backward[ref_index + 1][hyp_index]
            ):
                outcomes.append(None)
        reference_outcomes.append(tuple(outcomes))
    substitutions = 0
    deletions = 0
    insertions = 0
    forced: list[ForcedSubstitution] = []
    i, j = n, m
    while i or j:
        if (
            i
            and j
            and costs[i][j]
            == costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
        ):
            if reference[i - 1] != hypothesis[j - 1]:
                substitutions += 1
                paths_through_edge = forward[i - 1][j - 1] * backward[i][j]
                if paths_through_edge == optimal_path_count:
                    forced.append(
                        ForcedSubstitution(
                            reference_index=i - 1,
                            hypothesis_index=j - 1,
                            reference_character=reference[i - 1],
                            hypothesis_character=hypothesis[j - 1],
                        )
                    )
            i -= 1
            j -= 1
            continue
        if i and costs[i][j] == costs[i - 1][j] + 1:
            deletions += 1
            i -= 1
            continue
        if j and costs[i][j] == costs[i][j - 1] + 1:
            insertions += 1
            j -= 1
            continue
        raise RuntimeError("Levenshtein traceback is inconsistent")

    forced.reverse()
    if substitutions + deletions + insertions != costs[n][m]:
        raise RuntimeError("character edit counts do not equal edit distance")
    return CharacterAlignment(
        edit_distance=costs[n][m],
        optimal_path_count=optimal_path_count,
        substitution_count=substitutions,
        deletion_count=deletions,
        insertion_count=insertions,
        forced_substitutions=tuple(forced),
        reference_outcomes=tuple(reference_outcomes),
    )


def _pair_context_sensitive(
    reference: G2PPosition, hypothesis: G2PPosition
) -> bool | None:
    if reference.context_sensitive is True or hypothesis.context_sensitive is True:
        return True
    if reference.context_sensitive is None or hypothesis.context_sensitive is None:
        return None
    return False


def _component_operation(
    reference_unit: str, hypothesis_unit: str
) -> ComponentOperation:
    if reference_unit and not hypothesis_unit:
        return "deletion"
    if not reference_unit and hypothesis_unit:
        return "insertion"
    return "substitution"


def analyze_phonological_errors(reference: str, hypothesis: str) -> PhonologyAnalysis:
    """Return conservative Jyutping component events and raw coverage counts."""

    reference_mapping = map_text_to_syllables(reference)
    hypothesis_mapping = map_text_to_syllables(hypothesis)
    alignment = align_characters(reference, hypothesis)

    component_events: list[ComponentEvent] = []
    resolved_pairs = 0
    context_sensitive_pairs = 0
    context_uncheckable_pairs = 0
    equivalent_pairs = 0
    different_pairs = 0
    levels: tuple[ComponentLevel, ...] = ("onset", "nucleus", "coda", "tone")

    for substitution in alignment.forced_substitutions:
        ref_position = reference_mapping.positions[substitution.reference_index]
        hyp_position = hypothesis_mapping.positions[substitution.hypothesis_index]
        if not ref_position.resolved or not hyp_position.resolved:
            continue
        resolved_pairs += 1
        sensitive = _pair_context_sensitive(ref_position, hyp_position)
        if sensitive is True:
            context_sensitive_pairs += 1
        elif sensitive is None:
            context_uncheckable_pairs += 1

        if ref_position.jyutping == hyp_position.jyutping:
            equivalent_pairs += 1
            continue
        different_pairs += 1
        for level in levels:
            reference_unit = getattr(ref_position, level)
            hypothesis_unit = getattr(hyp_position, level)
            if reference_unit is None or hypothesis_unit is None:
                raise RuntimeError("resolved G2P position has a missing component")
            if reference_unit == hypothesis_unit:
                continue
            component_events.append(
                ComponentEvent(
                    level=level,
                    operation=_component_operation(reference_unit, hypothesis_unit),
                    reference_unit=reference_unit,
                    hypothesis_unit=hypothesis_unit,
                    reference_index=substitution.reference_index,
                    hypothesis_index=substitution.hypothesis_index,
                    reference_character=substitution.reference_character,
                    hypothesis_character=substitution.hypothesis_character,
                    reference_jyutping=ref_position.jyutping,
                    hypothesis_jyutping=hyp_position.jyutping,
                    context_sensitive=sensitive,
                )
            )

    forced_count = len(alignment.forced_substitutions)
    coverage = PhonologyCoverage(
        reference_character_count=len(reference),
        hypothesis_character_count=len(hypothesis),
        character_edit_distance=alignment.edit_distance,
        optimal_alignment_count=alignment.optimal_path_count,
        character_substitution_count=alignment.substitution_count,
        character_deletion_count=alignment.deletion_count,
        character_insertion_count=alignment.insertion_count,
        forced_substitution_count=forced_count,
        forced_substitution_g2p_resolved_count=resolved_pairs,
        forced_substitution_g2p_unresolved_count=forced_count - resolved_pairs,
        forced_substitution_context_sensitive_count=context_sensitive_pairs,
        forced_substitution_context_uncheckable_count=context_uncheckable_pairs,
        phonologically_equivalent_forced_substitution_count=equivalent_pairs,
        phonologically_different_forced_substitution_count=different_pairs,
        component_event_count=len(component_events),
    )
    return PhonologyAnalysis(
        reference=reference_mapping,
        hypothesis=hypothesis_mapping,
        alignment=alignment,
        component_events=tuple(component_events),
        coverage=coverage,
    )


__all__ = [
    "CharacterAlignment",
    "ComponentEvent",
    "ForcedSubstitution",
    "G2PCoverage",
    "G2PMapping",
    "G2PPosition",
    "PhonologyAnalysis",
    "PhonologyCoverage",
    "align_characters",
    "analyze_phonological_errors",
    "map_text_to_syllables",
]
