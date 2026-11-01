from __future__ import annotations

import pycantonese

from therapist_harness.phonology import (
    align_characters,
    analyze_phonological_errors,
    map_text_to_syllables,
)


def test_contextual_g2p_is_one_syllable_per_character() -> None:
    mapping = map_text_to_syllables("銀行")

    assert [position.character for position in mapping.positions] == ["銀", "行"]
    assert [position.jyutping for position in mapping.positions] == ["ngan4", "hong4"]
    assert mapping.positions[1].context_sensitive is True
    assert mapping.positions[1].tone == "4"
    assert mapping.coverage.character_count == 2
    assert mapping.coverage.resolved_count == 2
    assert mapping.coverage.context_sensitive_count == 1


def test_non_one_to_one_word_is_unresolved_without_character_fallback(
    monkeypatch,
) -> None:
    original = pycantonese.characters_to_jyutping

    def malformed(value):
        if value == "甲乙":
            return [("甲乙", "gaa3")]
        return original(value)

    monkeypatch.setattr(pycantonese, "characters_to_jyutping", malformed)
    mapping = map_text_to_syllables("甲乙")

    assert [position.jyutping for position in mapping.positions] == [None, None]
    assert {position.unresolved_reason for position in mapping.positions} == {
        "syllable_count_mismatch"
    }
    assert mapping.coverage.syllable_count_mismatch_count == 2
    assert mapping.coverage.resolved_count == 0


def test_forced_substitution_requires_the_same_edge_in_every_optimal_path() -> None:
    unique = align_characters("abc", "axc")
    assert unique.optimal_path_count == 1
    assert unique.substitution_count == 1
    assert [
        (item.reference_index, item.hypothesis_index)
        for item in unique.forced_substitutions
    ] == [(1, 1)]

    ambiguous = align_characters("a", "bb")
    assert ambiguous.edit_distance == 2
    assert ambiguous.optimal_path_count == 2
    assert ambiguous.substitution_count == 1
    assert ambiguous.forced_substitutions == ()


def test_only_forced_substitutions_create_component_events() -> None:
    analysis = analyze_phonological_errors("角", "哥")

    assert analysis.coverage.forced_substitution_count == 1
    assert analysis.coverage.phonologically_different_forced_substitution_count == 1
    assert [
        (event.level, event.operation, event.reference_unit, event.hypothesis_unit)
        for event in analysis.component_events
    ] == [
        ("coda", "deletion", "k", ""),
        ("tone", "substitution", "3", "1"),
    ]

    deletion = analyze_phonological_errors("角", "")
    assert deletion.coverage.character_deletion_count == 1
    assert deletion.coverage.forced_substitution_count == 0
    assert deletion.component_events == ()

    ambiguous = analyze_phonological_errors("啊", "啤啤")
    assert ambiguous.alignment.optimal_path_count > 1
    assert ambiguous.alignment.forced_substitutions == ()
    assert ambiguous.component_events == ()


def test_homophonic_substitution_is_covered_but_has_no_component_event() -> None:
    analysis = analyze_phonological_errors("角", "各")

    assert analysis.coverage.forced_substitution_g2p_resolved_count == 1
    assert analysis.coverage.phonologically_equivalent_forced_substitution_count == 1
    assert analysis.coverage.phonologically_different_forced_substitution_count == 0
    assert analysis.component_events == ()


def test_context_sensitivity_is_propagated_to_component_events() -> None:
    analysis = analyze_phonological_errors("銀行", "銀巷")

    assert analysis.coverage.forced_substitution_count == 1
    assert analysis.coverage.forced_substitution_context_sensitive_count == 1
    assert analysis.component_events
    assert all(event.context_sensitive is True for event in analysis.component_events)
