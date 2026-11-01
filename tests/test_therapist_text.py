from __future__ import annotations

import pytest

from therapist_harness.text import (
    cer_normalization_projection,
    edit_distance,
    identity_keys,
    parse_chinese_number,
    text_normalize,
    tts_phonology_text,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("十一", 11),
        ("一百零五", 105),
        ("廿", 20),
        ("廿一", 21),
        ("卅", 30),
        ("卅九", 39),
    ],
)
def test_parse_chinese_number_supported_forms(text: str, expected: int) -> None:
    assert parse_chinese_number(text) == expected


@pytest.mark.parametrize("text", ["二三廿", "二卅", "十廿", "卅百", "廿十"])
def test_malformed_special_tens_are_preserved(text: str) -> None:
    assert parse_chinese_number(text) is None
    assert text_normalize(text) == text


def test_parser_never_raises_for_its_declared_character_set() -> None:
    declared = "零〇一二兩两三四五六七八九十百千廿卅"
    for left in declared:
        for right in declared:
            parse_chinese_number(left + right)


def test_exact_asr_crash_regression_is_conservative() -> None:
    prediction = "我二三廿你你食"
    assert text_normalize(prediction) == prediction


def test_project_text_normalization_and_edit_distance() -> None:
    assert text_normalize(" 我冇三十四個。 ") == "我無34個"
    assert edit_distance("廣東話", "廣州話") == 1


def test_tts_phonology_text_preserves_spoken_lexemes_and_numbers() -> None:
    assert tts_phonology_text(" 我冇三十四個。 ") == "我冇三十四個"


def test_spoken_identity_detects_homophones_before_cer_rewrites() -> None:
    question = identity_keys("呢個咩字點讀呀。")
    carry = identity_keys("呢個孭字點讀呀。")

    assert question["text_normalize"] != carry["text_normalize"]
    assert question["jyutping"] == carry["jyutping"]
    assert question["detoned_jyutping"] == carry["detoned_jyutping"]
    assert "me1" in question["jyutping"].split()
    assert "mat1" not in question["jyutping"].split()


def test_cer_equivalent_words_retain_different_spoken_tones() -> None:
    colloquial = identity_keys("我冇時間。")
    formal = identity_keys("我無時間。")

    assert colloquial["text_normalize"] == formal["text_normalize"] == "我無時間"
    assert "mou5" in colloquial["jyutping"].split()
    assert "mou4" in formal["jyutping"].split()
    assert colloquial["jyutping"] != formal["jyutping"]


def test_normalized_positions_recover_only_forced_surface_carriers() -> None:
    projection = cer_normalization_projection("我冇時間。")
    index = projection.normalized.index("無")
    surface_index = projection.normalized_to_surface[index]
    assert surface_index is not None
    assert projection.surface[surface_index] == "冇"

    contracted = cer_normalization_projection("啱啱好。")
    assert contracted.normalized_to_surface == (None, 2)
