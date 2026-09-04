from __future__ import annotations

import pytest

from therapist_harness.text import (
    edit_distance,
    parse_chinese_number,
    text_normalize,
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
