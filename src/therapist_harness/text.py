"""Canonical text identities shared by Therapist Harness boundaries."""

from __future__ import annotations

import re

from opencc import OpenCC


PUNCTUATION = "，。！？、,.!?;；:：\"'“”‘’（）()【】[]《》<>"
_OPENCC = OpenCC("s2t")
CHINESE_DIGITS = {
    "零": "0",
    "〇": "0",
    "一": "1",
    "二": "2",
    "兩": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}
CHINESE_NUMBER_CHARS = "".join(CHINESE_DIGITS) + "十百千廿卅"
YUE_REPLACEMENTS = (
    ("冇", "無"),
    ("沒", "無"),
    ("咩嘢", "乜嘢"),
    ("咩野", "乜嘢"),
    ("咩", "乜"),
    ("野", "嘢"),
    ("唔洗", "唔使"),
    ("中意", "鍾意"),
    ("钟意", "鍾意"),
    ("啊", "呀"),
    ("吖", "呀"),
    ("畀", "俾"),
    ("岩", "啱"),
    ("啱啱", "啱"),
    ("的", "嘅"),
)


def parse_chinese_number(text: str) -> int | None:
    if not text or any(
        char not in CHINESE_DIGITS and char not in "十百千廿卅" for char in text
    ):
        return None
    if text.startswith("廿"):
        suffix = text[1:]
        if not suffix:
            return 20
        if len(suffix) == 1 and suffix in CHINESE_DIGITS:
            return 20 + int(CHINESE_DIGITS[suffix])
        return None
    if text.startswith("卅"):
        suffix = text[1:]
        if not suffix:
            return 30
        if len(suffix) == 1 and suffix in CHINESE_DIGITS:
            return 30 + int(CHINESE_DIGITS[suffix])
        return None
    if all(char in CHINESE_DIGITS for char in text):
        return None

    total = 0
    current = 0
    units = {"十": 10, "百": 100, "千": 1000}
    for char in text:
        if char in CHINESE_DIGITS:
            current = int(CHINESE_DIGITS[char])
            continue
        unit = units.get(char)
        if unit is None:
            return None
        if current == 0:
            current = 1
        total += current * unit
        current = 0
    return total + current


def _normalize_chinese_number_token(match: re.Match[str]) -> str:
    token = match.group(0)
    if all(char in CHINESE_DIGITS for char in token):
        return "".join(CHINESE_DIGITS[char] for char in token)
    parsed = parse_chinese_number(token)
    return str(parsed) if parsed is not None else token


def text_normalize(text: str) -> str:
    """Apply the exact normalization currently used for project ASR CER."""

    if not isinstance(text, str):
        raise TypeError("text_normalize requires a string")
    normalized = re.sub(r"\s+", "", text)
    normalized = normalized.translate(str.maketrans("", "", PUNCTUATION))
    normalized = _OPENCC.convert(normalized)
    number_pattern = f"[{re.escape(CHINESE_NUMBER_CHARS)}]{{2,}}"
    normalized = re.sub(number_pattern, _normalize_chinese_number_token, normalized)
    for source, target in YUE_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    return normalized.lower()


def edit_distance(reference: str, hypothesis: str) -> int:
    """Return character-level Levenshtein distance."""

    previous = list(range(len(hypothesis) + 1))
    for row_index, reference_char in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hypothesis_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + (reference_char != hypothesis_char),
                )
            )
        previous = current
    return previous[-1]


def _phonological_keys(normalized: str, *, source_label: str) -> tuple[str, str]:
    try:
        import pycantonese
    except ImportError as error:  # pragma: no cover - dependency is in requirements.txt
        raise RuntimeError(
            "PyCantonese is required for content-isolation checks"
        ) from error

    try:
        pieces = pycantonese.characters_to_jyutping(normalized)
    except Exception as error:
        raise RuntimeError(f"PyCantonese failed for {source_label}") from error

    toned: list[str] = []
    detoned: list[str] = []
    for piece in pieces:
        if not isinstance(piece, tuple) or len(piece) != 2:
            raise RuntimeError(
                f"PyCantonese returned an invalid item for {source_label}"
            )
        surface, jyutping = piece
        if not isinstance(surface, str) or not surface:
            raise RuntimeError(
                f"PyCantonese returned an invalid surface for {source_label}"
            )
        if jyutping is None:
            # Keep the unknown surface itself as part of the identity. Unrelated
            # unknowns must never collapse to a shared UNK token.
            identity = f"UNK:{surface}"
            toned.append(identity)
            detoned.append(identity)
            continue
        if not isinstance(jyutping, str) or not jyutping.strip():
            raise RuntimeError(
                f"PyCantonese returned an invalid Jyutping value for {source_label}"
            )
        for syllable in jyutping.split():
            token = syllable.strip().lower()
            if not token:
                raise RuntimeError(
                    f"PyCantonese returned an empty Jyutping token for {source_label}"
                )
            toned.append(token)
            detoned.append(re.sub(r"[0-9]+$", "", token))

    if not toned:
        raise ValueError(f"{source_label} has no Jyutping identity tokens")
    return " ".join(toned), " ".join(detoned)


def identity_keys(text: str, *, source_label: str = "text") -> dict[str, str]:
    """Return all four exact-match identities used for content isolation."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{source_label} must be non-empty text")
    raw_clean = text.strip()
    normalized = text_normalize(text)
    if not normalized:
        raise ValueError(f"{source_label} is empty after project text_normalize")
    jyutping, detoned_jyutping = _phonological_keys(
        normalized, source_label=source_label
    )
    return {
        "raw_clean": raw_clean,
        "text_normalize": normalized,
        "jyutping": jyutping,
        "detoned_jyutping": detoned_jyutping,
    }


__all__ = [
    "edit_distance",
    "identity_keys",
    "parse_chinese_number",
    "text_normalize",
]
