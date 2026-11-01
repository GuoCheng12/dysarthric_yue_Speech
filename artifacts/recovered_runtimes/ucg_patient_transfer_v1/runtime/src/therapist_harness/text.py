"""Canonical text identities shared by Therapist Harness boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

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


@dataclass(frozen=True)
class CerNormalizationProjection:
    """Conservative index provenance for project CER normalization.

    ``surface`` removes only whitespace and punctuation, so its characters are
    still the ones whose Cantonese readings may be inspected. ``normalized`` is
    the exact project CER string.  A surface index maps to a normalized index
    only when that diagonal correspondence occurs in every minimum-edit
    alignment between the two strings.  Ambiguous and length-changing regions
    deliberately map to ``None`` rather than guessing provenance.
    """

    surface: str
    normalized: str
    surface_to_normalized: tuple[int | None, ...]

    @property
    def normalized_to_surface(self) -> tuple[int | None, ...]:
        """Invert only forced one-to-one links; never invent a spoken carrier."""

        inverse: list[int | None] = [None] * len(self.normalized)
        for surface_index, normalized_index in enumerate(self.surface_to_normalized):
            if normalized_index is not None:
                inverse[normalized_index] = surface_index
        return tuple(inverse)


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


def tts_phonology_text(text: str) -> str:
    """Return the characters whose pronunciation will be checked before TTS.

    This deliberately does *not* reuse :func:`text_normalize`.  The CER
    normalizer contains equivalence rewrites such as ``冇 -> 無`` and number
    canonicalization; those rewrites are useful for scoring transcripts but can
    change the Cantonese phones that CosyVoice will actually receive.  Only
    whitespace and punctuation, which do not supply a segmental carrier, are
    removed here.  Stimulus policy separately requires Traditional Chinese.
    """

    if not isinstance(text, str):
        raise TypeError("tts_phonology_text requires a string")
    cleaned = re.sub(r"\s+", "", text)
    return cleaned.translate(str.maketrans("", "", PUNCTUATION)).lower()


def _forced_diagonal_projection(source: str, target: str) -> tuple[int | None, ...]:
    """Map source indices through diagonal edges shared by all optima."""

    n = len(source)
    m = len(target)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0] = i
    for j in range(1, m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i][j] = min(
                costs[i - 1][j - 1] + (source[i - 1] != target[j - 1]),
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
                and costs[i - 1][j - 1] + (source[i - 1] != target[j - 1])
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
                and costs[i][j] + (source[i] != target[j]) == costs[i + 1][j + 1]
            ):
                backward[i][j] += backward[i + 1][j + 1]
            if i < n and costs[i][j] + 1 == costs[i + 1][j]:
                backward[i][j] += backward[i + 1][j]
            if j < m and costs[i][j] + 1 == costs[i][j + 1]:
                backward[i][j] += backward[i][j + 1]

    optimal_path_count = forward[n][m]
    projection: list[int | None] = [None] * n
    for i in range(n):
        for j in range(m):
            if (
                costs[i][j] + (source[i] != target[j]) == costs[i + 1][j + 1]
                and forward[i][j] * backward[i + 1][j + 1] == optimal_path_count
            ):
                projection[i] = j
                break
    return tuple(projection)


def cer_normalization_projection(text: str) -> CerNormalizationProjection:
    """Return a conservative bridge from spoken surface to CER identity."""

    if not isinstance(text, str):
        raise TypeError("cer_normalization_projection requires a string")
    surface = tts_phonology_text(text)
    normalized = text_normalize(text)
    return CerNormalizationProjection(
        surface=surface,
        normalized=normalized,
        surface_to_normalized=_forced_diagonal_projection(surface, normalized),
    )


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


def _phonological_keys(surface: str, *, source_label: str) -> tuple[str, str]:
    try:
        import pycantonese
    except ImportError as error:  # pragma: no cover - dependency is in requirements.txt
        raise RuntimeError(
            "PyCantonese is required for content-isolation checks"
        ) from error

    try:
        pieces = pycantonese.characters_to_jyutping(surface)
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
    """Return scoring and spoken-content identities for content isolation.

    The CER key retains the project's transcript equivalences. Jyutping keys
    instead describe the words actually spoken, before those equivalences can
    change a syllable (for example, ``咩/me1 -> 乜/mat1``).
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{source_label} must be non-empty text")
    raw_clean = text.strip()
    normalized = text_normalize(text)
    if not normalized:
        raise ValueError(f"{source_label} is empty after project text_normalize")
    jyutping, detoned_jyutping = _phonological_keys(
        tts_phonology_text(text), source_label=source_label
    )
    return {
        "raw_clean": raw_clean,
        "text_normalize": normalized,
        "jyutping": jyutping,
        "detoned_jyutping": detoned_jyutping,
    }


__all__ = [
    "CerNormalizationProjection",
    "cer_normalization_projection",
    "edit_distance",
    "identity_keys",
    "parse_chinese_number",
    "text_normalize",
    "tts_phonology_text",
]
