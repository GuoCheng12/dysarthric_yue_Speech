#!/usr/bin/env python3
"""Check train/test prompt leakage for cleaned read-speech ASR splits.

The script checks three prompt identity layers:

1. raw text exact match
2. normalized text exact match
3. Jyutping sequence exact match

It can also join a three-way prediction table to summarize zero-shot, E2, and
new-method metrics on seen-prompt versus unseen-prompt test subsets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

try:
    import pycantonese
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency `pycantonese`. Install it with: pip install pycantonese"
    ) from exc

try:
    from opencc import OpenCC
except ImportError:  # pragma: no cover
    OpenCC = None


BASELINE_NAME = "E2_fullSFT_clean_pooled_epoch2"
DEFAULT_NEW_METHOD = "E3_LoRA_r16_best_dev_checkpoint_483"

PUNCTUATION = "，。！？、,.!?;；:：\"'“”‘’（）()【】[]《》<>"
FALLBACK_S2T = str.maketrans(
    {
        "妈": "媽",
        "汉": "漢",
        "红": "紅",
        "黄": "黃",
        "个": "個",
        "点": "點",
        "样": "樣",
        "边": "邊",
        "没": "沒",
        "会": "會",
        "动": "動",
        "电": "電",
        "脑": "腦",
        "数": "數",
        "学": "學",
        "声": "聲",
        "开": "開",
        "关": "關",
        "车": "車",
        "鱼": "魚",
        "饭": "飯",
        "过": "過",
        "为": "為",
        "后": "後",
        "来": "來",
        "这": "這",
        "实": "實",
        "试": "試",
        "钟": "鍾",
    }
)
OPENCC = OpenCC("s2t") if OpenCC is not None else None

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
YUE_REPLACEMENTS = [
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
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--three-way-csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--private-out-dir")
    p.add_argument("--text-column", default="clean_gt")
    p.add_argument("--raw-text-column", default="raw_gt")
    p.add_argument("--baseline-name", default=BASELINE_NAME)
    p.add_argument("--new-method-name", default=DEFAULT_NEW_METHOD)
    return p.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text_basic(text: str) -> str:
    text = re.sub(r"\s+", "", text or "")
    return text.translate(str.maketrans("", "", PUNCTUATION))


def to_traditional(text: str) -> str:
    if OPENCC is not None:
        return OPENCC.convert(text)
    return text.translate(FALLBACK_S2T)


def parse_chinese_number(text: str) -> int | None:
    if not text or any(ch not in CHINESE_DIGITS and ch not in "十百千廿卅" for ch in text):
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
    if all(ch in CHINESE_DIGITS for ch in text):
        return None

    total = 0
    current = 0
    units = {"十": 10, "百": 100, "千": 1000}
    for ch in text:
        if ch in CHINESE_DIGITS:
            current = int(CHINESE_DIGITS[ch])
        else:
            if current == 0:
                current = 1
            total += current * units[ch]
            current = 0
    return total + current


def normalize_chinese_number_token(match: re.Match[str]) -> str:
    token = match.group(0)
    if all(ch in CHINESE_DIGITS for ch in token):
        return "".join(CHINESE_DIGITS[ch] for ch in token)
    parsed = parse_chinese_number(token)
    return str(parsed) if parsed is not None else token


def normalize_numbers(text: str) -> str:
    pattern = f"[{re.escape(CHINESE_NUMBER_CHARS)}]{{2,}}"
    return re.sub(pattern, normalize_chinese_number_token, text)


def text_normalize(text: str) -> str:
    text = normalize_text_basic(text)
    text = to_traditional(text)
    text = normalize_numbers(text)
    for source, target in YUE_REPLACEMENTS:
        text = text.replace(source, target)
    return text.lower()


def jyutping_sequence(text: str) -> str:
    normalized = text_normalize(text)
    pieces = pycantonese.characters_to_jyutping(normalized)
    syllables: list[str] = []
    for surface, jyutping in pieces:
        if jyutping is None:
            syllables.append(f"UNK:{surface}")
            continue
        syllables.extend(jyutping.split())
    return " ".join(syllables)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: object) -> float:
    if value in ("", None):
        return math.nan
    return float(value)


def fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return statistics.mean(values) if values else math.nan


def sha16(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def make_keys(row: dict[str, str], raw_col: str, clean_col: str) -> dict[str, str]:
    raw = row.get(raw_col, "")
    clean = row.get(clean_col, "")
    return {
        "raw": raw.strip(),
        "normalized": text_normalize(clean),
        "jyutping": jyutping_sequence(clean),
    }


def build_index(rows: list[dict[str, str]], raw_col: str, clean_col: str) -> dict[str, dict[str, list[dict[str, str]]]]:
    index = {"raw": defaultdict(list), "normalized": defaultdict(list), "jyutping": defaultdict(list)}
    for row in rows:
        keys = make_keys(row, raw_col, clean_col)
        for layer, key in keys.items():
            if key:
                index[layer][key].append(row)
    return index


def method_metrics(row: dict[str, str], method: str, baseline_name: str) -> tuple[float, bool]:
    if method == "zero_shot":
        return as_float(row["zero_shot_cer"]), as_bool(row["zero_shot_critical"])
    if method == "baseline":
        return as_float(row[f"{baseline_name}_cer"]), as_bool(row[f"{baseline_name}_critical"])
    if method == "new":
        return as_float(row["new_method_cer"]), as_bool(row["new_method_critical"])
    raise ValueError(method)


def summarize_subset(
    rows: list[dict[str, str]],
    layer: str,
    subset: str,
    baseline_name: str,
    new_method_name: str,
) -> dict[str, object]:
    out: dict[str, object] = {
        "leakage_layer": layer,
        "test_subset": subset,
        "sample_count": len(rows),
    }
    for label, method in [
        ("zero_shot", "zero_shot"),
        (baseline_name, "baseline"),
        (new_method_name, "new"),
    ]:
        cers: list[float] = []
        crits: list[bool] = []
        for row in rows:
            cer, critical = method_metrics(row, method, baseline_name)
            cers.append(cer)
            crits.append(critical)
        out[f"{label}_cer"] = fmt(mean(cers))
        out[f"{label}_critical_rate"] = fmt(sum(crits) / len(crits)) if crits else ""
        out[f"{label}_critical_count"] = sum(crits)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    private_out_dir = Path(args.private_out_dir) if args.private_out_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)
    if private_out_dir:
        private_out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_csv(args.train_csv)
    test_rows = read_csv(args.test_csv)
    train_index = build_index(train_rows, args.raw_text_column, args.text_column)

    three_way_by_utt: dict[str, dict[str, str]] = {}
    if args.three_way_csv:
        three_way_by_utt = {row["utt_id"]: row for row in read_csv(args.three_way_csv)}

    sanitized_rows: list[dict[str, object]] = []
    private_rows: list[dict[str, object]] = []
    layer_seen_counts = Counter()
    any_seen_count = 0
    joined_metric_rows: list[dict[str, str]] = []

    for i, row in enumerate(test_rows, 1):
        keys = make_keys(row, args.raw_text_column, args.text_column)
        seen: dict[str, bool] = {}
        match_counts: dict[str, int] = {}
        matched_train_ids: dict[str, str] = {}
        for layer, key in keys.items():
            matches = train_index[layer].get(key, [])
            seen[layer] = bool(matches)
            match_counts[layer] = len(matches)
            matched_train_ids[layer] = ";".join(match.get("utt_id", "") for match in matches[:20])
            if seen[layer]:
                layer_seen_counts[layer] += 1
        any_seen = any(seen.values())
        if any_seen:
            any_seen_count += 1

        sanitized_rows.append(
            {
                "row_no": i,
                "sample_key": f"test_{i:04d}",
                "utt_id_sha256_16": sha16(row["utt_id"]),
                "speaker_sha256_12": hashlib.sha256(row.get("speaker_id", "").encode()).hexdigest()[:12],
                "disease_tag": row.get("disease_tag", ""),
                "duration": row.get("duration", ""),
                "duration_bucket": row.get("duration_bucket", ""),
                "zero_shot_bucket": row.get("zero_shot_bucket", ""),
                "seen_raw_exact": seen["raw"],
                "seen_normalized_exact": seen["normalized"],
                "seen_jyutping_exact": seen["jyutping"],
                "seen_any_layer": any_seen,
                "train_raw_match_count": match_counts["raw"],
                "train_normalized_match_count": match_counts["normalized"],
                "train_jyutping_match_count": match_counts["jyutping"],
            }
        )
        private_rows.append(
            {
                "row_no": i,
                "utt_id": row["utt_id"],
                "speaker_id": row.get("speaker_id", ""),
                "raw_gt": row.get(args.raw_text_column, ""),
                "clean_gt": row.get(args.text_column, ""),
                "normalized_text": keys["normalized"],
                "jyutping_sequence": keys["jyutping"],
                "seen_raw_exact": seen["raw"],
                "seen_normalized_exact": seen["normalized"],
                "seen_jyutping_exact": seen["jyutping"],
                "seen_any_layer": any_seen,
                "matched_train_utt_ids_raw": matched_train_ids["raw"],
                "matched_train_utt_ids_normalized": matched_train_ids["normalized"],
                "matched_train_utt_ids_jyutping": matched_train_ids["jyutping"],
            }
        )
        if row["utt_id"] in three_way_by_utt:
            metric_row = dict(three_way_by_utt[row["utt_id"]])
            metric_row.update(
                {
                    "seen_raw_exact": str(seen["raw"]),
                    "seen_normalized_exact": str(seen["normalized"]),
                    "seen_jyutping_exact": str(seen["jyutping"]),
                    "seen_any_layer": str(any_seen),
                }
            )
            joined_metric_rows.append(metric_row)

    layer_rows = []
    for layer in ["raw", "normalized", "jyutping"]:
        count = layer_seen_counts[layer]
        layer_rows.append(
            {
                "leakage_layer": layer,
                "seen_prompt_test_count": count,
                "unseen_prompt_test_count": len(test_rows) - count,
                "seen_prompt_test_rate": fmt(count / len(test_rows)),
            }
        )
    layer_rows.append(
        {
            "leakage_layer": "any_layer",
            "seen_prompt_test_count": any_seen_count,
            "unseen_prompt_test_count": len(test_rows) - any_seen_count,
            "seen_prompt_test_rate": fmt(any_seen_count / len(test_rows)),
        }
    )

    write_csv(out_dir / "test_prompt_leakage_summary.csv", layer_rows)
    write_csv(out_dir / "test_prompt_leakage_per_sample_sanitized.csv", sanitized_rows)

    if private_out_dir:
        write_csv(private_out_dir / "test_prompt_leakage_private_matches.csv", private_rows)

    if joined_metric_rows:
        metric_rows = []
        for layer, field in [
            ("raw", "seen_raw_exact"),
            ("normalized", "seen_normalized_exact"),
            ("jyutping", "seen_jyutping_exact"),
            ("any_layer", "seen_any_layer"),
        ]:
            seen_rows = [r for r in joined_metric_rows if as_bool(r[field])]
            unseen_rows = [r for r in joined_metric_rows if not as_bool(r[field])]
            metric_rows.append(
                summarize_subset(seen_rows, layer, "seen-prompt test", args.baseline_name, args.new_method_name)
            )
            metric_rows.append(
                summarize_subset(unseen_rows, layer, "unseen-prompt test", args.baseline_name, args.new_method_name)
            )
        write_csv(out_dir / "test_seen_unseen_metrics_by_layer.csv", metric_rows)


if __name__ == "__main__":
    main()
