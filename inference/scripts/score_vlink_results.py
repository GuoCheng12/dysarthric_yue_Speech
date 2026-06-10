#!/usr/bin/env python3
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPO_ROOT / "src", Path("/data/qwen3-asr/src")):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break
from eval_asr_dir import (  # noqa: E402
    CHINESE_DIGITS,
    CHINESE_NUMBER_CHARS,
    YUE_REPLACEMENTS,
    edit_distance,
    normalize_text,
    parse_chinese_number,
    text_normalize,
    to_traditional,
)

OUT = Path("/data/qwen3-asr/inference/outputs")
PREFIX = "vlink_data_raw_qwen3_asr_cantonese"
JSONL = OUT / f"{PREFIX}.jsonl"
CSV = OUT / f"{PREFIX}.csv"
SUMMARY_BY = OUT / f"{PREFIX}_summary_by_speaker.csv"
SUMMARY_OVERALL = OUT / f"{PREFIX}_summary_overall.csv"
SUMMARY_MD = OUT / f"{PREFIX}_summary.md"
REQ_BY = OUT / "vlink_data_raw_qwen3_asr_requested_stats_by_speaker.csv"
REQ_OVERALL = OUT / "vlink_data_raw_qwen3_asr_requested_stats_overall.csv"
REQ_MD = OUT / "vlink_data_raw_qwen3_asr_requested_stats.md"
CRITICAL_THRESHOLD = 0.5

RESULT_FIELDS = [
    "speaker",
    "session",
    "audio_id",
    "rel_path",
    "wav_path",
    "gt",
    "gt_source",
    "pred",
    "language_arg",
    "detected_language",
    "textnorm_cer",
    "critical_error",
    "critical_error_reason",
    "inference_error",
    "duration_sec",
    "sample_rate",
    "channels",
    "inference_time_sec",
    "textnorm_gt",
    "textnorm_pred",
    "textnorm_char_distance",
    "critical_error_rule",
]

SUMMARY_FIELDS = [
    "speaker",
    "samples_total",
    "inferred_total",
    "gt_total",
    "avg_textnorm_cer",
    "critical_error_rate",
    "critical_error_count",
    "critical_error_labeled",
]

REQUESTED_FIELDS = [
    "speaker",
    "samples_total",
    "inferred_total",
    "inference_error_count",
    "avg_textnorm_cer",
    "avg_critical_error",
    "critical_error_count",
    "critical_error_labeled",
]


def md_escape(value):
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def parse_bool(value):
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def score_row(row):
    gt = row.get("gt") or ""
    pred = row.get("pred") or ""
    textnorm_gt = safe_text_normalize(gt)
    textnorm_pred = safe_text_normalize(pred)
    dist = edit_distance(textnorm_gt, textnorm_pred)
    cer = round(dist / max(1, len(textnorm_gt)), 6)

    inference_error = row.get("inference_error") or ""
    if inference_error:
        critical = True
        reason = "inference_error"
    elif not pred:
        critical = True
        reason = "blank_prediction"
    elif cer >= CRITICAL_THRESHOLD:
        critical = True
        reason = f"textnorm_cer>={CRITICAL_THRESHOLD:g}"
    else:
        critical = False
        reason = ""

    row["textnorm_cer"] = cer
    row["critical_error"] = critical
    row["critical_error_reason"] = reason
    row["textnorm_gt"] = textnorm_gt
    row["textnorm_pred"] = textnorm_pred
    row["textnorm_char_distance"] = dist
    row["critical_error_rule"] = f"inference_error_or_blank_or_textnorm_cer>={CRITICAL_THRESHOLD:g}"
    return row


def safe_text_normalize(text):
    try:
        return text_normalize(text)
    except Exception:
        text = normalize_text(text)
        text = to_traditional(text)
        text = safe_normalize_numbers(text)
        for source, target in YUE_REPLACEMENTS:
            text = text.replace(source, target)
        return text.lower()


def safe_normalize_numbers(text):
    pattern = f"[{re.escape(CHINESE_NUMBER_CHARS)}]{{2,}}"

    def replace_token(match):
        token = match.group(0)
        if all(ch in CHINESE_DIGITS for ch in token):
            return "".join(CHINESE_DIGITS[ch] for ch in token)
        try:
            parsed = parse_chinese_number(token)
        except Exception:
            return token
        return str(parsed) if parsed is not None else token

    return re.sub(pattern, replace_token, text)


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def make_summary(name, rows):
    cers = [float(r["textnorm_cer"]) for r in rows if r.get("textnorm_cer") not in ("", None)]
    critical = [parse_bool(r.get("critical_error")) for r in rows]
    critical = [v for v in critical if v is not None]
    return {
        "speaker": name,
        "samples_total": len(rows),
        "inferred_total": sum(1 for r in rows if r.get("pred")),
        "gt_total": sum(1 for r in rows if r.get("gt")),
        "avg_textnorm_cer": round(statistics.mean(cers), 6) if cers else "",
        "critical_error_rate": round(sum(critical) / len(critical), 6) if critical else "",
        "critical_error_count": int(sum(critical)) if critical else "",
        "critical_error_labeled": len(critical),
        "inference_error_count": sum(1 for r in rows if r.get("inference_error")),
    }


def make_requested(row):
    return {
        "speaker": row["speaker"],
        "samples_total": row["samples_total"],
        "inferred_total": row["inferred_total"],
        "inference_error_count": row["inference_error_count"],
        "avg_textnorm_cer": row["avg_textnorm_cer"],
        "avg_critical_error": row["critical_error_rate"],
        "critical_error_count": row["critical_error_count"],
        "critical_error_labeled": row["critical_error_labeled"],
    }


def main():
    rows = []
    with JSONL.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(score_row(json.loads(line)))

    with JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(CSV, rows, RESULT_FIELDS)

    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    summary_rows = [make_summary(sp, by_speaker[sp]) for sp in sorted(by_speaker)]
    overall = make_summary("__OVERALL__", rows)
    write_csv(SUMMARY_BY, summary_rows, SUMMARY_FIELDS)
    write_csv(SUMMARY_OVERALL, [overall], SUMMARY_FIELDS)

    requested_rows = [make_requested(row) for row in summary_rows]
    requested_overall = make_requested(overall)
    write_csv(REQ_BY, requested_rows, REQUESTED_FIELDS)
    write_csv(REQ_OVERALL, [requested_overall], REQUESTED_FIELDS)

    with SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Qwen3-ASR vlink_data_raw Summary\n\n")
        f.write("- TextNorm_CER: character error rate after text normalization.\n")
        f.write(
            f"- Critical_Error: 1 if inference_error/blank prediction or TextNorm_CER >= {CRITICAL_THRESHOLD:g}; otherwise 0.\n"
        )
        f.write("- Semantic Similarity is not computed in this pass.\n\n")
        f.write("## Overall\n\n")
        f.write(
            "| samples_total | inferred_total | gt_total | avg_textnorm_cer | critical_error_rate | critical_error_count | critical_error_labeled |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        f.write(
            "| {samples_total} | {inferred_total} | {gt_total} | {avg_textnorm_cer} | {critical_error_rate} | {critical_error_count} | {critical_error_labeled} |\n\n".format(
                **overall
            )
        )
        f.write("## By Speaker\n\n")
        f.write(
            "| speaker | samples_total | inferred_total | gt_total | avg_textnorm_cer | critical_error_rate | critical_error_count | critical_error_labeled |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            safe = {**row, "speaker": md_escape(row["speaker"])}
            f.write(
                "| {speaker} | {samples_total} | {inferred_total} | {gt_total} | {avg_textnorm_cer} | {critical_error_rate} | {critical_error_count} | {critical_error_labeled} |\n".format(
                    **safe
                )
            )

    with REQ_MD.open("w", encoding="utf-8") as f:
        f.write("# Requested Qwen3-ASR Stats\n\n")
        f.write(
            f"Critical_Error rule: inference_error/blank prediction or TextNorm_CER >= {CRITICAL_THRESHOLD:g}. Semantic Similarity is not computed.\n\n"
        )
        f.write("## Overall\n\n")
        f.write(
            "| speaker | samples_total | inferred_total | inference_error_count | avg_textnorm_cer | avg_critical_error | critical_error_count | critical_error_labeled |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        f.write(
            "| {speaker} | {samples_total} | {inferred_total} | {inference_error_count} | {avg_textnorm_cer} | {avg_critical_error} | {critical_error_count} | {critical_error_labeled} |\n".format(
                **requested_overall
            )
        )
        f.write("\n## By Speaker\n\n")
        f.write(
            "| speaker | samples_total | inferred_total | inference_error_count | avg_textnorm_cer | avg_critical_error | critical_error_count | critical_error_labeled |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in requested_rows:
            safe = {**row, "speaker": md_escape(row["speaker"])}
            f.write(
                "| {speaker} | {samples_total} | {inferred_total} | {inference_error_count} | {avg_textnorm_cer} | {avg_critical_error} | {critical_error_count} | {critical_error_labeled} |\n".format(
                    **safe
                )
            )

    print(
        json.dumps(
            {
                "rows": len(rows),
                "overall_avg_textnorm_cer": overall["avg_textnorm_cer"],
                "overall_avg_critical_error": requested_overall["avg_critical_error"],
                "critical_error_count": overall["critical_error_count"],
                "critical_threshold": CRITICAL_THRESHOLD,
                "csv": str(CSV),
                "jsonl": str(JSONL),
                "requested_stats": str(REQ_MD),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
