#!/usr/bin/env python3
"""Build canonical zero-shot vs E2 baseline vs new-method comparison tables."""

import argparse
import csv
import math
from pathlib import Path


BASELINE_NAME = "E2_fullSFT_clean_pooled_epoch2"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value):
    if value is None or value == "":
        return math.nan
    return float(value)


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fmt_float(value):
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6f}"


def mean(values):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return math.nan
    return sum(values) / len(values)


def critical_rate(rows, key):
    if not rows:
        return math.nan
    return sum(as_bool(r.get(key)) for r in rows) / len(rows)


def critical_count(rows, key):
    return sum(as_bool(r.get(key)) for r in rows)


def group_rows(rows, key):
    buckets = {}
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        buckets.setdefault(value, []).append(row)
    return [(f"{key}={name}", group) for name, group in sorted(buckets.items())]


def summarize_group(name, rows):
    zero_cer = mean([as_float(r.get("zero_shot_cer")) for r in rows])
    e2_cer = mean([as_float(r.get(f"{BASELINE_NAME}_cer")) for r in rows])
    new_cer = mean([as_float(r.get("new_method_cer")) for r in rows])

    zero_crit_rate = critical_rate(rows, "zero_shot_critical")
    e2_crit_rate = critical_rate(rows, f"{BASELINE_NAME}_critical")
    new_crit_rate = critical_rate(rows, "new_method_critical")

    return {
        "group": name,
        "sample_count": str(len(rows)),
        "zero_shot_cer": fmt_float(zero_cer),
        f"{BASELINE_NAME}_cer": fmt_float(e2_cer),
        "new_method_cer": fmt_float(new_cer),
        "delta_new_vs_zero_cer": fmt_float(new_cer - zero_cer),
        f"delta_new_vs_{BASELINE_NAME}_cer": fmt_float(new_cer - e2_cer),
        "zero_shot_critical_rate": fmt_float(zero_crit_rate),
        f"{BASELINE_NAME}_critical_rate": fmt_float(e2_crit_rate),
        "new_method_critical_rate": fmt_float(new_crit_rate),
        "delta_new_vs_zero_critical_rate": fmt_float(new_crit_rate - zero_crit_rate),
        f"delta_new_vs_{BASELINE_NAME}_critical_rate": fmt_float(new_crit_rate - e2_crit_rate),
        "zero_shot_critical_count": str(critical_count(rows, "zero_shot_critical")),
        f"{BASELINE_NAME}_critical_count": str(critical_count(rows, f"{BASELINE_NAME}_critical")),
        "new_method_critical_count": str(critical_count(rows, "new_method_critical")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--new-predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--new-method-name", default="new_method")
    parser.add_argument("--new-pred-col", default="fine_tuned_pred")
    parser.add_argument("--new-cer-col", default="fine_tuned_cer")
    parser.add_argument("--new-critical-col", default="fine_tuned_critical_error")
    args = parser.parse_args()

    baseline_rows = read_csv(args.baseline_predictions)
    new_rows = read_csv(args.new_predictions)
    new_by_utt = {r["utt_id"]: r for r in new_rows}

    rows = []
    missing = []
    for base in baseline_rows:
        utt_id = base["utt_id"]
        new = new_by_utt.get(utt_id)
        if new is None:
            missing.append(utt_id)
            continue

        zero_cer = as_float(base.get("zero_shot_cer"))
        e2_cer = as_float(base.get("fine_tuned_cer"))
        new_cer = as_float(new.get(args.new_cer_col))

        rows.append({
            "utt_id": utt_id,
            "speaker_id": base.get("speaker_id", ""),
            "disease_tag": base.get("disease_tag", ""),
            "duration": base.get("duration", ""),
            "duration_bucket": base.get("duration_bucket", ""),
            "zero_shot_bucket": base.get("zero_shot_bucket", ""),
            "task_type": base.get("task_type", ""),
            "clean_gt": base.get("clean_gt", ""),
            "zero_shot_predict": base.get("raw_pred", ""),
            f"{BASELINE_NAME}_predict": base.get("fine_tuned_pred", ""),
            "new_method_name": args.new_method_name,
            "new_method_predict": new.get(args.new_pred_col, ""),
            "zero_shot_cer": fmt_float(zero_cer),
            f"{BASELINE_NAME}_cer": fmt_float(e2_cer),
            "new_method_cer": fmt_float(new_cer),
            "delta_new_vs_zero_cer": fmt_float(new_cer - zero_cer),
            f"delta_new_vs_{BASELINE_NAME}_cer": fmt_float(new_cer - e2_cer),
            "zero_shot_critical": str(as_bool(base.get("zero_shot_critical"))),
            f"{BASELINE_NAME}_critical": str(as_bool(base.get("fine_tuned_critical_error"))),
            "new_method_critical": str(as_bool(new.get(args.new_critical_col))),
            "audio_path": base.get("audio_path", ""),
        })

    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(f"Missing {len(missing)} baseline utt_id rows in new predictions: {preview}")

    out_dir = Path(args.out_dir)
    per_utt_fields = [
        "utt_id",
        "speaker_id",
        "disease_tag",
        "duration",
        "duration_bucket",
        "zero_shot_bucket",
        "task_type",
        "clean_gt",
        "zero_shot_predict",
        f"{BASELINE_NAME}_predict",
        "new_method_name",
        "new_method_predict",
        "zero_shot_cer",
        f"{BASELINE_NAME}_cer",
        "new_method_cer",
        "delta_new_vs_zero_cer",
        f"delta_new_vs_{BASELINE_NAME}_cer",
        "zero_shot_critical",
        f"{BASELINE_NAME}_critical",
        "new_method_critical",
        "audio_path",
    ]
    write_csv(out_dir / "three_way_per_utterance.csv", rows, per_utt_fields)

    group_sets = [("overall", rows)]
    for key in ("zero_shot_bucket", "disease_tag", "duration_bucket", "speaker_id"):
        group_sets.extend(group_rows(rows, key))

    summary_rows = [summarize_group(name, group) for name, group in group_sets]
    summary_fields = list(summary_rows[0].keys())
    write_csv(out_dir / "three_way_summary_by_group.csv", summary_rows, summary_fields)


if __name__ == "__main__":
    main()
