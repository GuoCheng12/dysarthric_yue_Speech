#!/usr/bin/env python3
"""Summarize focus metrics from a three-way ASR comparison table."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


BASELINE_NAME = "E2_fullSFT_clean_pooled_epoch2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--three-way", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-name", default=BASELINE_NAME)
    parser.add_argument("--top-k-speakers", type=int, default=20)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value):
    if value in ("", None):
        return None
    return float(value)


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fmt(value):
    if value is None:
        return ""
    return f"{value:.6f}"


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return statistics.mean(values)


def method_cer(row, method, baseline_name):
    if method == "zero_shot":
        return as_float(row["zero_shot_cer"])
    if method == "baseline":
        return as_float(row[f"{baseline_name}_cer"])
    if method == "new":
        return as_float(row["new_method_cer"])
    raise ValueError(method)


def method_critical(row, method, baseline_name):
    if method == "zero_shot":
        return as_bool(row["zero_shot_critical"])
    if method == "baseline":
        return as_bool(row[f"{baseline_name}_critical"])
    if method == "new":
        return as_bool(row["new_method_critical"])
    raise ValueError(method)


def cer_summary(name, rows, baseline_name):
    zero = mean([method_cer(row, "zero_shot", baseline_name) for row in rows])
    base = mean([method_cer(row, "baseline", baseline_name) for row in rows])
    new = mean([method_cer(row, "new", baseline_name) for row in rows])
    return {
        "metric": name,
        "sample_count": len(rows),
        "zero_shot_cer": fmt(zero),
        f"{baseline_name}_cer": fmt(base),
        "new_method_cer": fmt(new),
        f"delta_new_vs_{baseline_name}_cer": fmt(new - base if new is not None and base is not None else None),
        "delta_new_vs_zero_cer": fmt(new - zero if new is not None and zero is not None else None),
    }


def count_regressions(rows, method, baseline_name):
    return sum(
        (not method_critical(row, "zero_shot", baseline_name))
        and method_critical(row, method, baseline_name)
        for row in rows
    )


def main():
    args = parse_args()
    rows = read_csv(args.three_way)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_name = args.baseline_name

    hard_rows = [r for r in rows if r.get("zero_shot_bucket") == "hard"]
    medium_rows = [r for r in rows if r.get("zero_shot_bucket") == "medium"]

    focus_rows = [
        cer_summary("hard_cer", hard_rows, baseline_name),
        cer_summary("medium_cer", medium_rows, baseline_name),
    ]

    medium_regression = {
        "metric": "medium_regression_count",
        "sample_count": len(medium_rows),
        "zero_shot_regression_count": "0",
        f"{baseline_name}_regression_count": str(count_regressions(medium_rows, "baseline", baseline_name)),
        "new_method_regression_count": str(count_regressions(medium_rows, "new", baseline_name)),
    }

    focus_fields = list(focus_rows[0].keys())
    write_csv(out_dir / "focus_cer_metrics.csv", focus_rows, focus_fields)
    write_csv(out_dir / "medium_regression_metrics.csv", [medium_regression], list(medium_regression.keys()))

    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"]].append(row)

    speaker_rows = []
    for speaker_id, group in by_speaker.items():
        base = mean([method_cer(row, "baseline", baseline_name) for row in group])
        new = mean([method_cer(row, "new", baseline_name) for row in group])
        zero = mean([method_cer(row, "zero_shot", baseline_name) for row in group])
        base_crit = sum(method_critical(row, "baseline", baseline_name) for row in group)
        new_crit = sum(method_critical(row, "new", baseline_name) for row in group)
        hard_count = sum(row.get("zero_shot_bucket") == "hard" for row in group)
        speaker_rows.append({
            "speaker_id": speaker_id,
            "sample_count": len(group),
            "hard_count": hard_count,
            "zero_shot_cer": fmt(zero),
            f"{baseline_name}_cer": fmt(base),
            "new_method_cer": fmt(new),
            f"delta_new_vs_{baseline_name}_cer": fmt(new - base if new is not None and base is not None else None),
            f"{baseline_name}_critical_count": base_crit,
            "new_method_critical_count": new_crit,
            f"delta_new_vs_{baseline_name}_critical_count": new_crit - base_crit,
        })

    speaker_rows.sort(
        key=lambda r: (
            float(r[f"delta_new_vs_{baseline_name}_cer"]) if r[f"delta_new_vs_{baseline_name}_cer"] else -999,
            int(r[f"delta_new_vs_{baseline_name}_critical_count"]),
        ),
        reverse=True,
    )
    speaker_fields = list(speaker_rows[0].keys()) if speaker_rows else []
    write_csv(out_dir / "per_speaker_worst_cases.csv", speaker_rows[: args.top_k_speakers], speaker_fields)

    lines = ["# Three-Way Focus Metrics", ""]
    lines.append("## CER Focus")
    lines.append("")
    lines.append("| metric | n | zero-shot CER | E2 full-SFT CER | new method CER | new - E2 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in focus_rows:
        lines.append(
            f"| {row['metric']} | {row['sample_count']} | {row['zero_shot_cer']} | "
            f"{row[f'{baseline_name}_cer']} | {row['new_method_cer']} | "
            f"{row[f'delta_new_vs_{baseline_name}_cer']} |"
        )
    lines.append("")
    lines.append("## Medium Regression")
    lines.append("")
    lines.append(
        f"- E2 full-SFT medium regressions: {medium_regression[f'{baseline_name}_regression_count']}"
    )
    lines.append(f"- New method medium regressions: {medium_regression['new_method_regression_count']}")
    lines.append("")
    lines.append("## Per-Speaker Worst Cases")
    lines.append("")
    lines.append("See `per_speaker_worst_cases.csv`.")
    (out_dir / "focus_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
