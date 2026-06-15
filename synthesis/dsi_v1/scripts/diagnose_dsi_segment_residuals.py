#!/usr/bin/env python3
"""Diagnose DSI residual errors by approximate Jyutping segments.

The diagnostic is intentionally approximate: without a forced aligner, each
Jyutping syllable receives a proportional slice of the normal-TTS mel timeline.
The target mel is already DTW-aligned to that same timeline by Step B/E1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


INITIALS = [
    "gw",
    "kw",
    "ng",
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "w",
    "z",
    "c",
    "s",
    "j",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-group-count", type=int, default=5)
    parser.add_argument("--include-private-text", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def safe_float(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}"


def load_pycantonese():
    try:
        import pycantonese  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "pycantonese is required for Jyutping diagnostics. Install it in the active environment."
        ) from exc
    return pycantonese


def expand_jyutping(clean_text: str) -> list[dict[str, str]]:
    pycantonese = load_pycantonese()
    out: list[dict[str, str]] = []
    for surface, jyutping_phrase in pycantonese.characters_to_jyutping(clean_text):
        if not jyutping_phrase:
            continue
        syllables = [item.strip().lower() for item in jyutping_phrase.split() if item.strip()]
        chars = list(surface)
        for idx, syllable in enumerate(syllables):
            char = chars[idx] if idx < len(chars) else surface
            if not re.search(r"[1-6]$", syllable):
                continue
            initial, final, tone = parse_jyutping_syllable(syllable)
            out.append(
                {
                    "char": char,
                    "jyutping": syllable,
                    "initial": initial,
                    "final": final,
                    "tone": tone,
                }
            )
    return out


def parse_jyutping_syllable(syllable: str) -> tuple[str, str, str]:
    match = re.match(r"^([a-z]+)([1-6])$", syllable)
    if not match:
        return "unknown", "unknown", "unknown"
    body, tone = match.groups()
    for initial in INITIALS:
        if body.startswith(initial):
            final = body[len(initial) :] or "zero_final"
            return initial, final, tone
    return "zero_initial", body, tone


def segment_bounds(frame_count: int, token_count: int) -> list[tuple[int, int]]:
    if token_count <= 0:
        return []
    bounds = []
    for idx in range(token_count):
        start = int(round(idx * frame_count / token_count))
        end = int(round((idx + 1) * frame_count / token_count))
        if end <= start:
            end = min(frame_count, start + 1)
        bounds.append((start, end))
    return bounds


def band_mean_abs(x: torch.Tensor, start: int, end: int) -> float:
    return float(torch.mean(torch.abs(x[:, start:end])).item())


def temporal_delta_l1(pred_residual: torch.Tensor, target_residual: torch.Tensor) -> float:
    if pred_residual.shape[0] < 2:
        return 0.0
    pred_delta = pred_residual[1:] - pred_residual[:-1]
    target_delta = target_residual[1:] - target_residual[:-1]
    return float(torch.mean(torch.abs(pred_delta - target_delta)).item())


def tensor_metrics(norm_mel: torch.Tensor, target_mel: torch.Tensor, pred_mel: torch.Tensor) -> dict[str, float]:
    target_residual = target_mel - norm_mel
    pred_residual = pred_mel - norm_mel
    pred_error = pred_mel - target_mel
    norm_error_l1 = float(torch.mean(torch.abs(norm_mel - target_mel)).item())
    pred_error_l1 = float(torch.mean(torch.abs(pred_error)).item())
    target_abs = float(torch.mean(torch.abs(target_residual)).item())
    pred_abs = float(torch.mean(torch.abs(pred_residual)).item())
    return {
        "norm_error_l1": norm_error_l1,
        "pred_error_l1": pred_error_l1,
        "target_residual_l1": target_abs,
        "pred_residual_l1": pred_abs,
        "relative_l1_gain": (norm_error_l1 - pred_error_l1) / norm_error_l1 if norm_error_l1 > 1e-8 else 0.0,
        "gap_ratio": pred_error_l1 / target_abs if target_abs > 1e-8 else 0.0,
        "residual_scale_ratio": pred_abs / target_abs if target_abs > 1e-8 else 0.0,
        "low_band_gap_l1": band_mean_abs(pred_error, 0, 27),
        "mid_band_gap_l1": band_mean_abs(pred_error, 27, 54),
        "high_band_gap_l1": band_mean_abs(pred_error, 54, 80),
        "temporal_delta_gap_l1": temporal_delta_l1(pred_residual, target_residual),
    }


def sanitize_prediction_filename(path: Path) -> str:
    return path.stem.replace("__", "/")


def load_prediction_rows(prediction_dir: Path, limit: int) -> list[Path]:
    files = sorted(prediction_dir.glob("*.pt"))
    if limit > 0:
        files = files[:limit]
    if not files:
        raise ValueError(f"No prediction .pt files found under {prediction_dir}")
    return files


def aggregate(rows: list[dict[str, Any]], group_key: str, min_count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, ""))].append(row)

    out: list[dict[str, Any]] = []
    metric_fields = [
        "norm_error_l1",
        "pred_error_l1",
        "target_residual_l1",
        "pred_residual_l1",
        "relative_l1_gain",
        "gap_ratio",
        "residual_scale_ratio",
        "low_band_gap_l1",
        "mid_band_gap_l1",
        "high_band_gap_l1",
        "temporal_delta_gap_l1",
    ]
    for key, items in grouped.items():
        if len(items) < min_count:
            continue
        samples = {str(item["utt_id"]) for item in items}
        patients = {str(item["patient_id"]) for item in items}
        record: dict[str, Any] = {
            group_key: key,
            "segment_count": len(items),
            "sample_count": len(samples),
            "patient_count": len(patients),
            "frame_count": sum(int(item["frame_count"]) for item in items),
        }
        for metric in metric_fields:
            vals = [float(item[metric]) for item in items if math.isfinite(float(item[metric]))]
            record[f"mean_{metric}"] = float(np.mean(vals)) if vals else float("nan")
            record[f"median_{metric}"] = float(np.median(vals)) if vals else float("nan")
        out.append(record)

    out.sort(key=lambda row: (float(row.get("mean_gap_ratio", 0.0)), float(row.get("mean_pred_error_l1", 0.0))), reverse=True)
    return out


def build_overall(rows: list[dict[str, Any]], sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_fields = [
        "norm_error_l1",
        "pred_error_l1",
        "target_residual_l1",
        "pred_residual_l1",
        "relative_l1_gain",
        "gap_ratio",
        "residual_scale_ratio",
        "low_band_gap_l1",
        "mid_band_gap_l1",
        "high_band_gap_l1",
        "temporal_delta_gap_l1",
    ]
    out = [
        {"metric": "sample_count", "value": len(sample_rows)},
        {"metric": "segment_count", "value": len(rows)},
        {"metric": "patient_count", "value": len({row["patient_id"] for row in sample_rows})},
        {"metric": "jyutping_syllable_count", "value": len({row["jyutping"] for row in rows})},
        {"metric": "initial_count", "value": len({row["initial"] for row in rows})},
        {"metric": "final_count", "value": len({row["final"] for row in rows})},
    ]
    for metric in metric_fields:
        vals = [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))]
        out.append({"metric": f"mean_{metric}", "value": safe_float(float(np.mean(vals)) if vals else float("nan"))})
        out.append({"metric": f"median_{metric}", "value": safe_float(float(np.median(vals)) if vals else float("nan"))})
    return out


def numeric_format_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, float):
                item[key] = safe_float(value)
            else:
                item[key] = value
        formatted.append(item)
    return formatted


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = {
        row["utt_id"]: row
        for row in read_csv(Path(args.feature_manifest))
        if row.get("status") == "generated" and (not args.split or row.get("split") in set(args.split))
    }
    prediction_files = load_prediction_rows(Path(args.prediction_dir), args.limit)
    segment_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    unknown_text_rows = 0

    for pred_path in prediction_files:
        payload = torch.load(pred_path, map_location="cpu", weights_only=False)
        utt_id = str(payload["utt_id"])
        feature_row = feature_rows.get(utt_id)
        if feature_row is None:
            continue
        clean_text = feature_row.get("clean_text", "")
        tokens = expand_jyutping(clean_text)
        if not tokens:
            unknown_text_rows += 1
            continue

        norm_mel = payload["norm_mel"].float()
        target_mel = payload["target_mel"].float()
        pred_mel = payload["pred_mel"].float()
        frame_count = int(norm_mel.shape[0])
        bounds = segment_bounds(frame_count, len(tokens))
        sample_metric = tensor_metrics(norm_mel, target_mel, pred_mel)
        sample_rows.append(
            {
                "utt_id": utt_id,
                "patient_id": feature_row.get("patient_id", ""),
                "split": feature_row.get("split", ""),
                "zero_shot_bucket": feature_row.get("zero_shot_bucket", ""),
                "frame_count": frame_count,
                "syllable_count": len(tokens),
                "clean_text": clean_text if args.include_private_text else "",
                "jyutping_sequence": " ".join(token["jyutping"] for token in tokens) if args.include_private_text else "",
                **sample_metric,
            }
        )

        for idx, (token, (start, end)) in enumerate(zip(tokens, bounds), start=1):
            seg_norm = norm_mel[start:end]
            seg_target = target_mel[start:end]
            seg_pred = pred_mel[start:end]
            metrics = tensor_metrics(seg_norm, seg_target, seg_pred)
            segment_rows.append(
                {
                    "utt_id": utt_id,
                    "patient_id": feature_row.get("patient_id", ""),
                    "split": feature_row.get("split", ""),
                    "zero_shot_bucket": feature_row.get("zero_shot_bucket", ""),
                    "token_index": idx,
                    "char": token["char"] if args.include_private_text else "",
                    "jyutping": token["jyutping"],
                    "initial": token["initial"],
                    "final": token["final"],
                    "tone": token["tone"],
                    "frame_start": start,
                    "frame_end": end,
                    "frame_count": end - start,
                    **metrics,
                }
            )

    if not segment_rows:
        raise ValueError("No segment rows were produced")

    segment_fields = [
        "utt_id",
        "patient_id",
        "split",
        "zero_shot_bucket",
        "token_index",
        "char",
        "jyutping",
        "initial",
        "final",
        "tone",
        "frame_start",
        "frame_end",
        "frame_count",
        "norm_error_l1",
        "pred_error_l1",
        "target_residual_l1",
        "pred_residual_l1",
        "relative_l1_gain",
        "gap_ratio",
        "residual_scale_ratio",
        "low_band_gap_l1",
        "mid_band_gap_l1",
        "high_band_gap_l1",
        "temporal_delta_gap_l1",
    ]
    sample_fields = [
        "utt_id",
        "patient_id",
        "split",
        "zero_shot_bucket",
        "frame_count",
        "syllable_count",
        "clean_text",
        "jyutping_sequence",
        "norm_error_l1",
        "pred_error_l1",
        "target_residual_l1",
        "pred_residual_l1",
        "relative_l1_gain",
        "gap_ratio",
        "residual_scale_ratio",
        "low_band_gap_l1",
        "mid_band_gap_l1",
        "high_band_gap_l1",
        "temporal_delta_gap_l1",
    ]
    write_csv(out_dir / "segment_metrics_private.csv", numeric_format_rows(segment_rows), segment_fields)
    write_csv(out_dir / "sample_metrics_private.csv", numeric_format_rows(sample_rows), sample_fields)
    write_csv(out_dir / "overall_summary.csv", build_overall(segment_rows, sample_rows), ["metric", "value"])

    aggregate_fields = [
        "segment_count",
        "sample_count",
        "patient_count",
        "frame_count",
        "mean_norm_error_l1",
        "median_norm_error_l1",
        "mean_pred_error_l1",
        "median_pred_error_l1",
        "mean_target_residual_l1",
        "median_target_residual_l1",
        "mean_pred_residual_l1",
        "median_pred_residual_l1",
        "mean_relative_l1_gain",
        "median_relative_l1_gain",
        "mean_gap_ratio",
        "median_gap_ratio",
        "mean_residual_scale_ratio",
        "median_residual_scale_ratio",
        "mean_low_band_gap_l1",
        "median_low_band_gap_l1",
        "mean_mid_band_gap_l1",
        "median_mid_band_gap_l1",
        "mean_high_band_gap_l1",
        "median_high_band_gap_l1",
        "mean_temporal_delta_gap_l1",
        "median_temporal_delta_gap_l1",
    ]
    for group_key, filename, is_private in [
        ("initial", "by_initial.csv", False),
        ("final", "by_final.csv", False),
        ("tone", "by_tone.csv", False),
        ("zero_shot_bucket", "by_zero_shot_bucket.csv", False),
        ("jyutping", "by_jyutping_private.csv", True),
    ]:
        grouped = numeric_format_rows(aggregate(segment_rows, group_key, args.min_group_count))
        write_csv(out_dir / filename, grouped, [group_key, *aggregate_fields])

    worst = sorted(
        segment_rows,
        key=lambda row: (float(row["gap_ratio"]), float(row["pred_error_l1"]), float(row["target_residual_l1"])),
        reverse=True,
    )[:200]
    write_csv(out_dir / "worst_segments_private.csv", numeric_format_rows(worst), segment_fields)

    config = {
        "feature_manifest": args.feature_manifest,
        "prediction_dir": args.prediction_dir,
        "out_dir": args.out_dir,
        "splits": args.split,
        "prediction_files_seen": len(prediction_files),
        "sample_rows": len(sample_rows),
        "segment_rows": len(segment_rows),
        "unknown_text_rows": unknown_text_rows,
        "min_group_count": args.min_group_count,
        "alignment_note": "Syllables are assigned proportional slices of the normal-TTS mel timeline; target mel is already DTW-aligned to that timeline.",
        "privacy_note": "Files with suffix _private may include utterance IDs, Jyutping tokens, or text depending on flags and should not be committed publicly.",
    }
    (out_dir / "diagnostic_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
