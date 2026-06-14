#!/usr/bin/env python3
"""QA checks for generated DSI V1 normal-TTS pair outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--generated-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-sample-rate", type=int, default=16000)
    parser.add_argument("--target-rms-dbfs", type=float, default=-23.0)
    parser.add_argument("--rms-warning-tolerance", type=float, default=3.0)
    parser.add_argument("--min-duration-sec", type=float, default=0.2)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--expected-model-family", default="cosyvoice3")
    parser.add_argument("--expected-mode", default="instruct2")
    parser.add_argument("--expected-speed", type=float, default=0.9)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rms_dbfs(speech: torch.Tensor) -> float:
    rms = torch.sqrt(torch.mean(speech.float() ** 2)).clamp_min(1e-12)
    return float(20.0 * torch.log10(rms))


def add_issue(
    issues: list[dict[str, str]],
    severity: str,
    code: str,
    utt_id: str,
    path: str,
    message: str,
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "utt_id": utt_id,
            "path": path,
            "message": message,
        }
    )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def summarize_values(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "p01": percentile(values, 0.01),
        "p05": percentile(values, 0.05),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def check_expected_metadata(
    row: dict[str, str],
    args: argparse.Namespace,
    issues: list[dict[str, str]],
) -> None:
    utt_id = row.get("utt_id", "")
    path = row.get("norm_wav_path", "")
    if row.get("generation_status") != "generated":
        add_issue(issues, "error", "generation_status", utt_id, path, f"generation_status={row.get('generation_status')!r}")
    if row.get("tts_model_family") != args.expected_model_family:
        add_issue(issues, "error", "tts_model_family", utt_id, path, f"tts_model_family={row.get('tts_model_family')!r}")
    if row.get("tts_mode") != args.expected_mode:
        add_issue(issues, "error", "tts_mode", utt_id, path, f"tts_mode={row.get('tts_mode')!r}")

    speed = parse_float(row.get("tts_speed"))
    if speed is None or abs(speed - args.expected_speed) > 1e-4:
        add_issue(issues, "error", "tts_speed", utt_id, path, f"tts_speed={row.get('tts_speed')!r}")

    expected_sr = str(args.expected_sample_rate)
    if row.get("postprocess_target_sample_rate") != expected_sr:
        add_issue(
            issues,
            "error",
            "target_sample_rate",
            utt_id,
            path,
            f"postprocess_target_sample_rate={row.get('postprocess_target_sample_rate')!r}",
        )

    target_rms = parse_float(row.get("postprocess_target_rms_dbfs"))
    if target_rms is None or abs(target_rms - args.target_rms_dbfs) > 1e-4:
        add_issue(
            issues,
            "error",
            "target_rms_dbfs",
            utt_id,
            path,
            f"postprocess_target_rms_dbfs={row.get('postprocess_target_rms_dbfs')!r}",
        )


def main() -> None:
    args = parse_args()
    pair_manifest = Path(args.pair_manifest)
    generated_csv = Path(args.generated_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(pair_manifest)
    generated_rows = read_csv(generated_csv)
    generated_by_utt = {row["utt_id"]: row for row in generated_rows}

    issues: list[dict[str, str]] = []
    per_wav: list[dict[str, Any]] = []

    manifest_ids = [row["utt_id"] for row in manifest_rows]
    generated_ids = [row["utt_id"] for row in generated_rows]
    for utt_id, count in Counter(manifest_ids).items():
        if count > 1:
            add_issue(issues, "error", "duplicate_manifest_utt_id", utt_id, "", f"count={count}")
    for utt_id, count in Counter(generated_ids).items():
        if count > 1:
            add_issue(issues, "error", "duplicate_generated_utt_id", utt_id, "", f"count={count}")

    manifest_set = set(manifest_ids)
    generated_set = set(generated_ids)
    for utt_id in sorted(manifest_set - generated_set):
        add_issue(issues, "error", "missing_generated_row", utt_id, "", "utt_id exists in manifest but not generated csv")
    for utt_id in sorted(generated_set - manifest_set):
        add_issue(issues, "error", "extra_generated_row", utt_id, "", "utt_id exists in generated csv but not manifest")

    norm_paths: Counter[str] = Counter()
    durations: list[float] = []
    rms_values: list[float] = []
    peaks: list[float] = []
    generation_seconds: list[float] = []

    for manifest_row in manifest_rows:
        utt_id = manifest_row["utt_id"]
        row = generated_by_utt.get(utt_id)
        if row is None:
            continue

        check_expected_metadata(row, args, issues)

        norm_path = row.get("norm_wav_path", "")
        dys_path = row.get("dys_wav_path", "")
        norm_paths[norm_path] += 1
        if dys_path and not Path(dys_path).is_file():
            add_issue(issues, "error", "missing_dys_wav", utt_id, dys_path, "dys_wav_path does not exist")

        actual_sample_rate = ""
        actual_num_frames = ""
        actual_duration = ""
        actual_rms = ""
        actual_peak = ""
        file_size = ""

        if not norm_path:
            add_issue(issues, "error", "empty_norm_wav_path", utt_id, "", "norm_wav_path is empty")
        elif not Path(norm_path).is_file():
            add_issue(issues, "error", "missing_norm_wav", utt_id, norm_path, "norm_wav_path does not exist")
        else:
            try:
                file_size = str(Path(norm_path).stat().st_size)
                info = torchaudio.info(norm_path)
                actual_sample_rate = str(info.sample_rate)
                actual_num_frames = str(info.num_frames)
                duration = info.num_frames / info.sample_rate if info.sample_rate else 0.0
                actual_duration = f"{duration:.6f}"
                speech, loaded_sample_rate = torchaudio.load(norm_path)
                if loaded_sample_rate != info.sample_rate:
                    add_issue(
                        issues,
                        "error",
                        "load_sample_rate_mismatch",
                        utt_id,
                        norm_path,
                        f"info={info.sample_rate}, load={loaded_sample_rate}",
                    )
                actual_rms_value = rms_dbfs(speech)
                actual_peak_value = float(torch.max(torch.abs(speech))) if speech.numel() else 0.0
                actual_rms = f"{actual_rms_value:.6f}"
                actual_peak = f"{actual_peak_value:.6f}"
                durations.append(duration)
                rms_values.append(actual_rms_value)
                peaks.append(actual_peak_value)

                if info.sample_rate != args.expected_sample_rate:
                    add_issue(issues, "error", "sample_rate", utt_id, norm_path, f"actual_sample_rate={info.sample_rate}")
                if info.num_frames <= 0:
                    add_issue(issues, "error", "empty_audio", utt_id, norm_path, "num_frames <= 0")
                if duration < args.min_duration_sec:
                    add_issue(issues, "warning", "too_short", utt_id, norm_path, f"duration={duration:.6f}")
                if duration > args.max_duration_sec:
                    add_issue(issues, "warning", "too_long", utt_id, norm_path, f"duration={duration:.6f}")
                if abs(actual_rms_value - args.target_rms_dbfs) > args.rms_warning_tolerance:
                    add_issue(issues, "warning", "rms_out_of_range", utt_id, norm_path, f"rms_dbfs={actual_rms_value:.6f}")
                if actual_peak_value >= 0.999:
                    add_issue(issues, "warning", "peak_clip_risk", utt_id, norm_path, f"peak={actual_peak_value:.6f}")
                if actual_peak_value <= 1e-6:
                    add_issue(issues, "error", "silent_audio", utt_id, norm_path, f"peak={actual_peak_value:.6f}")

                csv_sr = row.get("norm_sample_rate")
                if csv_sr and csv_sr != actual_sample_rate:
                    add_issue(issues, "error", "csv_sample_rate_mismatch", utt_id, norm_path, f"csv={csv_sr}, actual={actual_sample_rate}")
                csv_frames = row.get("norm_num_frames")
                if csv_frames and csv_frames != actual_num_frames:
                    add_issue(issues, "error", "csv_num_frames_mismatch", utt_id, norm_path, f"csv={csv_frames}, actual={actual_num_frames}")
                csv_duration = parse_float(row.get("norm_duration"))
                if csv_duration is not None and abs(csv_duration - duration) > 0.02:
                    add_issue(
                        issues,
                        "error",
                        "csv_duration_mismatch",
                        utt_id,
                        norm_path,
                        f"csv={csv_duration:.6f}, actual={duration:.6f}",
                    )
            except Exception as exc:  # noqa: BLE001
                add_issue(issues, "error", "audio_read_error", utt_id, norm_path, repr(exc))

        seconds = parse_float(row.get("generation_seconds"))
        if seconds is not None:
            generation_seconds.append(seconds)

        per_wav.append(
            {
                "utt_id": utt_id,
                "patient_id": row.get("patient_id", ""),
                "split": row.get("split", ""),
                "zero_shot_bucket": row.get("zero_shot_bucket", ""),
                "norm_wav_path": norm_path,
                "dys_wav_path": dys_path,
                "file_size": file_size,
                "actual_sample_rate": actual_sample_rate,
                "actual_num_frames": actual_num_frames,
                "actual_duration": actual_duration,
                "actual_rms_dbfs": actual_rms,
                "actual_peak": actual_peak,
                "csv_norm_sample_rate": row.get("norm_sample_rate", ""),
                "csv_norm_num_frames": row.get("norm_num_frames", ""),
                "csv_norm_duration": row.get("norm_duration", ""),
                "csv_rms_dbfs": row.get("postprocess_output_rms_dbfs", ""),
                "csv_peak": row.get("postprocess_output_peak", ""),
                "generation_seconds": row.get("generation_seconds", ""),
                "generation_status": row.get("generation_status", ""),
            }
        )

    for norm_path, count in norm_paths.items():
        if count > 1:
            add_issue(issues, "error", "duplicate_norm_wav_path", "", norm_path, f"count={count}")

    issue_counts = Counter(issue["severity"] for issue in issues)
    issue_code_counts = Counter(issue["code"] for issue in issues)
    split_counts = Counter(row.get("split", "") for row in generated_rows)
    patient_counts = Counter(row.get("patient_id", "") for row in generated_rows)
    bucket_counts = Counter(row.get("zero_shot_bucket", "") for row in generated_rows)

    summary: dict[str, Any] = {
        "pair_manifest": str(pair_manifest),
        "generated_csv": str(generated_csv),
        "manifest_rows": len(manifest_rows),
        "generated_rows": len(generated_rows),
        "per_wav_rows": len(per_wav),
        "unique_patient_count": len(patient_counts),
        "split_counts": dict(sorted(split_counts.items())),
        "zero_shot_bucket_counts": dict(sorted(bucket_counts.items())),
        "expected_sample_rate": args.expected_sample_rate,
        "target_rms_dbfs": args.target_rms_dbfs,
        "issue_count": len(issues),
        "error_count": issue_counts.get("error", 0),
        "warning_count": issue_counts.get("warning", 0),
        "issue_code_counts": dict(sorted(issue_code_counts.items())),
        "duration_sec": summarize_values(durations),
        "rms_dbfs": summarize_values(rms_values),
        "peak": summarize_values(peaks),
        "generation_seconds": summarize_values(generation_seconds),
    }

    (out_dir / "qa_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(
        out_dir / "qa_summary.csv",
        [{"key": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value} for key, value in summary.items()],
        ["key", "value"],
    )
    write_csv(
        out_dir / "qa_issues.csv",
        issues,
        ["severity", "code", "utt_id", "path", "message"],
    )
    write_csv(
        out_dir / "qa_per_wav.csv",
        per_wav,
        [
            "utt_id",
            "patient_id",
            "split",
            "zero_shot_bucket",
            "norm_wav_path",
            "dys_wav_path",
            "file_size",
            "actual_sample_rate",
            "actual_num_frames",
            "actual_duration",
            "actual_rms_dbfs",
            "actual_peak",
            "csv_norm_sample_rate",
            "csv_norm_num_frames",
            "csv_norm_duration",
            "csv_rms_dbfs",
            "csv_peak",
            "generation_seconds",
            "generation_status",
        ],
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
