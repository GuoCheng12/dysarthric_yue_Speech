#!/usr/bin/env python3
"""Audit DSI V1 residual feature manifests before residual-generator training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


REQUIRED_TENSORS = {
    "norm_mel": 80,
    "dys_mel_aligned": 80,
    "residual_mel": 80,
    "norm_ssl": 768,
    "dys_ssl_aligned": 768,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-limit", type=int, default=0, help="Audit a deterministic sample; 0 audits all rows.")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--residual-tolerance", type=float, default=0.05)
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


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def add_issue(issues: list[dict[str, str]], severity: str, code: str, utt_id: str, path: str, message: str) -> None:
    issues.append({"severity": severity, "code": code, "utt_id": utt_id, "path": path, "message": message})


def select_rows(rows: list[dict[str, str]], sample_limit: int, seed: int) -> list[dict[str, str]]:
    if sample_limit <= 0 or sample_limit >= len(rows):
        return rows
    rng = random.Random(seed)
    picked: list[dict[str, str]] = []
    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_split[row.get("split", "")].append(row)
    for split in ["train", "dev", "test"]:
        split_rows = by_split.get(split, [])
        if split_rows:
            picked.append(split_rows[0])
            picked.append(split_rows[-1])
    remaining = [row for row in rows if row not in picked]
    rng.shuffle(remaining)
    picked.extend(remaining[: max(sample_limit - len(picked), 0)])
    return picked[:sample_limit]


def audit_tensor_payload(
    row: dict[str, str],
    tolerance: float,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    utt_id = row["utt_id"]
    feature_path = row["feature_path"]
    result: dict[str, Any] = {
        "utt_id": utt_id,
        "split": row.get("split", ""),
        "patient_id": row.get("patient_id", ""),
        "zero_shot_bucket": row.get("zero_shot_bucket", ""),
        "feature_path": feature_path,
        "status": "ok",
        "aligned_frames": "",
        "norm_mel_dim": "",
        "ssl_dim": "",
        "residual_l1_mean": "",
        "residual_l2_mean": "",
        "residual_recompute_max_abs_error": "",
        "dtw_path_len": "",
        "dtw_norm_cost": "",
    }

    path = Path(feature_path)
    if not path.is_file():
        add_issue(issues, "error", "missing_feature_file", utt_id, feature_path, "feature_path does not exist")
        result["status"] = "error"
        return result

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        add_issue(issues, "error", "load_error", utt_id, feature_path, repr(exc))
        result["status"] = "error"
        return result

    for key, expected_dim in REQUIRED_TENSORS.items():
        if key not in payload:
            add_issue(issues, "error", "missing_tensor", utt_id, feature_path, f"missing {key}")
            result["status"] = "error"
            continue
        tensor = payload[key]
        if tensor.ndim != 2:
            add_issue(issues, "error", "tensor_rank", utt_id, feature_path, f"{key} shape={tuple(tensor.shape)}")
            result["status"] = "error"
            continue
        if tensor.shape[1] != expected_dim:
            add_issue(issues, "error", "tensor_dim", utt_id, feature_path, f"{key} dim={tensor.shape[1]}, expected={expected_dim}")
            result["status"] = "error"
        if not torch.isfinite(tensor.float()).all():
            add_issue(issues, "error", "nonfinite_tensor", utt_id, feature_path, key)
            result["status"] = "error"

    if any(key not in payload for key in REQUIRED_TENSORS):
        return result

    lengths = {key: int(payload[key].shape[0]) for key in REQUIRED_TENSORS}
    if len(set(lengths.values())) != 1:
        add_issue(issues, "error", "time_axis_mismatch", utt_id, feature_path, json.dumps(lengths, ensure_ascii=False))
        result["status"] = "error"

    norm_mel = payload["norm_mel"].float()
    dys_mel = payload["dys_mel_aligned"].float()
    residual = payload["residual_mel"].float()
    recomputed = dys_mel - norm_mel
    max_abs_error = float(torch.max(torch.abs(recomputed - residual)))
    if max_abs_error > tolerance:
        add_issue(
            issues,
            "error",
            "residual_mismatch",
            utt_id,
            feature_path,
            f"max_abs_error={max_abs_error:.6f}, tolerance={tolerance}",
        )
        result["status"] = "error"

    if "dtw_norm_to_dys_path" not in payload:
        add_issue(issues, "error", "missing_dtw_path", utt_id, feature_path, "missing dtw_norm_to_dys_path")
        result["status"] = "error"
        dtw_path_len = ""
    else:
        dtw_path = payload["dtw_norm_to_dys_path"]
        dtw_path_len = int(dtw_path.shape[0]) if getattr(dtw_path, "ndim", 0) == 2 else ""
        if getattr(dtw_path, "ndim", 0) != 2 or dtw_path.shape[1] != 2:
            add_issue(issues, "error", "dtw_path_shape", utt_id, feature_path, f"shape={tuple(dtw_path.shape)}")
            result["status"] = "error"

    residual_l1 = float(torch.mean(torch.abs(residual)))
    residual_l2 = float(torch.sqrt(torch.mean(residual**2)))
    result.update(
        {
            "aligned_frames": next(iter(lengths.values())),
            "norm_mel_dim": int(payload["norm_mel"].shape[1]),
            "ssl_dim": int(payload["norm_ssl"].shape[1]),
            "residual_l1_mean": f"{residual_l1:.6f}",
            "residual_l2_mean": f"{residual_l2:.6f}",
            "residual_recompute_max_abs_error": f"{max_abs_error:.6f}",
            "dtw_path_len": dtw_path_len,
            "dtw_norm_cost": f"{float(payload.get('dtw_norm_cost', math.nan)):.6f}",
        }
    )
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(Path(args.feature_manifest))
    selected = select_rows(rows, args.sample_limit, args.seed)
    issues: list[dict[str, str]] = []

    manifest_status = Counter(row.get("status", "") for row in rows)
    split_counts = Counter(row.get("split", "") for row in rows)
    bucket_counts = Counter(row.get("zero_shot_bucket", "") for row in rows)
    train_patients = {row.get("patient_id", "") for row in rows if row.get("split") == "train"}
    dev_patients = {row.get("patient_id", "") for row in rows if row.get("split") == "dev"}
    test_patients = {row.get("patient_id", "") for row in rows if row.get("split") == "test"}
    patient_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        patient_split_counts[row.get("patient_id", "")][row.get("split", "")] += 1

    for row in rows:
        if row.get("status") != "generated":
            add_issue(issues, "error", "manifest_status", row.get("utt_id", ""), row.get("feature_path", ""), f"status={row.get('status')!r}")
        if not row.get("feature_path"):
            add_issue(issues, "error", "empty_feature_path", row.get("utt_id", ""), "", "feature_path is empty")

    per_feature = [audit_tensor_payload(row, args.residual_tolerance, issues) for row in selected]
    residual_l1 = [float(row["residual_l1_mean"]) for row in per_feature if row.get("residual_l1_mean")]
    dtw_cost = [float(row["dtw_norm_cost"]) for row in per_feature if row.get("dtw_norm_cost")]
    frames = [float(row["aligned_frames"]) for row in per_feature if row.get("aligned_frames")]

    patient_rows = []
    for patient, counts in sorted(patient_split_counts.items()):
        patient_rows.append(
            {
                "patient_id": patient,
                "train": counts.get("train", 0),
                "dev": counts.get("dev", 0),
                "test": counts.get("test", 0),
                "in_train": str(patient in train_patients),
            }
        )

    summary = {
        "feature_manifest": str(Path(args.feature_manifest)),
        "manifest_rows": len(rows),
        "audited_tensor_rows": len(selected),
        "sample_limit": args.sample_limit,
        "manifest_status_counts": dict(sorted(manifest_status.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "zero_shot_bucket_counts": dict(sorted(bucket_counts.items())),
        "unique_patient_count": len(patient_split_counts),
        "dev_patient_count": len(dev_patients),
        "test_patient_count": len(test_patients),
        "dev_patients_not_in_train": sorted(dev_patients - train_patients),
        "test_patients_not_in_train": sorted(test_patients - train_patients),
        "issue_count": len(issues),
        "error_count": sum(1 for issue in issues if issue["severity"] == "error"),
        "warning_count": sum(1 for issue in issues if issue["severity"] == "warning"),
        "issue_code_counts": dict(sorted(Counter(issue["code"] for issue in issues).items())),
        "residual_l1_mean": summarize(residual_l1),
        "dtw_norm_cost": summarize(dtw_cost),
        "aligned_frames": summarize(frames),
    }

    (out_dir / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "audit_issues.csv", issues, ["severity", "code", "utt_id", "path", "message"])
    write_csv(
        out_dir / "audit_per_feature.csv",
        per_feature,
        [
            "utt_id",
            "split",
            "patient_id",
            "zero_shot_bucket",
            "feature_path",
            "status",
            "aligned_frames",
            "norm_mel_dim",
            "ssl_dim",
            "residual_l1_mean",
            "residual_l2_mean",
            "residual_recompute_max_abs_error",
            "dtw_path_len",
            "dtw_norm_cost",
        ],
    )
    write_csv(out_dir / "patient_split_counts.csv", patient_rows, ["patient_id", "train", "dev", "test", "in_train"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
