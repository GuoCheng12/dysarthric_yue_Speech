#!/usr/bin/env python3
"""Rebuild DSI residual features in CosyVoice3 HiFT-compatible mel space."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-feature-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--n-fft", type=int, default=1920)
    parser.add_argument("--win-length", type=int, default=1920)
    parser.add_argument("--hop-length", type=int, default=480)
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument("--f-max", type=float, default=-1.0, help="Negative means None, matching CosyVoice3 config.")
    parser.add_argument("--dtw-metric", default="cosine")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--utt-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--flush-every", type=int, default=25)
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


def safe_utt_id(utt_id: str) -> str:
    return utt_id.replace("/", "__").replace(" ", "_")


def output_dtype(dtype: str) -> torch.dtype:
    return torch.float16 if dtype == "float16" else torch.float32


def fmax_value(raw_fmax: float) -> float | None:
    return None if raw_fmax < 0 else raw_fmax


def load_wav_mono(path: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)
    return wav.squeeze(0).contiguous().clamp(-1.0, 1.0)


def cosyvoice_log_mel(
    wav: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    hop_length: int,
    win_length: int,
    f_min: float,
    f_max: float | None,
) -> torch.Tensor:
    """Match CosyVoice3/Matcha mel frontend used by HiFT training.

    This mirrors `matcha.utils.audio.mel_spectrogram`: reflect pad by
    `(n_fft - hop_size) / 2`, use STFT with `center=False`, take magnitude,
    multiply a librosa mel basis, and log-compress magnitudes.
    """
    if wav.ndim != 1:
        raise ValueError(f"Expected 1D wav, got shape={tuple(wav.shape)}")
    device = wav.device
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
    )
    mel_basis_t = torch.from_numpy(mel_basis).float().to(device)
    window = torch.hann_window(win_length).to(device)
    pad = int((n_fft - hop_length) / 2)
    y = F.pad(wav.unsqueeze(0).unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel = torch.matmul(mel_basis_t, magnitude.squeeze(0))
    return torch.log(torch.clamp(mel, min=1e-5)).transpose(0, 1).contiguous()


def zscore_frames(x: torch.Tensor) -> np.ndarray:
    x = x.float()
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-5)
    return ((x - mean) / std).cpu().numpy().astype(np.float32)


def run_dtw(norm_mel: torch.Tensor, dys_mel: torch.Tensor, metric: str) -> tuple[np.ndarray, float]:
    x = zscore_frames(norm_mel).T
    y = zscore_frames(dys_mel).T
    cost, path = librosa.sequence.dtw(X=x, Y=y, metric=metric, backtrack=True)
    path = path[::-1].astype(np.int32)
    norm_cost = float(cost[-1, -1] / max(len(path), 1))
    return path, norm_cost


def align_to_norm_timeline(seq: torch.Tensor, path: np.ndarray, norm_len: int, side: str) -> torch.Tensor:
    if side not in {"norm", "dys"}:
        raise ValueError(f"side must be norm or dys, got {side}")
    source_col = 0 if side == "norm" else 1
    target_col = 0
    buckets: list[list[int]] = [[] for _ in range(norm_len)]
    max_source = seq.shape[0] - 1
    for pair in path:
        target_idx = int(pair[target_col])
        source_idx = int(pair[source_col])
        if 0 <= target_idx < norm_len and 0 <= source_idx <= max_source:
            buckets[target_idx].append(source_idx)

    aligned = []
    last = None
    for idx, bucket in enumerate(buckets):
        if bucket:
            item = seq[torch.tensor(bucket, dtype=torch.long)].float().mean(dim=0)
            last = item
        elif last is not None:
            item = last
        else:
            nearest = min(max(idx, 0), max_source)
            item = seq[nearest].float()
            last = item
        aligned.append(item)
    return torch.stack(aligned, dim=0)


def interpolate_time(seq: torch.Tensor, target_len: int) -> torch.Tensor:
    if seq.shape[0] == target_len:
        return seq.float()
    x = seq.float().transpose(0, 1).unsqueeze(0)
    y = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return y.squeeze(0).transpose(0, 1).contiguous()


def tensor_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def load_existing_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    return {row["utt_id"]: row for row in read_csv(path)}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    feature_root = out_dir / "features"
    out_manifest = Path(args.out_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)

    rows = [row for row in read_csv(Path(args.source_feature_manifest)) if row.get("status") == "generated"]
    if args.split:
        allowed = set(args.split)
        rows = [row for row in rows if row.get("split") in allowed]
    if args.utt_id:
        allowed = set(args.utt_id)
        rows = [row for row in rows if row.get("utt_id") in allowed]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No rows selected for CosyVoice mel feature rebuild")

    fields = [
        "utt_id",
        "patient_id",
        "split",
        "zero_shot_bucket",
        "prompt_id",
        "clean_text",
        "norm_wav_path",
        "dys_wav_path",
        "source_feature_path",
        "feature_path",
        "status",
        "error",
        "norm_mel_frames",
        "dys_mel_frames",
        "norm_ssl_frames",
        "dys_ssl_frames",
        "dtw_path_len",
        "dtw_norm_cost",
        "residual_l1_mean",
        "residual_l2_mean",
        "seconds",
    ]
    existing = load_existing_manifest(out_manifest)
    manifest_by_utt: dict[str, dict[str, Any]] = dict(existing)
    save_dtype = output_dtype(args.dtype)
    f_max = fmax_value(args.f_max)

    residual_l1_values: list[float] = []
    dtw_cost_values: list[float] = []
    errors = 0
    started_all = time.time()
    for idx, row in enumerate(rows, start=1):
        started = time.time()
        utt_id = row["utt_id"]
        out_path = feature_root / f"{safe_utt_id(utt_id)}.pt"
        if out_path.is_file() and not args.overwrite:
            old = manifest_by_utt.get(utt_id)
            if old and old.get("status") == "generated":
                continue
        try:
            source_payload = torch.load(row["feature_path"], map_location="cpu", weights_only=False)
            norm_wav = load_wav_mono(row["norm_wav_path"], args.sample_rate)
            dys_wav = load_wav_mono(row["dys_wav_path"], args.sample_rate)
            norm_mel = cosyvoice_log_mel(
                norm_wav,
                sample_rate=args.sample_rate,
                n_fft=args.n_fft,
                n_mels=args.n_mels,
                hop_length=args.hop_length,
                win_length=args.win_length,
                f_min=args.f_min,
                f_max=f_max,
            )
            dys_mel = cosyvoice_log_mel(
                dys_wav,
                sample_rate=args.sample_rate,
                n_fft=args.n_fft,
                n_mels=args.n_mels,
                hop_length=args.hop_length,
                win_length=args.win_length,
                f_min=args.f_min,
                f_max=f_max,
            )
            path, dtw_cost = run_dtw(norm_mel, dys_mel, args.dtw_metric)
            dys_mel_aligned = align_to_norm_timeline(dys_mel, path, norm_mel.shape[0], "dys")
            residual = dys_mel_aligned - norm_mel
            norm_ssl = interpolate_time(source_payload["norm_ssl"].float(), norm_mel.shape[0])
            dys_ssl_aligned = interpolate_time(source_payload["dys_ssl_aligned"].float(), norm_mel.shape[0])
            residual_l1 = float(torch.mean(torch.abs(residual)))
            residual_l2 = float(torch.sqrt(torch.mean(residual**2)))
            residual_l1_values.append(residual_l1)
            dtw_cost_values.append(dtw_cost)

            payload = {
                "utt_id": utt_id,
                "patient_id": row.get("patient_id", ""),
                "split": row.get("split", ""),
                "zero_shot_bucket": row.get("zero_shot_bucket", ""),
                "prompt_id": row.get("prompt_id", ""),
                "clean_text": row.get("clean_text", ""),
                "norm_wav_path": row.get("norm_wav_path", ""),
                "dys_wav_path": row.get("dys_wav_path", ""),
                "source_feature_path": row.get("feature_path", ""),
                "mel_frontend": {
                    "name": "cosyvoice3_matcha_mel_spectrogram",
                    "sample_rate": args.sample_rate,
                    "n_fft": args.n_fft,
                    "n_mels": args.n_mels,
                    "hop_length": args.hop_length,
                    "win_length": args.win_length,
                    "f_min": args.f_min,
                    "f_max": f_max,
                    "center": False,
                    "pad": int((args.n_fft - args.hop_length) / 2),
                    "scale": "log_magnitude",
                },
                "norm_mel": norm_mel.to(save_dtype),
                "dys_mel_aligned": dys_mel_aligned.to(save_dtype),
                "residual_mel": residual.to(save_dtype),
                "norm_ssl": norm_ssl.to(save_dtype),
                "dys_ssl_aligned": dys_ssl_aligned.to(save_dtype),
                "dtw_norm_to_dys_path": torch.from_numpy(path.astype(np.int32)),
                "dtw_norm_cost": dtw_cost,
                "residual_l1_mean": residual_l1,
                "residual_l2_mean": residual_l2,
            }
            torch.save(payload, out_path)
            manifest_by_utt[utt_id] = {
                **{field: row.get(field, "") for field in fields},
                "source_feature_path": row.get("feature_path", ""),
                "feature_path": str(out_path),
                "status": "generated",
                "error": "",
                "norm_mel_frames": norm_mel.shape[0],
                "dys_mel_frames": dys_mel.shape[0],
                "norm_ssl_frames": norm_ssl.shape[0],
                "dys_ssl_frames": dys_ssl_aligned.shape[0],
                "dtw_path_len": int(path.shape[0]),
                "dtw_norm_cost": f"{dtw_cost:.6f}",
                "residual_l1_mean": f"{residual_l1:.6f}",
                "residual_l2_mean": f"{residual_l2:.6f}",
                "seconds": f"{time.time() - started:.3f}",
            }
        except Exception as exc:  # noqa: BLE001
            errors += 1
            manifest_by_utt[utt_id] = {
                **{field: row.get(field, "") for field in fields},
                "source_feature_path": row.get("feature_path", ""),
                "feature_path": str(out_path),
                "status": "error",
                "error": repr(exc),
                "seconds": f"{time.time() - started:.3f}",
            }

        if idx % args.flush_every == 0 or idx == len(rows):
            manifest_rows = [manifest_by_utt[key] for key in sorted(manifest_by_utt)]
            write_csv(out_manifest, manifest_rows, fields)
            status_counts = Counter(str(row.get("status", "")) for row in manifest_rows)
            print(
                json.dumps(
                    {
                        "processed": idx,
                        "selected": len(rows),
                        "status_counts": dict(sorted(status_counts.items())),
                        "errors": errors,
                        "elapsed_seconds": time.time() - started_all,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    manifest_rows = [manifest_by_utt[key] for key in sorted(manifest_by_utt)]
    status_counts = Counter(str(row.get("status", "")) for row in manifest_rows)
    split_counts = Counter(row.get("split", "") for row in manifest_rows if row.get("status") == "generated")
    bucket_counts = Counter(row.get("zero_shot_bucket", "") for row in manifest_rows if row.get("status") == "generated")
    summary = {
        "source_feature_manifest": str(Path(args.source_feature_manifest)),
        "out_dir": str(out_dir),
        "out_manifest": str(out_manifest),
        "selected_rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "zero_shot_bucket_counts": dict(sorted(bucket_counts.items())),
        "mel_frontend": {
            "name": "cosyvoice3_matcha_mel_spectrogram",
            "sample_rate": args.sample_rate,
            "n_fft": args.n_fft,
            "n_mels": args.n_mels,
            "hop_length": args.hop_length,
            "win_length": args.win_length,
            "f_min": args.f_min,
            "f_max": f_max,
            "center": False,
            "pad": int((args.n_fft - args.hop_length) / 2),
            "scale": "log_magnitude",
        },
        "ssl_source": "reused_from_step_b_hubert_features_and_interpolated_to_cosy_mel_frames",
        "dtype": args.dtype,
        "dtw_metric": args.dtw_metric,
        "residual_l1_mean": tensor_summary(residual_l1_values),
        "dtw_norm_cost": tensor_summary(dtw_cost_values),
        "elapsed_seconds": time.time() - started_all,
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
