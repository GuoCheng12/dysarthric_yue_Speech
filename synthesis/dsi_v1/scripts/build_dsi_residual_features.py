#!/usr/bin/env python3
"""Build DSI V1 mel/SSL/DTW residual feature files from TTS-patient pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--ssl-model", required=True, help="Hugging Face repo id or local model directory.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--win-length", type=int, default=400)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument("--f-min", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=7600.0)
    parser.add_argument("--dtw-metric", default="cosine")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default="auto")
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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_wav_mono(path: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)
    return wav.squeeze(0).contiguous()


def make_mel_transform(args: argparse.Namespace) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        f_min=args.f_min,
        f_max=args.f_max,
        n_mels=args.n_mels,
        center=True,
        power=2.0,
        normalized=False,
    )


def log_mel(wav: torch.Tensor, mel_transform: torchaudio.transforms.MelSpectrogram) -> torch.Tensor:
    mel = mel_transform(wav.unsqueeze(0)).squeeze(0).transpose(0, 1)
    return torch.log(mel.clamp_min(1e-6))


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


def extract_ssl(
    wav: torch.Tensor,
    extractor: Any,
    model: torch.nn.Module,
    device: torch.device,
    sample_rate: int,
) -> torch.Tensor:
    inputs = extractor(
        wav.cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=False,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.squeeze(0).detach().cpu()


def output_dtype(dtype: str) -> torch.dtype:
    return torch.float16 if dtype == "float16" else torch.float32


def tensor_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def load_existing_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    rows = read_csv(path)
    return {row["utt_id"]: row for row in rows}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    feature_root = out_dir / "features"
    out_manifest = Path(args.out_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)

    rows = read_csv(Path(args.generated_csv))
    if args.split:
        allowed = set(args.split)
        rows = [row for row in rows if row.get("split") in allowed]
    if args.utt_id:
        allowed = set(args.utt_id)
        rows = [row for row in rows if row.get("utt_id") in allowed]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No rows selected for feature extraction")

    device = resolve_device(args.device)
    print(json.dumps({"event": "load_ssl_model", "ssl_model": args.ssl_model, "device": str(device)}, ensure_ascii=False), flush=True)
    extractor = AutoFeatureExtractor.from_pretrained(args.ssl_model)
    ssl_model = AutoModel.from_pretrained(args.ssl_model).to(device)
    ssl_model.eval()

    mel_transform = make_mel_transform(args)
    save_dtype = output_dtype(args.dtype)

    fields = [
        "utt_id",
        "patient_id",
        "split",
        "zero_shot_bucket",
        "prompt_id",
        "clean_text",
        "norm_wav_path",
        "dys_wav_path",
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
    output_rows: list[dict[str, Any]] = []
    completed = 0
    errors = 0
    residual_l1_values: list[float] = []
    dtw_cost_values: list[float] = []

    for idx, row in enumerate(rows, start=1):
        start = time.time()
        utt_id = row["utt_id"]
        split = row.get("split", "")
        feature_path = feature_root / split / f"{safe_utt_id(utt_id)}.pt"
        feature_path.parent.mkdir(parents=True, exist_ok=True)

        out_row: dict[str, Any] = {
            "utt_id": utt_id,
            "patient_id": row.get("patient_id", ""),
            "split": split,
            "zero_shot_bucket": row.get("zero_shot_bucket", ""),
            "prompt_id": row.get("prompt_id", ""),
            "clean_text": row.get("clean_text", ""),
            "norm_wav_path": row.get("norm_wav_path", ""),
            "dys_wav_path": row.get("dys_wav_path", ""),
            "feature_path": str(feature_path),
            "status": "",
            "error": "",
        }

        try:
            if feature_path.is_file() and not args.overwrite:
                previous = existing.get(utt_id, {})
                out_row.update(previous)
                out_row["status"] = "exists"
                completed += 1
            else:
                norm_wav = load_wav_mono(row["norm_wav_path"], args.sample_rate)
                dys_wav = load_wav_mono(row["dys_wav_path"], args.sample_rate)
                norm_mel = log_mel(norm_wav, mel_transform)
                dys_mel = log_mel(dys_wav, mel_transform)
                path, dtw_cost = run_dtw(norm_mel, dys_mel, args.dtw_metric)

                norm_ssl_raw = extract_ssl(norm_wav, extractor, ssl_model, device, args.sample_rate)
                dys_ssl_raw = extract_ssl(dys_wav, extractor, ssl_model, device, args.sample_rate)

                norm_len = norm_mel.shape[0]
                norm_ssl = interpolate_time(norm_ssl_raw, norm_len)
                dys_ssl_mel_time = interpolate_time(dys_ssl_raw, dys_mel.shape[0])
                dys_mel_aligned = align_to_norm_timeline(dys_mel, path, norm_len, side="dys")
                dys_ssl_aligned = align_to_norm_timeline(dys_ssl_mel_time, path, norm_len, side="dys")
                residual_mel = dys_mel_aligned.float() - norm_mel.float()

                residual_l1 = float(residual_mel.abs().mean())
                residual_l2 = float(torch.sqrt(torch.mean(residual_mel**2)))
                payload = {
                    "utt_id": utt_id,
                    "patient_id": row.get("patient_id", ""),
                    "split": split,
                    "zero_shot_bucket": row.get("zero_shot_bucket", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "clean_text": row.get("clean_text", ""),
                    "norm_wav_path": row.get("norm_wav_path", ""),
                    "dys_wav_path": row.get("dys_wav_path", ""),
                    "sample_rate": args.sample_rate,
                    "mel_config": {
                        "n_mels": args.n_mels,
                        "n_fft": args.n_fft,
                        "win_length": args.win_length,
                        "hop_length": args.hop_length,
                        "f_min": args.f_min,
                        "f_max": args.f_max,
                        "log_floor": 1e-6,
                    },
                    "ssl_model": args.ssl_model,
                    "dtw_metric": args.dtw_metric,
                    "dtw_norm_to_dys_path": torch.from_numpy(path.astype(np.int32)),
                    "dtw_norm_cost": dtw_cost,
                    "norm_mel": norm_mel.to(save_dtype),
                    "dys_mel_aligned": dys_mel_aligned.to(save_dtype),
                    "residual_mel": residual_mel.to(save_dtype),
                    "norm_ssl": norm_ssl.to(save_dtype),
                    "dys_ssl_aligned": dys_ssl_aligned.to(save_dtype),
                    "raw_lengths": {
                        "norm_samples": int(norm_wav.numel()),
                        "dys_samples": int(dys_wav.numel()),
                        "norm_mel_frames": int(norm_mel.shape[0]),
                        "dys_mel_frames": int(dys_mel.shape[0]),
                        "norm_ssl_frames": int(norm_ssl_raw.shape[0]),
                        "dys_ssl_frames": int(dys_ssl_raw.shape[0]),
                        "aligned_frames": int(norm_len),
                    },
                    "residual_stats": {
                        "l1_mean": residual_l1,
                        "l2_mean": residual_l2,
                    },
                }
                torch.save(payload, feature_path)
                out_row.update(
                    {
                        "status": "generated",
                        "norm_mel_frames": int(norm_mel.shape[0]),
                        "dys_mel_frames": int(dys_mel.shape[0]),
                        "norm_ssl_frames": int(norm_ssl_raw.shape[0]),
                        "dys_ssl_frames": int(dys_ssl_raw.shape[0]),
                        "dtw_path_len": int(path.shape[0]),
                        "dtw_norm_cost": f"{dtw_cost:.6f}",
                        "residual_l1_mean": f"{residual_l1:.6f}",
                        "residual_l2_mean": f"{residual_l2:.6f}",
                    }
                )
                residual_l1_values.append(residual_l1)
                dtw_cost_values.append(dtw_cost)
                completed += 1
        except Exception as exc:  # noqa: BLE001
            out_row["status"] = "error"
            out_row["error"] = repr(exc)
            errors += 1

        out_row["seconds"] = f"{time.time() - start:.3f}"
        output_rows.append(out_row)
        print(json.dumps({"idx": idx, "utt_id": utt_id, "status": out_row["status"], "feature_path": out_row["feature_path"]}, ensure_ascii=False), flush=True)

        if args.flush_every > 0 and len(output_rows) % args.flush_every == 0:
            write_csv(out_manifest, output_rows, fields)

    write_csv(out_manifest, output_rows, fields)
    summary = {
        "generated_csv": str(Path(args.generated_csv)),
        "out_dir": str(out_dir),
        "out_manifest": str(out_manifest),
        "ssl_model": args.ssl_model,
        "sample_rate": args.sample_rate,
        "mel_config": {
            "n_mels": args.n_mels,
            "n_fft": args.n_fft,
            "win_length": args.win_length,
            "hop_length": args.hop_length,
            "f_min": args.f_min,
            "f_max": args.f_max,
        },
        "selected_rows": len(rows),
        "completed": completed,
        "errors": errors,
        "status_counts": dict(Counter(row.get("status", "") for row in output_rows)),
        "split_counts": dict(Counter(row.get("split", "") for row in output_rows)),
        "zero_shot_bucket_counts": dict(Counter(row.get("zero_shot_bucket", "") for row in output_rows)),
        "residual_l1_mean": tensor_summary(residual_l1_values),
        "dtw_norm_cost": tensor_summary(dtw_cost_values),
        "feature_dtype": args.dtype,
        "device": str(device),
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
