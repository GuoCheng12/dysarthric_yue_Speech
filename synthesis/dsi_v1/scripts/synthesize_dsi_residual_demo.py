#!/usr/bin/env python3
"""Export Step D diagnostic mel and waveform demos from a trained residual generator."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch

from train_dsi_residual_generator import DeterministicResidualGenerator, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--limit-per-split", type=int, default=3)
    parser.add_argument("--utt-id", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--win-length", type=int, default=400)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument("--f-min", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=7600.0)
    parser.add_argument("--griffinlim-iters", type=int, default=64)
    parser.add_argument("--log-mel-min", type=float, default=-13.8)
    parser.add_argument("--log-mel-max", type=float, default=8.0)
    parser.add_argument("--copy-reference-wavs", action="store_true")
    parser.add_argument("--skip-wav", action="store_true")
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


def safe_id(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def select_rows(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    generated = [row for row in rows if row.get("status") == "generated"]
    if args.utt_id:
        allowed = set(args.utt_id)
        return [row for row in generated if row.get("utt_id") in allowed]

    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for split in args.split:
        split_rows = [row for row in generated if row.get("split") == split]
        split_rows.sort(key=lambda row: (row.get("zero_shot_bucket", ""), row.get("patient_id", ""), row.get("utt_id", "")))
        rng.shuffle(split_rows)
        if args.limit_per_split > 0:
            split_rows = split_rows[: args.limit_per_split]
        selected.extend(split_rows)
    selected.sort(key=lambda row: (row.get("split", ""), row.get("patient_id", ""), row.get("utt_id", "")))
    return selected


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[DeterministicResidualGenerator, dict[str, int], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    patient_vocab = checkpoint["patient_vocab"]
    model = DeterministicResidualGenerator(
        patient_count=len(patient_vocab),
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_layers=int(config.get("num_layers", 4)),
        dropout=float(config.get("dropout", 0.05)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, patient_vocab, checkpoint


def mel_to_wave(args: argparse.Namespace, log_mel: torch.Tensor) -> torch.Tensor:
    log_mel = log_mel.float().clamp(args.log_mel_min, args.log_mel_max)
    mel_power = torch.exp(log_mel).transpose(0, 1).clamp_min(1e-8).cpu().numpy()
    wav_np = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        window="hann",
        center=True,
        power=2.0,
        n_iter=args.griffinlim_iters,
        fmin=args.f_min,
        fmax=args.f_max,
        htk=True,
        norm=None,
    )
    wav = torch.from_numpy(np.asarray(wav_np, dtype=np.float32)).unsqueeze(0)
    if wav.numel() == 0:
        return wav
    wav = wav - wav.mean()
    peak = wav.abs().max().clamp_min(1e-8)
    wav = wav / peak * 0.95
    return wav.cpu()


def wav_stats(wav: torch.Tensor) -> dict[str, float]:
    wav = wav.float()
    peak = float(wav.abs().max()) if wav.numel() else 0.0
    rms = float(torch.sqrt(torch.mean(wav**2)).item()) if wav.numel() else 0.0
    duration = float(wav.shape[-1] / 16000.0) if wav.ndim == 2 else 0.0
    return {"peak": peak, "rms": rms, "duration_seconds": duration}


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> dict[str, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav.squeeze(0).cpu().numpy(), sample_rate)
    return wav_stats(wav)


def copy_reference(src: str, dst: Path) -> None:
    if not src:
        return
    path = Path(src)
    if path.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not args.split:
        args.split = ["dev"]
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    wav_dir = out_dir / "wav"
    ref_dir = out_dir / "reference_wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_wav:
        wav_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(Path(args.feature_manifest))
    selected = select_rows(args, rows)
    if not selected:
        raise ValueError("No rows selected")

    device = resolve_device(args.device)
    model, patient_vocab, checkpoint = load_model(Path(args.checkpoint), device)
    summary_rows: list[dict[str, Any]] = []

    for row in selected:
        utt_id = row["utt_id"]
        patient_id = row.get("patient_id", "")
        if patient_id not in patient_vocab:
            raise KeyError(f"Patient {patient_id!r} not found in checkpoint patient vocab")
        payload = torch.load(row["feature_path"], map_location="cpu", weights_only=False)
        norm_mel = payload["norm_mel"].float()
        norm_ssl = payload["norm_ssl"].float()
        target_mel = payload["dys_mel_aligned"].float()
        target_residual = payload["residual_mel"].float()
        patient_idx = torch.tensor([patient_vocab[patient_id]], dtype=torch.long, device=device)
        pred_residual = model(
            norm_mel.unsqueeze(0).to(device),
            norm_ssl.unsqueeze(0).to(device),
            patient_idx,
        ).squeeze(0).cpu()
        pred_mel = norm_mel + pred_residual

        baseline_l1 = float(torch.mean(torch.abs(norm_mel - target_mel)))
        pred_l1 = float(torch.mean(torch.abs(pred_mel - target_mel)))
        residual_l1 = float(torch.mean(torch.abs(pred_residual - target_residual)))
        gain = (baseline_l1 - pred_l1) / baseline_l1 if baseline_l1 > 0 else 0.0

        item_id = safe_id(utt_id)
        pred_path = pred_dir / f"{item_id}.pt"
        torch.save(
            {
                "utt_id": utt_id,
                "patient_id": patient_id,
                "split": row.get("split", ""),
                "zero_shot_bucket": row.get("zero_shot_bucket", ""),
                "feature_path": row["feature_path"],
                "norm_mel": norm_mel.half(),
                "target_mel": target_mel.half(),
                "target_residual": target_residual.half(),
                "pred_residual": pred_residual.half(),
                "pred_mel": pred_mel.half(),
                "baseline_l1_norm_to_target": baseline_l1,
                "pred_l1_to_target": pred_l1,
                "relative_l1_gain": gain,
            },
            pred_path,
        )

        wav_metrics: dict[str, float] = {}
        if not args.skip_wav:
            for name, mel in [("norm_gl", norm_mel), ("target_aligned_gl", target_mel), ("pred_gl", pred_mel)]:
                stats = save_wav(wav_dir / f"{item_id}_{name}.wav", mel_to_wave(args, mel), args.sample_rate)
                wav_metrics[f"{name}_peak"] = stats["peak"]
                wav_metrics[f"{name}_rms"] = stats["rms"]
                wav_metrics[f"{name}_duration_seconds"] = stats["duration_seconds"]

        if args.copy_reference_wavs:
            copy_reference(row.get("norm_wav_path", ""), ref_dir / f"{item_id}_norm_original.wav")
            copy_reference(row.get("dys_wav_path", ""), ref_dir / f"{item_id}_dys_original.wav")

        summary_rows.append(
            {
                "utt_id": utt_id,
                "patient_id": patient_id,
                "split": row.get("split", ""),
                "zero_shot_bucket": row.get("zero_shot_bucket", ""),
                "frames": int(norm_mel.shape[0]),
                "approx_duration_seconds": f"{norm_mel.shape[0] * args.hop_length / args.sample_rate:.3f}",
                "baseline_l1_norm_to_target": f"{baseline_l1:.6f}",
                "pred_l1_to_target": f"{pred_l1:.6f}",
                "residual_l1": f"{residual_l1:.6f}",
                "relative_l1_gain": f"{gain:.6f}",
                "target_residual_abs_mean": f"{float(torch.mean(torch.abs(target_residual))):.6f}",
                "pred_residual_abs_mean": f"{float(torch.mean(torch.abs(pred_residual))):.6f}",
                "norm_mel_min": f"{float(norm_mel.min()):.6f}",
                "norm_mel_max": f"{float(norm_mel.max()):.6f}",
                "target_mel_min": f"{float(target_mel.min()):.6f}",
                "target_mel_max": f"{float(target_mel.max()):.6f}",
                "pred_mel_min": f"{float(pred_mel.min()):.6f}",
                "pred_mel_max": f"{float(pred_mel.max()):.6f}",
                "prediction_path": str(pred_path),
                **{key: f"{value:.6f}" for key, value in wav_metrics.items()},
            }
        )

    fields = [
        "utt_id",
        "patient_id",
        "split",
        "zero_shot_bucket",
        "frames",
        "approx_duration_seconds",
        "baseline_l1_norm_to_target",
        "pred_l1_to_target",
        "residual_l1",
        "relative_l1_gain",
        "target_residual_abs_mean",
        "pred_residual_abs_mean",
        "norm_mel_min",
        "norm_mel_max",
        "target_mel_min",
        "target_mel_max",
        "pred_mel_min",
        "pred_mel_max",
        "prediction_path",
        "norm_gl_peak",
        "norm_gl_rms",
        "norm_gl_duration_seconds",
        "target_aligned_gl_peak",
        "target_aligned_gl_rms",
        "target_aligned_gl_duration_seconds",
        "pred_gl_peak",
        "pred_gl_rms",
        "pred_gl_duration_seconds",
    ]
    write_csv(out_dir / "step_d_demo_summary.csv", summary_rows, fields)
    config = {
        "feature_manifest": args.feature_manifest,
        "checkpoint": args.checkpoint,
        "out_dir": args.out_dir,
        "selected_rows": len(selected),
        "splits": args.split,
        "limit_per_split": args.limit_per_split,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "griffinlim_iters": args.griffinlim_iters,
        "diagnostic_note": "Griffin-Lim output is for mel sanity checking only, not final vocoder quality.",
    }
    (out_dir / "step_d_demo_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "rows": len(summary_rows), "summary": str(out_dir / "step_d_demo_summary.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
