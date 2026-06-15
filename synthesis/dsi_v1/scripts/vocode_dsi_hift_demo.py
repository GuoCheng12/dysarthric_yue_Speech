#!/usr/bin/env python3
"""Vocode Step D residual-generator mel predictions with CosyVoice HiFT."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True, help="Directory containing Step D prediction .pt files.")
    parser.add_argument("--feature-manifest", required=True, help="Step B feature manifest with norm/dys wav paths.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cosyvoice-repo", default="/data/qwen3-asr/third_party/CosyVoice")
    parser.add_argument("--matcha-repo", default="/data/qwen3-asr/third_party/CosyVoice/third_party/Matcha-TTS")
    parser.add_argument("--cosyvoice-config", default="/data/qwen3-asr/models/tts/Fun-CosyVoice3-0.5B-2512/cosyvoice3.yaml")
    parser.add_argument("--hift-checkpoint", default="/data/qwen3-asr/models/tts/Fun-CosyVoice3-0.5B-2512/hift.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration", choices=["global", "per-bin", "none"], default="per-bin")
    parser.add_argument("--mel-kind", action="append", default=[], help="One of norm_mel,target_mel,pred_mel. Defaults to all.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--copy-reference-wavs", action="store_true")
    return parser.parse_args()


def add_import_paths(cosyvoice_repo: str, matcha_repo: str) -> None:
    for path in [cosyvoice_repo, matcha_repo]:
        if path and path not in sys.path:
            sys.path.insert(0, path)


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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_hift(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    add_import_paths(args.cosyvoice_repo, args.matcha_repo)
    with open(args.cosyvoice_config, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})
    hift = configs["hift"]
    state = torch.load(args.hift_checkpoint, map_location="cpu", weights_only=True)
    state = {key.replace("generator.", ""): value for key, value in state.items()}
    hift.load_state_dict(state, strict=True)
    hift.to(device).eval()
    return hift, configs


def load_wav_mono(path: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)
    return wav.squeeze(0).contiguous()


def cosyvoice_mel_from_wav(path: str, configs: dict[str, Any], device: torch.device) -> torch.Tensor:
    from matcha.utils.audio import mel_spectrogram

    sample_rate = int(configs["sample_rate"])
    wav = load_wav_mono(path, sample_rate).unsqueeze(0).to(device)
    wav = wav.clamp(-1.0, 1.0)
    mel = mel_spectrogram(
        wav,
        n_fft=1920,
        num_mels=80,
        sampling_rate=sample_rate,
        hop_size=480,
        win_size=1920,
        fmin=0,
        fmax=None,
        center=False,
    )
    return mel.squeeze(0).detach().cpu().transpose(0, 1).contiguous()


def match_length(mel: torch.Tensor, target_frames: int) -> torch.Tensor:
    if mel.shape[0] == target_frames:
        return mel.float()
    x = mel.float().transpose(0, 1).unsqueeze(0)
    y = F.interpolate(x, size=target_frames, mode="linear", align_corners=False)
    return y.squeeze(0).transpose(0, 1).contiguous()


def calibrate_mel(x: torch.Tensor, source_norm: torch.Tensor, target_norm: torch.Tensor, mode: str) -> torch.Tensor:
    x = x.float()
    source_norm = source_norm.float()
    target_norm = target_norm.float()
    if mode == "none":
        return x
    if mode == "global":
        src_mean = source_norm.mean()
        src_std = source_norm.std().clamp_min(1e-4)
        tgt_mean = target_norm.mean()
        tgt_std = target_norm.std().clamp_min(1e-4)
        return (x - src_mean) / src_std * tgt_std + tgt_mean
    src_mean = source_norm.mean(dim=0, keepdim=True)
    src_std = source_norm.std(dim=0, keepdim=True).clamp_min(1e-4)
    tgt_mean = target_norm.mean(dim=0, keepdim=True)
    tgt_std = target_norm.std(dim=0, keepdim=True).clamp_min(1e-4)
    return (x - src_mean) / src_std * tgt_std + tgt_mean


@torch.inference_mode()
def vocode(hift: torch.nn.Module, mel: torch.Tensor, device: torch.device) -> torch.Tensor:
    speech_feat = mel.float().transpose(0, 1).unsqueeze(0).to(device)
    wav, _ = hift.inference(speech_feat)
    wav = wav.detach().cpu().float()
    wav = wav.clamp(-0.99, 0.99)
    return wav


def wav_stats(wav: torch.Tensor, sample_rate: int) -> dict[str, float]:
    wav = wav.float()
    return {
        "peak": float(wav.abs().max()) if wav.numel() else 0.0,
        "rms": float(torch.sqrt(torch.mean(wav**2))) if wav.numel() else 0.0,
        "duration_seconds": float(wav.shape[-1] / sample_rate) if wav.ndim == 2 else 0.0,
    }


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> dict[str, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav.squeeze(0).cpu().numpy(), sample_rate)
    return wav_stats(wav, sample_rate)


def copy_reference(src: str, dst: Path) -> None:
    if not src:
        return
    path = Path(src)
    if path.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def mel_stats(prefix: str, mel: torch.Tensor) -> dict[str, str]:
    return {
        f"{prefix}_min": f"{float(mel.min()):.6f}",
        f"{prefix}_mean": f"{float(mel.mean()):.6f}",
        f"{prefix}_max": f"{float(mel.max()):.6f}",
        f"{prefix}_std": f"{float(mel.std()):.6f}",
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav_hift"
    ref_dir = out_dir / "reference_wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    mel_kinds = args.mel_kind or ["norm_mel", "target_mel", "pred_mel"]
    feature_rows = {row["utt_id"]: row for row in read_csv(Path(args.feature_manifest))}
    prediction_files = sorted(Path(args.prediction_dir).glob("*.pt"))
    if args.limit > 0:
        prediction_files = prediction_files[: args.limit]
    if not prediction_files:
        raise ValueError(f"No prediction .pt files found under {args.prediction_dir}")

    device = resolve_device(args.device)
    hift, configs = load_hift(args, device)
    sample_rate = int(configs["sample_rate"])
    rows: list[dict[str, Any]] = []

    for pred_path in prediction_files:
        payload = torch.load(pred_path, map_location="cpu", weights_only=False)
        utt_id = payload["utt_id"]
        manifest_row = feature_rows.get(utt_id)
        if manifest_row is None:
            raise KeyError(f"{utt_id} not found in feature manifest")
        item_id = safe_id(utt_id)
        norm_mel = payload["norm_mel"].float()
        cosy_norm = cosyvoice_mel_from_wav(manifest_row["norm_wav_path"], configs, device)
        cosy_norm = match_length(cosy_norm, norm_mel.shape[0])

        row_base: dict[str, Any] = {
            "utt_id": utt_id,
            "patient_id": payload.get("patient_id", manifest_row.get("patient_id", "")),
            "split": payload.get("split", manifest_row.get("split", "")),
            "zero_shot_bucket": payload.get("zero_shot_bucket", manifest_row.get("zero_shot_bucket", "")),
            "frames": int(norm_mel.shape[0]),
            "sample_rate": sample_rate,
            "calibration": args.calibration,
            "prediction_path": str(pred_path),
            **mel_stats("source_norm_mel", norm_mel),
            **mel_stats("target_cosy_norm_mel", cosy_norm),
        }

        if args.copy_reference_wavs:
            copy_reference(manifest_row.get("norm_wav_path", ""), ref_dir / f"{item_id}_norm_original.wav")
            copy_reference(manifest_row.get("dys_wav_path", ""), ref_dir / f"{item_id}_dys_original.wav")

        for mel_kind in mel_kinds:
            mel = payload[mel_kind].float()
            calibrated = calibrate_mel(mel, norm_mel, cosy_norm, args.calibration)
            wav = vocode(hift, calibrated, device)
            wav_path = wav_dir / f"{item_id}_{mel_kind.replace('_mel', '')}_hift_{args.calibration}.wav"
            stats = save_wav(wav_path, wav, sample_rate)
            rows.append(
                {
                    **row_base,
                    "mel_kind": mel_kind,
                    "wav_path": str(wav_path),
                    **mel_stats("input_mel", mel),
                    **mel_stats("calibrated_mel", calibrated),
                    "wav_peak": f"{stats['peak']:.6f}",
                    "wav_rms": f"{stats['rms']:.6f}",
                    "wav_duration_seconds": f"{stats['duration_seconds']:.6f}",
                }
            )

    fields = [
        "utt_id",
        "patient_id",
        "split",
        "zero_shot_bucket",
        "frames",
        "sample_rate",
        "calibration",
        "mel_kind",
        "wav_path",
        "prediction_path",
        "source_norm_mel_min",
        "source_norm_mel_mean",
        "source_norm_mel_max",
        "source_norm_mel_std",
        "target_cosy_norm_mel_min",
        "target_cosy_norm_mel_mean",
        "target_cosy_norm_mel_max",
        "target_cosy_norm_mel_std",
        "input_mel_min",
        "input_mel_mean",
        "input_mel_max",
        "input_mel_std",
        "calibrated_mel_min",
        "calibrated_mel_mean",
        "calibrated_mel_max",
        "calibrated_mel_std",
        "wav_peak",
        "wav_rms",
        "wav_duration_seconds",
    ]
    write_csv(out_dir / "hift_vocoder_summary.csv", rows, fields)
    config = {
        "prediction_dir": args.prediction_dir,
        "feature_manifest": args.feature_manifest,
        "out_dir": args.out_dir,
        "cosyvoice_config": args.cosyvoice_config,
        "hift_checkpoint": args.hift_checkpoint,
        "sample_rate": sample_rate,
        "calibration": args.calibration,
        "mel_kinds": mel_kinds,
        "prediction_count": len(prediction_files),
        "wav_count": len(rows),
        "note": "HiFT is trained on CosyVoice 24k log-magnitude mel. This demo calibrates Step D mel using each normal TTS utterance; final quality still requires matched-feature training.",
    }
    (out_dir / "hift_vocoder_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "wav_count": len(rows), "summary": str(out_dir / "hift_vocoder_summary.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
