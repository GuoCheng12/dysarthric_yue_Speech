#!/usr/bin/env python3
"""Generate normal Cantonese TTS waveforms for a DSI V1 pair manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from importlib import metadata
from pathlib import Path

import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--cosyvoice-repo", required=True)
    parser.add_argument(
        "--pythonpath-prepend",
        action="append",
        default=[],
        help="Extra import path to prepend before importing CosyVoice; useful for dependency overlays.",
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--mode", choices=["cached_zero_shot", "sft", "instruct2"], default="cached_zero_shot")
    parser.add_argument("--speaker-id", default="my_zero_shot_spk")
    parser.add_argument("--prompt-wav", default="")
    parser.add_argument(
        "--prompt-text",
        default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    )
    parser.add_argument(
        "--instruction",
        default="You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
    )
    parser.add_argument(
        "--text-frontend",
        choices=["true", "false"],
        default="false",
        help="Whether to run CosyVoice text normalization. Use false for already-clean Cantonese prompts.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of rows to synthesize.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Write the output CSV every N processed rows so long runs are resumable.",
    )
    parser.add_argument("--fp16", action="store_true", help="Enable CosyVoice fp16 inference.")
    parser.add_argument("--speed", type=float, default=1.0, help="CosyVoice speech speed multiplier.")
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=0,
        help="Optional output sample rate after synthesis, e.g. 16000.",
    )
    parser.add_argument(
        "--target-rms-dbfs",
        type=float,
        default=None,
        help="Optional RMS loudness normalization target in dBFS, e.g. -23.0.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def import_cosyvoice(cosyvoice_repo: Path, pythonpath_prepend: list[str]):
    matcha = cosyvoice_repo / "third_party" / "Matcha-TTS"
    for path in reversed([*pythonpath_prepend, str(cosyvoice_repo), str(matcha)]):
        if path:
            sys.path.insert(0, path)
    from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: PLC0415
    from cosyvoice.utils.file_utils import load_wav  # noqa: PLC0415

    return CosyVoice2, load_wav


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def concat_outputs(outputs: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    speeches = [item["tts_speech"].detach().cpu() for item in outputs]
    if not speeches:
        raise RuntimeError("CosyVoice returned no speech chunks")
    speeches = [x if x.dim() == 2 else x.reshape(1, -1) for x in speeches]
    return torch.cat(speeches, dim=1)


def rms_dbfs(speech: torch.Tensor) -> float:
    rms = torch.sqrt(torch.mean(speech.float() ** 2)).clamp_min(1e-12)
    return float(20.0 * torch.log10(rms))


def postprocess_speech(
    speech: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
    target_rms_dbfs: float | None,
) -> tuple[torch.Tensor, int, float, float]:
    if target_sample_rate > 0 and target_sample_rate != sample_rate:
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(speech)
        sample_rate = target_sample_rate

    if target_rms_dbfs is not None:
        current_rms = torch.sqrt(torch.mean(speech.float() ** 2)).clamp_min(1e-12)
        target_rms = 10 ** (target_rms_dbfs / 20.0)
        speech = speech * (target_rms / current_rms)
        peak = torch.max(torch.abs(speech)).clamp_min(1e-12)
        if peak > 0.99:
            speech = speech * (0.99 / peak)

    return speech, sample_rate, rms_dbfs(speech), float(torch.max(torch.abs(speech)))


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.pair_manifest))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("pair manifest has no rows")

    prompt_wav = Path(args.prompt_wav) if args.prompt_wav else None
    if args.mode == "instruct2" and (prompt_wav is None or not prompt_wav.exists()):
        raise FileNotFoundError("--prompt-wav is required for mode=instruct2")

    text_frontend = args.text_frontend == "true"
    CosyVoice2, load_wav = import_cosyvoice(Path(args.cosyvoice_repo), args.pythonpath_prepend)
    model = CosyVoice2(
        args.model_dir,
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=args.fp16,
    )
    prompt_speech_16k = load_wav(str(prompt_wav), 16000) if prompt_wav is not None else None
    transformers_version = package_version("transformers")
    tokenizers_version = package_version("tokenizers")

    generated_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        text = row["clean_text"].strip()
        out_wav = Path(row["norm_wav_path"])
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        start = time.time()
        status = "exists"
        if args.overwrite or not out_wav.exists():
            if args.mode == "cached_zero_shot":
                outputs = list(
                    model.inference_zero_shot(
                        text,
                        "",
                        "",
                        zero_shot_spk_id=args.speaker_id,
                        stream=False,
                        text_frontend=text_frontend,
                        speed=args.speed,
                    )
                )
            elif args.mode == "sft":
                outputs = list(
                    model.inference_sft(
                        text,
                        args.speaker_id,
                        stream=False,
                        text_frontend=text_frontend,
                        speed=args.speed,
                    )
                )
            else:
                if prompt_speech_16k is None:
                    raise RuntimeError("--prompt-wav is required for mode=instruct2")
                outputs = list(
                    model.inference_instruct2(
                        text,
                        args.instruction,
                        prompt_speech_16k,
                        stream=False,
                        text_frontend=text_frontend,
                        speed=args.speed,
                    )
                )
            speech = concat_outputs(outputs)
            speech, save_sample_rate, output_rms_dbfs, output_peak = postprocess_speech(
                speech,
                model.sample_rate,
                args.target_sample_rate,
                args.target_rms_dbfs,
            )
            torchaudio.save(str(out_wav), speech, save_sample_rate)
            status = "generated"
        else:
            existing_speech, save_sample_rate = torchaudio.load(str(out_wav))
            output_rms_dbfs = rms_dbfs(existing_speech)
            output_peak = float(torch.max(torch.abs(existing_speech)))

        wav_info = torchaudio.info(str(out_wav))
        row = {
            **row,
            "tts_model": Path(args.model_dir).name,
            "tts_mode": args.mode,
            "tts_speaker_id": args.speaker_id,
            "tts_instruction": args.instruction,
            "tts_prompt_wav": str(prompt_wav or ""),
            "tts_prompt_text": args.prompt_text,
            "tts_text_frontend": str(text_frontend),
            "tts_speed": f"{args.speed:.6f}",
            "tts_transformers_version": transformers_version,
            "tts_tokenizers_version": tokenizers_version,
            "postprocess_target_sample_rate": str(args.target_sample_rate or ""),
            "postprocess_target_rms_dbfs": "" if args.target_rms_dbfs is None else f"{args.target_rms_dbfs:.6f}",
            "postprocess_output_rms_dbfs": f"{output_rms_dbfs:.6f}",
            "postprocess_output_peak": f"{output_peak:.6f}",
            "norm_sample_rate": str(wav_info.sample_rate),
            "norm_num_frames": str(wav_info.num_frames),
            "norm_duration": f"{wav_info.num_frames / wav_info.sample_rate:.6f}",
            "generation_status": status,
            "generation_seconds": f"{time.time() - start:.3f}",
        }
        generated_rows.append(row)
        print(json.dumps({"idx": idx, "utt_id": row["utt_id"], "status": status, "wav": str(out_wav)}, ensure_ascii=False), flush=True)
        if args.flush_every > 0 and len(generated_rows) % args.flush_every == 0:
            write_csv(Path(args.out_csv), generated_rows)

    write_csv(Path(args.out_csv), generated_rows)


if __name__ == "__main__":
    main()
