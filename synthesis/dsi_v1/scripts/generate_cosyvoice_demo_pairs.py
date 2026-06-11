#!/usr/bin/env python3
"""Generate normal Cantonese TTS waveforms for a DSI V1 pair manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--cosyvoice-repo", required=True)
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
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of rows to synthesize.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Enable CosyVoice fp16 inference.")
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


def import_cosyvoice(cosyvoice_repo: Path):
    matcha = cosyvoice_repo / "third_party" / "Matcha-TTS"
    sys.path.insert(0, str(cosyvoice_repo))
    sys.path.insert(0, str(matcha))
    from cosyvoice.cli.cosyvoice import AutoModel  # noqa: PLC0415

    return AutoModel


def concat_outputs(outputs: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    speeches = [item["tts_speech"].detach().cpu() for item in outputs]
    if not speeches:
        raise RuntimeError("CosyVoice returned no speech chunks")
    speeches = [x if x.dim() == 2 else x.reshape(1, -1) for x in speeches]
    return torch.cat(speeches, dim=1)


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

    AutoModel = import_cosyvoice(Path(args.cosyvoice_repo))
    model = AutoModel(model_dir=args.model_dir, fp16=args.fp16)

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
                        text_frontend=True,
                    )
                )
            elif args.mode == "sft":
                outputs = list(
                    model.inference_sft(
                        text,
                        args.speaker_id,
                        stream=False,
                        text_frontend=True,
                    )
                )
            else:
                outputs = list(
                    model.inference_instruct2(
                        text,
                        args.instruction,
                        str(prompt_wav),
                        stream=False,
                        text_frontend=True,
                    )
                )
            speech = concat_outputs(outputs)
            torchaudio.save(str(out_wav), speech, model.sample_rate)
            status = "generated"

        wav_info = torchaudio.info(str(out_wav))
        row = {
            **row,
            "tts_model": Path(args.model_dir).name,
            "tts_mode": args.mode,
            "tts_speaker_id": args.speaker_id,
            "tts_instruction": args.instruction,
            "tts_prompt_wav": str(prompt_wav or ""),
            "tts_prompt_text": args.prompt_text,
            "norm_sample_rate": str(wav_info.sample_rate),
            "norm_num_frames": str(wav_info.num_frames),
            "norm_duration": f"{wav_info.num_frames / wav_info.sample_rate:.6f}",
            "generation_status": status,
            "generation_seconds": f"{time.time() - start:.3f}",
        }
        generated_rows.append(row)
        print(json.dumps({"idx": idx, "utt_id": row["utt_id"], "status": status, "wav": str(out_wav)}, ensure_ascii=False), flush=True)

    write_csv(Path(args.out_csv), generated_rows)


if __name__ == "__main__":
    main()
