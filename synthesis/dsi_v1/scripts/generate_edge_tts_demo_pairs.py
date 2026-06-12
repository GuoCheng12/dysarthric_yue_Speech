#!/usr/bin/env python3
"""Generate normal Hong Kong Cantonese TTS waveforms with edge-tts."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import edge_tts
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--voice", default="zh-HK-HiuGaaiNeural")
    parser.add_argument("--rate", default="+0%")
    parser.add_argument("--volume", default="+0%")
    parser.add_argument("--pitch", default="+0Hz")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-mp3", action="store_true")
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


async def synthesize_mp3(text: str, out_mp3: Path, voice: str, rate: str, volume: str, pitch: str) -> None:
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
    )
    await communicate.save(str(out_mp3))


def mp3_to_wav(mp3_path: Path, wav_path: Path) -> tuple[int, int]:
    wav, sr = torchaudio.load(str(mp3_path))
    torchaudio.save(str(wav_path), wav, sr)
    info = torchaudio.info(str(wav_path))
    return info.sample_rate, info.num_frames


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.pair_manifest))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("pair manifest has no rows")

    generated_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        out_wav = Path(row["norm_wav_path"])
        out_mp3 = out_wav.with_suffix(".mp3")
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        start = time.time()
        status = "exists"
        if args.overwrite or not out_wav.exists():
            asyncio.run(
                synthesize_mp3(
                    row["clean_text"],
                    out_mp3,
                    voice=args.voice,
                    rate=args.rate,
                    volume=args.volume,
                    pitch=args.pitch,
                )
            )
            sample_rate, num_frames = mp3_to_wav(out_mp3, out_wav)
            if not args.keep_mp3:
                out_mp3.unlink(missing_ok=True)
            status = "generated"
        else:
            info = torchaudio.info(str(out_wav))
            sample_rate, num_frames = info.sample_rate, info.num_frames

        out_row = {
            **row,
            "tts_backend": "edge-tts",
            "tts_voice": args.voice,
            "tts_rate": args.rate,
            "tts_volume": args.volume,
            "tts_pitch": args.pitch,
            "norm_sample_rate": str(sample_rate),
            "norm_num_frames": str(num_frames),
            "norm_duration": f"{num_frames / sample_rate:.6f}",
            "generation_status": status,
            "generation_seconds": f"{time.time() - start:.3f}",
        }
        generated_rows.append(out_row)
        print(
            json.dumps(
                {"idx": idx, "utt_id": row["utt_id"], "status": status, "wav": str(out_wav)},
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_csv(Path(args.out_csv), generated_rows)


if __name__ == "__main__":
    main()
