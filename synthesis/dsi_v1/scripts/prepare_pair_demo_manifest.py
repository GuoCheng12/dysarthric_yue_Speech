#!/usr/bin/env python3
"""Build prompt-disjoint pair-data manifests for DSI V1."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "utt_id",
    "speaker_id",
    "clean_gt",
    "audio",
    "split",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        action="append",
        required=True,
        help="Input prompt-disjoint JSONL manifest. Can be repeated.",
    )
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--tts-root", required=True)
    parser.add_argument("--per-split", type=int, default=2)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use every input row instead of sampling per split.",
    )
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument(
        "--prefer-buckets",
        default="easy,medium,hard",
        help="Comma-separated zero-shot bucket priority for demo diversity.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_COLUMNS - set(row)
            if missing:
                raise ValueError(f"{path}:{line_no} missing columns: {sorted(missing)}")
            row["_source_manifest"] = str(path)
            rows.append(row)
    return rows


def safe_utt_id(utt_id: str) -> str:
    return utt_id.replace("/", "__").replace(" ", "_")


def pick_demo_rows(rows: list[dict[str, Any]], per_split: int, seed: int, prefer_buckets: list[str]) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        split = str(row["split"])
        by_split.setdefault(split, []).append(row)

    picked: list[dict[str, Any]] = []
    for split in ["train", "dev", "test"]:
        split_rows = by_split.get(split, [])
        if not split_rows:
            raise ValueError(f"no rows for split={split}")

        chosen: list[dict[str, Any]] = []
        used_speakers: set[str] = set()
        for bucket in prefer_buckets:
            candidates = [
                r
                for r in split_rows
                if str(r.get("zero_shot_bucket", "")) == bucket
                and str(r["speaker_id"]) not in used_speakers
                and Path(str(r["audio"])).exists()
            ]
            rng.shuffle(candidates)
            if candidates:
                row = candidates[0]
                chosen.append(row)
                used_speakers.add(str(row["speaker_id"]))
            if len(chosen) >= per_split:
                break

        if len(chosen) < per_split:
            candidates = [r for r in split_rows if Path(str(r["audio"])).exists() and r not in chosen]
            rng.shuffle(candidates)
            chosen.extend(candidates[: per_split - len(chosen)])

        if len(chosen) < per_split:
            raise ValueError(f"split={split} has only {len(chosen)} usable rows")
        picked.extend(chosen[:per_split])
    return picked


def pick_all_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing_audio = [str(row["utt_id"]) for row in rows if not Path(str(row["audio"])).exists()]
    if missing_audio:
        preview = ", ".join(missing_audio[:10])
        raise FileNotFoundError(f"{len(missing_audio)} rows have missing audio. First rows: {preview}")
    return sorted(rows, key=lambda row: (str(row["split"]), str(row["speaker_id"]), str(row["utt_id"])))


def build_pair_rows(rows: list[dict[str, Any]], tts_root: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        utt_id = str(row["utt_id"])
        split = str(row["split"])
        norm_wav = tts_root / split / f"{safe_utt_id(utt_id)}.wav"
        out.append(
            {
                "utt_id": utt_id,
                "patient_id": str(row["speaker_id"]),
                "clean_text": str(row["clean_gt"]),
                "jyutping": str(row.get("jyutping", "")),
                "dys_wav_path": str(row["audio"]),
                "norm_wav_path": str(norm_wav),
                "split": split,
                "prompt_id": str(row.get("prompt_id", "")),
                "zero_shot_bucket": str(row.get("zero_shot_bucket", "")),
                "duration": str(row.get("duration", "")),
                "source_manifest": str(row.get("_source_manifest", "")),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    for item in args.input_jsonl:
        all_rows.extend(read_jsonl(Path(item)))

    prefer_buckets = [x.strip() for x in args.prefer_buckets.split(",") if x.strip()]
    picked = pick_all_rows(all_rows) if args.all else pick_demo_rows(all_rows, args.per_split, args.seed, prefer_buckets)
    pair_rows = build_pair_rows(picked, Path(args.tts_root))

    write_csv(Path(args.out_csv), pair_rows)
    write_jsonl(Path(args.out_jsonl), pair_rows)

    counts: dict[str, int] = {}
    for row in pair_rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    print(json.dumps({"rows": len(pair_rows), "counts_by_split": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
