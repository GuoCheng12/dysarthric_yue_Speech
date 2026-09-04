#!/usr/bin/env python3
"""Decode one frozen Setting D JSONL split with base Qwen3-ASR or one LoRA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from qwen_asr import Qwen3ASRModel


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from therapist_harness.evidence import evaluate_snapshot  # noqa: E402
from therapist_harness.text import edit_distance, text_normalize  # noqa: E402


REQUIRED_FIELDS = (
    "utt_id",
    "speaker_id",
    "prompt_id",
    "clean_gt",
    "zero_shot_bucket",
    "audio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--split-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--language", default="Cantonese", choices=("Cantonese",))
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--critical-cer-threshold", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_split(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"line {line_number}: blank rows are forbidden")
            try:
                row = json.loads(line, object_pairs_hook=_json_object)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"line {line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(f"line {line_number}: row must be an object")
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(f"line {line_number}: missing fields {missing}")
            if any(
                not isinstance(row[field], str) or not row[field].strip()
                for field in REQUIRED_FIELDS
            ):
                raise ValueError(
                    f"line {line_number}: required fields must be non-empty strings"
                )
            utterance_id = row["utt_id"].strip()
            if utterance_id in seen_ids:
                raise ValueError(f"duplicate utt_id: {utterance_id}")
            audio = Path(row["audio"])
            if not audio.is_absolute() or not audio.is_file():
                raise FileNotFoundError(f"invalid audio for {utterance_id}: {audio}")
            seen_ids.add(utterance_id)
            rows.append(row)
    if not rows:
        raise ValueError("split is empty")
    return rows


def load_model(args: argparse.Namespace) -> Qwen3ASRModel:
    if not torch.cuda.is_available():
        raise RuntimeError("the maintained evaluator requires one CUDA GPU")
    if not args.base_model_path.is_dir():
        raise FileNotFoundError(args.base_model_path)
    wrapper = Qwen3ASRModel.from_pretrained(
        str(args.base_model_path),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    if args.adapter_path is not None:
        if not args.adapter_path.is_dir():
            raise FileNotFoundError(args.adapter_path)
        try:
            from peft import PeftModel
        except ImportError as error:  # pragma: no cover - environment diagnostic
            raise RuntimeError("PEFT is required to evaluate an adapter") from error
        wrapper.model = PeftModel.from_pretrained(
            wrapper.model, str(args.adapter_path)
        ).merge_and_unload()
    return wrapper


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    jsonl = output_dir / "predictions.jsonl"
    with jsonl.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    fields = list(rows[0])
    with (output_dir / "predictions.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows = read_split(args.split_jsonl)
    model = load_model(args)
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        decoded = list(
            model.transcribe(
                audio=[row["audio"] for row in batch],
                language=[args.language] * len(batch),
            )
        )
        if len(decoded) != len(batch):
            raise RuntimeError(
                f"decoder returned {len(decoded)} results for batch size {len(batch)}"
            )
        for row, result in zip(batch, decoded, strict=True):
            prediction = result.text
            if not isinstance(prediction, str):
                raise TypeError(f"decoder returned non-string text for {row['utt_id']}")
            reference = text_normalize(row["clean_gt"])
            normalized_prediction = text_normalize(prediction)
            edits = edit_distance(reference, normalized_prediction)
            cer = edits / len(reference)
            predictions.append(
                {
                    **row,
                    "prediction": prediction,
                    "detected_language": result.language,
                    "normalized_reference_characters": len(reference),
                    "edit_count": edits,
                    "cer": cer,
                    "critical": (
                        not normalized_prediction or cer >= args.critical_cer_threshold
                    ),
                    "exact": edits == 0,
                }
            )
        print(
            json.dumps({"decoded": len(predictions), "total": len(rows)}),
            flush=True,
        )

    if len(predictions) != len(rows):
        raise RuntimeError("prediction count differs from the frozen split")
    metrics = evaluate_snapshot(
        predictions,
        prediction_column="prediction",
        critical_cer_threshold=args.critical_cer_threshold,
    )
    summary = {
        "schema_version": "1",
        "split_path": str(args.split_jsonl.resolve()),
        "split_sha256": sha256_file(args.split_jsonl),
        "base_model_path": str(args.base_model_path.resolve()),
        "adapter_path": (
            str(args.adapter_path.resolve()) if args.adapter_path is not None else None
        ),
        "language": args.language,
        "critical_cer_threshold": args.critical_cer_threshold,
        "prediction_count": len(predictions),
        "metrics": metrics.model_dump(mode="json"),
    }
    write_outputs(args.out_dir, predictions, summary)


if __name__ == "__main__":
    main()
