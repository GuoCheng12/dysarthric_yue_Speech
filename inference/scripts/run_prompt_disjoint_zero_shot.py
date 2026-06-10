#!/usr/bin/env python3
"""Run zero-shot Qwen3-ASR inference for a prompt-disjoint manifest.

The script keeps private per-utterance predictions under /data and writes two
main performance tables:

- test_set_performance.csv
- full_dataset_performance.csv

It is resumable: existing rows in predictions_all.jsonl are reused by utt_id.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel


REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPO_ROOT / "src", Path("/data/qwen3-asr/src")):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

try:
    from eval_asr_dir import (
        CHINESE_DIGITS,
        CHINESE_NUMBER_CHARS,
        YUE_REPLACEMENTS,
        edit_distance,
        normalize_text,
        parse_chinese_number,
        text_normalize,
        to_traditional,
    )
except Exception:  # pragma: no cover - only used in degraded local checks
    CHINESE_DIGITS = {}
    CHINESE_NUMBER_CHARS = ""
    YUE_REPLACEMENTS = []

    def text_normalize(text: str) -> str:
        return "".join(str(text).lower().split())

    def normalize_text(text: str) -> str:
        return "".join(str(text).split())

    def to_traditional(text: str) -> str:
        return text

    def parse_chinese_number(text: str) -> int | None:
        return None

    def edit_distance(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


DEFAULT_MANIFEST = "/data/qwen3-asr/inference/outputs/prompt_disjoint_v1/prompt_disjoint_manifest.csv"
DEFAULT_MODEL = "/data/qwen3-asr/models/Qwen3-ASR-1.7B"
DEFAULT_OUT_DIR = "/data/qwen3-asr/inference/outputs/prompt_disjoint_v1/e1_zero_shot"
CRITICAL_THRESHOLD = 0.5

PRIVATE_FIELDS = [
    "utt_id",
    "speaker_id",
    "disease_tag",
    "split",
    "prompt_id",
    "rel_path",
    "audio_path",
    "clean_gt",
    "raw_gt",
    "pred",
    "detected_language",
    "language_arg",
    "textnorm_gt",
    "textnorm_pred",
    "textnorm_char_distance",
    "textnorm_cer",
    "critical_error",
    "critical_error_reason",
    "inference_error",
    "duration",
    "duration_bucket",
    "e1_zero_shot_bucket",
    "source_zero_shot_cer",
    "source_zero_shot_critical",
    "source_zero_shot_bucket",
    "task_type",
    "inference_time_sec",
]

PERFORMANCE_FIELDS = [
    "scope",
    "group_type",
    "group_value",
    "sample_count",
    "inferred_count",
    "inference_error_count",
    "avg_textnorm_cer",
    "avg_critical_error",
    "critical_error_count",
    "critical_error_labeled",
    "avg_duration_sec",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--language", default="Cantonese")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--summary-only", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_jsonl_by_utt(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["utt_id"])] = row
    return rows


def as_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: object) -> bool | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def cer_bucket(cer: float) -> str:
    if cer < 0.2:
        return "easy"
    if cer < 0.5:
        return "medium"
    return "hard"


def safe_normalize_numbers(text: str) -> str:
    if not CHINESE_NUMBER_CHARS:
        return text
    pattern = f"[{re.escape(CHINESE_NUMBER_CHARS)}]{{2,}}"

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        if CHINESE_DIGITS and all(ch in CHINESE_DIGITS for ch in token):
            return "".join(CHINESE_DIGITS[ch] for ch in token)
        try:
            parsed = parse_chinese_number(token)
        except Exception:
            return token
        return str(parsed) if parsed is not None else token

    return re.sub(pattern, replace_token, text)


def safe_text_normalize(text: str) -> str:
    try:
        return text_normalize(text)
    except Exception:
        text = normalize_text(text)
        text = to_traditional(text)
        text = safe_normalize_numbers(text)
        for source, target in YUE_REPLACEMENTS:
            text = text.replace(source, target)
        return text.lower()


def score_prediction(row: dict[str, str], pred: str, inference_error: str = "") -> dict[str, object]:
    ref = row.get("clean_gt") or ""
    textnorm_gt = safe_text_normalize(ref)
    textnorm_pred = safe_text_normalize(pred or "")
    dist = edit_distance(textnorm_gt, textnorm_pred)
    cer = dist / max(1, len(textnorm_gt))

    if inference_error:
        critical = True
        reason = "inference_error"
    elif not textnorm_pred:
        critical = True
        reason = "blank_prediction"
    elif cer >= CRITICAL_THRESHOLD:
        critical = True
        reason = f"textnorm_cer>={CRITICAL_THRESHOLD:g}"
    else:
        critical = False
        reason = ""

    return {
        "textnorm_gt": textnorm_gt,
        "textnorm_pred": textnorm_pred,
        "textnorm_char_distance": dist,
        "textnorm_cer": round(cer, 6),
        "critical_error": critical,
        "critical_error_reason": reason,
        "e1_zero_shot_bucket": cer_bucket(cer),
    }


def build_private_row(
    source: dict[str, str],
    pred: str,
    detected_language: object,
    language: str,
    inference_time_sec: float | str,
    inference_error: str = "",
) -> dict[str, object]:
    scored = score_prediction(source, pred, inference_error)
    return {
        "utt_id": source["utt_id"],
        "speaker_id": source.get("speaker_id", ""),
        "disease_tag": source.get("disease_tag", ""),
        "split": source.get("split", ""),
        "prompt_id": source.get("prompt_id", ""),
        "rel_path": source.get("rel_path", ""),
        "audio_path": source.get("audio_path", ""),
        "clean_gt": source.get("clean_gt", ""),
        "raw_gt": source.get("raw_gt", ""),
        "pred": pred,
        "detected_language": detected_language or "",
        "language_arg": language,
        **scored,
        "inference_error": inference_error,
        "duration": source.get("duration", ""),
        "duration_bucket": source.get("duration_bucket", ""),
        "source_zero_shot_cer": source.get("zero_shot_cer", ""),
        "source_zero_shot_critical": source.get("zero_shot_critical", ""),
        "source_zero_shot_bucket": source.get("zero_shot_bucket", ""),
        "task_type": source.get("task_type", ""),
        "inference_time_sec": inference_time_sec,
    }


def format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize(scope: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = [summarize_group(scope, "overall", "overall", rows)]
    for field in ("e1_zero_shot_bucket", "disease_tag", "duration_bucket", "split"):
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(field, ""))].append(row)
        for key in sorted(groups):
            out.append(summarize_group(scope, field, key, groups[key]))
    return out


def summarize_group(
    scope: str,
    group_type: str,
    group_value: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    cers = [v for v in (as_float(row.get("textnorm_cer")) for row in rows) if v is not None]
    critical = [v for v in (as_bool(row.get("critical_error")) for row in rows) if v is not None]
    durations = [v for v in (as_float(row.get("duration")) for row in rows) if v is not None and not math.isnan(v)]
    avg_cer = mean(cers)
    avg_critical = mean([float(v) for v in critical])
    avg_duration = mean(durations)
    return {
        "scope": scope,
        "group_type": group_type,
        "group_value": group_value,
        "sample_count": len(rows),
        "inferred_count": sum(1 for row in rows if row.get("pred")),
        "inference_error_count": sum(1 for row in rows if row.get("inference_error")),
        "avg_textnorm_cer": round(avg_cer, 6) if avg_cer is not None else "",
        "avg_critical_error": round(avg_critical, 6) if avg_critical is not None else "",
        "critical_error_count": int(sum(critical)) if critical else "",
        "critical_error_labeled": len(critical),
        "avg_duration_sec": round(avg_duration, 6) if avg_duration is not None else "",
    }


def write_outputs(out_dir: Path, manifest_rows: list[dict[str, str]], results: dict[str, dict[str, object]]) -> None:
    all_rows = [results[row["utt_id"]] for row in manifest_rows if row["utt_id"] in results]
    all_rows.sort(key=lambda row: (str(row.get("split", "")), str(row.get("utt_id", ""))))
    test_rows = [row for row in all_rows if row.get("split") == "test"]

    write_csv(out_dir / "predictions_all_private.csv", all_rows, PRIVATE_FIELDS)
    write_csv(out_dir / "predictions_test_private.csv", test_rows, PRIVATE_FIELDS)
    write_csv(out_dir / "full_dataset_performance.csv", summarize("full_dataset", all_rows), PERFORMANCE_FIELDS)
    write_csv(out_dir / "test_set_performance.csv", summarize("test_set", test_rows), PERFORMANCE_FIELDS)

    report = {
        "all_predictions": len(all_rows),
        "test_predictions": len(test_rows),
        "full_dataset_performance": str(out_dir / "full_dataset_performance.csv"),
        "test_set_performance": str(out_dir / "test_set_performance.csv"),
        "critical_rule": f"inference_error_or_blank_or_textnorm_cer>={CRITICAL_THRESHOLD:g}",
    }
    (out_dir / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(manifest_path)
    if args.limit:
        manifest_rows = manifest_rows[: args.limit]
    if not manifest_rows:
        raise SystemExit(f"Empty manifest: {manifest_path}")

    jsonl_path = out_dir / "predictions_all.jsonl"
    results = read_jsonl_by_utt(jsonl_path)
    remaining = [row for row in manifest_rows if row["utt_id"] not in results]
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "rows": len(manifest_rows),
                "existing_results": len(results),
                "remaining": len(remaining),
                "jsonl": str(jsonl_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.summary_only:
        write_outputs(out_dir, manifest_rows, results)
        return 0

    if remaining:
        if not torch.cuda.is_available():
            raise SystemExit("GPU is required for this inference pass, but torch.cuda.is_available() is False.")

        model = Qwen3ASRModel.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

        with jsonl_path.open("a", encoding="utf-8") as f:
            for start in range(0, len(remaining), args.batch_size):
                batch = remaining[start : start + args.batch_size]
                batch_start = time.time()
                print(f"running {start + 1}-{start + len(batch)} / {len(remaining)} remaining", flush=True)
                try:
                    transcriptions = model.transcribe(
                        audio=[row["audio_path"] for row in batch],
                        language=[args.language] * len(batch),
                    )
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "batch_error": format_exception(exc),
                                "fallback": "single_file",
                                "start": start + 1,
                                "end": start + len(batch),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    for row in batch:
                        single_start = time.time()
                        try:
                            transcription = model.transcribe(audio=[row["audio_path"]], language=[args.language])[0]
                            pred = getattr(transcription, "text", "") or ""
                            detected_language = getattr(transcription, "language", "")
                            result = build_private_row(
                                row,
                                pred,
                                detected_language,
                                args.language,
                                round(time.time() - single_start, 3),
                            )
                        except Exception as single_exc:
                            result = build_private_row(
                                row,
                                "",
                                "",
                                args.language,
                                round(time.time() - single_start, 3),
                                format_exception(single_exc),
                            )
                            print(
                                json.dumps(
                                    {"file_error": result["inference_error"], "utt_id": row["utt_id"]},
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                        results[row["utt_id"]] = result
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                else:
                    elapsed = round(time.time() - batch_start, 3)
                    for row, transcription in zip(batch, transcriptions):
                        pred = getattr(transcription, "text", "") or ""
                        detected_language = getattr(transcription, "language", "")
                        result = build_private_row(row, pred, detected_language, args.language, elapsed)
                        results[row["utt_id"]] = result
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()

    write_outputs(out_dir, manifest_rows, results)
    print(
        json.dumps(
            {
                "completed": len(results),
                "out_dir": str(out_dir),
                "test_set_performance": str(out_dir / "test_set_performance.csv"),
                "full_dataset_performance": str(out_dir / "full_dataset_performance.csv"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
