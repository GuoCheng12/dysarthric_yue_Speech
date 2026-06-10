#!/usr/bin/env python3
"""Evaluate a Qwen3-ASR LoRA adapter on an Experiment 2 split."""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency `peft`. Install it with: pip install -U peft") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPO_ROOT / "src", Path("/data/qwen3-asr/src")):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

try:
    from eval_asr_dir import edit_distance, text_normalize
except Exception:
    def text_normalize(text):
        return "".join(str(text).lower().split())

    def edit_distance(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--language", default="Cantonese")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--merge-adapter", type=int, default=1)
    return parser.parse_args()


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cer(ref, hyp):
    ref_n = text_normalize(ref or "")
    hyp_n = text_normalize(hyp or "")
    return edit_distance(ref_n, hyp_n) / max(1, len(ref_n))


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def critical_error(pred_text, error_rate):
    return not text_normalize(pred_text or "") or error_rate >= 0.5


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize_group(name, rows):
    tuned = [float(row["fine_tuned_cer"]) for row in rows if row.get("fine_tuned_cer") != ""]
    zero = [float(row["zero_shot_cer"]) for row in rows if row.get("zero_shot_cer") != ""]
    tuned_critical = [
        parse_bool(row["fine_tuned_critical_error"])
        for row in rows
        if row.get("fine_tuned_critical_error") != ""
    ]
    zero_critical = [
        parse_bool(row["zero_shot_critical"])
        for row in rows
        if row.get("zero_shot_critical") != ""
    ]
    tuned_mean = statistics.mean(tuned) if tuned else None
    zero_mean = statistics.mean(zero) if zero else None
    tuned_critical_rate = statistics.mean(tuned_critical) if tuned_critical else None
    zero_critical_rate = statistics.mean(zero_critical) if zero_critical else None
    return {
        "group": name,
        "sample_count": len(rows),
        "zero_shot_cer": round(zero_mean, 6) if zero_mean is not None else "",
        "fine_tuned_cer": round(tuned_mean, 6) if tuned_mean is not None else "",
        "delta_cer": round(tuned_mean - zero_mean, 6) if tuned_mean is not None and zero_mean is not None else "",
        "zero_shot_critical_rate": round(zero_critical_rate, 6) if zero_critical_rate is not None else "",
        "fine_tuned_critical_rate": round(tuned_critical_rate, 6) if tuned_critical_rate is not None else "",
        "delta_critical_rate": (
            round(tuned_critical_rate - zero_critical_rate, 6)
            if tuned_critical_rate is not None and zero_critical_rate is not None
            else ""
        ),
        "zero_shot_critical_count": sum(zero_critical) if zero_critical else "",
        "fine_tuned_critical_count": sum(tuned_critical) if tuned_critical else "",
    }


def summarize(rows):
    out = [summarize_group("overall", rows)]
    for field in ["zero_shot_bucket", "disease_tag", "duration_bucket", "speaker_id"]:
        groups = defaultdict(list)
        for row in rows:
            groups[row.get(field, "")].append(row)
        for key in sorted(groups):
            out.append(summarize_group(f"{field}={key}", groups[key]))
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.split_csv)
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.base_model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    asr_wrapper.model = PeftModel.from_pretrained(asr_wrapper.model, args.adapter_path)
    if args.merge_adapter == 1 and hasattr(asr_wrapper.model, "merge_and_unload"):
        asr_wrapper.model = asr_wrapper.model.merge_and_unload()

    results = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        trans = asr_wrapper.transcribe(
            audio=[row["audio_path"] for row in batch],
            language=[args.language] * len(batch),
        )
        for row, pred in zip(batch, trans):
            pred_text = pred.text
            fine_cer = cer(row["clean_gt"], pred_text)
            result = dict(row)
            result["fine_tuned_pred"] = pred_text
            result["fine_tuned_detected_language"] = pred.language
            result["fine_tuned_cer"] = round(fine_cer, 6)
            result["fine_tuned_critical_error"] = critical_error(pred_text, fine_cer)
            result["delta_cer"] = (
                round(fine_cer - float(row["zero_shot_cer"]), 6)
                if row.get("zero_shot_cer") not in ("", None)
                else ""
            )
            results.append(result)
        print(json.dumps({"done": len(results), "total": len(rows)}, ensure_ascii=False), flush=True)

    fields = list(rows[0].keys()) + [
        "fine_tuned_pred",
        "fine_tuned_detected_language",
        "fine_tuned_cer",
        "fine_tuned_critical_error",
        "delta_cer",
    ]
    write_csv(out_dir / "predictions.csv", results, fields)
    write_csv(
        out_dir / "summary_by_group.csv",
        summarize(results),
        [
            "group",
            "sample_count",
            "zero_shot_cer",
            "fine_tuned_cer",
            "delta_cer",
            "zero_shot_critical_rate",
            "fine_tuned_critical_rate",
            "delta_critical_rate",
            "zero_shot_critical_count",
            "fine_tuned_critical_count",
        ],
    )


if __name__ == "__main__":
    main()
