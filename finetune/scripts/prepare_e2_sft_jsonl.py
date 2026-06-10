#!/usr/bin/env python3
"""Convert Experiment 2 clean manifest CSV into Qwen3-ASR SFT JSONL files."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "dev", "test")
PREFIX_TEMPLATE = "language {language}<asr_text>"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/data/qwen3-asr/inference/outputs/experiment2/clean_manifest.csv")
    parser.add_argument("--out-dir", default="/data/qwen3-asr/finetune/data")
    parser.add_argument("--language", default="Cantonese")
    parser.add_argument("--smoke-train-size", type=int, default=32)
    parser.add_argument("--smoke-dev-size", type=int, default=16)
    return parser.parse_args()


def read_manifest(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def is_trainable(row):
    return (
        row.get("main_train_eligible") == "true"
        and row.get("task_type") == "read_sentence"
        and row.get("exclude_from_training") == "false"
        and row.get("clean_gt")
        and row.get("audio_path")
        and row.get("split") in SPLITS
    )


def make_json_row(row, language):
    prefix = PREFIX_TEMPLATE.format(language=language)
    return {
        "audio": row["audio_path"],
        "text": prefix + row["clean_gt"],
        "prompt": "",
        "utt_id": row["utt_id"],
        "speaker_id": row["speaker_id"],
        "disease_tag": row["disease_tag"],
        "task_type": row["task_type"],
        "duration": row["duration"],
        "duration_bucket": row["duration_bucket"],
        "zero_shot_cer": row["zero_shot_cer"],
        "zero_shot_critical": row["zero_shot_critical"],
        "zero_shot_bucket": row["zero_shot_bucket"],
        "clean_gt": row["clean_gt"],
        "raw_gt": row["raw_gt"],
        "split": row["split"],
        "prompt_id": row["prompt_id"],
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_smoke(rows, target_size):
    by_bucket = defaultdict(list)
    for row in sorted(rows, key=lambda x: (x["zero_shot_bucket"], x["utt_id"])):
        by_bucket[row["zero_shot_bucket"]].append(row)

    order = ["hard", "medium", "easy", "unknown"]
    selected = []
    while len(selected) < target_size:
        advanced = False
        for bucket in order:
            if by_bucket[bucket]:
                selected.append(by_bucket[bucket].pop(0))
                advanced = True
                if len(selected) >= target_size:
                    break
        if not advanced:
            break
    return selected


def validate(rows):
    errors = []
    seen = set()
    for row in rows:
        if not Path(row["audio"]).exists():
            errors.append(f"missing_audio:{row['utt_id']}:{row['audio']}")
        if not row["text"].startswith("language Cantonese<asr_text>"):
            errors.append(f"bad_prefix:{row['utt_id']}")
        if not row["clean_gt"]:
            errors.append(f"empty_clean_gt:{row['utt_id']}")
        key = (row["speaker_id"], row["prompt_id"], row["split"])
        if key in seen:
            pass
        seen.add(key)
    return errors


def write_report(path, all_rows, split_rows, smoke_train, smoke_dev, errors):
    lines = ["# Experiment 2 SFT JSONL Report", ""]
    lines.append("## Files")
    for split in SPLITS:
        lines.append(f"- `e2_{split}.jsonl`: {len(split_rows[split])} rows")
    lines.append(f"- `e2_smoke_train.jsonl`: {len(smoke_train)} rows")
    lines.append(f"- `e2_smoke_dev.jsonl`: {len(smoke_dev)} rows")
    lines.append("")
    lines.append("## Counts")
    lines.append("| split | rows |")
    lines.append("|---|---:|")
    for split in SPLITS:
        lines.append(f"| {split} | {len(split_rows[split])} |")
    lines.append("")
    lines.append("## Zero-Shot Bucket by Split")
    lines.append("| split | bucket | rows |")
    lines.append("|---|---|---:|")
    for split in SPLITS:
        counts = Counter(row["zero_shot_bucket"] for row in split_rows[split])
        for bucket, count in sorted(counts.items()):
            lines.append(f"| {split} | {bucket} | {count} |")
    lines.append("")
    lines.append("## Disease Tag by Split")
    lines.append("| split | disease_tag | rows |")
    lines.append("|---|---|---:|")
    for split in SPLITS:
        counts = Counter(row["disease_tag"] for row in split_rows[split])
        for disease, count in sorted(counts.items()):
            lines.append(f"| {split} | {disease} | {count} |")
    lines.append("")
    lines.append("## Validation")
    if errors:
        lines.append(f"- Errors: {len(errors)}")
        for item in errors[:50]:
            lines.append(f"  - `{item}`")
    else:
        lines.append("- Errors: 0")
    lines.append("")
    lines.append("## Training Target")
    lines.append("- Use `text` as the target for the official Qwen3-ASR SFT script.")
    lines.append("- `text` is `language Cantonese<asr_text>` + `clean_gt`.")
    lines.append("- Metadata fields are kept in JSONL for traceability; the official script removes unused columns.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    rows = [row for row in read_manifest(args.manifest) if is_trainable(row)]
    json_rows = [make_json_row(row, args.language) for row in rows]

    split_rows = {split: [] for split in SPLITS}
    for row in json_rows:
        split_rows[row["split"]].append(row)

    for split in SPLITS:
        write_jsonl(out_dir / f"e2_{split}.jsonl", split_rows[split])

    smoke_train = stratified_smoke(split_rows["train"], args.smoke_train_size)
    smoke_dev = stratified_smoke(split_rows["dev"], args.smoke_dev_size)
    write_jsonl(out_dir / "e2_smoke_train.jsonl", smoke_train)
    write_jsonl(out_dir / "e2_smoke_dev.jsonl", smoke_dev)

    errors = validate(json_rows)
    write_report(out_dir / "e2_sft_jsonl_report.md", json_rows, split_rows, smoke_train, smoke_dev, errors)

    print(f"train/dev/test: {len(split_rows['train'])}/{len(split_rows['dev'])}/{len(split_rows['test'])}")
    print(f"smoke train/dev: {len(smoke_train)}/{len(smoke_dev)}")
    print(f"validation_errors: {len(errors)}")
    if errors:
        for item in errors[:20]:
            print(item)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
