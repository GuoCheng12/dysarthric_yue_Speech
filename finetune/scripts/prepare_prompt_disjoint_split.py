#!/usr/bin/env python3
"""Prepare prompt-disjoint clean read-sentence splits for Qwen3-ASR SFT.

Prompt groups are connected components over three text identity layers:

- raw `clean_gt`
- normalized `clean_gt`
- Jyutping sequence

This guarantees that a prompt-equivalent sentence cannot appear in more than one
split even when Cantonese orthography differs but pronunciation is identical.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import check_prompt_leakage as prompt_norm


SPLITS = ("train", "dev", "test")
PREFIX_TEMPLATE = "language {language}<asr_text>"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", action="append", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--jsonl-out-dir", required=True)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--trials", type=int, default=30000)
    p.add_argument("--language", default="Cantonese")
    p.add_argument("--smoke-train-size", type=int, default=32)
    p.add_argument("--smoke-dev-size", type=int, default=16)
    return p.parse_args()


def read_csvs(paths: list[str]) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                utt_id = row["utt_id"]
                if utt_id in seen:
                    duplicates.append(utt_id)
                    continue
                seen.add(utt_id)
                rows.append(row)
    return rows, duplicates


def sha16(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def text_keys(row: dict[str, str]) -> dict[str, str]:
    raw_clean = (row.get("clean_gt") or "").strip()
    normalized = prompt_norm.text_normalize(raw_clean)
    jyutping = prompt_norm.jyutping_sequence(raw_clean)
    return {
        "raw_clean": raw_clean,
        "normalized": normalized,
        "jyutping": jyutping,
    }


def assign_prompt_ids(rows: list[dict[str, str]]) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    uf = UnionFind()
    row_keys: dict[str, dict[str, str]] = {}
    key_to_nodes: dict[tuple[str, str], list[str]] = defaultdict(list)

    for row in rows:
        node = f"utt:{row['utt_id']}"
        keys = text_keys(row)
        row_keys[row["utt_id"]] = keys
        uf.find(node)
        for layer, key in keys.items():
            layer_node = f"{layer}:{key}"
            uf.union(node, layer_node)
            key_to_nodes[(layer, key)].append(node)

    component_members: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        root = uf.find(f"utt:{row['utt_id']}")
        component_members[root].append(row)

    root_to_prompt_id: dict[str, str] = {}
    prompt_key_rows: list[dict[str, str]] = []
    for root, members in component_members.items():
        raws = sorted({row_keys[row["utt_id"]]["raw_clean"] for row in members})
        normalized = sorted({row_keys[row["utt_id"]]["normalized"] for row in members})
        jyutping = sorted({row_keys[row["utt_id"]]["jyutping"] for row in members})
        prompt_key = "RAW=" + "||".join(raws) + "\nNORM=" + "||".join(normalized) + "\nJYUT=" + "||".join(jyutping)
        prompt_id = sha16(prompt_key)
        root_to_prompt_id[root] = prompt_id
        prompt_key_rows.append(
            {
                "prompt_id": prompt_id,
                "sample_count": str(len(members)),
                "raw_clean_variant_count": str(len(raws)),
                "normalized_variant_count": str(len(normalized)),
                "jyutping_variant_count": str(len(jyutping)),
                "raw_clean_hashes": ";".join(sha16(x) for x in raws),
                "normalized_hashes": ";".join(sha16(x) for x in normalized),
                "jyutping_hashes": ";".join(sha16(x) for x in jyutping),
            }
        )

    by_prompt: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        keys = row_keys[row["utt_id"]]
        root = uf.find(f"utt:{row['utt_id']}")
        prompt_id = root_to_prompt_id[root]
        enriched = dict(row)
        enriched["prompt_id"] = prompt_id
        enriched["prompt_raw_clean_sha16"] = sha16(keys["raw_clean"])
        enriched["prompt_normalized_sha16"] = sha16(keys["normalized"])
        enriched["prompt_jyutping_sha16"] = sha16(keys["jyutping"])
        by_prompt[prompt_id].append(enriched)

    return dict(by_prompt), prompt_key_rows


def split_prompt_counts(prompt_count: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    dev_count = round(prompt_count * ratios[1])
    test_count = round(prompt_count * ratios[2])
    train_count = prompt_count - dev_count - test_count
    return {"train": train_count, "dev": dev_count, "test": test_count}


def flatten_split(by_prompt: dict[str, list[dict[str, str]]], assignment: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    out = {split: [] for split in SPLITS}
    for prompt_id, split in assignment.items():
        out[split].extend(by_prompt[prompt_id])
    for split in SPLITS:
        out[split].sort(key=lambda row: row["utt_id"])
        for row in out[split]:
            row["split"] = split
    return out


def count_field(rows: list[dict[str, str]], field: str) -> Counter:
    return Counter(row.get(field, "") for row in rows)


def score_split(
    split_rows: dict[str, list[dict[str, str]]],
    all_rows: list[dict[str, str]],
    ratios: tuple[float, float, float],
) -> float:
    total = len(all_rows)
    score = 0.0
    for split, ratio in zip(SPLITS, ratios):
        target = total * ratio
        score += 20.0 * abs(len(split_rows[split]) - target) / max(1.0, target)

    for field, weight in [("zero_shot_bucket", 7.0), ("duration_bucket", 3.0), ("disease_tag", 2.0)]:
        global_counts = count_field(all_rows, field)
        global_props = {key: value / total for key, value in global_counts.items()}
        for split in SPLITS:
            rows = split_rows[split]
            counts = count_field(rows, field)
            n = max(1, len(rows))
            for key, global_prop in global_props.items():
                score += weight * abs(counts.get(key, 0) / n - global_prop)

    for split in ("dev", "test"):
        rows = split_rows[split]
        distinct_speakers = len({row["speaker_id"] for row in rows})
        hard_count = sum(row["zero_shot_bucket"] == "hard" for row in rows)
        kaho_count = sum(row["speaker_id"] == "vlink_Kaho" for row in rows)
        score += max(0, 25 - distinct_speakers) * 1.0
        score += max(0, 35 - hard_count) * 0.3
        score += max(0, 3 - kaho_count) * 2.0
    return score


def search_assignment(
    by_prompt: dict[str, list[dict[str, str]]],
    ratios: tuple[float, float, float],
    seed: int,
    trials: int,
) -> tuple[dict[str, str], float]:
    prompt_ids = sorted(by_prompt)
    counts = split_prompt_counts(len(prompt_ids), ratios)
    all_rows = [row for rows in by_prompt.values() for row in rows]
    rng = random.Random(seed)
    best_assignment: dict[str, str] | None = None
    best_score = math.inf

    for _ in range(trials):
        shuffled = prompt_ids[:]
        rng.shuffle(shuffled)
        assignment: dict[str, str] = {}
        train_end = counts["train"]
        dev_end = train_end + counts["dev"]
        for prompt_id in shuffled[:train_end]:
            assignment[prompt_id] = "train"
        for prompt_id in shuffled[train_end:dev_end]:
            assignment[prompt_id] = "dev"
        for prompt_id in shuffled[dev_end:]:
            assignment[prompt_id] = "test"
        split_rows = flatten_split(by_prompt, assignment)
        score = score_split(split_rows, all_rows, ratios)
        if score < best_score:
            best_assignment = assignment
            best_score = score

    assert best_assignment is not None
    return best_assignment, best_score


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_json_row(row: dict[str, str], language: str) -> dict[str, str]:
    return {
        "audio": row["audio_path"],
        "text": PREFIX_TEMPLATE.format(language=language) + row["clean_gt"],
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


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_smoke(rows: list[dict[str, str]], target_size: int) -> list[dict[str, str]]:
    by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=lambda x: (x["zero_shot_bucket"], x["utt_id"])):
        by_bucket[row["zero_shot_bucket"]].append(row)

    selected: list[dict[str, str]] = []
    while len(selected) < target_size:
        advanced = False
        for bucket in ("hard", "medium", "easy", "unknown"):
            if by_bucket[bucket]:
                selected.append(by_bucket[bucket].pop(0))
                advanced = True
                if len(selected) >= target_size:
                    break
        if not advanced:
            break
    return selected


def split_audit_rows(
    split_rows: dict[str, list[dict[str, str]]],
    assignment: dict[str, str],
) -> list[dict[str, object]]:
    out = []
    for split in SPLITS:
        rows = split_rows[split]
        zero = count_field(rows, "zero_shot_bucket")
        duration = count_field(rows, "duration_bucket")
        disease = count_field(rows, "disease_tag")
        out.append(
            {
                "split": split,
                "sample_count": len(rows),
                "prompt_count": sum(value == split for value in assignment.values()),
                "speaker_count": len({row["speaker_id"] for row in rows}),
                "kaho_count": sum(row["speaker_id"] == "vlink_Kaho" for row in rows),
                "zero_easy": zero.get("easy", 0),
                "zero_medium": zero.get("medium", 0),
                "zero_hard": zero.get("hard", 0),
                "duration_very_short": duration.get("very_short", 0),
                "duration_short": duration.get("short", 0),
                "duration_medium": duration.get("medium", 0),
                "duration_long": duration.get("long", 0),
                "duration_very_long": duration.get("very_long", 0),
                "disease_HD": disease.get("HD", 0),
                "disease_SCA": disease.get("SCA", 0),
                "disease_unknown": disease.get("unknown", 0),
            }
        )
    return out


def speaker_count_rows(split_rows: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    out = []
    for split in SPLITS:
        by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in split_rows[split]:
            by_speaker[row["speaker_id"]].append(row)
        for speaker_id, rows in sorted(by_speaker.items()):
            buckets = count_field(rows, "zero_shot_bucket")
            out.append(
                {
                    "split": split,
                    "speaker_id": speaker_id,
                    "disease_tag": rows[0].get("disease_tag", ""),
                    "sample_count": len(rows),
                    "zero_easy": buckets.get("easy", 0),
                    "zero_medium": buckets.get("medium", 0),
                    "zero_hard": buckets.get("hard", 0),
                }
            )
    return out


def overlap_audit(split_rows: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    rows = []
    for layer in ("prompt_id", "prompt_raw_clean_sha16", "prompt_normalized_sha16", "prompt_jyutping_sha16"):
        sets = {split: {row[layer] for row in split_rows[split]} for split in SPLITS}
        rows.append(
            {
                "layer": layer,
                "train_dev_overlap": len(sets["train"] & sets["dev"]),
                "train_test_overlap": len(sets["train"] & sets["test"]),
                "dev_test_overlap": len(sets["dev"] & sets["test"]),
            }
        )
    return rows


def write_report(path: Path, audit: list[dict[str, object]], overlap: list[dict[str, object]], duplicates: list[str], score: float) -> None:
    lines = ["# Prompt-Disjoint Split V1 Report", ""]
    lines.append(f"- split search score: `{score:.6f}`")
    lines.append(f"- duplicate input `utt_id` rows skipped: `{len(duplicates)}`")
    if duplicates:
        lines.append(f"- duplicate preview: `{', '.join(duplicates[:10])}`")
    lines.append("")
    lines.append("## Split Counts")
    lines.append("")
    lines.append("| split | samples | prompts | speakers | Kaho | easy | medium | hard | HD | SCA | unknown |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in audit:
        lines.append(
            f"| {row['split']} | {row['sample_count']} | {row['prompt_count']} | "
            f"{row['speaker_count']} | {row['kaho_count']} | {row['zero_easy']} | "
            f"{row['zero_medium']} | {row['zero_hard']} | {row['disease_HD']} | "
            f"{row['disease_SCA']} | {row['disease_unknown']} |"
        )
    lines.append("")
    lines.append("## Prompt Overlap Audit")
    lines.append("")
    lines.append("| layer | train-dev | train-test | dev-test |")
    lines.append("|---|---:|---:|---:|")
    for row in overlap:
        lines.append(f"| {row['layer']} | {row['train_dev_overlap']} | {row['train_test_overlap']} | {row['dev_test_overlap']} |")
    lines.append("")
    lines.append("Step 4 has not been started. Use these split files for the next zero-shot, full-SFT, and LoRA runs after explicit confirmation.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ratios = (args.train_ratio, args.dev_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit(f"Ratios must sum to 1.0, got {ratios}")

    out_dir = Path(args.out_dir)
    jsonl_out_dir = Path(args.jsonl_out_dir)
    rows, duplicates = read_csvs(args.input_csv)
    by_prompt, prompt_key_rows = assign_prompt_ids(rows)
    assignment, best_score = search_assignment(by_prompt, ratios, args.seed, args.trials)
    split_rows = flatten_split(by_prompt, assignment)

    manifest_rows = []
    for split in SPLITS:
        manifest_rows.extend(split_rows[split])
    manifest_rows.sort(key=lambda row: (row["split"], row["utt_id"]))
    fieldnames = list(manifest_rows[0].keys())
    write_csv(out_dir / "prompt_disjoint_manifest.csv", manifest_rows, fieldnames)
    for split in SPLITS:
        write_csv(out_dir / f"prompt_disjoint_{split}.csv", split_rows[split], fieldnames)

    prompt_rows = []
    for prompt_id, rows_for_prompt in sorted(by_prompt.items()):
        split = assignment[prompt_id]
        buckets = count_field(rows_for_prompt, "zero_shot_bucket")
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "split": split,
                "sample_count": len(rows_for_prompt),
                "speaker_count": len({row["speaker_id"] for row in rows_for_prompt}),
                "kaho_count": sum(row["speaker_id"] == "vlink_Kaho" for row in rows_for_prompt),
                "zero_easy": buckets.get("easy", 0),
                "zero_medium": buckets.get("medium", 0),
                "zero_hard": buckets.get("hard", 0),
            }
        )
    write_csv(out_dir / "prompt_group_split_assignment.csv", prompt_rows)
    write_csv(out_dir / "prompt_group_key_audit_sanitized.csv", sorted(prompt_key_rows, key=lambda r: r["prompt_id"]))

    audit = split_audit_rows(split_rows, assignment)
    overlap = overlap_audit(split_rows)
    write_csv(out_dir / "split_audit_by_split.csv", audit)
    write_csv(out_dir / "split_speaker_counts.csv", speaker_count_rows(split_rows))
    write_csv(out_dir / "prompt_overlap_audit.csv", overlap)

    for split in SPLITS:
        json_rows = [make_json_row(row, args.language) for row in split_rows[split]]
        write_jsonl(jsonl_out_dir / f"prompt_disjoint_{split}.jsonl", json_rows)
    smoke_train = [make_json_row(row, args.language) for row in stratified_smoke(split_rows["train"], args.smoke_train_size)]
    smoke_dev = [make_json_row(row, args.language) for row in stratified_smoke(split_rows["dev"], args.smoke_dev_size)]
    write_jsonl(jsonl_out_dir / "prompt_disjoint_smoke_train.jsonl", smoke_train)
    write_jsonl(jsonl_out_dir / "prompt_disjoint_smoke_dev.jsonl", smoke_dev)

    write_report(out_dir / "prompt_disjoint_split_report.md", audit, overlap, duplicates, best_score)

    print(f"samples: {len(rows)}")
    print(f"prompt_groups: {len(by_prompt)}")
    print(f"duplicates_skipped: {len(duplicates)}")
    print(f"best_score: {best_score:.6f}")
    for row in audit:
        print(row)
    print("overlap:", overlap)


if __name__ == "__main__":
    main()
