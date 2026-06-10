#!/usr/bin/env python3
"""Build automated Experiment 2 cleaning artifacts from Qwen3-ASR results.

This script is intentionally conservative:
- It never edits Experiment 1 outputs.
- It preserves raw_gt/raw_pred and writes cleaned text into clean_gt.
- It excludes obvious prompt/answer mismatches and non-main-task rows from the
  first pooled read-sentence fine-tuning run, while keeping them in the manifest.
"""

import csv
import hashlib
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


INPUT_CSV = Path("outputs/vlink_data_raw_qwen3_asr_cantonese.csv")
OUT_DIR = Path("outputs/experiment2")

MANIFEST_CSV = OUT_DIR / "clean_manifest.csv"
TRAIN_CSV = OUT_DIR / "clean_read_sentence_train.csv"
DEV_CSV = OUT_DIR / "clean_read_sentence_dev.csv"
TEST_CSV = OUT_DIR / "clean_read_sentence_test.csv"
ALL_TRAINABLE_CSV = OUT_DIR / "clean_read_sentence_all.csv"
EXCLUDED_CSV = OUT_DIR / "excluded_rows.csv"
REVIEW_CSV = OUT_DIR / "review_queue.csv"
REPORT_MD = OUT_DIR / "cleaning_report.md"

CRITICAL_THRESHOLD = 0.5
SHORT_WORD_MAX_CHARS = 6
LONG_DURATION_SEC = 30.0
SHORT_DURATION_SEC = 0.5

PUNCTUATION_RE = re.compile(r"[，。！？!?、,.；;：:「」『』“”\"'‘’（）()\[\]【】《》〈〉…~～\-_—]+")
SLASH_ANNOTATION_RE = re.compile(r"/[^/]+/")
LEADING_NUMBERED_PROMPT_RE = re.compile(r"^\s*[\[（(【]?\s*\d+[\.\．、\)）:：]\s*")
NUMBERED_QA_RE = re.compile(r"^\s*[\[（(【]?\s*\d+[\.\．、\)）:：]\s*.*[?？]\s*[\]）)】]?\s*$")
BRACKET_WRAPPER_RE = re.compile(r"^\s*[\[（(【]\s*(.*?)\s*[\]）)】]\s*$")
SPACE_RE = re.compile(r"\s+")

# Strong automatic exclusion markers. These are not used as evidence of ASR
# failure; they are used to protect the first pooled fine-tuning run from
# obvious non-target speech in the audio/transcript.
META_MARKERS = [
    "再講一次",
    "再講",
    "講多一次",
    "講多次",
    "再讀",
    "讀啊",
    "讀呀",
    "大聲啲",
    "大聲",
    "開始緊",
    "開始錄",
]
START_META_RE = re.compile(r"^\s*開始[，,。 ]")


FIELDS = [
    "utt_id",
    "speaker_id",
    "speaker_suffix",
    "disease_tag",
    "session",
    "audio_id",
    "rel_path",
    "audio_path",
    "raw_gt",
    "clean_gt",
    "raw_pred",
    "detected_language",
    "duration",
    "duration_bucket",
    "zero_shot_cer",
    "zero_shot_critical",
    "zero_shot_bucket",
    "task_type",
    "quality_label",
    "cleaning_flags",
    "exclude_from_training",
    "exclusion_reason",
    "main_train_eligible",
    "split",
    "prompt_id",
    "clean_gt_char_len",
]


def stable_int(key):
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def prompt_id(clean_gt):
    return hashlib.md5(clean_gt.encode("utf-8")).hexdigest()[:12]


def parse_float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_speaker(speaker):
    if "_" in speaker:
        suffix = speaker.split("_", 1)[1]
    else:
        suffix = "unknown"
    disease = suffix if suffix in {"HD", "SCA"} else "unknown"
    return suffix, disease


def strip_wrapper(text):
    match = BRACKET_WRAPPER_RE.match(text)
    if match:
        return match.group(1)
    return text


def clean_reference(raw_gt):
    text = (raw_gt or "").strip()
    text = strip_wrapper(text)
    text = LEADING_NUMBERED_PROMPT_RE.sub("", text)
    text = SLASH_ANNOTATION_RE.sub("", text)
    text = PUNCTUATION_RE.sub("", text)
    text = SPACE_RE.sub("", text)
    return text.strip()


def char_len_for_task(clean_gt):
    return len(clean_gt or "")


def has_meta_speech(row):
    raw_gt = row.get("gt") or ""
    raw_pred = row.get("pred") or ""
    joined = f"{raw_gt}\n{raw_pred}"
    if START_META_RE.search(raw_pred):
        return True
    for marker in META_MARKERS:
        if marker in joined:
            # Do not treat lexical "一開始" in the reference as a meta command.
            if marker == "大聲" and marker in raw_gt and marker not in raw_pred:
                continue
            if marker.startswith("開始") and "一開始" in raw_gt and marker not in raw_pred:
                continue
            return True
    return False


def duration_bucket(duration):
    if duration is None:
        return "missing"
    if duration <= 0:
        return "zero"
    if duration < 2:
        return "very_short"
    if duration < 5:
        return "short"
    if duration < 10:
        return "medium"
    if duration < 20:
        return "long"
    return "very_long"


def zero_shot_bucket(cer, critical):
    if critical or (cer is not None and cer >= CRITICAL_THRESHOLD):
        return "hard"
    if cer is None:
        return "unknown"
    if cer < 0.2:
        return "easy"
    return "medium"


def classify_row(row):
    speaker = row["speaker"]
    speaker_suffix, disease = parse_speaker(speaker)
    raw_gt = row.get("gt") or ""
    raw_pred = row.get("pred") or ""
    clean_gt = clean_reference(raw_gt)
    clean_len = char_len_for_task(clean_gt)
    duration = parse_float(row.get("duration_sec"))
    cer = parse_float(row.get("textnorm_cer"))
    critical = parse_bool(row.get("critical_error"))

    is_qna = bool(NUMBERED_QA_RE.match(raw_gt))
    flags = []
    exclusion_reasons = []
    task_type = "read_sentence"

    if is_qna:
        task_type = "open_question_answer"
        flags.append("prompt_answer_mismatch")
        exclusion_reasons.append("prompt_answer_mismatch")
    elif clean_len <= SHORT_WORD_MAX_CHARS:
        task_type = "short_word"
        flags.append("short_word_or_isolated_phrase")
        exclusion_reasons.append("not_main_read_sentence_task")

    if not clean_gt:
        flags.append("empty_clean_gt")
        exclusion_reasons.append("empty_clean_gt")

    if duration is None:
        flags.append("missing_duration")
        exclusion_reasons.append("missing_duration")
    elif duration <= 0:
        flags.append("duration_zero")
        exclusion_reasons.append("duration_zero")
    elif duration < SHORT_DURATION_SEC:
        flags.append("duration_very_short")

    if duration is not None and duration >= LONG_DURATION_SEC:
        flags.append("duration_very_long")

    if has_meta_speech(row):
        if task_type == "read_sentence":
            task_type = "meta_or_instruction"
        flags.append("meta_or_instruction_candidate")
        exclusion_reasons.append("meta_or_instruction_candidate")

    if critical:
        flags.append("zero_shot_critical")

    main_train_eligible = (
        task_type == "read_sentence"
        and not exclusion_reasons
        and bool(clean_gt)
        and duration is not None
        and duration > 0
    )

    if task_type == "short_word" and not any(reason != "not_main_read_sentence_task" for reason in exclusion_reasons):
        split = "analysis_only"
    elif main_train_eligible:
        split = "pending"
    else:
        split = "excluded"

    if not flags:
        quality_label = "ok"
    elif "prompt_answer_mismatch" in flags:
        quality_label = "prompt_answer_mismatch"
    elif "meta_or_instruction_candidate" in flags:
        quality_label = "meta_or_instruction_candidate"
    elif "duration_zero" in flags or "empty_clean_gt" in flags:
        quality_label = "invalid_for_training"
    elif "short_word_or_isolated_phrase" in flags:
        quality_label = "short_word_analysis_only"
    else:
        quality_label = "needs_attention"

    utt_id = f"{speaker}/{row['audio_id']}"
    return {
        "utt_id": utt_id,
        "speaker_id": speaker,
        "speaker_suffix": speaker_suffix,
        "disease_tag": disease,
        "session": row.get("session") or "",
        "audio_id": row.get("audio_id") or "",
        "rel_path": row.get("rel_path") or "",
        "audio_path": row.get("wav_path") or "",
        "raw_gt": raw_gt,
        "clean_gt": clean_gt,
        "raw_pred": raw_pred,
        "detected_language": row.get("detected_language") or "",
        "duration": "" if duration is None else duration,
        "duration_bucket": duration_bucket(duration),
        "zero_shot_cer": "" if cer is None else cer,
        "zero_shot_critical": str(critical).lower(),
        "zero_shot_bucket": zero_shot_bucket(cer, critical),
        "task_type": task_type,
        "quality_label": quality_label,
        "cleaning_flags": ";".join(sorted(set(flags))),
        "exclude_from_training": str(not main_train_eligible).lower(),
        "exclusion_reason": ";".join(sorted(set(exclusion_reasons))),
        "main_train_eligible": str(main_train_eligible).lower(),
        "split": split,
        "prompt_id": prompt_id(clean_gt) if clean_gt else "",
        "clean_gt_char_len": clean_len,
    }


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def assign_speaker_prompt_splits(rows):
    """Assign train/dev/test for main-eligible rows.

    Strategy:
    - Work inside each speaker so pooled fine-tuning sees as many speakers as possible.
    - Keep identical prompts for the same speaker in the same split.
    - For speakers with enough prompt groups, create same-patient unseen-prompt test rows.
    - Very small speakers are kept in train; they are useful for pooled adaptation but
      not reliable per-speaker evaluation units.
    """
    by_speaker = defaultdict(list)
    for row in rows:
        if row["main_train_eligible"] == "true":
            by_speaker[row["speaker_id"]].append(row)

    for speaker, speaker_rows in by_speaker.items():
        by_prompt = defaultdict(list)
        for row in speaker_rows:
            by_prompt[row["prompt_id"]].append(row)
        prompt_ids = sorted(by_prompt, key=lambda pid: stable_int(f"{speaker}|{pid}"))
        n = len(prompt_ids)

        if n >= 10:
            n_test = max(1, round(n * 0.1))
            n_dev = max(1, round(n * 0.1))
        elif n >= 4:
            n_test = 1
            n_dev = 0
        else:
            n_test = 0
            n_dev = 0

        test_ids = set(prompt_ids[:n_test])
        dev_ids = set(prompt_ids[n_test : n_test + n_dev])
        for prompt in prompt_ids:
            if prompt in test_ids:
                split = "test"
            elif prompt in dev_ids:
                split = "dev"
            else:
                split = "train"
            for row in by_prompt[prompt]:
                row["split"] = split


def group_mean(rows, field):
    values = []
    for row in rows:
        value = parse_float(row.get(field))
        if value is not None:
            values.append(value)
    return round(statistics.mean(values), 6) if values else ""


def write_report(rows):
    total = len(rows)
    task_counts = Counter(row["task_type"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    disease_counts = Counter(row["disease_tag"] for row in rows)
    quality_counts = Counter(row["quality_label"] for row in rows)
    reason_counts = Counter()
    for row in rows:
        for reason in (row.get("exclusion_reason") or "").split(";"):
            if reason:
                reason_counts[reason] += 1

    eligible = [row for row in rows if row["main_train_eligible"] == "true"]
    excluded = [row for row in rows if row["exclude_from_training"] == "true"]
    short_words = [row for row in rows if row["task_type"] == "short_word"]
    qna = [row for row in rows if row["task_type"] == "open_question_answer"]
    review = build_review_rows(rows)

    lines = []
    lines.append("# Experiment 2 Automated Cleaning Report\n")
    lines.append("## Inputs\n")
    lines.append(f"- Source CSV: `{INPUT_CSV}`")
    lines.append("- Original Experiment 1 outputs are not modified.")
    lines.append("- `raw_gt` and `raw_pred` are preserved; training should use `clean_gt` only.\n")
    lines.append("## Overall Counts\n")
    lines.append(f"- Total rows: {total}")
    lines.append(f"- Main read-sentence train/dev/test eligible rows: {len(eligible)}")
    lines.append(f"- Excluded from first pooled read-sentence training: {len(excluded)}")
    lines.append(f"- Q&A prompt-answer mismatch rows excluded: {len(qna)}")
    lines.append(f"- Short-word / isolated-phrase rows preserved as analysis-only: {len(short_words)}")
    lines.append(f"- Review queue rows: {len(review)}\n")

    lines.append("## Task Type Counts\n")
    lines.append("| task_type | count |")
    lines.append("|---|---:|")
    for key, count in sorted(task_counts.items()):
        lines.append(f"| {key} | {count} |")
    lines.append("")

    lines.append("## Split Counts\n")
    lines.append("| split | count |")
    lines.append("|---|---:|")
    for key, count in sorted(split_counts.items()):
        lines.append(f"| {key} | {count} |")
    lines.append("")

    lines.append("## Disease Tag Counts\n")
    lines.append("| disease_tag | count |")
    lines.append("|---|---:|")
    for key, count in sorted(disease_counts.items()):
        lines.append(f"| {key} | {count} |")
    lines.append("")

    lines.append("## Quality Label Counts\n")
    lines.append("| quality_label | count |")
    lines.append("|---|---:|")
    for key, count in sorted(quality_counts.items()):
        lines.append(f"| {key} | {count} |")
    lines.append("")

    lines.append("## Exclusion Reasons\n")
    lines.append("| reason | count |")
    lines.append("|---|---:|")
    for key, count in sorted(reason_counts.items()):
        lines.append(f"| {key} | {count} |")
    lines.append("")

    lines.append("## Main Eligible Zero-Shot CER by Bucket\n")
    lines.append("| zero_shot_bucket | rows | avg_zero_shot_cer |")
    lines.append("|---|---:|---:|")
    by_bucket = defaultdict(list)
    for row in eligible:
        by_bucket[row["zero_shot_bucket"]].append(row)
    for key in ["easy", "medium", "hard", "unknown"]:
        group = by_bucket.get(key, [])
        if group:
            lines.append(f"| {key} | {len(group)} | {group_mean(group, 'zero_shot_cer')} |")
    lines.append("")

    lines.append("## Generated Files\n")
    for path in [MANIFEST_CSV, ALL_TRAINABLE_CSV, TRAIN_CSV, DEV_CSV, TEST_CSV, EXCLUDED_CSV, REVIEW_CSV]:
        lines.append(f"- `{path}`")
    lines.append("")

    lines.append("## Automatic Cleaning Rules\n")
    lines.append("- Exclude numbered question prompts such as `[1. 可以描述一下今天的天氣嗎？]` as `open_question_answer`.")
    lines.append("- Remove slash annotations such as `/kyt6/` from `clean_gt`.")
    lines.append("- Remove leading numbering, wrapper brackets, spaces, and punctuation from `clean_gt`.")
    lines.append(f"- Mark rows with `clean_gt` length <= {SHORT_WORD_MAX_CHARS} as `short_word`; they are preserved but not used in the first pooled read-sentence training.")
    lines.append("- Exclude zero-duration rows and strong meta/instruction candidates from first pooled read-sentence training.")
    lines.append("- Do not exclude hard samples only because Qwen zero-shot failed; `zero_shot_critical` is kept for stratified evaluation.\n")
    lines.append("## Split Rule\n")
    lines.append("- Splits are assigned inside each speaker, grouped by `prompt_id`, so the same speaker's identical prompt does not cross train/dev/test.")
    lines.append("- Speakers with at least 10 prompt groups receive train/dev/test rows.")
    lines.append("- Speakers with 4-9 prompt groups receive train/test rows.")
    lines.append("- Speakers with fewer than 4 prompt groups are kept in train for pooled adaptation and are not treated as reliable per-speaker evaluation units.\n")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_review_rows(rows):
    review = []
    for row in rows:
        flags = set((row.get("cleaning_flags") or "").split(";"))
        if "" in flags:
            flags.remove("")
        if row["task_type"] in {"open_question_answer", "meta_or_instruction"}:
            review.append(row)
            continue
        if "empty_clean_gt" in flags or "duration_zero" in flags or "duration_very_long" in flags:
            review.append(row)
            continue
        if row["zero_shot_bucket"] == "hard" and row["main_train_eligible"] == "true":
            review.append(row)
    return review


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with INPUT_CSV.open(encoding="utf-8", newline="") as f:
        source_rows = list(csv.DictReader(f))

    rows = [classify_row(row) for row in source_rows]
    assign_speaker_prompt_splits(rows)
    eligible = [row for row in rows if row["main_train_eligible"] == "true"]
    train_rows = [row for row in eligible if row["split"] == "train"]
    dev_rows = [row for row in eligible if row["split"] == "dev"]
    test_rows = [row for row in eligible if row["split"] == "test"]
    excluded_rows = [row for row in rows if row["exclude_from_training"] == "true"]
    review_rows = build_review_rows(rows)

    write_csv(MANIFEST_CSV, rows, FIELDS)
    write_csv(ALL_TRAINABLE_CSV, eligible, FIELDS)
    write_csv(TRAIN_CSV, train_rows, FIELDS)
    write_csv(DEV_CSV, dev_rows, FIELDS)
    write_csv(TEST_CSV, test_rows, FIELDS)
    write_csv(EXCLUDED_CSV, excluded_rows, FIELDS)
    write_csv(REVIEW_CSV, review_rows, FIELDS)
    write_report(rows)

    print(f"Wrote {MANIFEST_CSV} ({len(rows)} rows)")
    print(f"Main eligible read_sentence rows: {len(eligible)}")
    print(f"Train/dev/test: {len(train_rows)}/{len(dev_rows)}/{len(test_rows)}")
    print(f"Excluded from first pooled training: {len(excluded_rows)}")
    print(f"Review queue: {len(review_rows)}")


if __name__ == "__main__":
    main()
