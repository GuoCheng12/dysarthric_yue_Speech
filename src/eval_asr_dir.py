#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import re
import sys

import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel

try:
    from opencc import OpenCC
except ImportError:  # pragma: no cover - fallback for minimal environments
    OpenCC = None


PUNCTUATION = "，。！？、,.!?;；:：\"'“”‘’（）()【】[]《》<>"
FALLBACK_S2T = str.maketrans(
    {
        "妈": "媽",
        "汉": "漢",
        "红": "紅",
        "黄": "黃",
        "个": "個",
        "点": "點",
        "样": "樣",
        "边": "邊",
        "没": "沒",
        "会": "會",
        "动": "動",
        "电": "電",
        "脑": "腦",
        "数": "數",
        "学": "學",
        "声": "聲",
        "体": "體",
        "育": "育",
        "开": "開",
        "关": "關",
        "车": "車",
        "鱼": "魚",
        "饭": "飯",
        "过": "過",
        "为": "為",
        "后": "後",
        "来": "來",
        "这": "這",
        "实": "實",
        "试": "試",
        "钟": "鍾",
        "意": "意",
    }
)
_OPENCC = OpenCC("s2t") if OpenCC is not None else None

CHINESE_DIGITS = {
    "零": "0",
    "〇": "0",
    "一": "1",
    "二": "2",
    "兩": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}
CHINESE_NUMBER_CHARS = "".join(CHINESE_DIGITS) + "十百千廿卅"
YUE_REPLACEMENTS = [
    ("冇", "無"),
    ("沒", "無"),
    ("咩嘢", "乜嘢"),
    ("咩野", "乜嘢"),
    ("咩", "乜"),
    ("野", "嘢"),
    ("唔洗", "唔使"),
    ("中意", "鍾意"),
    ("钟意", "鍾意"),
    ("啊", "呀"),
    ("吖", "呀"),
    ("畀", "俾"),
    ("岩", "啱"),
    ("啱啱", "啱"),
    ("的", "嘅"),
]


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return text.translate(str.maketrans("", "", PUNCTUATION))


def to_traditional(text: str) -> str:
    if _OPENCC is not None:
        return _OPENCC.convert(text)
    return text.translate(FALLBACK_S2T)


def parse_chinese_number(text: str) -> int | None:
    if not text or any(ch not in CHINESE_DIGITS and ch not in "十百千廿卅" for ch in text):
        return None
    if text.startswith("廿"):
        suffix = text[1:]
        if not suffix:
            return 20
        if len(suffix) == 1 and suffix in CHINESE_DIGITS:
            return 20 + int(CHINESE_DIGITS[suffix])
        return None
    if text.startswith("卅"):
        suffix = text[1:]
        if not suffix:
            return 30
        if len(suffix) == 1 and suffix in CHINESE_DIGITS:
            return 30 + int(CHINESE_DIGITS[suffix])
        return None
    if all(ch in CHINESE_DIGITS for ch in text):
        return None

    total = 0
    current = 0
    units = {"十": 10, "百": 100, "千": 1000}
    for ch in text:
        if ch in CHINESE_DIGITS:
            current = int(CHINESE_DIGITS[ch])
        else:
            unit = units[ch]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
    total += current
    return total


def normalize_chinese_number_token(match: re.Match[str]) -> str:
    token = match.group(0)
    if all(ch in CHINESE_DIGITS for ch in token):
        return "".join(CHINESE_DIGITS[ch] for ch in token)

    parsed = parse_chinese_number(token)
    return str(parsed) if parsed is not None else token


def normalize_numbers(text: str) -> str:
    pattern = f"[{re.escape(CHINESE_NUMBER_CHARS)}]{{2,}}"
    return re.sub(pattern, normalize_chinese_number_token, text)


def text_normalize(text: str) -> str:
    text = normalize_text(text)
    text = to_traditional(text)
    text = normalize_numbers(text)
    for source, target in YUE_REPLACEMENTS:
        text = text.replace(source, target)
    return text.lower()


def md_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-evaluate Qwen3-ASR on a directory of wav/lab pairs.")
    parser.add_argument("data_dir", help="Directory containing .wav and .lab files with matching stems.")
    parser.add_argument("--model", default="/data/qwen3-asr/models/Qwen3-ASR-1.7B")
    parser.add_argument("--language", default="Cantonese")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out-dir", default="/data/qwen3-asr/results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(data_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No .wav files found in {data_dir}")

    pairs = []
    for wav in wavs:
        lab = wav.with_suffix(".lab")
        if not lab.is_file():
            raise SystemExit(f"Missing GT lab file for {wav}: {lab}")
        info = sf.info(str(wav))
        pairs.append(
            {
                "id": wav.stem,
                "wav": str(wav),
                "lab": str(lab),
                "duration_sec": round(info.frames / info.samplerate, 3),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "gt": lab.read_text(encoding="utf-8", errors="replace").strip(),
            }
        )

    if not torch.cuda.is_available():
        raise SystemExit("需要重新开启挂载 GPU 的 DevBox 后再跑 inference (torch.cuda.is_available() is False).")

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    results = []
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        print(f"running {start + 1}-{start + len(batch)} / {len(pairs)}", flush=True)
        transcriptions = model.transcribe(
            audio=[item["wav"] for item in batch],
            language=[args.language] * len(batch),
        )
        for item, transcription in zip(batch, transcriptions):
            pred = getattr(transcription, "text", "") or ""
            detected_language = getattr(transcription, "language", None)
            gt = item["gt"]
            dist = edit_distance(gt, pred)
            norm_gt = normalize_text(gt)
            norm_pred = normalize_text(pred)
            norm_dist = edit_distance(norm_gt, norm_pred)
            textnorm_gt = text_normalize(gt)
            textnorm_pred = text_normalize(pred)
            textnorm_dist = edit_distance(textnorm_gt, textnorm_pred)
            row = {
                **item,
                "language_arg": args.language,
                "detected_language": detected_language,
                "pred": pred,
                "exact": gt == pred,
                "char_distance": dist,
                "cer": round(dist / max(1, len(gt)), 4),
                "norm_exact": norm_gt == norm_pred,
                "norm_char_distance": norm_dist,
                "norm_cer": round(norm_dist / max(1, len(norm_gt)), 4),
                "textnorm_gt": textnorm_gt,
                "textnorm_pred": textnorm_pred,
                "textnorm_exact": textnorm_gt == textnorm_pred,
                "textnorm_char_distance": textnorm_dist,
                "textnorm_cer": round(textnorm_dist / max(1, len(textnorm_gt)), 4),
            }
            results.append(row)
            print(json.dumps({"id": row["id"], "gt": gt, "pred": pred, "cer": row["cer"]}, ensure_ascii=False), flush=True)

    safe_name = data_dir.name
    csv_path = out_dir / f"{safe_name}_qwen3_asr_eval.csv"
    md_path = out_dir / f"{safe_name}_qwen3_asr_eval.md"
    jsonl_path = out_dir / f"{safe_name}_qwen3_asr_eval.jsonl"

    fieldnames = [
        "id",
        "duration_sec",
        "gt",
        "pred",
        "language_arg",
        "detected_language",
        "exact",
        "char_distance",
        "cer",
        "norm_exact",
        "norm_char_distance",
        "norm_cer",
        "textnorm_exact",
        "textnorm_char_distance",
        "textnorm_cer",
        "textnorm_gt",
        "textnorm_pred",
        "wav",
        "lab",
        "sample_rate",
        "channels",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_dist = sum(row["char_distance"] for row in results)
    total_chars = sum(max(1, len(row["gt"])) for row in results)
    total_norm_dist = sum(row["norm_char_distance"] for row in results)
    total_norm_chars = sum(max(1, len(normalize_text(row["gt"]))) for row in results)
    total_textnorm_dist = sum(row["textnorm_char_distance"] for row in results)
    total_textnorm_chars = sum(max(1, len(row["textnorm_gt"])) for row in results)
    exact = sum(1 for row in results if row["exact"])
    norm_exact = sum(1 for row in results if row["norm_exact"])
    textnorm_exact = sum(1 for row in results if row["textnorm_exact"])

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Qwen3-ASR Evaluation: vlink_Kaho/20250929\n\n")
        f.write(f"- samples: {len(results)}\n")
        f.write(f"- language_arg: {args.language}\n")
        f.write(f"- exact_match: {exact}/{len(results)} ({exact / len(results):.4f})\n")
        f.write(f"- corpus_cer: {total_dist / total_chars:.4f}\n")
        f.write(f"- normalized_exact_match: {norm_exact}/{len(results)} ({norm_exact / len(results):.4f})\n")
        f.write(f"- normalized_corpus_cer: {total_norm_dist / total_norm_chars:.4f}\n")
        f.write(f"- textnorm_exact_match: {textnorm_exact}/{len(results)} ({textnorm_exact / len(results):.4f})\n")
        f.write(f"- textnorm_corpus_cer: {total_textnorm_dist / total_textnorm_chars:.4f}\n\n")
        f.write("| id | dur_s | GT | ASR | exact | CER | norm_exact | norm_CER | textnorm_exact | TextNorm_CER |\n")
        f.write("|---:|---:|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in results:
            f.write(
                f"| {md_escape(row['id'])} | {row['duration_sec']} | {md_escape(row['gt'])} | "
                f"{md_escape(row['pred'])} | {row['exact']} | {row['cer']:.4f} | "
                f"{row['norm_exact']} | {row['norm_cer']:.4f} | "
                f"{row['textnorm_exact']} | {row['textnorm_cer']:.4f} |\n"
            )

    print(json.dumps(
        {
            "samples": len(results),
            "exact_match": exact,
            "exact_rate": round(exact / len(results), 4),
            "corpus_cer": round(total_dist / total_chars, 4),
            "normalized_exact_match": norm_exact,
            "normalized_exact_rate": round(norm_exact / len(results), 4),
            "normalized_corpus_cer": round(total_norm_dist / total_norm_chars, 4),
            "textnorm_exact_match": textnorm_exact,
            "textnorm_exact_rate": round(textnorm_exact / len(results), 4),
            "textnorm_corpus_cer": round(total_textnorm_dist / total_textnorm_chars, 4),
            "csv": str(csv_path),
            "markdown": str(md_path),
            "jsonl": str(jsonl_path),
        },
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
