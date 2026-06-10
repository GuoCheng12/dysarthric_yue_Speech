#!/usr/bin/env python3
import argparse
import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
import statistics
import sys
import time

import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel


DEFAULT_DATA_ROOT = "/data/qwen3-asr/datasets/Speech_data/vlink_data_raw"
DEFAULT_MODEL = "/data/qwen3-asr/models/Qwen3-ASR-1.7B"
DEFAULT_OUT_DIR = "/data/qwen3-asr/inference/outputs"

RESULT_FIELDS = [
    "speaker",
    "session",
    "audio_id",
    "rel_path",
    "wav_path",
    "gt",
    "gt_source",
    "pred",
    "language_arg",
    "detected_language",
    "textnorm_cer",
    "critical_error",
    "critical_error_reason",
    "inference_error",
    "duration_sec",
    "sample_rate",
    "channels",
    "inference_time_sec",
]

SUMMARY_FIELDS = [
    "speaker",
    "samples_total",
    "inferred_total",
    "gt_total",
    "avg_textnorm_cer",
    "critical_error_rate",
    "critical_error_count",
    "critical_error_labeled",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def load_transcript_map(directory: Path) -> dict[str, str]:
    transcript_path = directory / "transcript.txt"
    if not transcript_path.is_file():
        return {}

    mapping = {}
    lines = transcript_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        if not line.strip() or line.strip() == "filename,transcript":
            continue
        if "," not in line:
            continue
        filename, transcript = line.split(",", 1)
        mapping[filename.strip()] = transcript.strip()
    return mapping


def discover_manifest(data_root: Path) -> list[dict[str, object]]:
    transcript_maps: dict[Path, dict[str, str]] = {}
    manifest = []
    for wav in sorted(data_root.rglob("*.wav")):
        rel = wav.relative_to(data_root)
        parts = rel.parts
        speaker = parts[0]
        session = parts[1] if len(parts) > 2 else ""
        lab = wav.with_suffix(".lab")
        gt = ""
        gt_source = ""

        if lab.is_file():
            gt = read_text(lab)
            gt_source = str(lab)
        else:
            transcript_dir = wav.parent
            if transcript_dir not in transcript_maps:
                transcript_maps[transcript_dir] = load_transcript_map(transcript_dir)
            gt = transcript_maps[transcript_dir].get(wav.name, "")
            gt_source = str(transcript_dir / "transcript.txt") if gt else ""

        try:
            info = sf.info(str(wav))
            duration_sec = round(info.frames / info.samplerate, 3)
            sample_rate = info.samplerate
            channels = info.channels
        except Exception:
            duration_sec = ""
            sample_rate = ""
            channels = ""

        manifest.append(
            {
                "speaker": speaker,
                "session": session,
                "audio_id": wav.stem,
                "rel_path": str(rel),
                "wav_path": str(wav),
                "gt": gt,
                "gt_source": gt_source,
                "duration_sec": duration_sec,
                "sample_rate": sample_rate,
                "channels": channels,
            }
        )
    return manifest


def read_jsonl_results(path: Path) -> OrderedDict[str, dict[str, object]]:
    results: OrderedDict[str, dict[str, object]] = OrderedDict()
    if not path.is_file():
        return results
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        results[row["rel_path"]] = row
    return results


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_optional_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_optional_bool(value: object) -> bool | None:
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


def summarize_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_speaker: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker"])].append(row)

    summaries = []
    for speaker in sorted(by_speaker):
        group = by_speaker[speaker]
        summaries.append(make_summary(speaker, group))
    overall = make_summary("__OVERALL__", rows)
    return summaries, overall


def make_summary(name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    textnorm_values = [
        value
        for value in (parse_optional_float(row.get("textnorm_cer")) for row in rows)
        if value is not None
    ]
    critical_values = [
        value
        for value in (parse_optional_bool(row.get("critical_error")) for row in rows)
        if value is not None
    ]
    return {
        "speaker": name,
        "samples_total": len(rows),
        "inferred_total": sum(1 for row in rows if row.get("pred") not in ("", None)),
        "gt_total": sum(1 for row in rows if row.get("gt") not in ("", None)),
        "avg_textnorm_cer": round(statistics.mean(textnorm_values), 6) if textnorm_values else "",
        "critical_error_rate": round(sum(critical_values) / len(critical_values), 6) if critical_values else "",
        "critical_error_count": sum(critical_values) if critical_values else "",
        "critical_error_labeled": len(critical_values),
    }


def md_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def write_summary_markdown(path: Path, summaries: list[dict[str, object]], overall: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Qwen3-ASR vlink_data_raw Summary\n\n")
        f.write("TextNorm_CER and Critical_Error are intentionally left blank until the next scoring pass.\n\n")
        f.write("## Overall\n\n")
        f.write("| samples_total | inferred_total | gt_total | avg_textnorm_cer | critical_error_rate | critical_error_count | critical_error_labeled |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        f.write(
            f"| {overall['samples_total']} | {overall['inferred_total']} | {overall['gt_total']} | "
            f"{overall['avg_textnorm_cer']} | {overall['critical_error_rate']} | "
            f"{overall['critical_error_count']} | {overall['critical_error_labeled']} |\n\n"
        )
        f.write("## By Speaker\n\n")
        f.write("| speaker | samples_total | inferred_total | gt_total | avg_textnorm_cer | critical_error_rate | critical_error_count | critical_error_labeled |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            f.write(
                f"| {md_escape(row['speaker'])} | {row['samples_total']} | {row['inferred_total']} | "
                f"{row['gt_total']} | {row['avg_textnorm_cer']} | {row['critical_error_rate']} | "
                f"{row['critical_error_count']} | {row['critical_error_labeled']} |\n"
            )


def materialize_outputs(manifest: list[dict[str, object]], results: OrderedDict[str, dict[str, object]], out_dir: Path, prefix: str) -> None:
    rows = []
    for item in manifest:
        result = results.get(str(item["rel_path"]))
        if result is None:
            row = {
                **item,
                "pred": "",
                "language_arg": "",
                "detected_language": "",
                "textnorm_cer": "",
                "critical_error": "",
                "critical_error_reason": "",
                "inference_error": "",
                "inference_time_sec": "",
            }
        else:
            row = {**item, **result}
        rows.append(row)

    write_csv(out_dir / f"{prefix}.csv", rows, RESULT_FIELDS)
    summaries, overall = summarize_rows(rows)
    write_csv(out_dir / f"{prefix}_summary_by_speaker.csv", summaries, SUMMARY_FIELDS)
    write_csv(out_dir / f"{prefix}_summary_overall.csv", [overall], SUMMARY_FIELDS)
    write_summary_markdown(out_dir / f"{prefix}_summary.md", summaries, overall)


def build_result_row(
    item: dict[str, object],
    transcription: object | None,
    language: str,
    inference_time_sec: float | str,
    inference_error: str = "",
) -> dict[str, object]:
    if transcription is None:
        pred = ""
        detected_language = ""
    else:
        pred = getattr(transcription, "text", "") or ""
        detected_language = getattr(transcription, "language", None)

    return {
        **item,
        "pred": pred,
        "language_arg": language,
        "detected_language": detected_language,
        "textnorm_cer": "",
        "critical_error": "",
        "critical_error_reason": "",
        "inference_error": inference_error,
        "inference_time_sec": inference_time_sec,
    }


def format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Qwen3-ASR inference for vlink_data_raw.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="vlink_data_raw_qwen3_asr_cantonese")
    parser.add_argument("--language", default="Cantonese")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit. 0 means all samples.")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = discover_manifest(data_root)
    if args.limit:
        manifest = manifest[: args.limit]
    if not manifest:
        raise SystemExit(f"No wav files found under {data_root}")

    manifest_csv = out_dir / f"{args.prefix}_manifest.csv"
    write_csv(
        manifest_csv,
        [
            {
                **item,
                "pred": "",
                "language_arg": args.language,
                "detected_language": "",
                "textnorm_cer": "",
                "critical_error": "",
                "critical_error_reason": "",
                "inference_error": "",
                "inference_time_sec": "",
            }
            for item in manifest
        ],
        RESULT_FIELDS,
    )

    jsonl_path = out_dir / f"{args.prefix}.jsonl"
    results = read_jsonl_results(jsonl_path)
    remaining = [item for item in manifest if str(item["rel_path"]) not in results]

    print(
        json.dumps(
            {
                "manifest": len(manifest),
                "existing_results": len(results),
                "remaining": len(remaining),
                "jsonl": str(jsonl_path),
                "manifest_csv": str(manifest_csv),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.summary_only:
        materialize_outputs(manifest, results, out_dir, args.prefix)
        return 0

    if remaining:
        if not torch.cuda.is_available():
            raise SystemExit("需要重新开启挂载 GPU 的 DevBox 后再跑 inference (torch.cuda.is_available() is False).")

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
                        audio=[str(item["wav_path"]) for item in batch],
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
                    for item in batch:
                        single_start = time.time()
                        try:
                            transcription = model.transcribe(
                                audio=[str(item["wav_path"])],
                                language=[args.language],
                            )[0]
                            row = build_result_row(
                                item,
                                transcription,
                                args.language,
                                round(time.time() - single_start, 3),
                            )
                        except Exception as single_exc:
                            row = build_result_row(
                                item,
                                None,
                                args.language,
                                round(time.time() - single_start, 3),
                                format_exception(single_exc),
                            )
                            print(
                                json.dumps(
                                    {
                                        "file_error": row["inference_error"],
                                        "rel_path": row["rel_path"],
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                        results[str(item["rel_path"])] = row
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                else:
                    elapsed = round(time.time() - batch_start, 3)
                    for item, transcription in zip(batch, transcriptions):
                        row = build_result_row(item, transcription, args.language, elapsed)
                        results[str(item["rel_path"])] = row
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()

    materialize_outputs(manifest, results, out_dir, args.prefix)
    print(
        json.dumps(
            {
                "completed": len(results),
                "csv": str(out_dir / f"{args.prefix}.csv"),
                "summary": str(out_dir / f"{args.prefix}_summary.md"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
