#!/usr/bin/env python3
"""Frozen greedy ASR transcription for the SG three-text review.

Run as a separate ASR process, never inside the SG Transformers overlay.
Inputs contain audio only: target text and TTS reference are not available to
this recognizer. A decoding cap or technical error cannot become an SG verdict.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from therapist_harness.store import sha256_directory, sha256_file, write_immutable_json


MAX_NEW_TOKENS = 512
EOS_IDS = (151645, 151643)
DECODE_CONFIG = {"do_sample": False, "num_beams": 1, "language": "Cantonese",
                 "context": "", "max_new_tokens": MAX_NEW_TOKENS, "seed": 42,
                 "eos_token_ids": list(EOS_IDS)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio", type=Path, help="One audio; output is one transcription receipt")
    inputs.add_argument("--manifest", type=Path, help="Audio-only JSONL; output contains records, model loads once")
    inputs.add_argument("--worker", action="store_true", help="Read audio/output requests as JSON lines on stdin; keep ASR loaded")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True,
                        help="Frozen seed42 initial R-only adapter, independent of current-ASR preference sampling")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def audio_inputs(args):
    if args.audio is not None:
        values = [{"audio": str(args.audio)}]
    else:
        values = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError("audio manifest must not be empty")
    result = []
    for index, row in enumerate(values):
        if not isinstance(row, dict) or set(row) - {"audio", "audio_id", "audio_sha256"}:
            raise ValueError("transcription inputs accept audio/audio_id/audio_sha256 only; no target or reference text")
        audio = Path(row["audio"]).resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        digest = sha256_file(audio)
        if row.get("audio_sha256", digest) != digest:
            raise ValueError("audio differs from supplied SHA256")
        result.append({"audio": str(audio), "audio_id": row.get("audio_id", str(index)), "audio_sha256": digest})
    return result


class QwenGreedyTranscriber:
    def __init__(self, args):
        import torch
        from peft import PeftModel
        from qwen_asr import Qwen3ASRModel
        from transformers import set_seed

        set_seed(42)
        dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
        self.wrapper = Qwen3ASRModel.from_pretrained(
            str(args.base_model), dtype=dtype, device_map=args.device,
            max_inference_batch_size=1, max_new_tokens=MAX_NEW_TOKENS,
        )
        self.wrapper.model = PeftModel.from_pretrained(
            self.wrapper.model, str(args.adapter), is_trainable=False,
        ).merge_and_unload()
        self.wrapper.model.eval().requires_grad_(False)

    def transcribe(self, audio):
        import torch
        from qwen_asr.inference.utils import normalize_audio_input, parse_asr_output

        wav = normalize_audio_input(audio)
        # These are fixed constants, never target- or reference-derived context.
        prompt = self.wrapper._build_text_prompt(context="", force_language="Cantonese")
        features = self.wrapper.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        features = features.to(self.wrapper.model.device).to(self.wrapper.model.dtype)
        prefix_length = features["input_ids"].shape[1]
        with torch.inference_mode():
            output = self.wrapper.model.generate(
                **features, do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
            )
        tokens = output.sequences[0, prefix_length:].tolist()
        raw = self.wrapper.processor.batch_decode(
            [tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        language, prediction = parse_asr_output(raw, user_language="Cantonese")
        return {"generated_token_ids": tokens, "raw_output": raw,
                "prediction": prediction, "detected_language": language}


def transcription_record(audio, decoded, identity, *, actual_transcription=False):
    """Pure receipt construction; test callers are explicitly non-live by default."""
    tokens = decoded["generated_token_ids"]
    ended = bool(tokens and tokens[-1] in EOS_IDS)
    capped = len(tokens) >= MAX_NEW_TOKENS
    # Even EOS exactly at the budget limit is recorded as capped, not accepted.
    complete = ended and not capped
    return {**identity, **audio, **decoded, "schema_version": "sg_review_asr_transcription_v1",
            "status": "complete" if complete else "technical_error",
            "actual_transcription": actual_transcription,
            "decode_config": dict(DECODE_CONFIG), "ended_with_eos": ended,
            "reached_max_new_tokens": capped,
            "technical_error": None if complete else ("asr_token_limit" if capped else "asr_missing_eos"),
            "review_usable": complete and actual_transcription,
            "scope": "ASR transcript only; not direct audio inspection or word-fidelity proof"}


def run(args, *, decoder_factory=None):
    """One model load for all audio. Injected test decoders never produce live flags."""
    if args.output.exists():
        raise FileExistsError("transcription output is immutable; use a new path")
    inputs = audio_inputs(args)
    identity = {"model": str(args.base_model.resolve()), "base_model_sha256": sha256_directory(args.base_model),
                "adapter": str(args.adapter.resolve()), "adapter_sha256": sha256_directory(args.adapter),
                "source_sha256": sha256_file(Path(__file__))}
    actual = decoder_factory is None
    records = []
    try:
        decoder = (QwenGreedyTranscriber if actual else decoder_factory)(args)
        for audio in inputs:
            try:
                decoded = decoder.transcribe(audio["audio"])
                if sha256_file(Path(audio["audio"])) != audio["audio_sha256"]:
                    raise RuntimeError("audio changed during transcription")
                record = transcription_record(audio, decoded, identity, actual_transcription=actual)
            except Exception as error:
                record = {**identity, **audio, "schema_version": "sg_review_asr_transcription_v1",
                          "status": "technical_error", "actual_transcription": False,
                          "decode_config": dict(DECODE_CONFIG), "technical_error": type(error).__name__,
                          "prediction": None, "raw_output": None, "generated_token_ids": [],
                          "ended_with_eos": False, "reached_max_new_tokens": False, "review_usable": False}
            records.append(record)
    except Exception as error:
        records = [{**identity, **audio, "schema_version": "sg_review_asr_transcription_v1",
                    "status": "technical_error", "actual_transcription": False,
                    "decode_config": dict(DECODE_CONFIG), "technical_error": type(error).__name__,
                    "prediction": None, "raw_output": None, "generated_token_ids": [],
                    "ended_with_eos": False, "reached_max_new_tokens": False, "review_usable": False}
                   for audio in inputs]
    adapter_unchanged = sha256_directory(args.adapter) == identity["adapter_sha256"]
    for record in records:
        record["adapter_files_unchanged"] = adapter_unchanged
        if not adapter_unchanged:
            record.update(status="technical_error", technical_error="adapter_changed", review_usable=False)
    if args.audio is not None:
        receipt = records[0]
    else:
        receipt = {**identity, "schema_version": "sg_review_asr_batch_v1",
                   "status": "complete" if all(r["status"] == "complete" for r in records) else "technical_error",
                   "actual_transcription": all(r["actual_transcription"] for r in records),
                   "decode_config": dict(DECODE_CONFIG), "records": records,
                   "adapter_files_unchanged": adapter_unchanged}
    write_immutable_json(args.output, receipt)
    return receipt


def worker(args, *, decoder_factory=None, requests=None, responses=None):
    """One independent ASR process; JSON-lines pipes are the request/response queue."""
    requests = sys.stdin if requests is None else requests
    responses = sys.stdout if responses is None else responses
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError("worker session output is immutable")
    identity = {"model": str(args.base_model.resolve()), "base_model_sha256": sha256_directory(args.base_model),
                "adapter": str(args.adapter.resolve()), "adapter_sha256": sha256_directory(args.adapter),
                "source_sha256": sha256_file(Path(__file__))}
    hash_seconds = time.perf_counter() - started
    load_start = time.perf_counter()
    actual = decoder_factory is None
    with redirect_stdout(sys.stderr):
        decoder = (QwenGreedyTranscriber if actual else decoder_factory)(args)
    load_seconds = time.perf_counter() - load_start
    session = {**identity, "schema_version": "sg_review_asr_worker_v1", "status": "ready", "pid": os.getpid(),
               "actual_transcription": actual, "decode_config": dict(DECODE_CONFIG),
               "timing": {"initial_hash_seconds": hash_seconds, "model_load_seconds": load_seconds,
                          "startup_seconds": time.perf_counter() - started}}
    write_immutable_json(args.output, session)
    session_sha = sha256_file(args.output)

    def reply(value):
        responses.write(json.dumps(value, ensure_ascii=False) + "\n")
        responses.flush()

    reply({"status": "ready", "session": str(args.output.resolve()), "session_sha256": session_sha})
    receipts = []
    closed = False
    try:
        for line in requests:
            request = json.loads(line)
            if request == {"command": "close"}:
                closed = True
                break
            if not isinstance(request, dict) or set(request) != {"audio", "output"}:
                raise ValueError("worker accepts audio/output only; no target, reference or context")
            output = Path(request["output"])
            if not output.is_absolute() or output.exists():
                raise ValueError("request output must be a fresh absolute path")
            audio = Path(request["audio"])
            if not audio.is_absolute() or not audio.is_file():
                raise ValueError("request audio must be an existing absolute file")
            begin = time.perf_counter()
            audio_record = {"audio": str(audio), "audio_id": "0", "audio_sha256": sha256_file(audio)}
            decode_start = time.perf_counter()
            with redirect_stdout(sys.stderr):
                decoded = decoder.transcribe(str(audio))
            decode_seconds = time.perf_counter() - decode_start
            if sha256_file(audio) != audio_record["audio_sha256"]:
                raise RuntimeError("audio changed during worker transcription")
            receipt = transcription_record(audio_record, decoded, identity, actual_transcription=actual)
            receipt.update(worker_session_path=str(args.output.resolve()), worker_session_sha256=session_sha,
                           timing={"decode_seconds": decode_seconds, "request_seconds": time.perf_counter() - begin})
            write_immutable_json(output, receipt)
            digest = sha256_file(output)
            receipts.append({"path": str(output), "sha256": digest})
            reply({"status": receipt["status"], "output": str(output), "sha256": digest})
            if receipt["status"] != "complete":
                break
    finally:
        verify_start = time.perf_counter()
        base_unchanged = sha256_directory(args.base_model) == identity["base_model_sha256"]
        adapter_unchanged = sha256_directory(args.adapter) == identity["adapter_sha256"]
        final = {"schema_version": "sg_review_asr_worker_close_v1", "session_sha256": session_sha,
                 "status": "complete" if closed and base_unchanged and adapter_unchanged else "technical_error",
                 "base_files_unchanged": base_unchanged, "adapter_files_unchanged": adapter_unchanged,
                 "receipts": receipts, "timing": {"final_hash_seconds": time.perf_counter() - verify_start,
                    "session_seconds": time.perf_counter() - started}}
        final_path = args.output.with_suffix(".closed.json")
        write_immutable_json(final_path, final)
    reply({"status": final["status"], "closed": str(final_path)})
    return final


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        result = worker(args)
        return 0 if result["status"] == "complete" else 1
    receipt = run(args)
    print(json.dumps({"status": receipt["status"], "actual_transcription": receipt["actual_transcription"],
                      "output": str(args.output.resolve())}))
    return 0 if receipt["status"] == "complete" and receipt["actual_transcription"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
