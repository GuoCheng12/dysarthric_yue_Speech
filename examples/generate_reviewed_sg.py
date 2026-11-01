#!/usr/bin/env python3
"""Fixed SG generation followed by a small explicitly identified coarse review.

Technical generation errors stop the worker. Only a review's obvious_failure
routes zero-shot -> one zero-shot retry -> instruct2 -> exclude. No ASR CER,
phoneme check, target rewrite, or reference replacement is performed here.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from therapist_harness.audio_review import AudioReviewer, RULEBOOK, next_generation_action  # noqa: E402
from therapist_harness.store import sha256_file, write_immutable_bytes, write_immutable_json  # noqa: E402

INSTRUCTION = "You are a helpful assistant. 请用广东话表达。<|endofprompt|>"


class InstructProxy:
    """Keep the acoustic reference while using CosyVoice's instruct2 text path."""
    def __init__(self, model):
        self.model = model
        self.sample_rate = model.sample_rate

    def inference_zero_shot(self, target, _reference_text_prompt, reference, **kwargs):
        return self.model.inference_instruct2(target, INSTRUCTION, reference, **kwargs)


def run_attempts(initial_seed, generate, review):
    """Pure finite routing, also testable without creating any review receipts.

    Production callbacks verify a live review in the declared evidence mode.
    Exceptions intentionally propagate; technical failures do not trigger retry.
    """
    results = []
    for attempt, mode in enumerate(("zero_shot", "zero_shot", "instruct2")):
        seed = initial_seed + attempt
        generation = generate(mode, seed, attempt)
        judgment = review(generation, attempt)
        action = next_generation_action(judgment, attempt)
        results.append({"attempt": attempt, "mode": mode, "seed": seed,
                        "generation": generation, "review": judgment.to_dict() if hasattr(judgment, "to_dict") else judgment.model_dump(),
                        "action": action})
        if action in ("keep", "exclude", "quarantine"):
            return results, action
    raise AssertionError("finite SG rulebook must terminate")


def _install_empty_token_guard(backend):
    """Convert the known completed-empty-token backend hang into a technical error."""
    original = backend.token2wav

    def token2wav(*args, **kwargs):
        token, uid = kwargs["token"], kwargs["uuid"]
        if token.shape[1] == 0 and backend.llm_end_dict[uid] is True:
            with backend.lock:
                for cache in (backend.tts_speech_token_dict, backend.llm_end_dict, backend.hift_cache_dict):
                    cache.pop(uid)
            raise RuntimeError("SG completed with no speech tokens")
        return original(*args, **kwargs)

    backend.token2wav = token2wav


def _write_jsonl(path, rows):
    write_immutable_bytes(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--audio-review-model")
    parser.add_argument("--audio-review-base-url")
    parser.add_argument("--audio-review-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--review-mode", choices=("audio", "asr_transcript"), default="audio")
    parser.add_argument("--review-asr-model", type=Path)
    parser.add_argument("--review-asr-adapter", type=Path)
    parser.add_argument("--review-asr-adapter-sha256")
    parser.add_argument("--review-asr-base-model-sha256")
    parser.add_argument("--review-asr-device", default="cuda:0")
    parser.add_argument("--review-asr-worker", action="store_true")
    parser.add_argument("--review-allow-pending", action="store_true")
    parser.add_argument("--generation-plan", type=Path, help="Bound per-sample mode and instruction choices")
    parser.add_argument("--reuse-manifest", type=Path, help="Verified migration entries for already completed assignments")
    parser.add_argument("--generate-only-smoke", action="store_true",
                        help="Exactly one technical generation; unreviewed, never admitted to training.")
    return parser.parse_args(argv)


def transcribe_for_review(audio, output, args):
    """The ASR subprocess receives audio only, with no target/reference hints.

    A new process uses the ASR environment instead of the loaded SG Transformers
    overlay. Its complete receipt is subsequently validated by TranscriptReviewer.
    """
    command = [sys.executable, str(REPO / "examples/transcribe_sg_review.py"),
        "--audio", str(audio), "--output", str(output),
        "--base-model", str(args.review_asr_model), "--adapter", str(args.review_asr_adapter),
        "--device", args.review_asr_device]
    with output.with_suffix(".log").open("x", encoding="utf-8") as handle:
        subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return output


def run(args):
    # Importing this module does not load Transformers. The SG loader inserts its
    # frozen 4.51 overlay before loading the runtime; ASR uses a separate process.
    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts

    manifest = tts.load_frozen_manifest(args.manifest)
    if args.generate_only_smoke and len(manifest.samples) != 1:
        raise ValueError("generate-only smoke requires exactly one assignment")
    if not args.generate_only_smoke and (not args.audio_review_model or not args.audio_review_base_url):
        raise ValueError("live SG admission requires audio-review model and base URL")
    if not args.generate_only_smoke and args.review_mode == "asr_transcript" and not all((
            args.review_asr_model, args.review_asr_adapter, args.review_asr_adapter_sha256,
            args.review_asr_base_model_sha256)):
        raise ValueError("transcript review requires the fixed review ASR model, adapter and their hashes")
    if any(sample.seed > 2**63 - 3 for sample in manifest.samples):
        raise ValueError("initial seed must leave room for two signed-int64 retry seeds")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    raw_rows = {row["sample_id"]: row for row in (json.loads(line) for line in args.manifest.read_text().splitlines())}
    choices = {}
    if getattr(args, "generation_plan", None):
        from therapist_harness.generation_design import load_generation_choices
        choices = load_generation_choices(args.generation_plan, raw_rows)
    started = time.time()
    outputs = []
    reviews = []
    accepted = []
    stage = "review_initialization"
    active_sample = None
    reviewer = None
    asr_worker = None
    runtime = None
    reuse = {}
    if args.reuse_manifest is not None:
        reuse = {x["sample_id"]: x for x in json.loads(args.reuse_manifest.read_text())["assignments"]}
    timing = {"worker_startup_seconds": 0.0, "tts_load_seconds": 0.0, "reused_count": 0}
    rulebook = ("pass keeps the audio; pending quarantines without automatic retry; obvious_failure retries once, "
                "then tries instruct2, then excludes. Unclear patient speech and ordinary ASR errors alone are not failure. "
                "All judgments are coarse, not verbatim verification.") if getattr(args, 'review_allow_pending', False) else RULEBOOK
    config = {
        "schema_version": "reviewed_sg_v2" if args.review_mode == "asr_transcript" else "reviewed_sg_v1", "manifest": {"path": str(args.manifest), "sha256": manifest.sha256,
        "sample_count": len(manifest.samples)}, "rulebook": rulebook,
        "modes": ["zero_shot", "zero_shot", "instruct2"], "seed_rule": "assignment seed + attempt (0,1,2)",
        "instruct2_instruction": INSTRUCTION, "technical_failure_policy": "stop_no_automatic_pass_or_quality_retry",
        "generate_only_smoke": args.generate_only_smoke, "audio_review_model": args.audio_review_model,
        "audio_review_base_url": args.audio_review_base_url,
        "review_mode": args.review_mode,
        "review_allow_pending": getattr(args, "review_allow_pending", False),
        "generation_plan": {"path": str(args.generation_plan), "sha256": sha256_file(args.generation_plan)} if getattr(args, "generation_plan", None) else None,
        "review_asr_execution": "persistent_worker" if args.review_asr_worker else "oneshot",
        "reuse_manifest": {"path": str(args.reuse_manifest), "sha256": sha256_file(args.reuse_manifest)} if args.reuse_manifest else None,
        "review_asr_adapter": str(args.review_asr_adapter) if args.review_asr_adapter else None,
        "review_asr_adapter_sha256": args.review_asr_adapter_sha256,
        "review_asr_base_model_sha256": args.review_asr_base_model_sha256,
        "source_sha256": {str(Path(__file__)): sha256_file(Path(__file__)), str(Path(tts.__file__)): sha256_file(Path(tts.__file__)),
                          **{str(path): sha256_file(path) for path in (
                              REPO / "src/therapist_harness/audio_review.py",
                              REPO / "src/therapist_harness/transcript_review.py",
                              REPO / "src/therapist_harness/generation_design.py",
                              REPO / "src/therapist_harness/prompts/transcript_review.md",
                              REPO / "src/therapist_harness/prompts/transcript_review_v2.md",
                              REPO / "examples/transcribe_sg_review.py")}},
    }
    write_immutable_json(args.out_dir / "generation_config.json", config)
    try:
        if not args.generate_only_smoke:
            if args.review_mode == "asr_transcript":
                from therapist_harness.transcript_review import TranscriptReviewer
                reviewer = TranscriptReviewer(model=args.audio_review_model, base_url=args.audio_review_base_url,
                    api_key_env=args.audio_review_api_key_env, transcription_model=str(args.review_asr_model.resolve()),
                    transcription_base_model_sha256=args.review_asr_base_model_sha256,
                    transcription_adapter_sha256=args.review_asr_adapter_sha256,
                    allow_pending=getattr(args, "review_allow_pending", False))
            else:
                reviewer = AudioReviewer(model=args.audio_review_model, base_url=args.audio_review_base_url,
                                         api_key_env=args.audio_review_api_key_env)
                capability = reviewer.verify_audio_capability(manifest.samples[0].reference_wav)
                write_immutable_json(args.out_dir / "audio_capability.json", capability)
                if not capability["passed"] or not capability["live_api"]:
                    raise RuntimeError("live audio capability has not been demonstrated")
        stage = "runtime_load"
        if args.review_asr_worker and args.review_mode == "asr_transcript" and len(reuse) < len(manifest.samples):
            from therapist_harness.review_worker import ReviewASRWorker
            asr_worker = ReviewASRWorker(base_model=args.review_asr_model, adapter=args.review_asr_adapter,
                session_path=args.out_dir / "review_worker.json", device=args.review_asr_device)
            timing["worker_startup_seconds"] = asr_worker.startup_seconds
        load_start = time.perf_counter()
        model, runtime = tts.load_fixed_runtime()
        timing["tts_load_seconds"] = time.perf_counter() - load_start
        _install_empty_token_guard(model.model)
        runtime["reviewed_generation"] = {"rulebook": rulebook, "instruct2_instruction": INSTRUCTION,
                                          "empty_token_guard": "completed empty tokens raise a technical error; no retry"}
        write_immutable_json(args.out_dir / "runtime.json", runtime)
        for index, sample in enumerate(manifest.samples):
            active_sample = sample.sample_id
            # Manifest IDs may contain slashes; attempt filenames use a stable
            # index and ID hash, while every receipt preserves the original ID.
            stem = f"{index:04d}_{hashlib.sha256(sample.sample_id.encode()).hexdigest()[:12]}"
            attempt_dir = args.out_dir / "attempts" / stem
            attempt_dir.mkdir(parents=True)
            attempt_timings = []

            def generate(mode, seed, attempt):
                nonlocal stage
                begin = time.perf_counter()
                stage = "generation"
                if sha256_file(sample.reference_wav) != sample.reference_wav_sha256:
                    raise RuntimeError("TTS reference changed before generation")
                attempt_sample = replace(sample, output_wav=attempt_dir / f"attempt_{attempt}_{mode}.wav", seed=seed)
                selected_model = model if mode == "zero_shot" else InstructProxy(model)
                effective_mode = mode
                if sample.sample_id in choices:
                    from therapist_harness.generation_design import ConfiguredGenerator
                    choice = choices[sample.sample_id]
                    effective_mode = choice["mode"] if attempt < 2 else "instruct2"
                    selected_model = ConfiguredGenerator(model, effective_mode, choice["instruction_id"], sample.reference_text)
                record, temporary = tts._stage_audio_once(attempt_sample, selected_model)
                try:
                    os.link(temporary, attempt_sample.output_wav)
                finally:
                    temporary.unlink(missing_ok=True)
                record["generation_mode"] = effective_mode
                if sample.sample_id in choices:
                    record["generation_choice"] = choices[sample.sample_id]
                    record["effective_prompt_text"] = selected_model.effective_prompt
                    record["effective_instruction_id"] = selected_model.instruction_id
                record["attempt"] = attempt
                write_immutable_json(attempt_dir / f"attempt_{attempt}.generation.json", record)
                attempt_timings.append({"attempt": attempt, "tts_seconds": time.perf_counter() - begin})
                return record

            def inspect(record, attempt):
                nonlocal stage
                stage = "coarse_review"
                if args.review_mode == "asr_transcript":
                    stage = "review_transcription"
                    begin = time.perf_counter()
                    output = attempt_dir / f"attempt_{attempt}.transcription.json"
                    receipt = asr_worker.transcribe(record["output_wav"], output) if asr_worker else transcribe_for_review(record["output_wav"], output, args)
                    attempt_timings[-1]["asr_request_seconds"] = time.perf_counter() - begin
                    decoded_timing = json.loads(receipt.read_text()).get("timing", {})
                    attempt_timings[-1]["asr_decode_seconds"] = decoded_timing.get("decode_seconds")
                    attempt_timings[-1]["asr_wait_or_overhead_seconds"] = max(0.0,
                        attempt_timings[-1]["asr_request_seconds"] - decoded_timing["decode_seconds"]) if "decode_seconds" in decoded_timing else None
                    stage = "transcript_review"
                    begin = time.perf_counter()
                    judgment = reviewer.review(record["output_wav"], sample.target_text,
                        sample.reference_wav, sample.reference_text, receipt)
                    from therapist_harness.transcript_review import validate_transcript_review
                    validate_transcript_review(judgment.to_dict(), expected_adapter_sha256=args.review_asr_adapter_sha256,
                        expected_base_model_sha256=args.review_asr_base_model_sha256, require_worker_closed=False)
                    attempt_timings[-1]["agent_seconds"] = time.perf_counter() - begin
                else:
                    judgment = reviewer.review(record["output_wav"], sample.target_text,
                                               sample.reference_wav, sample.reference_text)
                    if not judgment.audio_inspected:
                        raise RuntimeError("review did not inspect the actual generated audio")
                if judgment.candidate_audio_sha256 != record["output_wav_sha256"]:
                    raise RuntimeError("review is bound to different generated audio")
                if (judgment.target_text != sample.target_text
                        or judgment.reference_audio_sha256 != sample.reference_wav_sha256
                        or judgment.reference_text != sample.reference_text):
                    raise RuntimeError("review conditioning differs from the generation assignment")
                write_immutable_json(attempt_dir / f"attempt_{attempt}.review.json", judgment.to_dict())
                return judgment

            if sample.sample_id in reuse:
                entry = reuse[sample.sample_id]
                source = Path(entry["assignment_result"])
                if sha256_file(source) != entry["assignment_result_sha256"]:
                    raise ValueError("reused assignment receipt changed")
                previous = json.loads(source.read_text())
                from therapist_harness.transcript_review import validate_transcript_review
                validate_transcript_review(previous["review"], args.review_asr_adapter_sha256, args.review_asr_base_model_sha256)
                expected = {"sample_id": sample.sample_id, "target_text": sample.target_text,
                            "reference_text": sample.reference_text, "reference_wav_sha256": sample.reference_wav_sha256,
                            "assignment_seed": sample.seed}
                if (any(previous.get(k) != v for k, v in expected.items())
                        or sha256_file(Path(previous["output_wav"])) != previous["output_wav_sha256"]
                        or previous["review"]["candidate_audio_sha256"] != previous["output_wav_sha256"]):
                    raise ValueError("reused assignment differs from its frozen identity/audio")
                attempts, action = previous["attempt_records"], previous["action"]
                if action not in {"keep", "exclude"}:
                    raise ValueError("only completed reviewed assignments can be reused")
                timing["reused_count"] += 1
            elif args.generate_only_smoke:
                record = generate("zero_shot", sample.seed, 0)
                attempts = [{"attempt": 0, "mode": "zero_shot", "seed": sample.seed,
                             "generation": record, "review": None, "action": "unreviewed_technical_smoke"}]
                action = "unreviewed_technical_smoke"
            else:
                attempts, action = run_attempts(sample.seed, generate, inspect)
            final = attempts[-1]
            final_review = final["review"]
            final_record = final["generation"]
            # This link fulfils the materialized assignment's output contract.
            # Its existence alone never permits sampling/training admission.
            os.link(final_record["output_wav"], sample.output_wav)
            output = {**final_record, "output_wav": str(sample.output_wav),
                      "final_attempt_wav": final_record["output_wav"], "assignment_seed": sample.seed,
                      "review_verdict": final_review["verdict"] if final_review else "unreviewed",
                      "review": final_review, "admitted": action == "keep", "action": action,
                      "attempt_records": attempts, "execution_timing": attempt_timings,
                      "reuse_provenance": reuse.get(sample.sample_id)}
            outputs.append(output)
            if final_review is not None:
                reviews.append({"sample_id": sample.sample_id, "audio_id": sample.sample_id, **final_review})
            if action == "keep":
                original = raw_rows[sample.sample_id]
                accepted.append({**original, "source": "synthetic", "audio_id": sample.sample_id,
                                 "utt_id": sample.sample_id, "audio": str(sample.output_wav),
                                 "audio_sha256": final_record["output_wav_sha256"], "target_text": sample.target_text,
                                 "clean_gt": sample.target_text, "prompt": "", "language": "Cantonese",
                                 "reference_audio": str(sample.reference_wav), "reference_text": sample.reference_text,
                                 "generation_mode": final_record["generation_mode"], "generation_seed": final["seed"],
                                 "review": final_review, "chosen_status": "intended_target_after_coarse_review_not_word_verified"})
            write_immutable_json(attempt_dir / "assignment_result.json", output)
            print(json.dumps({"completed": index + 1, "requested": len(manifest.samples),
                              "sample_id": sample.sample_id, "action": action, "attempt_count": len(attempts),
                              "reused": sample.sample_id in reuse}), flush=True)
        stage = "finalization"
        if asr_worker is not None:
            asr_worker.close()
        if not args.generate_only_smoke and args.review_mode == "asr_transcript":
            from therapist_harness.transcript_review import validate_transcript_review
            for review in reviews:
                validate_transcript_review(review, args.review_asr_adapter_sha256, args.review_asr_base_model_sha256)
        if sha256_file(args.manifest) != manifest.sha256:
            raise RuntimeError("manifest changed during generation")
        summary = {**config, "status": "complete_unreviewed_technical_smoke" if args.generate_only_smoke else "complete",
                   "runtime": runtime, "output_count": len(outputs), "admitted_count": len(accepted),
                   "excluded_count": sum(o["action"] == "exclude" for o in outputs),
                   "pending_count": sum(o["action"] == "quarantine" for o in outputs),
                   "generation_attempt_count": sum(len(o["attempt_records"]) for o in outputs),
                   "new_generation_attempt_count": sum(len(o["attempt_records"]) for o in outputs if not o["reuse_provenance"]),
                   "reused_assignment_count": timing["reused_count"],
                   "new_audio_count": len(outputs) - timing["reused_count"],
                   "outputs": outputs, "elapsed_seconds": time.time() - started,
                   "timing": timing,
                   "admission_rule": "only accepted.jsonl with a live bound pass in the declared review mode is training input; coarse pass does not prove word fidelity"}
        if not args.generate_only_smoke:
            _write_jsonl(args.out_dir / "reviews.jsonl", reviews)
            _write_jsonl(args.out_dir / "accepted.jsonl", accepted)
        write_immutable_json(args.out_dir / "generation_summary.json", summary)
        return summary
    except Exception as error:
        failure = {**config, "status": "failed", "stage": stage, "failed_sample_id": active_sample,
                   "completed_sample_ids": [o["sample_id"] for o in outputs], "error_type": type(error).__name__,
                   "http_status": getattr(error, "status_code", None), "elapsed_seconds": time.time() - started,
                   "training_admission": "none_from_incomplete_batch"}
        if reviewer is not None and getattr(reviewer, "capability_evidence", None) is not None:
            failure["capability"] = reviewer.capability_evidence
        write_immutable_json(args.out_dir / "generation_failure.json", failure)
        raise
    finally:
        if asr_worker is not None and not asr_worker.closed:
            asr_worker.abort()


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run(args)
        print(json.dumps({"status": result["status"], "generated": result["output_count"],
                          "admitted": result["admitted_count"], "output": str(args.out_dir)}), flush=True)
        return 0
    except Exception as error:
        # API errors can contain echoed request bodies. Persist only compact
        # diagnostic categories, never audio base64 or credentials.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "http_status": getattr(error, "status_code", None), "output": str(args.out_dir)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
