"""Current-ASR samples to audio preferences, with train/feedback isolation.

The only negatives are observed ASR transcripts. A coarse SG review permits
using the intended target as chosen; it does not establish word-level fidelity.
This module has no torch/model imports so pair construction also runs on CPU.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .store import sha256_file
from .text import text_normalize


def target_text(row: Mapping[str, Any]) -> str:
    value = next((row[k] for k in ("target_text", "chosen", "clean_gt", "text") if k in row), None)
    if not isinstance(value, str):
        raise ValueError("audio input requires a target text string")
    if value.startswith("language ") and "<asr_text>" in value:
        value = value.split("<asr_text>", 1)[1]
    if not text_normalize(value):
        raise ValueError("chosen text is empty after project normalization")
    return value


def _audio_hash(row: Mapping[str, Any]) -> str:
    path = Path(row["audio"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"audio must be an existing absolute file: {path}")
    actual = sha256_file(path)
    if row.get("audio_sha256", actual) != actual:
        raise ValueError(f"audio hash mismatch: {path}")
    return actual


def validate_sampling_inputs(
    inputs: Sequence[Mapping[str, Any]],
    real_train_rows: Sequence[Mapping[str, Any]],
    feedback_rows: Sequence[Mapping[str, Any]],
    review_receipts: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Bind real GT to the train pool and synthetic targets to an audio review.

    Split counts are owned by the frozen outer protocol (2117/511 in production),
    so this boundary also supports small, genuinely disjoint smoke fixtures.
    """
    if not real_train_rows or not feedback_rows:
        raise ValueError("both the real training pool and feedback split are required")
    real_by_hash: dict[str, set[str]] = defaultdict(set)
    for row in real_train_rows:
        real_by_hash[_audio_hash(row)].add(target_text(row))
    feedback_hashes = {_audio_hash(row) for row in feedback_rows}
    feedback_ids = {r["utt_id"] for r in feedback_rows if r.get("utt_id")}
    feedback_texts = {text_normalize(target_text(row)) for row in feedback_rows}
    if set(real_by_hash) & feedback_hashes:
        raise ValueError("real training pool overlaps feedback audio")
    if {text_normalize(t) for texts in real_by_hash.values() for t in texts} & feedback_texts:
        raise ValueError("real training pool overlaps feedback content")

    reviews = {r["candidate_audio_sha256"]: dict(r) for r in review_receipts}
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    for original in inputs:
        row = dict(original)
        audio_hash = _audio_hash(row)
        audio_id = str(row.get("audio_id") or row.get("utt_id") or audio_hash)
        if audio_id in seen_ids or audio_hash in seen_audio:
            raise ValueError(f"duplicate input audio or audio_id: {audio_id}")
        seen_ids.add(audio_id)
        seen_audio.add(audio_hash)
        if audio_hash in feedback_hashes or row.get("utt_id") in feedback_ids:
            raise ValueError(f"feedback audio cannot be sampled for training: {audio_id}")
        chosen = target_text(row)
        if text_normalize(chosen) in feedback_texts:
            raise ValueError(f"feedback content cannot be used for training: {audio_id}")
        if row.get("language", "Cantonese") != "Cantonese" or row.get("prompt", "") != "":
            raise ValueError("this protocol uses Cantonese and an empty context prompt")
        if row.get("source") == "real":
            if chosen not in real_by_hash.get(audio_hash, set()):
                raise ValueError(f"real audio/target is not in the training pool: {audio_id}")
            chosen_status = "real_training_transcript"
        elif row.get("source") == "synthetic":
            if audio_hash in real_by_hash:
                raise ValueError("synthetic audio is actually a real training recording")
            review = row.get("review") or reviews.get(audio_hash)
            if hasattr(review, "to_dict"):
                review = review.to_dict()
            if not isinstance(review, Mapping):
                raise ValueError(f"synthetic audio requires its audio review: {audio_id}")
            review_mode = review.get("review_mode", "audio")
            if review_mode == "asr_transcript":
                from .transcript_review import validate_transcript_review

                validate_transcript_review(review)
            elif review_mode != "audio" or review.get("audio_inspected") is not True:
                raise ValueError(f"synthetic audio has not passed audio inspection: {audio_id}")
            if review.get("verdict") != "pass":
                raise ValueError(f"synthetic audio has not passed its coarse review: {audio_id}")
            if review.get("candidate_audio_sha256") != audio_hash or review.get("target_text") != chosen:
                raise ValueError(f"audio review is bound to different audio/target: {audio_id}")
            reference_path = row.get("reference_audio") or row.get("reference_wav") or row.get("prompt_wav")
            reference_text = row.get("reference_text", row.get("prompt_text"))
            if not reference_path or not isinstance(reference_text, str):
                raise ValueError("synthetic input requires TTS reference audio and text")
            reference_hash = _audio_hash({"audio": reference_path})
            if reference_hash not in real_by_hash or reference_text not in real_by_hash[reference_hash]:
                raise ValueError("TTS reference must belong to the real training pool")
            if review.get("reference_audio_sha256") != reference_hash or review.get("reference_text") != reference_text:
                raise ValueError("audio review is bound to a different TTS reference")
            row["review"] = dict(review)
            row["reference_audio"] = str(reference_path)
            row["reference_audio_sha256"] = reference_hash
            chosen_status = "intended_target_after_coarse_review_not_word_verified"
        else:
            raise ValueError("source must be real or synthetic")
        result.append({**row, "audio_id": audio_id, "utt_id": row.get("utt_id", audio_id),
                       "audio_sha256": audio_hash, "target_text": chosen,
                       "chosen_status": chosen_status, "prompt": "", "language": "Cantonese"})
    if not result:
        raise ValueError("sampling inputs are empty")
    return result


def sample_seed(seed: int, audio_id: str, sample_index: int) -> int:
    """Stable per-draw RNG independent of source manifest order and batching."""
    payload = json.dumps([seed, audio_id, sample_index], ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def build_preference_pairs(
    inputs: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    current_asr_sha256: str,
    *,
    expected_samples: int = 4,
    greedy_candidates: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Construct one pair per distinct normalized wrong transcript per audio.

    Call validate_sampling_inputs first. All-correct audio contributes no DPO
    pair; repeated identical errors contribute one pair. Empty actual ASR output
    is a legitimate rejected response (EOS), not a fabricated negative.
    """
    if expected_samples < 1:
        raise ValueError("expected_samples must be positive")
    by_id = {row["audio_id"]: row for row in inputs}
    if not by_id or len(by_id) != len(inputs):
        raise ValueError("input audio IDs must be nonempty and unique")
    if len(current_asr_sha256) != 64:
        raise ValueError("current ASR must have a SHA256 identity")
    sampled: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    normal = {}
    if greedy_candidates is not None:
        normal = {x["audio_id"]:x for x in greedy_candidates}
        if len(normal) != len(greedy_candidates) or set(normal) != set(by_id):
            raise ValueError("normal decodes must cover the same unique audio inputs")
        for audio_id, value in normal.items():
            if (value.get("current_asr_sha256") != current_asr_sha256
                    or value.get("audio_sha256") != by_id[audio_id]["audio_sha256"]
                    or value.get("do_sample") is not False or value.get("num_beams") != 1
                    or value.get("language") != "Cantonese" or value.get("prompt") != ""):
                raise ValueError("normal decoding differs from the current parent/audio/conditioning")
    for candidate in candidates:
        audio_id = candidate["audio_id"]
        if audio_id not in by_id:
            raise ValueError(f"sample refers to unknown audio: {audio_id}")
        if candidate.get("current_asr_sha256") != current_asr_sha256:
            raise ValueError("candidate came from a different current ASR")
        if candidate.get("audio_sha256") != by_id[audio_id]["audio_sha256"]:
            raise ValueError("candidate came from different audio")
        if not isinstance(candidate.get("prediction"), str):
            raise ValueError("candidate prediction must be a string")
        if candidate.get("language", "Cantonese") != "Cantonese" or candidate.get("prompt", "") != "":
            raise ValueError("candidate conditioning differs from training")
        sampled[audio_id].append(candidate)

    pairs: list[dict[str, Any]] = []
    audio_audit: list[dict[str, Any]] = []
    for audio_id, row in by_id.items():
        draws = sampled[audio_id]
        indices = [draw["sample_index"] for draw in draws]
        if len(draws) != expected_samples or sorted(indices) != list(range(expected_samples)):
            raise ValueError(f"expected exactly {expected_samples} unique draws for {audio_id}")
        chosen = target_text(row)
        normalized_chosen = text_normalize(chosen)
        errors: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        correct = 0
        for draw in sorted(draws, key=lambda value: value["sample_index"]):
            normalized = text_normalize(draw["prediction"])
            if normalized == normalized_chosen:
                correct += 1
            else:
                errors[normalized].append(draw)
        for error_index, (normalized_rejected, matches) in enumerate(errors.items()):
            pairs.append({
                "schema_version": "asr_preference_v1", "pair_id": f"{audio_id}::error_{error_index:02d}",
                "audio_id": audio_id, "utt_id": row.get("utt_id", audio_id),
                "audio": row["audio"], "audio_sha256": row["audio_sha256"],
                "source": row["source"], "chosen": chosen, "rejected": matches[0]["prediction"],
                "prompt": "", "language": "Cantonese", "pair_weight": 1.0 / len(errors),
                "chosen_status": row["chosen_status"], "current_asr_sha256": current_asr_sha256,
                "sample_ids": [draw.get("sample_id", f"{audio_id}::{draw['sample_index']}") for draw in matches],
                "sample_indices": [draw["sample_index"] for draw in matches],
                "normalized_chosen": normalized_chosen, "normalized_rejected": normalized_rejected,
                "metadata": {key: row[key] for key in ("speaker_id", "prompt_id", "proposal_id", "roles", "role", "target_ids", "review", "reference_audio_sha256") if key in row},
            })
        audio_audit.append({"audio_id": audio_id, "source": row["source"], "samples": len(draws),
                            "correct_samples": correct, "distinct_errors": len(errors),
                            "pair_count": len(errors), "status": "all_correct_no_pair" if not errors else "paired"})
        if normal:
            greedy_correct = text_normalize(normal[audio_id]["prediction"]) == normalized_chosen
            audio_audit[-1].update(greedy_correct=greedy_correct,
                error_origin="greedy_error" if not greedy_correct else ("stochastic_only" if errors else "all_correct"))
    counts = Counter(row["source"] for row in inputs)
    return pairs, {
        "schema_version": "asr_preference_audit_v1", "current_asr_sha256": current_asr_sha256,
        "normalization": "therapist_harness.text.text_normalize", "audio_count": len(inputs),
        "sample_count": len(candidates), "pair_count": len(pairs), "source_audio_counts": dict(counts),
        "paired_audio_count": sum(r["pair_count"] > 0 for r in audio_audit),
        "all_correct_audio_count": sum(r["pair_count"] == 0 for r in audio_audit),
        "pair_weight_rule": "each contributing audio sums to one across its distinct errors",
        "chosen_caveat": "coarse synthetic review does not prove word-level target fidelity",
        "audio_results": audio_audit,
    }


def sft_rows(inputs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The SFT counterpart retains every admitted audio, including all-correct."""
    return [{**row, "text": "language Cantonese<asr_text>" + target_text(row),
             "chosen": target_text(row)} for row in inputs]
