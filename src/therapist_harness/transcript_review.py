"""Explicit transcript-based SG review authorized as a temporary audio-review mode.

The text Agent never receives audio or runs ASR with target/reference hints.
Its brief decision is not direct listening and is not word-level verification.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping

from httpx import Timeout
from openai import OpenAI
from pydantic import Field

from .audio_review import ReviewJudgment


PROMPT_PATH = Path(__file__).with_name("prompts") / "transcript_review.md"
SHA_PATTERN = r"^[0-9a-f]{64}$"


class ThreeWayReviewJudgment(ReviewJudgment):
    verdict: Literal["pass", "obvious_failure", "pending"]


class TranscriptReview(ReviewJudgment):
    verdict: Literal["pass", "obvious_failure", "pending"]
    review_mode: Literal["asr_transcript"] = "asr_transcript"
    audio_inspected: Literal[False] = False
    transcript_inspected: bool
    candidate_audio_sha256: str = Field(pattern=SHA_PATTERN)
    target_text: str = Field(min_length=1)
    reference_audio_sha256: str = Field(pattern=SHA_PATTERN)
    reference_text: str
    asr_transcription: str
    asr_raw_output: str
    transcription_model: str = Field(min_length=1)
    transcription_base_model_sha256: str = Field(pattern=SHA_PATTERN)
    transcription_adapter_sha256: str = Field(pattern=SHA_PATTERN)
    transcription_receipt_path: str = Field(min_length=1)
    transcription_receipt_sha256: str = Field(pattern=SHA_PATTERN)
    model: str = Field(min_length=1)
    response_id: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=SHA_PATTERN)
    fidelity_claim: Literal["coarse_only_not_word_verified"] = "coarse_only_not_word_verified"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_transcription(path: Path, *, require_worker_closed=False) -> tuple[dict[str, Any], str]:
    if not path.is_absolute() or not path.is_file():
        raise ValueError("transcription receipt must be an existing absolute file")
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if receipt.get("status") != "complete" or receipt.get("actual_transcription") is not True:
        raise ValueError("review requires a completed actual ASR transcription")
    if (receipt.get("ended_with_eos") is not True
            or receipt.get("reached_max_new_tokens") is not False
            or not isinstance(receipt.get("prediction"), str)):
        raise ValueError("technically incomplete ASR transcription belongs upstream")
    conditioning = receipt.get("decode_config", {})
    if conditioning.get("language") != "Cantonese" or conditioning.get("context") != "":
        raise ValueError("review ASR must use Cantonese with no target/reference context")
    if conditioning.get("do_sample") is not False or conditioning.get("num_beams") != 1:
        raise ValueError("review ASR must use the frozen greedy decoding configuration")
    audio = Path(receipt["audio"])
    if not audio.is_absolute() or not audio.is_file():
        raise ValueError("transcribed audio must be an existing absolute file")
    if _sha(audio.read_bytes()) != receipt.get("audio_sha256"):
        raise ValueError("transcription audio hash changed")
    if "worker_session_path" in receipt:
        session_path = Path(receipt["worker_session_path"])
        session_raw = session_path.read_bytes()
        session = json.loads(session_raw)
        if _sha(session_raw) != receipt.get("worker_session_sha256"):
            raise ValueError("review worker session hash changed")
        if (session.get("status") != "ready" or session.get("actual_transcription") is not True
                or any(session.get(k) != receipt.get(k) for k in (
                    "model", "adapter", "base_model_sha256", "adapter_sha256", "source_sha256", "decode_config"))):
            raise ValueError("transcription differs from its frozen worker session")
        final_path = session_path.with_suffix(".closed.json")
        if require_worker_closed or final_path.exists():
            final = json.loads(final_path.read_bytes())
            if (final.get("status") != "complete" or final.get("session_sha256") != receipt["worker_session_sha256"]
                    or final.get("base_files_unchanged") is not True or final.get("adapter_files_unchanged") is not True
                    or {"path": str(path), "sha256": _sha(raw)} not in final.get("receipts", [])):
                raise ValueError("review worker did not finalize this transcription with unchanged models")
    return receipt, _sha(raw)


def _validate_transcription_binding(review: TranscriptReview, receipt: Mapping[str, Any]) -> None:
    bindings = {
        "audio_sha256": review.candidate_audio_sha256,
        "prediction": review.asr_transcription,
        "raw_output": review.asr_raw_output,
        "model": review.transcription_model,
        "base_model_sha256": review.transcription_base_model_sha256,
        "adapter_sha256": review.transcription_adapter_sha256,
    }
    if any(receipt.get(key) != expected for key, expected in bindings.items()):
        raise ValueError("review differs from its actual transcription or fixed ASR identity")


def validate_transcript_review(
    review: Mapping[str, Any],
    expected_adapter_sha256: str | None = None,
    expected_base_model_sha256: str | None = None,
    *, require_worker_closed: bool = True,
) -> None:
    """Recheck receipt/audio/ASR bindings without ASR decoding or complex review.

    Callers separately bind chosen and TTS reference to the generation assignment.
    Final review ledgers may append sample_id/audio_id outside this receipt schema.
    """
    parsed = TranscriptReview.model_validate({
        key: review[key] for key in TranscriptReview.model_fields if key in review
    })
    if review.get("review_mode") != "asr_transcript" or review.get("audio_inspected") is not False:
        raise ValueError("transcript review must explicitly declare that audio was not inspected")
    if review.get("transcript_inspected") is not True:
        raise ValueError("a mock text review is not an actual transcript inspection")
    receipt, digest = _load_transcription(Path(parsed.transcription_receipt_path), require_worker_closed=require_worker_closed)
    if digest != parsed.transcription_receipt_sha256:
        raise ValueError("transcription receipt hash changed")
    _validate_transcription_binding(parsed, receipt)
    if expected_adapter_sha256 is not None and parsed.transcription_adapter_sha256 != expected_adapter_sha256:
        raise ValueError("transcription ASR adapter differs from the frozen review ASR")
    if expected_base_model_sha256 is not None and parsed.transcription_base_model_sha256 != expected_base_model_sha256:
        raise ValueError("transcription base model differs from the frozen review ASR")


class TranscriptReviewer:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        transcription_model: str,
        transcription_base_model_sha256: str,
        transcription_adapter_sha256: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 180,
        client: Any = None,
        allow_pending: bool = False,
    ) -> None:
        self.model = model
        self.transcription_model = transcription_model
        self.transcription_base_model_sha256 = transcription_base_model_sha256
        self.transcription_adapter_sha256 = transcription_adapter_sha256
        self.judgment_type = ThreeWayReviewJudgment if allow_pending else ReviewJudgment
        self.prompt = (PROMPT_PATH.with_name("transcript_review_v2.md") if allow_pending else PROMPT_PATH).read_text("utf-8")
        self._injected_client = client is not None
        if client is None:
            key = os.environ.get(api_key_env)
            if not key:
                raise RuntimeError(f"missing API key environment variable {api_key_env}")
            client = OpenAI(api_key=key, base_url=base_url, max_retries=4,
                            timeout=Timeout(timeout_seconds, connect=20.0),
                            default_headers={"User-Agent": "curl/8.0"})
        self.client = client

    def review(
        self,
        candidate_audio: str | Path,
        target_text: str,
        reference_audio: str | Path,
        reference_text: str,
        transcription_receipt: str | Path,
    ) -> TranscriptReview:
        receipt_path = Path(transcription_receipt).resolve()
        receipt, receipt_digest = _load_transcription(receipt_path)
        candidate_digest = _sha(Path(candidate_audio).read_bytes())
        if receipt.get("audio_sha256") != candidate_digest:
            raise ValueError("ASR transcription belongs to different candidate audio")
        expected_asr = {"model": self.transcription_model,
                        "base_model_sha256": self.transcription_base_model_sha256,
                        "adapter_sha256": self.transcription_adapter_sha256}
        if any(receipt.get(key) != value for key, value in expected_asr.items()):
            raise ValueError("transcription did not use the frozen review ASR")
        if not target_text.strip():
            raise ValueError("target text is required")
        reference_digest = _sha(Path(reference_audio).read_bytes())
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "low"},
            max_output_tokens=1024,
            store=False,
            input=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": json.dumps({
                    "asr_transcription": receipt["prediction"],
                    "target_text": target_text,
                    "reference_text": reference_text,
                }, ensure_ascii=False)},
            ],
            text={"format": {"type": "json_schema", "name": "sg_transcript_review",
                             "strict": True, "schema": self.judgment_type.model_json_schema()}},
        )
        if response.status != "completed" or not response.output_text:
            raise RuntimeError("transcript review did not complete with a judgment")
        judgment = self.judgment_type.model_validate_json(response.output_text)
        result = TranscriptReview(
            **judgment.model_dump(), transcript_inspected=not self._injected_client,
            candidate_audio_sha256=candidate_digest, target_text=target_text,
            reference_audio_sha256=reference_digest, reference_text=reference_text,
            asr_transcription=receipt["prediction"], asr_raw_output=receipt["raw_output"],
            transcription_model=self.transcription_model,
            transcription_base_model_sha256=self.transcription_base_model_sha256,
            transcription_adapter_sha256=self.transcription_adapter_sha256,
            transcription_receipt_path=str(receipt_path), transcription_receipt_sha256=receipt_digest,
            model=self.model, response_id=response.id, prompt_sha256=_sha(self.prompt.encode("utf-8")),
        )
        if not self._injected_client:
            validate_transcript_review(result.to_dict(), self.transcription_adapter_sha256,
                                       self.transcription_base_model_sha256, require_worker_closed=False)
        return result
