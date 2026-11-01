"""Coarse, audio-input SG review. This module does not run an ASR quality filter.

The live capability probe checks that the configured endpoint distinguishes sound
from silence using audio bytes. It is not a validation of clinical judgment or
word-for-word Cantonese transcription. No transcript-only fallback is available.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


PROMPT_PATH = Path(__file__).with_name("prompts") / "audio_review.md"
RULEBOOK = (
    "Only obvious_failure triggers regeneration: retry zero_shot once with the next "
    "seed, then try instruct2 once with the following seed. Review each new audio. "
    "After three obvious failures exclude the item. A pass keeps the audio, including "
    "unclear patient speech and uncertain suspected artifacts. Inaccessible audio is "
    "a technical error, never an automatic pass or a patient-quality failure."
)


class ReviewJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["pass", "obvious_failure"]
    reason: str = Field(min_length=1, max_length=480)


class AudioReview(ReviewJudgment):
    audio_inspected: bool
    candidate_audio_sha256: str
    target_text: str
    reference_audio_sha256: str
    reference_text: str
    model: str
    response_id: str
    prompt_sha256: str
    capability_probe_sha256: str
    fidelity_claim: Literal["coarse_only_not_word_verified"] = "coarse_only_not_word_verified"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def next_generation_action(
    review: AudioReview | ReviewJudgment, attempt: int
) -> Literal["keep", "retry_zero_shot", "retry_instruct2", "exclude", "quarantine"]:
    """Attempts 0, 1, 2 are initial zero-shot, zero-shot retry, instruct2 fallback."""
    if attempt not in (0, 1, 2):
        raise ValueError("review attempt must be 0, 1, or 2")
    if review.verdict == "pass":
        return "keep"
    if review.verdict == "pending":
        return "quarantine"
    return ("retry_zero_shot", "retry_instruct2", "exclude")[attempt]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _input_audio(value: bytes, audio_format: str = "wav") -> dict[str, Any]:
    return {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(value).decode("ascii"),
            "format": audio_format,
        },
    }


def _read_audio(path: Path) -> tuple[bytes, str]:
    audio_format = path.suffix.lower().lstrip(".")
    if audio_format not in ("wav", "mp3"):
        raise ValueError("audio review accepts WAV or MP3 attachments")
    value = path.read_bytes()
    if not value:
        raise ValueError("empty audio attachment")
    return value, audio_format


def _parse_json(response: Any) -> dict[str, Any]:
    if not response.choices or response.choices[0].finish_reason != "stop":
        raise RuntimeError("audio review did not finish normally")
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("audio reviewer returned no text")
    # Some audio-capable models return fenced JSON despite the short instruction.
    if content.strip().startswith("```json") and content.strip().endswith("```"):
        content = content.strip()[7:-3].strip()
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("audio reviewer must return a JSON object")
    return parsed


class AudioReviewer:
    """One configured audio model, one capability check, then simple listening calls.

    The client is injectable for unit tests; mocked tests never count as a live
    capability smoke. API credentials are read from an environment variable only.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 180,
        client: Any = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.prompt = PROMPT_PATH.read_text("utf-8")
        self.prompt_sha256 = _sha256(self.prompt.encode("utf-8"))
        self.capability_evidence: dict[str, Any] | None = None
        self._injected_client = client is not None
        if client is None:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(f"missing API key environment variable {api_key_env}")
            client = OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=timeout_seconds,
                max_retries=0,
                # Match the existing MAS client: the configured gateway rejects
                # the SDK's default User-Agent at its WAF.
                default_headers={"User-Agent": "curl/8.0"},
            )
        self.client = client

    def _call(self, system: str, content: list[dict[str, Any]]) -> Any:
        # Audio input is sent directly; no ASR transcript is substituted or added.
        return self.client.chat.completions.create(
            model=self.model,
            modalities=["text"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            max_tokens=300,
            temperature=0,
        )

    def verify_audio_capability(self, speech_audio: str | Path) -> dict[str, Any]:
        """A live audio-only speech/silence sanity check, independent of target text.

        Both requests use exactly the same textual instruction and WAV format. The
        speech clip must contain speech; the control has matching length and sample
        rate but zero samples. Only hashes and compact replies enter the receipt.
        """
        self.capability_evidence = None
        speech_audio = Path(speech_audio)
        samples, rate = sf.read(speech_audio, dtype="float32", always_2d=True)
        samples = samples[: rate * 8]
        if len(samples) == 0 or not np.any(samples):
            raise ValueError("capability probe requires a nonempty speech recording")
        prompt = (
            "Listen to the attached audio directly. Return JSON only with exactly "
            "label and reason. label must be audible_speech, silence, or other. "
            "Use audible_speech when any person is speaking, including unclear "
            "speech. Do not infer sound from this instruction. Keep reason brief."
        )
        evidence: list[dict[str, Any]] = []
        # Do not reveal which member is speech in either prompt or attachment name.
        for expected, waveform in (
            ("silence", np.zeros_like(samples)),
            ("audible_speech", samples),
        ):
            buf = io.BytesIO()
            sf.write(buf, waveform, rate, format="WAV", subtype="PCM_16")
            wav = buf.getvalue()
            response = self._call(prompt, [_input_audio(wav)])
            observed = _parse_json(response)
            evidence.append({
                "audio_sha256": _sha256(wav),
                "expected": expected,
                "observed": observed,
                "response_id": str(response.id),
            })
        passed = all(item["observed"].get("label") == item["expected"] for item in evidence)
        result = {
            "model": self.model,
            "base_url": self.base_url,
            "passed": passed,
            "live_api": not self._injected_client,
            "input_modality": "actual_wav_bytes",
            "scope": "audio_access_sanity_only_not_Cantonese_or_clinical_validation",
            "probe_prompt_sha256": _sha256(prompt.encode("utf-8")),
            "checks": evidence,
        }
        self.capability_evidence = result
        if not passed:
            raise RuntimeError("configured reviewer failed the audio-only capability probe")
        return result

    def review(
        self,
        candidate_audio: str | Path,
        target_text: str,
        reference_audio: str | Path,
        reference_text: str = "",
    ) -> AudioReview:
        if not self.capability_evidence or not self.capability_evidence["passed"]:
            raise RuntimeError("run verify_audio_capability before reviewing SG audio")
        if not target_text.strip():
            raise ValueError("target text is required")
        candidate, candidate_format = _read_audio(Path(candidate_audio))
        reference, reference_format = _read_audio(Path(reference_audio))
        response = self._call(self.prompt, [
            {"type": "text", "text": "Candidate target text: " + json.dumps(target_text, ensure_ascii=False)
             + "\nThe next attachment is the CANDIDATE recording."},
            _input_audio(candidate, candidate_format),
            {"type": "text", "text": "TTS reference text (may be unavailable): "
             + json.dumps(reference_text, ensure_ascii=False)
             + "\nThe next attachment is the TTS REFERENCE recording."},
            _input_audio(reference, reference_format),
        ])
        payload = _parse_json(response)
        if payload.get("error") == "audio_unavailable":
            raise RuntimeError("audio reviewer could not access/hear an attachment")
        judgment = ReviewJudgment.model_validate(payload)
        return AudioReview(
            **judgment.model_dump(),
            audio_inspected=bool(self.capability_evidence["live_api"]),
            candidate_audio_sha256=_sha256(candidate),
            target_text=target_text,
            reference_audio_sha256=_sha256(reference),
            reference_text=reference_text,
            model=self.model,
            response_id=str(response.id),
            prompt_sha256=self.prompt_sha256,
            capability_probe_sha256=_sha256(json.dumps(
                self.capability_evidence, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")),
        )
