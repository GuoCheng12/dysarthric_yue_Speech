from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from httpx import Timeout
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema
from openai.types.responses.response import Response

from .grounding import validate_stimulus_grounding
from .outcome import validate_reflection, validate_reflection_pack
from .round_schema import ReflectionPack, ReflectionV2
from .schema import (
    EvidencePack,
    Proposal,
    RunProtocol,
    canonical_sha256,
    validate_evidence_pack,
    validate_proposal,
)


PRESCRIBE_TASK = (
    "Prescribe one next-round synthetic practice batch from the "
    "supplied evidence. Return only the Proposal JSON required by "
    "the response schema; do not execute synthesis or training."
)
REFLECT_TASK = (
    "Reflect on this completed round using only the supplied "
    "ReflectionPack and active Skill. Return only the Reflection "
    "JSON required by the response schema, with qualitative "
    "next-round guidance, not a new Proposal."
)


def serialize_response(response: Response) -> dict[str, Any]:
    """Persist API field names so the SDK can validate the saved Response again."""
    return response.model_dump(mode="json", by_alias=True)


def build_therapist_request(
    protocol: RunProtocol,
    evidence: EvidencePack,
    *,
    system_prompt: str,
    skill_text: str,
) -> dict[str, Any]:
    validate_evidence_pack(evidence, protocol)
    evidence_payload = evidence.model_dump(mode="json")
    serialized = json.dumps(evidence_payload, ensure_ascii=False)
    lowered = serialized.casefold()
    forbidden_fragments = ("/data/", "audio_path", "speaker_id", "patient_id")
    found = [value for value in forbidden_fragments if value in lowered]
    if re.search(r"\bvlink[0-9a-z_/-]*", lowered):
        found.append("vlink identifier")
    if found:
        raise ValueError(
            f"agent evidence contains private identifiers or paths: {found}"
        )

    return {
        "model": protocol.agent.model,
        "reasoning": {"effort": protocol.agent.reasoning_effort},
        "store": protocol.agent.store,
        "stream": True,
        "max_output_tokens": protocol.agent.max_output_tokens,
        "input": [
            {"role": "system", "content": system_prompt},
            {
                "role": "developer",
                "content": "Current therapist skill:\n" + skill_text,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": PRESCRIBE_TASK,
                        "evidence_sha256": canonical_sha256(evidence),
                        "evidence": evidence_payload,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "therapist_proposal",
                "strict": True,
                "schema": to_strict_json_schema(Proposal),
            }
        },
    }


def call_therapist(
    protocol: RunProtocol,
    evidence: EvidencePack,
    *,
    system_prompt_path: Path,
    skill_path: Path,
) -> tuple[Proposal, dict[str, Any], dict[str, Any]]:
    request = build_therapist_request(
        protocol,
        evidence,
        system_prompt=system_prompt_path.read_text("utf-8"),
        skill_text=skill_path.read_text("utf-8"),
    )
    response = execute_therapist_request(protocol, request)
    response_payload = serialize_response(response)
    proposal = parse_therapist_response(response, protocol, evidence)
    return proposal, request, response_payload


def execute_therapist_request(
    protocol: RunProtocol, request: dict[str, Any]
) -> Response:
    """Execute one logical request with bounded SDK retries before streaming.

    Interrupted streams and invalid model output still propagate to the caller.
    """

    api_key = os.environ.get(protocol.agent.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"missing API key environment variable {protocol.agent.api_key_env}"
        )
    client = OpenAI(
        api_key=api_key,
        base_url=str(protocol.agent.base_url),
        max_retries=protocol.agent.max_retries,
        timeout=Timeout(float(protocol.agent.timeout_seconds), connect=20.0),
        default_headers={"User-Agent": "curl/8.0"},
    )
    response: Response | None = None
    with client.responses.create(**request) as stream:
        for event in stream:
            if event.type in {
                "response.completed",
                "response.failed",
                "response.incomplete",
            }:
                response = event.response
                break
    if response is None:
        raise RuntimeError("Therapist response stream ended without a terminal event")
    return response


def terminal_output_text(response: Response) -> str:
    """Prefer the provider's final answer while preserving legacy unphased output."""
    final = [item for item in response.output
             if item.type == 'message' and getattr(item, 'phase', None) == 'final_answer']
    if not final:
        return response.output_text
    return ''.join(part.text for item in final for part in item.content if part.type == 'output_text')


def parse_therapist_response(
    response: Response,
    protocol: RunProtocol,
    evidence: EvidencePack,
) -> Proposal:
    """Parse and fully validate one terminal API response."""

    if response.status != "completed":
        raise RuntimeError(f"Therapist response ended with status={response.status}")
    if not response.output_text:
        raise RuntimeError("Therapist response completed without output text")
    proposal = Proposal.model_validate_json(terminal_output_text(response))
    validate_proposal(proposal, evidence, protocol)
    validate_stimulus_grounding(proposal, evidence)
    return proposal


def build_reflection_request(
    protocol: RunProtocol,
    evidence: EvidencePack,
    proposal: Proposal,
    pack: ReflectionPack,
    *,
    system_prompt: str,
    skill_text: str,
) -> dict[str, Any]:
    """Build the sole LLM request allowed after deterministic reassessment."""

    validate_reflection_pack(pack, protocol, evidence, proposal)
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    skill_hash = hashlib.sha256(skill_text.encode("utf-8")).hexdigest()
    if prompt_hash != pack.provenance.reflection_prompt_file_sha256:
        raise ValueError("reflection prompt hash differs from ReflectionPack")
    if skill_hash != pack.provenance.skill_file_sha256:
        raise ValueError("Therapist skill hash differs from ReflectionPack")
    instruction_text = (system_prompt + "\n" + skill_text).casefold()
    instruction_forbidden = [
        value
        for value in ("/data/", "speaker_id", "patient_id", "reference_wav")
        if value in instruction_text
    ]
    if re.search(r"\bvlink[0-9a-z_/-]*", instruction_text):
        instruction_forbidden.append("vlink identifier")
    if instruction_forbidden:
        raise ValueError(
            "reflection instructions contain private identifiers or paths: "
            f"{instruction_forbidden}"
        )
    payload = pack.model_dump(mode="json")
    return {
        "model": protocol.agent.model,
        "reasoning": {"effort": protocol.agent.reasoning_effort},
        "store": protocol.agent.store,
        "stream": True,
        "max_output_tokens": protocol.agent.max_output_tokens,
        "input": [
            {"role": "system", "content": system_prompt},
            {
                "role": "developer",
                "content": (
                    "Frozen Therapist skill used for this round "
                    "(interpretation only; it does not authorize a new prescription):\n"
                    + skill_text
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": REFLECT_TASK,
                        "reflection_pack_sha256": canonical_sha256(pack),
                        "reflection_pack": payload,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "therapist_reflection",
                "strict": True,
                "schema": to_strict_json_schema(ReflectionV2),
            }
        },
    }


def parse_reflection_response(response: Response, pack: ReflectionPack) -> ReflectionV2:
    """Parse one reflection and enforce Compiler-owned semantics."""

    if response.status != "completed":
        raise RuntimeError(f"Therapist reflection ended with status={response.status}")
    if not response.output_text:
        raise RuntimeError("Therapist reflection completed without output text")
    reflection = ReflectionV2.model_validate_json(terminal_output_text(response))
    validate_reflection(reflection, pack)
    return reflection


def call_reflection(
    protocol: RunProtocol,
    evidence: EvidencePack,
    proposal: Proposal,
    pack: ReflectionPack,
    *,
    system_prompt_path: Path,
    skill_path: Path,
) -> tuple[ReflectionV2, dict[str, Any], dict[str, Any]]:
    request = build_reflection_request(
        protocol,
        evidence,
        proposal,
        pack,
        system_prompt=system_prompt_path.read_text("utf-8"),
        skill_text=skill_path.read_text("utf-8"),
    )
    response = execute_therapist_request(protocol, request)
    reflection = parse_reflection_response(response, pack)
    return reflection, request, serialize_response(response)
