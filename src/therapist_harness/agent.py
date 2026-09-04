from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from .schema import (
    EvidencePack,
    Proposal,
    RunProtocol,
    canonical_sha256,
    validate_proposal,
)


def build_therapist_request(
    protocol: RunProtocol,
    evidence: EvidencePack,
    *,
    system_prompt: str,
    skill_text: str,
) -> dict[str, Any]:
    if evidence.run_id != protocol.run_id:
        raise ValueError("evidence run does not match protocol")
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
                "schema": Proposal.model_json_schema(),
            }
        },
    }


def call_therapist(
    protocol: RunProtocol,
    evidence: EvidencePack,
    *,
    system_prompt_path: Path,
    skill_path: Path,
) -> tuple[Proposal, dict[str, Any]]:
    api_key = os.environ.get(protocol.agent.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"missing API key environment variable {protocol.agent.api_key_env}"
        )
    request = build_therapist_request(
        protocol,
        evidence,
        system_prompt=system_prompt_path.read_text("utf-8"),
        skill_text=skill_path.read_text("utf-8"),
    )
    client = OpenAI(
        api_key=api_key,
        base_url=str(protocol.agent.base_url),
        max_retries=protocol.agent.max_retries,
        default_headers={"User-Agent": "curl/8.0"},
    )
    response = client.responses.create(**request)
    proposal = Proposal.model_validate_json(response.output_text)
    validate_proposal(proposal, evidence, protocol)
    return proposal, request
