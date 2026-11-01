"""Real-data curriculum: one sampling action, two LLM roles, measured memory.

The existing Therapist prescribes and reflects. The previously planned Skill
Optimizer now summarizes experience only. Neither role controls the trainer.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import math
from pathlib import Path
import signal
import time
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from openai.lib._pydantic import to_strict_json_schema
import httpx
from openai import APIConnectionError, APIError, APITimeoutError
from pydantic import BaseModel, ConfigDict, Field

from .agent import execute_therapist_request, serialize_response, terminal_output_text
from .schema import AgentConfig
from .store import sha256_file, write_immutable_json

PROMPTS = Path(__file__).with_name("prompts")
HISTORY_ROUNDS = 4
REQUEST_DEADLINE_SECONDS = 600


@contextmanager
def request_deadline(seconds=REQUEST_DEADLINE_SECONDS):
    """Bound total stream time, including streams that keep sending heartbeats."""
    def expired(signum, frame):
        raise TimeoutError('MAS request exceeded its wall-clock deadline')
    previous_handler = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


class SamplingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class GroupProbability(SamplingModel):
    group_id: str
    probability: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1)


class SamplingDecision(SamplingModel):
    round_index: int = Field(ge=0)
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnosis: str = Field(min_length=1, max_length=1800)
    allocations: list[GroupProbability] = Field(min_length=1)
    expected_change: str = Field(min_length=1, max_length=1000)
    history_used: str = Field(min_length=1, max_length=1000)


class SamplingReflection(SamplingModel):
    round_index: int = Field(ge=0)
    outcome_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_changes: str = Field(min_length=1, max_length=1800)
    evidence_ids: list[str] = Field(min_length=1)
    explanation: str = Field(min_length=1, max_length=1400)
    next_guidance: str = Field(min_length=1, max_length=1000)


class SamplingExperience(SamplingModel):
    experience: str = Field(min_length=1, max_length=1800)
    supporting_rounds: list[int] = Field(min_length=1)


def payload_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_agent_config(config: AgentConfig) -> None:
    if config.model != "gpt-5.6-sol":
        raise ValueError("all real-sampling Agent calls require gpt-5.6-sol")
    if urlparse(str(config.base_url)).hostname != "api.laicaishare.xyz":
        raise ValueError("real-sampling Agent calls require the configured laicai gateway")


def compact_history(history) -> list[dict[str, Any]]:
    """Full episodes stay on disk; only four recent summaries enter each call."""
    return [{key: episode[key] for key in ("round_index", "decision", "reflection", "experience")
             if key in episode} for episode in list(history)[-HISTORY_ROUNDS:]]


def validate_decision(decision: SamplingDecision, evidence: dict) -> dict[str, float]:
    _validate_dev_evidence(evidence)
    if decision.round_index != evidence["round_index"] or decision.evidence_sha256 != payload_hash(evidence):
        raise ValueError("sampling decision is bound to a different round or evidence")
    groups = evidence["groups"]
    expected = {group["group_id"] for group in groups}
    if len(expected) != len(groups) or any(group["train_count"] <= 0 for group in groups):
        raise ValueError("fixed sampling groups must be unique and nonempty")
    weights = {item.group_id: item.probability for item in decision.allocations}
    if len(weights) != len(decision.allocations) or set(weights) != expected:
        raise ValueError("sampling decision must include every fixed group exactly once")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("sampling probabilities must sum to one; no silent renormalization")
    known = set(evidence["observation_ids"])
    if any(not set(item.evidence_ids) <= known for item in decision.allocations):
        raise ValueError("sampling decision cites an unknown observation")
    return weights


def _validate_dev_evidence(evidence: dict) -> None:
    if evidence.get("feedback_role") != "dev":
        raise ValueError("MAS may receive dev feedback only; train and test feedback are excluded")


def build_sampling_request(config: AgentConfig, *, phase: str, context: dict) -> dict:
    validate_agent_config(config)
    output_type = {"prescribe": SamplingDecision, "reflect": SamplingReflection,
                   "experience": SamplingExperience}[phase]
    system = (PROMPTS / "real_sampling_instruction.md").read_text("utf-8")
    task = (PROMPTS / f"real_sampling_{phase}.md").read_text("utf-8")
    return {
        "model": config.model,
        "reasoning": {"effort": config.reasoning_effort},
        "store": False,
        "stream": True,
        "max_output_tokens": config.max_output_tokens,
        "input": [{"role": "system", "content": system},
                  {"role": "developer", "content": task},
                  {"role": "user", "content": json.dumps(context, ensure_ascii=False, sort_keys=True, allow_nan=False)}],
        "text": {"format": {"type": "json_schema", "name": f"real_sampling_{phase}",
                              "strict": True, "schema": to_strict_json_schema(output_type)}},
    }


def _execute_recorded(config, request, directory):
    """Retry timeouts and transport disconnections, preserving identical requests."""
    request_sha = sha256_file(directory / 'request.json')
    for attempt in range(1, 4):
        started = time.perf_counter()
        record = dict(attempt=attempt, request_sha256=request_sha, request_model=request['model'],
                      started_unix=time.time())
        record['deadline_seconds'] = REQUEST_DEADLINE_SECONDS
        write_immutable_json(directory / 'attempts' / f'{attempt:03d}.started.json', record)
        try:
            with request_deadline():
                response = execute_therapist_request(SimpleNamespace(agent=config), request)
        except Exception as error:
            stream_error = isinstance(error, APIError) and str(error) == 'stream_read_error'
            timeout_error = isinstance(error, (TimeoutError, httpx.TimeoutException, APITimeoutError))
            connection_error = isinstance(error, (APIConnectionError, httpx.NetworkError, httpx.RemoteProtocolError))
            retry = (stream_error or timeout_error or connection_error) and attempt < 3
            record.update(seconds=time.perf_counter()-started, response_status='no_terminal_response',
                          error_type=type(error).__name__, stream_read_error=stream_error,
                          timeout=timeout_error, connection_error=connection_error, retry=retry)
            write_immutable_json(directory / 'attempts' / f'{attempt:03d}.json', record)
            print(json.dumps(dict(stage='agent_attempt_failed', directory=str(directory), **record)), flush=True)
            if not retry:
                raise
            time.sleep(2)
        else:
            record.update(seconds=time.perf_counter()-started, response_status=response.status,
                          response_model=response.model, response_id=response.id, retry=False)
            write_immutable_json(directory / 'attempts' / f'{attempt:03d}.json', record)
            return response, attempt


def _call(config, *, phase, context, directory, validate, request=None, output_type=None, role=None):
    directory = Path(directory)
    validate_agent_config(config)
    request = request or build_sampling_request(config, phase=phase, context=context)
    if request['model'] != config.model:
        raise ValueError('request must use the configured Agent model')
    role = role or ("Skill Optimizer" if phase == "experience" else "Therapist")
    write_immutable_json(directory / "request.json", request)
    started = time.perf_counter()
    try:
        response, attempts = _execute_recorded(config, request, directory)
    except Exception as error:
        write_immutable_json(directory / "receipt.json", {
            "schema_version": "real_sampling_agent_call_v1",
            "role": role,
            "phase": phase, "provider_base_url": str(config.base_url),
            "request_model": request["model"], "response_model": None,
            "response_status": "no_terminal_response", "validated": False,
            "seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "attempts": len(list((directory / 'attempts').glob('[0-9][0-9][0-9].json'))),
            "request_sha256": sha256_file(directory / "request.json"),
        })
        raise
    seconds = time.perf_counter() - started
    response_payload = serialize_response(response)
    write_immutable_json(directory / "response.json", response_payload)
    receipt = {
        "schema_version": "real_sampling_agent_call_v1",
        "role": role,
        "phase": phase,
        "provider_base_url": str(config.base_url),
        "request_model": request["model"],
        "response_model": response.model,
        "response_id": response.id,
        "response_status": response.status,
        "reasoning_effort": config.reasoning_effort,
        "seconds": seconds,
        "attempts": attempts,
        "usage": response_payload.get("usage"),
        "request_sha256": sha256_file(directory / "request.json"),
        "response_sha256": sha256_file(directory / "response.json"),
        "validated": False,
    }
    output_type = output_type or {"prescribe": SamplingDecision, "reflect": SamplingReflection,
                                 "experience": SamplingExperience}[phase]
    try:
        if response.status != "completed":
            raise RuntimeError(f"Agent response ended with status={response.status}")
        if response.model != config.model:
            raise ValueError("actual response model differs from required gpt-5.6-sol")
        parsed = output_type.model_validate_json(terminal_output_text(response))
        validate(parsed)
        write_immutable_json(directory / "decision.json", parsed.model_dump(mode="json"))
        receipt["decision_sha256"] = sha256_file(directory / "decision.json")
        receipt["validated"] = True
    finally:
        write_immutable_json(directory / "receipt.json", receipt)
    sources = {str(directory / name): sha256_file(directory / name)
               for name in ("request.json", "response.json", "decision.json", "receipt.json")}
    sources.update({str(path): sha256_file(path) for path in (directory / 'attempts').glob('*.json')})
    return parsed, sources, seconds


def choose_sampling(config: AgentConfig, evidence: dict, *, directory: Path, history=()) -> dict:
    """Prescribe probabilities over an externally frozen real-training catalog."""
    _validate_dev_evidence(evidence)
    context = {"evidence": evidence, "evidence_sha256": payload_hash(evidence),
               "history": compact_history(history)}
    decision, sources, seconds = _call(config, phase="prescribe", context=context,
        directory=directory, validate=lambda output: validate_decision(output, evidence))
    return {"group_weights": validate_decision(decision, evidence),
            "decision": decision.model_dump(mode="json"), "sources": sources, "seconds": seconds}


def reflect_sampling(config: AgentConfig, outcome: dict, *, directory: Path, history=()) -> dict:
    """Reflect on measured training, then update prose experience for the next call.

    outcome contains round_index, decision, one evidence pack with current and
    previous dev measurements, and observation_ids. The evidence includes
    actual group exposures. Test measurements never enter this interface.
    """
    _validate_dev_evidence(outcome["evidence"])
    def validate_reflection(output):
        if output.round_index != outcome["round_index"] or output.outcome_sha256 != payload_hash(outcome):
            raise ValueError("reflection is bound to a different completed stage")
        if not set(output.evidence_ids) <= set(outcome["observation_ids"]):
            raise ValueError("reflection cites an unknown observation")

    reflection, reflection_sources, reflection_seconds = _call(config, phase="reflect",
        context={"outcome": outcome, "outcome_sha256": payload_hash(outcome), "history": compact_history(history)},
        directory=Path(directory) / "reflection", validate=validate_reflection)
    episode = {"round_index": outcome["round_index"], "decision": outcome["decision"],
               "reflection": reflection.model_dump(mode="json"), "outcome_sha256": payload_hash(outcome)}
    episodes = compact_history([*history, episode])
    known_rounds = {item["round_index"] for item in episodes}

    def validate_experience(output):
        if not set(output.supporting_rounds) <= known_rounds:
            raise ValueError("experience cites an unavailable training round")
        if outcome["round_index"] not in output.supporting_rounds:
            raise ValueError("experience must incorporate the just-completed round")

    experience, experience_sources, experience_seconds = _call(config, phase="experience",
        context={"completed_rounds": episodes}, directory=Path(directory) / "experience",
        validate=validate_experience)
    episode.update(experience=experience.model_dump(mode="json"),
                   sources={**reflection_sources, **experience_sources},
                   seconds=reflection_seconds + experience_seconds)
    write_immutable_json(Path(directory) / "episode.json", episode)
    return episode
