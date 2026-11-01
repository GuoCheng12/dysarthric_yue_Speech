from __future__ import annotations

import json

import pytest
from openai.types.responses.response import Response
from pydantic import ValidationError

from therapist_harness import agent


def _structured_response() -> Response:
    return Response.model_validate(
        {
            "id": "resp_serialization_test",
            "created_at": 0.0,
            "model": "gpt-5.6-sol",
            "object": "response",
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "msg_serialization_test",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"ok":true}',
                            "annotations": [],
                        }
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "serialization_test",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                }
            },
        }
    )


def test_sdk_internal_schema_name_does_not_round_trip():
    response = _structured_response()
    payload = response.model_dump(mode="json")
    assert "schema_" in payload["text"]["format"]
    with pytest.raises(ValidationError):
        Response.model_validate_json(json.dumps(payload))


def test_structured_response_persists_api_alias_and_round_trips():
    response = _structured_response()
    payload = agent.serialize_response(response)
    assert "schema" in payload["text"]["format"]
    assert "schema_" not in payload["text"]["format"]
    restored = Response.model_validate_json(json.dumps(payload))
    assert restored == response
    assert restored.output_text == '{"ok":true}'
    assert agent.serialize_response(restored) == payload


@pytest.mark.parametrize("role", ["therapist", "reflection"])
def test_call_helpers_return_reloadable_responses(monkeypatch, tmp_path, role):
    response = _structured_response()
    prompt = tmp_path / "prompt.md"
    skill = tmp_path / "skill.md"
    prompt.write_text("Unit-test instruction.")
    skill.write_text("Unit-test Skill.")
    request = {"model": "gpt-5.6-sol"}
    monkeypatch.setattr(agent, f"build_{role}_request", lambda *a, **k: request)
    monkeypatch.setattr(agent, "execute_therapist_request", lambda *a: response)
    monkeypatch.setattr(agent, f"parse_{role}_response", lambda *a: "parsed")
    arguments = (None, None) if role == "therapist" else (None, None, None, None)
    parsed, sent, payload = getattr(agent, f"call_{role}")(
        *arguments,
        system_prompt_path=prompt,
        skill_path=skill,
    )
    assert (parsed, sent) == ("parsed", request)
    assert Response.model_validate_json(json.dumps(payload)) == response
