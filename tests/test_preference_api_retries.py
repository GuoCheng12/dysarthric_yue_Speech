"""Exercise real SDK retries with a local HTTP transport; no gateway calls."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from therapist_harness import agent, transcript_review


@pytest.mark.parametrize("failure,fail_count,expected_calls", [
    ("connect", 2, 3), ("connect", 5, 5),
    (503, 2, 3), (429, 2, 3), (401, 5, 1),
])
def test_mas_transport_retry_budget(monkeypatch, failure, fail_count, expected_calls):
    requests, delays = [], []

    def handle(request):
        requests.append(request.content)
        if len(requests) <= fail_count:
            if failure == "connect":
                raise httpx.ConnectError("connection reset", request=request)
            return httpx.Response(failure, json={"error": {"message": "test error"}})
        event = {"type": "response.completed", "response": {
            "id": "test-response", "status": "completed", "output": [],
        }}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="data: " + json.dumps(event) + "\n\n")

    clients = []
    def factory(**kwargs):
        assert kwargs["max_retries"] == 4
        assert kwargs["timeout"].connect == 20
        assert kwargs["timeout"].read == 900
        client = openai.OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        clients.append(client)
        return client

    monkeypatch.setenv("TEST_RETRY_KEY", "test-only")
    monkeypatch.setattr(agent, "OpenAI", factory)
    monkeypatch.setattr(openai._base_client.time, "sleep", delays.append)
    protocol = SimpleNamespace(agent=SimpleNamespace(api_key_env="TEST_RETRY_KEY",
        base_url="https://test.invalid/", max_retries=4, timeout_seconds=900))
    try:
        if failure == 401 or fail_count == 5:
            with pytest.raises((openai.APIConnectionError, openai.APIStatusError)):
                agent.execute_therapist_request(protocol, {"model": "test", "input": "same", "stream": True})
        else:
            result = agent.execute_therapist_request(protocol, {"model": "test", "input": "same", "stream": True})
            assert result.status == "completed"
        assert len(requests) == expected_calls
        assert len(set(requests)) == 1
        assert len(delays) == expected_calls - 1
        assert all(0 < delay <= 60 for delay in delays)
    finally:
        for client in clients:
            client.close()


def test_transcript_reviewer_enables_same_bounded_retries(monkeypatch):
    monkeypatch.setenv("TEST_RETRY_KEY", "test-only")
    reviewer = transcript_review.TranscriptReviewer(model="test", base_url="https://test.invalid/",
        transcription_model="test-asr", transcription_base_model_sha256="b" * 64,
        transcription_adapter_sha256="a" * 64, api_key_env="TEST_RETRY_KEY")
    try:
        assert reviewer.client.max_retries == 4
        assert reviewer.client.timeout.connect == 20
        assert reviewer.client.timeout.read == 180
    finally:
        reviewer.client.close()
