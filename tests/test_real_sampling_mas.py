import json

from openai.types.responses import Response
import pytest
from pydantic import ValidationError

from therapist_harness import real_sampling_mas as mas
from therapist_harness.schema import AgentConfig


def config(**changes):
    return AgentConfig(**{
        "provider": "OpenAI", "base_url": "https://api.laicaishare.xyz",
        "model": "gpt-5.6-sol", "reasoning_effort": "xhigh", "store": False,
        "max_retries": 0, "timeout_seconds": 900, "max_output_tokens": 8192,
        "api_key_env": "OPENAI_API_KEY", **changes,
    })


def evidence():
    return {"round_index": 0, "checkpoint_step": 0, "feedback_role": "dev",
            "groups": [{"group_id": "G1", "label": "原始识别较容易", "train_count": 12},
                       {"group_id": "G2", "label": "原始识别较困难", "train_count": 8}],
            "feedback": {"current": {"CER_percent": 20.0}, "previous": None},
            "observation_ids": ["feedback.current", "groups.G1", "groups.G2"]}


def decision(pack):
    return dict(round_index=pack["round_index"], evidence_sha256=mas.payload_hash(pack),
        diagnosis="当前仍有漏字，先比较两组的学习进展。",
        allocations=[dict(group_id=key, probability=prob, reason="保留练习并观察验证变化。", evidence_ids=[f"groups.{key}"])
                     for key, prob in [("G1", .4), ("G2", .6)]],
        expected_change="dev 诊断中的漏字减少。", history_used="没有已完成阶段。")


def response(payload, model="gpt-5.6-sol"):
    return Response.model_validate(dict(id="resp_test", object="response", created_at=1,
        status="completed", model=model, parallel_tool_calls=False, tool_choice="auto", tools=[],
        output=[dict(id="msg_test", type="message", role="assistant", status="completed",
                     content=[dict(type="output_text", text=json.dumps(payload, ensure_ascii=False), annotations=[])])]))


@pytest.mark.parametrize("mutation,match", [
    ("missing_group", "every fixed group"), ("duplicate_group", "every fixed group"),
    ("unknown_citation", "unknown observation"), ("wrong_sum", "sum to one"),
    ("stale_evidence", "different round or evidence"),
])
def test_decision_must_match_fixed_groups_and_actual_evidence(mutation, match):
    pack = evidence()
    value = decision(pack)
    if mutation == "missing_group":
        value["allocations"].pop()
    elif mutation == "duplicate_group":
        value["allocations"][1]["group_id"] = "G1"
    elif mutation == "unknown_citation":
        value["allocations"][0]["evidence_ids"] = ["invented"]
    elif mutation == "wrong_sum":
        value["allocations"][0]["probability"] = .7
    else:
        pack["feedback"]["current"]["CER_percent"] = 30.0
    with pytest.raises(ValueError, match=match):
        mas.validate_decision(mas.SamplingDecision(**value), pack)


def test_no_other_training_action_can_enter_output():
    value = decision(evidence())
    value["learning_rate"] = 1e-3
    with pytest.raises(ValidationError, match="Extra inputs"):
        mas.SamplingDecision(**value)
    with pytest.raises(ValidationError):
        mas.SamplingExperience(experience="一次观测不能证明普遍规律。", supporting_rounds=[0], group_weights={"G1": 1})


@pytest.mark.parametrize("role", ["test", "training_diagnostics"])
def test_test_feedback_cannot_be_sent_to_agent(monkeypatch, tmp_path, role):
    pack = evidence()
    pack["feedback_role"] = role
    monkeypatch.setattr(mas, "execute_therapist_request", lambda *args: pytest.fail("API must not be called"))
    with pytest.raises(ValueError, match="dev feedback only"):
        mas.choose_sampling(config(), pack, directory=tmp_path)
    with pytest.raises(ValueError, match="dev feedback only"):
        mas.reflect_sampling(config(), {"evidence": pack}, directory=tmp_path)
    assert not (tmp_path / "request.json").exists()


def test_requested_gateway_and_model_fixed_in_all_phases():
    for phase in ("prescribe", "reflect", "experience"):
        request = mas.build_sampling_request(config(), phase=phase, context={"数据": "今日食飯"})
        assert request["model"] == "gpt-5.6-sol"
        assert request["store"] is False
        assert request["text"]["format"]["strict"] is True
        assert "今日食飯" in request["input"][-1]["content"]
    with pytest.raises(ValueError, match="gpt-5.6-sol"):
        mas.build_sampling_request(config(model="another-model"), phase="prescribe", context={})
    with pytest.raises(ValueError, match="laicai"):
        mas.build_sampling_request(config(base_url="https://example.com"), phase="prescribe", context={})


def test_history_bounded_and_carries_last_experience():
    history = [{"round_index": n, "decision": {}, "reflection": {},
                "experience": {"experience": f"包含第零轮以来的经验，最近第{n}轮。"},
                "sources": {"large": "not passed"}} for n in range(8)]
    packed = mas.compact_history(history)
    assert [item["round_index"] for item in packed] == [4, 5, 6, 7]
    assert "第零轮" in packed[-1]["experience"]["experience"]
    assert all("sources" not in item for item in packed)


def test_actual_model_receipt_and_immutable_evidence(monkeypatch, tmp_path):
    pack = evidence()
    monkeypatch.setattr(mas, "execute_therapist_request", lambda protocol, request: response(decision(pack)))
    result = mas.choose_sampling(config(), pack, directory=tmp_path)
    assert result["group_weights"] == {"G1": .4, "G2": .6}
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["request_model"] == receipt["response_model"] == "gpt-5.6-sol"
    assert receipt["role"] == "Therapist" and receipt["validated"]
    for path, sha in result["sources"].items():
        assert mas.sha256_file(mas.Path(path)) == sha
    with pytest.raises(FileExistsError):
        mas.choose_sampling(config(), pack, directory=tmp_path)


def test_response_model_substitution_rejected_and_preserved(monkeypatch, tmp_path):
    pack = evidence()
    monkeypatch.setattr(mas, "execute_therapist_request", lambda protocol, request: response(decision(pack), model="other"))
    with pytest.raises(ValueError, match="actual response model"):
        mas.choose_sampling(config(), pack, directory=tmp_path)
    assert (tmp_path / "response.json").exists()
    assert not (tmp_path / "decision.json").exists()
    assert not json.loads((tmp_path / "receipt.json").read_text())["validated"]


def test_interrupted_call_keeps_attempt_receipt_without_secret(monkeypatch, tmp_path):
    def interrupt(*args):
        raise RuntimeError("private transport detail")
    monkeypatch.setattr(mas, "execute_therapist_request", interrupt)
    with pytest.raises(RuntimeError):
        mas.choose_sampling(config(), evidence(), directory=tmp_path)
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["response_model"] is None
    assert receipt["response_status"] == "no_terminal_response"
    assert "private transport detail" not in (tmp_path / "receipt.json").read_text()


def test_reflection_and_experience_are_real_separate_calls(monkeypatch, tmp_path):
    pack = evidence()
    outcome = {"round_index": 0, "decision": decision(pack),
               "evidence": {**pack, "feedback": {"current": {"CER_percent": 19.0},
                                                "previous": {"CER_percent": 20.0}}},
               "delivery": {"G1": 40, "G2": 60}, "observation_ids": ["paired.overall"]}
    phases = []

    def execute(protocol, request):
        phase = request["text"]["format"]["name"].removeprefix("real_sampling_")
        phases.append(phase)
        if phase == "reflect":
            return response(dict(round_index=0, outcome_sha256=mas.payload_hash(outcome),
                observed_changes="dev 错误率下降一个百分点。", evidence_ids=["paired.overall"],
                explanation="符合预期，但不能归因于某一组。", next_guidance="继续观察。"))
        context = json.loads(request["input"][-1]["content"])
        assert context["completed_rounds"][-1]["reflection"]["observed_changes"] == "dev 错误率下降一个百分点。"
        return response(dict(experience="本轮整体改善，不能据此认定某组独立有效。", supporting_rounds=[0]))

    monkeypatch.setattr(mas, "execute_therapist_request", execute)
    episode = mas.reflect_sampling(config(), outcome, directory=tmp_path)
    assert phases == ["reflect", "experience"]
    assert episode["experience"]["supporting_rounds"] == [0]
    roles = [json.loads((tmp_path / sub / "receipt.json").read_text())["role"] for sub in ("reflection", "experience")]
    assert roles == ["Therapist", "Skill Optimizer"]
    assert (tmp_path / "episode.json").exists()


def test_stream_failure_retries_identical_request_and_keeps_attempts(monkeypatch, tmp_path):
    import httpx
    requests = []
    pack = evidence()
    def execute(protocol, request):
        requests.append(json.dumps(request, sort_keys=True))
        if len(requests) == 1:
            raise mas.APIError('stream_read_error', request=httpx.Request('POST', 'https://api.laicaishare.xyz/responses'), body=None)
        return response(decision(pack))
    monkeypatch.setattr(mas, 'execute_therapist_request', execute)
    monkeypatch.setattr(mas.time, 'sleep', lambda seconds: None)
    result = mas.choose_sampling(config(), pack, directory=tmp_path)
    assert len(requests) == 2 and requests[0] == requests[1]
    first = json.loads((tmp_path / 'attempts/001.json').read_text())
    second = json.loads((tmp_path / 'attempts/002.json').read_text())
    assert first['retry'] and first['stream_read_error']
    assert first['request_sha256'] == second['request_sha256']
    assert json.loads((tmp_path / 'receipt.json').read_text())['attempts'] == 2
    assert len(result['sources']) == 8


def test_stream_retries_are_limited_to_three_and_preserve_failure(monkeypatch, tmp_path):
    import httpx
    def execute(*args):
        raise mas.APIError('stream_read_error', request=httpx.Request('POST', 'https://api.laicaishare.xyz/responses'), body=None)
    monkeypatch.setattr(mas, 'execute_therapist_request', execute)
    monkeypatch.setattr(mas.time, 'sleep', lambda seconds: None)
    with pytest.raises(mas.APIError):
        mas.choose_sampling(config(), evidence(), directory=tmp_path)
    assert len(list((tmp_path / 'attempts').glob('[0-9][0-9][0-9].json'))) == 3
    assert not json.loads((tmp_path / 'attempts/003.json').read_text())['retry']
    assert not json.loads((tmp_path / 'receipt.json').read_text())['validated']


@pytest.mark.parametrize('kind', ['read', 'sdk', 'deadline'])
def test_timeouts_retry_without_changing_request(monkeypatch, tmp_path, kind):
    import httpx
    calls = []
    def execute(protocol, request):
        calls.append(json.dumps(request, sort_keys=True))
        if len(calls) == 1:
            if kind == 'read':
                raise httpx.ReadTimeout('read operation timed out')
            if kind == 'sdk':
                raise mas.APITimeoutError(request=httpx.Request('POST', 'https://api.laicaishare.xyz/responses'))
            raise TimeoutError('total request deadline')
        return response(decision(evidence()))
    monkeypatch.setattr(mas, 'execute_therapist_request', execute)
    monkeypatch.setattr(mas.time, 'sleep', lambda seconds: None)
    mas.choose_sampling(config(), evidence(), directory=tmp_path)
    assert len(calls) == 2 and calls[0] == calls[1]
    receipt = json.loads((tmp_path / 'attempts/001.json').read_text())
    assert receipt['timeout'] and receipt['retry'] and receipt['deadline_seconds'] == 600


def test_total_deadline_interrupts_continuously_active_stream_and_restores_alarm():
    import signal
    import time
    previous = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match='wall-clock deadline'):
        with mas.request_deadline(0.05):
            while True:
                time.sleep(0.005)  # A heartbeat alone must not extend the total deadline.
    assert time.monotonic()-started < 0.5
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


@pytest.mark.parametrize('kind', ['sdk_connection', 'remote_protocol', 'read_disconnect', 'connect'])
def test_connection_failures_retry_identical_request(monkeypatch, tmp_path, kind):
    import httpx
    calls = []
    def execute(protocol, request):
        calls.append(json.dumps(request, sort_keys=True))
        if len(calls) == 1:
            if kind == 'sdk_connection':
                raise mas.APIConnectionError(request=httpx.Request('POST', 'https://api.laicaishare.xyz/responses'))
            failure = {'remote_protocol': httpx.RemoteProtocolError, 'read_disconnect': httpx.ReadError,
                       'connect': httpx.ConnectError}[kind]
            raise failure('Server disconnected without sending a response.')
        return response(decision(evidence()))
    monkeypatch.setattr(mas, 'execute_therapist_request', execute)
    monkeypatch.setattr(mas.time, 'sleep', lambda seconds: None)
    mas.choose_sampling(config(), evidence(), directory=tmp_path)
    assert len(calls) == 2 and calls[0] == calls[1]
    first = json.loads((tmp_path / 'attempts/001.json').read_text())
    second = json.loads((tmp_path / 'attempts/002.json').read_text())
    assert first['connection_error'] and first['retry']
    assert first['request_sha256'] == second['request_sha256']


def test_repeated_disconnects_stop_after_three_recorded_attempts(monkeypatch, tmp_path):
    import httpx
    def execute(*args):
        raise mas.APIConnectionError(request=httpx.Request('POST', 'https://api.laicaishare.xyz/responses'))
    monkeypatch.setattr(mas, 'execute_therapist_request', execute)
    monkeypatch.setattr(mas.time, 'sleep', lambda seconds: None)
    with pytest.raises(mas.APIConnectionError):
        mas.choose_sampling(config(), evidence(), directory=tmp_path)
    assert len(list((tmp_path / 'attempts').glob('[0-9][0-9][0-9].json'))) == 3
    assert not json.loads((tmp_path / 'attempts/003.json').read_text())['retry']
