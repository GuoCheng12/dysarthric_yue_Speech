import pytest
from pydantic import ValidationError

from therapist_harness.schema import RunProtocol, ActionContract
from therapist_harness.round_schema import TrainingReceipt
from test_therapist_outcome import _protocol, round_artifacts  # noqa: F401


def test_zero_replay_protocol_is_valid_but_negative_is_not(tmp_path):
    payload = _protocol(tmp_path).model_dump(mode="json")
    payload["real_replay_count_per_round"] = 0
    batch = payload["lora"]["per_device_batch_size"] * payload["lora"]["gradient_accumulation_steps"]
    payload["lora"]["optimizer_steps_per_round"] = (
        (payload["synthetic_count_per_round"] + batch - 1) // batch
        * payload["lora"]["training_passes_per_round"]
    )
    assert RunProtocol.model_validate(payload).real_replay_count_per_round == 0
    payload["real_replay_count_per_round"] = -1
    with pytest.raises(ValidationError):
        RunProtocol.model_validate(payload)


def test_zero_replay_receipt_and_agent_contract(round_artifacts):
    contract = round_artifacts.evidence.action_contract.model_dump(mode="json")
    contract["real_replay_count"] = 0
    assert ActionContract.model_validate(contract).real_replay_count == 0
    receipt = round_artifacts.training.model_dump(mode="json")
    receipt.update(real_rows=0, total_rows=receipt["synthetic_rows"])
    assert TrainingReceipt.model_validate(receipt).real_rows == 0
    receipt["real_rows"] = -1
    with pytest.raises(ValidationError):
        TrainingReceipt.model_validate(receipt)
