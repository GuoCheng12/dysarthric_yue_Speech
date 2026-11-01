from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from test_therapist_outcome import _assignments, _proposal, _protocol
from test_therapist_outcome import round_artifacts as round_artifacts
from test_therapist_reflection import (
    SKILL_TEXT,
    SYSTEM_PROMPT,
    ReflectionArtifacts,
    _reflection,
)
from therapist_harness.agent import build_reflection_request, build_therapist_request
from therapist_harness.materialize import build_materialization_sidecar
from therapist_harness.memory import compile_intervention_summary
from therapist_harness.outcome import (
    compile_evaluation_plan,
    compile_reflection_pack,
    compile_round_result,
)
from therapist_harness.round_schema import (
    SynthesisReceipt,
    SynthesizedSample,
    TrainingReceipt,
)
from therapist_harness.schema import canonical_sha256
from therapist_harness.store import (
    REQUIRED_ARTIFACTS,
    Event,
    RunStore,
    sha256_directory,
    sha256_file,
    write_immutable_json,
)


def _json(path: Path, payload) -> Path:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    write_immutable_json(path, payload)
    return path


def _rows(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _response(proposal, model: str) -> dict:
    return {
        "id": "resp_fixture",
        "created_at": 0.0,
        "model": model,
        "object": "response",
        "status": "completed",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": proposal.model_dump_json(),
                        "annotations": [],
                    }
                ],
            }
        ],
    }


@pytest.fixture()
def assessed(tmp_path: Path):
    protocol = _protocol(tmp_path)
    evidence, proposal, _ = _proposal(protocol)
    plan = compile_evaluation_plan(
        protocol,
        evidence,
        proposal,
        before_prediction_column="before",
        after_prediction_column="after",
    )
    store = RunStore.initialize(protocol, tmp_path / "run")
    store.advance("assessed", {"evidence": _json(tmp_path / "evidence.json", evidence)})
    proposed = {
        "proposal": _json(tmp_path / "proposal.json", proposal),
        "evaluation_plan": _json(tmp_path / "evaluation_plan.json", plan),
        "agent_request": _json(
            tmp_path / "agent_request.json",
            build_therapist_request(
                protocol,
                evidence,
                system_prompt="Describe the intervention.",
                skill_text="Only use measured evidence.",
            ),
        ),
        "agent_response": _json(
            tmp_path / "agent_response.json", _response(proposal, protocol.agent.model)
        ),
    }
    return store, protocol, evidence, proposal, proposed


@pytest.fixture()
def materialized(assessed, tmp_path: Path):
    store, protocol, evidence, proposal, proposed = assessed
    store.advance("proposed", proposed)
    assignments = _assignments(protocol, proposal, evidence)
    sidecar = build_materialization_sidecar(proposal, evidence, protocol, assignments)
    store.advance(
        "materialized",
        {
            "synthesis_manifest": _rows(tmp_path / "assignments.jsonl", assignments),
            "materialization_sidecar": _json(tmp_path / "sidecar.json", sidecar),
        },
    )
    return store, protocol, assignments, sidecar


def _synthesis_artifacts(
    tmp_path, protocol, assignments, sidecar, *, mode="recorded_execution"
):
    samples = []
    for index, row in enumerate(assignments):
        audio_path = Path(row["output_wav"])
        audio_path.write_bytes(f"generated-fixture-{index}".encode())
        samples.append(
            SynthesizedSample(
                sample_id=row["sample_id"],
                audio_file_sha256=sha256_file(audio_path),
                duration_seconds=1.0,
            )
        )
    receipt = SynthesisReceipt(
        schema_version="1",
        run_id=protocol.run_id,
        round_index=1,
        artifact_mode=mode,
        status="complete",
        assignments_canonical_sha256=sidecar.assignments_canonical_sha256,
        requested=len(samples),
        generated=len(samples),
        failed=0,
        samples=samples,
    )
    train_rows = [
        json.loads(line)
        for line in Path(protocol.real_train.path).read_text().splitlines()
    ]
    train_rows = [
        {"audio": row["audio"], "text": row["clean_gt"]} for row in train_rows
    ]
    train_rows.extend(
        {"audio": row["output_wav"], "text": row["target_text"]} for row in assignments
    )
    return {
        "synthesis_summary": _json(tmp_path / "synthesis_receipt.json", receipt),
        "train_manifest": _rows(tmp_path / "train_manifest.jsonl", train_rows),
        "training_index": _json(
            tmp_path / "training_index.json", {"fixture": "row index"}
        ),
    }


def test_typed_proposal_and_materialization_can_advance(materialized):
    store, _, _, _ = materialized
    assert store.current().phase == "materialized"
    assert len(store.events()) == 4
    store.verify_integrity()


def test_assessed_rejects_snapshot_not_from_initial_adapter(tmp_path: Path):
    protocol = _protocol(tmp_path)
    evidence, _, _ = _proposal(protocol)
    payload = evidence.model_dump(mode="json")
    payload["snapshots"]["current"]["checkpoint_sha256"] = "f" * 64
    store = RunStore.initialize(protocol, tmp_path / "run")
    with pytest.raises(ValueError, match="round 1 current snapshot must be M0"):
        store.advance("assessed", {"evidence": _json(tmp_path / "wrong.json", payload)})
    assert store.current().phase == "initialized"


def test_proposed_rejects_placeholder_without_registering_success(
    assessed, tmp_path: Path
):
    store, _, _, _, _ = assessed
    empty = _json(tmp_path / "empty.json", {})
    with pytest.raises(ValidationError):
        store.advance(
            "proposed", {name: empty for name in REQUIRED_ARTIFACTS["proposed"]}
        )
    assert store.current().phase == "assessed"


def test_proposed_must_equal_recorded_agent_output(assessed, tmp_path: Path):
    store, protocol, _, proposal, artifacts = assessed
    different = proposal.model_copy(
        update={"hypothesis": "A different intervention hypothesis."}
    )
    artifacts["agent_response"] = _json(
        tmp_path / "different_response.json", _response(different, protocol.agent.model)
    )
    with pytest.raises(ValueError, match="differs from Agent output"):
        store.advance("proposed", artifacts)


def test_later_stage_cannot_redefine_registered_evidence(assessed):
    store, _, _, _, artifacts = assessed
    artifacts["evidence"] = Path(store.current().artifact_paths["evidence"])
    with pytest.raises(ValueError, match="cannot redefine"):
        store.advance("proposed", artifacts)


def test_proposed_rejects_changed_user_task(assessed, tmp_path: Path):
    store, _, _, _, artifacts = assessed
    request = json.loads(artifacts["agent_request"].read_text("utf-8"))
    user = json.loads(request["input"][-1]["content"])
    user["task"] = "Change the training hyperparameters instead."
    request["input"][-1]["content"] = json.dumps(user)
    artifacts["agent_request"] = _json(tmp_path / "changed_task.json", request)
    with pytest.raises(ValueError, match="input differs from its registered source"):
        store.advance("proposed", artifacts)
    assert store.current().phase == "assessed"


def test_counterfactual_receipt_cannot_claim_real_synthesis(
    materialized, tmp_path: Path
):
    store, protocol, assignments, sidecar = materialized
    artifacts = _synthesis_artifacts(
        tmp_path, protocol, assignments, sidecar, mode="counterfactual_interface_smoke"
    )
    with pytest.raises(ValueError, match="counterfactual synthesis"):
        store.advance("synthesized", artifacts)
    assert store.current().phase == "materialized"


def test_synthesis_receipt_hash_must_match_generated_file(materialized, tmp_path: Path):
    store, protocol, assignments, sidecar = materialized
    artifacts = _synthesis_artifacts(tmp_path, protocol, assignments, sidecar)
    Path(assignments[0]["output_wav"]).write_bytes(b"different audio bytes")
    with pytest.raises(ValueError, match="generated audio hash"):
        store.advance("synthesized", artifacts)


def test_training_receipt_steps_must_match_trainer_state(materialized, tmp_path: Path):
    store, protocol, assignments, sidecar = materialized
    synthesized = _synthesis_artifacts(tmp_path, protocol, assignments, sidecar)
    store.advance("synthesized", synthesized)
    checkpoint = tmp_path / "new_checkpoint"
    checkpoint.mkdir()
    _json(
        checkpoint / "adapter_config.json",
        {
            "r": protocol.lora.rank,
            "lora_alpha": protocol.lora.alpha,
            "lora_dropout": protocol.lora.dropout,
            "target_modules": protocol.lora.target_modules,
        },
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture adapter weights")
    state = _json(checkpoint / "trainer_state.json", {"global_step": 999})
    receipt = TrainingReceipt(
        schema_version="1",
        run_id=protocol.run_id,
        round_index=1,
        status="complete",
        parent_checkpoint_directory_sha256=protocol.initial_adapter.sha256,
        new_checkpoint_directory_sha256=sha256_directory(checkpoint),
        train_manifest_file_sha256=sha256_file(synthesized["train_manifest"]),
        training_index_file_sha256=sha256_file(synthesized["training_index"]),
        trainer_state_file_sha256=sha256_file(state),
        real_rows=protocol.real_replay_count_per_round,
        synthetic_rows=protocol.synthetic_count_per_round,
        total_rows=protocol.real_replay_count_per_round
        + protocol.synthetic_count_per_round,
        expected_optimizer_steps=protocol.lora.optimizer_steps_per_round,
        actual_optimizer_steps=protocol.lora.optimizer_steps_per_round,
    )
    with pytest.raises(ValueError, match="training receipt differs"):
        store.advance(
            "trained",
            {
                "checkpoint": checkpoint,
                "training_receipt": _json(tmp_path / "training_receipt.json", receipt),
            },
        )
    assert store.current().phase == "synthesized"


def test_unimplemented_final_test_contract_is_explicitly_blocked(assessed):
    store, protocol, _, _, _ = assessed
    event = Event(
        sequence=2,
        phase="completed",
        round_index=1,
        protocol_sha256=canonical_sha256(protocol),
        artifact_sha256={},
        artifact_paths={},
        artifact_kinds={},
    )
    with pytest.raises(
        NotImplementedError, match="execution contract is not implemented"
    ):
        store._validate_transition(event, protocol)


@pytest.mark.parametrize("prediction_format", ["csv", "jsonl"])
def test_typed_source_bound_path_reaches_reflected(round_artifacts, tmp_path: Path, prediction_format):
    """Exercise file/receipt plumbing with fixture bytes, not real TTS/training."""

    item = round_artifacts
    witness = tmp_path / "witnesses"
    witness.mkdir()
    before_file = witness / f"before_predictions.{prediction_format}"
    if prediction_format == "jsonl":
        _rows(before_file, item.before_rows)
    else:
        with before_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(item.before_rows[0]))
            writer.writeheader()
            writer.writerows(item.before_rows)
    protocol = item.protocol.model_copy(
        update={
            "initial_adapter_dev_predictions": item.protocol.initial_adapter_dev_predictions.model_copy(
                update={"path": str(before_file), "sha256": sha256_file(before_file)}
            )
        }
    )
    evidence, proposal, target_id = _proposal(protocol)
    plan = compile_evaluation_plan(
        protocol,
        evidence,
        proposal,
        before_prediction_column="before",
        after_prediction_column="after",
    )
    assignments = _assignments(protocol, proposal, evidence)
    sidecar = build_materialization_sidecar(proposal, evidence, protocol, assignments)
    store = RunStore.initialize(protocol, tmp_path / "run")
    store.advance("assessed", {"evidence": _json(witness / "evidence.json", evidence)})
    prescription_prompt = "Describe the intervention."
    store.advance(
        "proposed",
        {
            "proposal": _json(witness / "proposal.json", proposal),
            "evaluation_plan": _json(witness / "plan.json", plan),
            "agent_request": _json(
                witness / "request.json",
                build_therapist_request(
                    protocol,
                    evidence,
                    system_prompt=prescription_prompt,
                    skill_text=SKILL_TEXT,
                ),
            ),
            "agent_response": _json(
                witness / "response.json", _response(proposal, protocol.agent.model)
            ),
        },
    )
    store.advance(
        "materialized",
        {
            "synthesis_manifest": _rows(witness / "assignments.jsonl", assignments),
            "materialization_sidecar": _json(witness / "sidecar.json", sidecar),
        },
    )
    synthesized = _synthesis_artifacts(witness, protocol, assignments, sidecar)
    store.advance("synthesized", synthesized)

    checkpoint = witness / "checkpoint"
    checkpoint.mkdir()
    _json(
        checkpoint / "adapter_config.json",
        {
            "r": protocol.lora.rank,
            "lora_alpha": protocol.lora.alpha,
            "lora_dropout": protocol.lora.dropout,
            "target_modules": protocol.lora.target_modules,
        },
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"test-only adapter bytes")
    state = _json(
        checkpoint / "trainer_state.json",
        {
            "global_step": protocol.lora.optimizer_steps_per_round,
        },
    )
    training = item.training.model_copy(
        update={
            "new_checkpoint_directory_sha256": sha256_directory(checkpoint),
            "train_manifest_file_sha256": sha256_file(synthesized["train_manifest"]),
            "training_index_file_sha256": sha256_file(synthesized["training_index"]),
            "trainer_state_file_sha256": sha256_file(state),
        }
    )
    store.advance(
        "trained",
        {
            "checkpoint": checkpoint,
            "training_receipt": _json(witness / "training.json", training),
        },
    )

    after_rows = [dict(row) for row in item.after_rows]
    next(row for row in after_rows if row["utt_id"] == "u2")["after"] = "阿聽日錯錯"
    after_file = witness / f"after_predictions.{prediction_format}"
    if prediction_format == "jsonl":
        _rows(after_file, after_rows)
    else:
        with after_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(after_rows[0]))
            writer.writeheader()
            writer.writerows(after_rows)
    synthesis = SynthesisReceipt.model_validate_json(
        synthesized["synthesis_summary"].read_text()
    )
    result = compile_round_result(
        protocol,
        evidence,
        proposal,
        plan,
        sidecar,
        synthesis,
        training,
        before_rows=item.before_rows,
        after_rows=after_rows,
        before_predictions_file_sha256=protocol.initial_adapter_dev_predictions.sha256,
        after_predictions_file_sha256=sha256_file(after_file),
    )
    pack = compile_reflection_pack(
        protocol,
        evidence,
        proposal,
        plan,
        sidecar,
        result,
        prescription_prompt_file_sha256=hashlib.sha256(
            prescription_prompt.encode()
        ).hexdigest(),
        reflection_prompt_file_sha256=hashlib.sha256(
            SYSTEM_PROMPT.encode()
        ).hexdigest(),
        skill_version="v000",
        skill_file_sha256=hashlib.sha256(SKILL_TEXT.encode()).hexdigest(),
    )
    store.advance(
        "reassessed",
        {
            "after_predictions": after_file,
            "round_result": _json(witness / "result.json", result),
            "reflection_pack": _json(witness / "reflection_pack.json", pack),
        },
    )
    reflection = _reflection(
        ReflectionArtifacts(
            protocol=protocol,
            evidence=evidence,
            proposal=proposal,
            pack=pack,
            target_id=target_id,
        )
    )
    store.advance(
        "reflected",
        {
            "reflection_request": _json(
                witness / "reflection_request.json",
                build_reflection_request(
                    protocol,
                    evidence,
                    proposal,
                    pack,
                    system_prompt=SYSTEM_PROMPT,
                    skill_text=SKILL_TEXT,
                ),
            ),
            "reflection_response": _json(
                witness / "reflection_response.json",
                _response(reflection, protocol.agent.model),
            ),
            "reflection": _json(witness / "reflection.json", reflection),
        },
    )
    assert store.current().phase == "reflected"
    assert len(store.events()) == 8
    ledger = compile_intervention_summary(
        protocol, evidence, proposal, pack, reflection
    )
    invalid_ledger = ledger.model_copy(update={"hypothesis_verdict": "supported"})
    with pytest.raises(ValueError, match="ledger differs"):
        store.advance(
            "committed",
            {"ledger_entry": _json(witness / "bad_ledger.json", invalid_ledger)},
        )
    store.advance("committed", {"ledger_entry": _json(witness / "ledger.json", ledger)})
    assert store.current().phase == "committed"
    assert store.intervention_history() == [ledger]
    assert (
        store.intervention_history()[0].episode_memory.next_round_guidance
        == reflection.next_round_guidance
    )
    assert len(store.events()) == 9
    store.verify_integrity()
