"""Preference-loop contracts using explicit counterfactual interface fixtures.

The synthetic before/after predictions below test plumbing, not model performance.
No fixture is a live listening decision, real training result, or clinical evidence.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from test_therapist_outcome import (
    SHA_A, SHA_B, SHA_C, _assignments, _prediction_rows, _proposal, _protocol,
)
from test_therapist_reflection import SKILL_TEXT, SYSTEM_PROMPT, _reflection
from test_therapist_reflection import mixed_round as mixed_round
from therapist_harness.agent import build_reflection_request, build_therapist_request
from therapist_harness.evidence_pack import build_evidence_pack
from therapist_harness.materialize import build_materialization_sidecar
from therapist_harness.memory import compile_intervention_summary, validate_summary_digest
from therapist_harness.outcome import (
    compile_evaluation_plan, compile_reflection_pack, compile_round_result,
    validate_reflection_pack,
)
from therapist_harness.preference_contract import (
    PreferenceDelivery, PreferenceLoopConfig, PreferenceSampleDelivery,
    validate_preference_training,
)
from therapist_harness.round_schema import SynthesisReceipt, SynthesizedSample, TrainingReceipt
from therapist_harness.schema import RunProtocol, canonical_sha256


COUNTERFACTUAL = "counterfactual_interface_smoke"


def _config():
    return PreferenceLoopConfig(
        audio_review_model="test-audio-endpoint-not-called",
        audio_review_base_url="https://example.invalid/v1",
        real_per_step=2,
        synthetic_per_step=1,
    )


def _delivery(protocol, sample_ids, *, first_pairs=1, second_verdict="pass"):
    return PreferenceDelivery(
        current_asr_sha256=protocol.initial_adapter.sha256,
        frozen_reference_asr_sha256=protocol.initial_adapter.sha256,
        reviews_sha256=SHA_A,
        candidates_sha256=SHA_B,
        pair_manifest_sha256=SHA_C,
        exposure_log_sha256=SHA_A,
        samples_per_audio=4,
        real_audio_considered=6,
        real_pair_audio_count=2,
        real_pair_count=3,
        real_training_exposures=4,
        synthetic_samples=[
            PreferenceSampleDelivery(sample_id=sample_ids[0], review_verdict="pass",
                                     pair_count=first_pairs, training_exposures=2 if first_pairs else 0),
            PreferenceSampleDelivery(sample_id=sample_ids[1], review_verdict=second_verdict,
                                     pair_count=0, training_exposures=0),
        ],
    )


def _training(protocol, delivery):
    synthetic_rows = sum(x.pair_count for x in delivery.synthetic_samples)
    return TrainingReceipt(
        schema_version="1", run_id=protocol.run_id, round_index=1,
        artifact_mode=COUNTERFACTUAL, status="complete",
        parent_checkpoint_directory_sha256=protocol.initial_adapter.sha256,
        new_checkpoint_directory_sha256=SHA_C,
        train_manifest_file_sha256=delivery.pair_manifest_sha256,
        training_index_file_sha256=SHA_B, trainer_state_file_sha256=SHA_C,
        real_rows=delivery.real_pair_count, synthetic_rows=synthetic_rows,
        total_rows=delivery.real_pair_count + synthetic_rows,
        expected_optimizer_steps=2, actual_optimizer_steps=2,
        preference_delivery=delivery,
    )


def _counterfactual_preference_round(tmp_path, *, first_pairs=1, second_verdict="pass"):
    protocol = _protocol(tmp_path)
    protocol = RunProtocol.model_validate({**protocol.model_dump(mode="python"),
                                           "preference_loop": _config()})
    evidence, proposal, target_id = _proposal(protocol)
    evidence = evidence.model_copy(update={"artifact_mode": COUNTERFACTUAL})
    proposal = proposal.model_copy(update={"evidence_sha256": canonical_sha256(evidence)})
    plan = compile_evaluation_plan(protocol, evidence, proposal,
                                   before_prediction_column="before", after_prediction_column="after")
    sidecar = build_materialization_sidecar(proposal, evidence, protocol,
                                            _assignments(protocol, proposal, evidence))
    synthesis = SynthesisReceipt(
        schema_version="1", run_id=protocol.run_id, round_index=1,
        artifact_mode=COUNTERFACTUAL, status="complete",
        assignments_canonical_sha256=sidecar.assignments_canonical_sha256,
        requested=2, generated=2, failed=0,
        samples=[SynthesizedSample(sample_id=x.sample_id, audio_file_sha256=SHA_B,
                                   duration_seconds=1.0) for x in sidecar.samples],
    )
    delivery = _delivery(protocol, [x.sample_id for x in sidecar.samples],
                         first_pairs=first_pairs, second_verdict=second_verdict)
    training = _training(protocol, delivery)
    before = _prediction_rows(after=False)
    after = _prediction_rows(after=True)
    # Fixed artificial target improvement plus easy regression exercises mixed
    # compiler facts. This is not presented as observed DPO performance.
    next(row for row in after if row["utt_id"] == "u2")["after"] = "阿聽日錯錯"
    result = compile_round_result(
        protocol, evidence, proposal, plan, sidecar, synthesis, training,
        before_rows=before, after_rows=after,
        before_predictions_file_sha256=protocol.initial_adapter_dev_predictions.sha256,
        after_predictions_file_sha256=SHA_B,
    )
    pack = compile_reflection_pack(
        protocol, evidence, proposal, plan, sidecar, result,
        prescription_prompt_file_sha256=SHA_A,
        reflection_prompt_file_sha256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        skill_version="v000", skill_file_sha256=hashlib.sha256(SKILL_TEXT.encode()).hexdigest(),
    )
    return SimpleNamespace(protocol=protocol, evidence=evidence, proposal=proposal,
                           target_id=target_id, plan=plan, sidecar=sidecar, synthesis=synthesis,
                           training=training, result=result, pack=pack, delivery=delivery,
                           before_rows=before, after_rows=after)


@pytest.mark.parametrize("first_pairs,second_verdict", [(1, "pass"), (0, "pass"), (1, "obvious_failure")])
def test_actual_pair_delivery_survives_result_reflection_ledger_and_next_input(
    tmp_path, first_pairs, second_verdict,
):
    item = _counterfactual_preference_round(tmp_path, first_pairs=first_pairs,
                                           second_verdict=second_verdict)
    assert item.result.intervention.overall.generated == 2
    assert item.result.training.synthetic_rows == first_pairs
    assert item.result.training.preference_delivery == item.delivery
    validate_reflection_pack(item.pack, item.protocol, item.evidence, item.proposal)
    assert item.pack.provenance.artifact_mode == COUNTERFACTUAL
    request = build_reflection_request(item.protocol, item.evidence, item.proposal, item.pack,
                                       system_prompt=SYSTEM_PROMPT, skill_text=SKILL_TEXT)
    sent_pack = json.loads(request["input"][2]["content"])["reflection_pack"]
    assert sent_pack["round_result"]["training"]["preference_delivery"] == item.delivery.model_dump(mode="json")
    summary = compile_intervention_summary(item.protocol, item.evidence, item.proposal,
                                           item.pack, _reflection(item))
    validate_summary_digest(summary)
    assert summary.episode_memory.artifact_mode == COUNTERFACTUAL
    assert summary.episode_memory.preference_delivery == item.delivery
    next_pack = build_evidence_pack(
        item.protocol, round_index=2,
        current_rows=item.after_rows, current_prediction_column="after",
        current_checkpoint_sha256=summary.new_checkpoint_sha256,
        current_predictions_sha256=summary.dev_predictions_sha256,
        m0_rows=item.before_rows, m0_prediction_column="before",
        m0_predictions_sha256=item.protocol.initial_adapter_dev_predictions.sha256,
        zero_shot_rows=item.before_rows, zero_shot_prediction_column="before",
        zero_shot_predictions_sha256=item.protocol.zero_shot_dev_predictions.sha256,
        previous_rows=item.before_rows, previous_prediction_column="before",
        previous_checkpoint_sha256=item.protocol.initial_adapter.sha256,
        previous_predictions_sha256=item.protocol.initial_adapter_dev_predictions.sha256,
        intervention_history=[summary], artifact_mode=COUNTERFACTUAL,
    )
    next_request = build_therapist_request(item.protocol, next_pack,
                                          system_prompt="Prescribe the next practice.", skill_text=SKILL_TEXT)
    episode = json.loads(next_request["input"][2]["content"])["evidence"]["intervention_history"][0]["episode_memory"]
    assert episode["preference_delivery"] == item.delivery.model_dump(mode="json")
    skipped = episode["preference_delivery"]["synthetic_samples"][1]
    assert skipped["pair_count"] == skipped["training_exposures"] == 0
    assert skipped["review_verdict"] == second_verdict
    assert "not_verbatim_verified" in episode["preference_delivery"]["chosen_label_semantics"]


@pytest.mark.parametrize("field", ["current_asr_sha256", "frozen_reference_asr_sha256"])
def test_reference_and_current_asr_hashes_require_valid_format_and_frozen_binding(tmp_path, field):
    protocol = _protocol(tmp_path).model_copy(update={"preference_loop": _config()})
    delivery = _delivery(protocol, ["s1", "s2"])
    payload = delivery.model_dump(mode="python")
    with pytest.raises(ValidationError):
        PreferenceDelivery.model_validate({**payload, field: "z" * 64})
    wrong = PreferenceDelivery.model_validate({**payload, field: SHA_B})
    with pytest.raises(ValueError, match="frozen initial|current parent"):
        validate_preference_training(protocol, _training(protocol, wrong), sample_ids=["s1", "s2"])


def test_later_current_parent_changes_without_moving_dpo_reference(tmp_path):
    protocol = _protocol(tmp_path).model_copy(update={"preference_loop": _config()})
    delivery = _delivery(protocol, ["s1", "s2"]).model_copy(update={"current_asr_sha256": SHA_B})
    training = _training(protocol, delivery).model_copy(
        update={"parent_checkpoint_directory_sha256": SHA_B}
    )
    validate_preference_training(protocol, training, sample_ids=["s1", "s2"])
    moving_reference = delivery.model_copy(update={"frozen_reference_asr_sha256": SHA_B})
    with pytest.raises(ValueError, match="frozen initial"):
        validate_preference_training(
            protocol, training.model_copy(update={"preference_delivery": moving_reference})
        )


@pytest.mark.parametrize("pairs,exposures", [(1, 0), (0, 1), (1, 1)])
def test_obvious_failure_cannot_enter_dpo(pairs, exposures):
    with pytest.raises(ValidationError, match="obvious generation failures"):
        PreferenceSampleDelivery(sample_id="bad", review_verdict="obvious_failure",
                                 pair_count=pairs, training_exposures=exposures)


def test_pairless_real_or_synthetic_audio_cannot_claim_training_exposure(tmp_path):
    with pytest.raises(ValidationError, match="pairless"):
        PreferenceSampleDelivery(sample_id="correct", review_verdict="pass",
                                 pair_count=0, training_exposures=1)
    protocol = _protocol(tmp_path)
    payload = _delivery(protocol, ["s1", "s2"]).model_dump(mode="python")
    payload.update(real_pair_audio_count=0, real_pair_count=0, real_training_exposures=1)
    with pytest.raises(ValidationError, match="real.*pair|pair.*real"):
        PreferenceDelivery.model_validate(payload)


def test_delivery_checks_pair_counts_and_requested_identities(tmp_path):
    protocol = _protocol(tmp_path).model_copy(update={"preference_loop": _config()})
    delivery = _delivery(protocol, ["s1", "s2"])
    training = _training(protocol, delivery)
    validate_preference_training(protocol, training, sample_ids=["s1", "s2"])
    with pytest.raises(ValueError, match="identities"):
        validate_preference_training(protocol, training, sample_ids=["s1", "different"])
    wrong_count = training.model_copy(update={"synthetic_rows": 2, "total_rows": 5})
    with pytest.raises(ValueError, match="actual preference pairs"):
        validate_preference_training(protocol, wrong_count)
    wrong_manifest = training.model_copy(update={"train_manifest_file_sha256": SHA_B})
    with pytest.raises(ValueError, match="manifest"):
        validate_preference_training(protocol, wrong_manifest)
    no_delivery = training.model_copy(update={"preference_delivery": None})
    with pytest.raises(ValueError, match="requires actual delivery"):
        validate_preference_training(protocol, no_delivery)


def test_opt_in_does_not_change_legacy_serialized_payloads_or_hashes(mixed_round):
    item = mixed_round
    legacy_protocol = item.protocol.model_dump(mode="json", exclude={"preference_loop"})
    assert item.protocol.preference_loop is None
    assert RunProtocol.model_validate(legacy_protocol).model_dump(mode="json") == legacy_protocol
    assert canonical_sha256(item.protocol) == canonical_sha256(legacy_protocol)
    legacy_training = item.pack.training.model_dump(mode="json", exclude={"preference_delivery"})
    assert TrainingReceipt.model_validate(legacy_training).model_dump(mode="json") == legacy_training
    assert canonical_sha256(item.pack.training) == canonical_sha256(legacy_training)
    summary = compile_intervention_summary(item.protocol, item.evidence, item.proposal,
                                           item.pack, _reflection(item))
    validate_summary_digest(summary)
    assert "preference_delivery" not in summary.model_dump(mode="json")["episode_memory"]
    assert "preference_delivery" not in item.pack.model_dump_json()
    assert "preference_loop" not in item.protocol.model_dump_json()


def test_preference_mode_does_not_relax_legacy_training_budget(tmp_path):
    protocol = _protocol(tmp_path)
    delivery = _delivery(protocol, ["s1", "s2"])
    with pytest.raises(ValueError, match="requires a preference-loop"):
        validate_preference_training(protocol, _training(protocol, delivery))
    legacy = _training(protocol, delivery).model_copy(update={"preference_delivery": None})
    with pytest.raises(ValueError, match="real-replay count"):
        validate_preference_training(protocol, legacy)
    payload = protocol.model_dump(mode="python")
    payload["lora"]["optimizer_steps_per_round"] = 3
    with pytest.raises(ValidationError, match="optimizer_steps_per_round"):
        RunProtocol.model_validate(payload)
    # A variable number of sampled errors needs its independently fixed step budget.
    enabled = RunProtocol.model_validate({**payload, "preference_loop": _config()})
    assert enabled.lora.optimizer_steps_per_round == 3
