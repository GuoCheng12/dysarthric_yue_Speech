from __future__ import annotations

from collections import Counter

import pytest

from therapist_harness.evidence_pack import build_evidence_pack
from therapist_harness.grounding import validate_stimulus_grounding
from therapist_harness.materialize import (
    build_materialization_sidecar,
    materialize_proposal,
)
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
from therapist_harness.schema import (
    ExpectedOutcome,
    RunProtocol,
    Stimulus,
    canonical_sha256,
    validate_proposal,
)

from test_therapist_outcome import (
    SHA_A,
    SHA_B,
    SHA_C,
    _prediction_rows,
    _proposal,
    _protocol,
)


def _maintenance_case(tmp_path, *, protection=None):
    original = _protocol(tmp_path)
    protocol = RunProtocol.model_validate(
        {
            **original.model_dump(),
            "synthetic_count_per_round": 8,
            "unique_stimulus_count_per_round": 4 if protection else 3,
            "proposal_constraints": {
                **original.proposal_constraints.model_dump(),
                "max_copies_per_stimulus": 3,
            },
            "lora": {**original.lora.model_dump(), "optimizer_steps_per_round": 4},
        }
    )
    before = _prediction_rows(after=False)
    zero = [dict(row) for row in before]
    if protection == "hard":
        for row in before[4:]:
            row["before"] = row["before"].replace("我", "")
    elif protection == "easy":
        for row in before[:2]:
            row["before"] = row["before"].replace("日", "")
    evidence = build_evidence_pack(
        protocol,
        round_index=1,
        current_rows=before,
        current_prediction_column="before",
        current_checkpoint_sha256=protocol.initial_adapter.sha256,
        current_predictions_sha256=protocol.initial_adapter_dev_predictions.sha256,
        m0_rows=before,
        m0_prediction_column="before",
        m0_predictions_sha256=protocol.initial_adapter_dev_predictions.sha256,
        zero_shot_rows=zero,
        zero_shot_prediction_column="before",
        zero_shot_predictions_sha256=protocol.zero_shot_dev_predictions.sha256,
    )
    _, base, target_id = _proposal(protocol)
    easy = next(
        item
        for item in evidence.capability_anchors
        if item.evidence_id == "ability.coda.k.easy"
    )
    medium = next(
        item
        for item in evidence.capability_anchors
        if item.evidence_id == "ability.tone.1.medium"
    )
    copies = 2 if protection else 3
    maintenance = [
        Stimulus(
            stimulus_id="maint-easy",
            text="我食碗粥先返屋企。",
            role="maintenance",
            evidence_ids=[easy.evidence_id],
            speaker_profile_id="all_balanced",
            copies=copies,
            rationale="Retain correct easy carrier recognition across new lexical contexts.",
        ),
        Stimulus(
            stimulus_id="maint-medium",
            text="今晚幫姑娘開窗先。",
            role="maintenance",
            evidence_ids=[medium.evidence_id],
            speaker_profile_id="all_balanced",
            copies=copies,
            rationale="Maintain an error-free medium capability without requiring a failure cluster.",
        ),
    ]
    outcomes = [
        *base.expected_outcomes,
        *[
            ExpectedOutcome(
                outcome_id=f"retain-{anchor.difficulty}",
                category="maintenance",
                subject_evidence_id=anchor.evidence_id,
                measure="capability_lost_count",
                desired_direction="non_increase",
                rationale="No pre-round correctly recognized carrier may be lost.",
            )
            for anchor in (easy, medium)
        ],
    ]
    stimuli = [
        base.stimuli[0].model_copy(update={"text": "阿叔記得飲暖水。"}),
        *maintenance,
    ]
    if protection:
        unit = "我" if protection == "hard" else "日"
        cluster = next(
            item
            for item in evidence.error_clusters
            if item.level == "character"
            and item.operation == "deletion"
            and item.reference_unit == unit
        )
        stimuli.append(
            Stimulus(
                stimulus_id="repair",
                text="我聽日記得帶水樽。",
                role="protection",
                evidence_ids=[cluster.evidence_id],
                speaker_profile_id="all_balanced",
                copies=2,
                rationale="Repair an observed adaptation regression.",
            )
        )
        outcomes.append(
            ExpectedOutcome(
                outcome_id="repair-regression",
                category="protection",
                subject_evidence_id=cluster.evidence_id,
                measure="cluster_error_count",
                desired_direction="non_increase",
                rationale="Do not worsen the observed regression.",
            )
        )
    proposal = base.model_copy(
        update={
            "evidence_sha256": canonical_sha256(evidence),
            "stimuli": stimuli,
            "expected_outcomes": outcomes,
        }
    )
    return protocol, evidence, proposal, before, (easy, medium), target_id


def test_zero_error_capability_can_support_a_grounded_maintenance_prescription(
    tmp_path,
):
    protocol, evidence, proposal, _, (easy, medium), _ = _maintenance_case(tmp_path)
    assert medium.current_correct_count == medium.opportunity_count
    assert medium.current_unresolved_count == medium.m0_lost_count == 0
    assert set(item.evidence_id for item in evidence.error_clusters).isdisjoint(
        {easy.evidence_id, medium.evidence_id}
    )
    validate_proposal(proposal, evidence, protocol)
    validate_stimulus_grounding(proposal, evidence)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("wrong_unit", "capability_carrier_missing"),
        ("error_citation", "maintenance must cite capability anchors"),
        ("nonmaintenance_anchor", "maintenance must cite capability anchors"),
        ("missing_medium", "maintenance must cover each available"),
        ("wrong_measure", "maintenance_outcomes_do_not_match"),
    ],
)
def test_maintenance_has_operational_validation_not_just_a_role_label(
    tmp_path, mutation, expected
):
    protocol, evidence, proposal, _, (easy, _), target_id = _maintenance_case(tmp_path)
    stimuli = list(proposal.stimuli)
    outcomes = list(proposal.expected_outcomes)
    if mutation == "wrong_unit":
        stimuli[1] = stimuli[1].model_copy(update={"text": "我哋去飲茶。"})
    elif mutation == "error_citation":
        stimuli[2] = stimuli[2].model_copy(update={"evidence_ids": [target_id]})
    elif mutation == "nonmaintenance_anchor":
        stimuli[0] = stimuli[0].model_copy(
            update={"evidence_ids": [target_id, easy.evidence_id]}
        )
    elif mutation == "missing_medium":
        stimuli[2] = stimuli[2].model_copy(update={"evidence_ids": [easy.evidence_id]})
    else:
        outcomes[-1] = outcomes[-1].model_copy(
            update={"measure": "cluster_error_count", "subject_evidence_id": target_id}
        )
    changed = proposal.model_copy(
        update={"stimuli": stimuli, "expected_outcomes": outcomes}
    )
    with pytest.raises(ValueError, match=expected):
        validate_proposal(changed, evidence, protocol)
        validate_stimulus_grounding(changed, evidence)


def test_hard_only_regression_cannot_be_mislabeled_easy_medium_protection(tmp_path):
    protocol, evidence, proposal, *_ = _maintenance_case(tmp_path, protection="hard")
    with pytest.raises(
        ValueError, match="lack an observed easy/medium adaptation regression"
    ):
        validate_proposal(proposal, evidence, protocol)


def test_observed_easy_regression_can_still_receive_repair_exercises(tmp_path):
    protocol, evidence, proposal, *_ = _maintenance_case(tmp_path, protection="easy")
    validate_proposal(proposal, evidence, protocol)
    validate_stimulus_grounding(proposal, evidence)


def test_materialization_tracks_maintenance_dose_in_fixed_reference_strata(tmp_path):
    protocol, evidence, proposal, *_ = _maintenance_case(tmp_path)
    assignments = materialize_proposal(
        proposal, evidence, protocol, output_root=tmp_path
    )
    sidecar = build_materialization_sidecar(proposal, evidence, protocol, assignments)
    assert len(assignments) == 8
    maintenance = [item for item in sidecar.samples if item.role == "maintenance"]
    assert len(maintenance) == 6
    # This fixture has two easy and two hard frozen references, no medium reference.
    assert Counter(item.reference_difficulty_group for item in maintenance) == {
        "easy": 3,
        "hard": 3,
    }
    dose = next(item for item in sidecar.by_role if item.role == "maintenance")
    assert (dose.unique_stimuli, dose.copies) == (2, 6)
    assert dose.carrier_exposures == sum(
        carrier.count for sample in maintenance for carrier in sample.carrier_counts
    )
    assert all(item.carrier_counts[0].count > 0 for item in maintenance)
    repeated = materialize_proposal(proposal, evidence, protocol, output_root=tmp_path)
    assert repeated == assignments
    assert protocol.real_replay_count_per_round == 6
    assert protocol.lora.optimizer_steps_per_round == 4


def test_complete_round_and_reflection_keep_retention_losses_despite_net_gains(
    tmp_path,
):
    protocol, evidence, proposal, before, (easy, medium), _ = _maintenance_case(
        tmp_path
    )
    plan = compile_evaluation_plan(
        protocol,
        evidence,
        proposal,
        before_prediction_column="before",
        after_prediction_column="after",
    )
    assignments = materialize_proposal(
        proposal, evidence, protocol, output_root=tmp_path
    )
    sidecar = build_materialization_sidecar(proposal, evidence, protocol, assignments)
    synthesis = SynthesisReceipt(
        schema_version="1",
        run_id=protocol.run_id,
        round_index=1,
        status="complete",
        assignments_canonical_sha256=sidecar.assignments_canonical_sha256,
        requested=8,
        generated=8,
        failed=0,
        samples=[
            SynthesizedSample(
                sample_id=item.sample_id, audio_file_sha256=SHA_B, duration_seconds=1.0
            )
            for item in sidecar.samples
        ],
    )
    training = TrainingReceipt(
        schema_version="1",
        run_id=protocol.run_id,
        round_index=1,
        status="complete",
        parent_checkpoint_directory_sha256=protocol.initial_adapter.sha256,
        new_checkpoint_directory_sha256=SHA_C,
        train_manifest_file_sha256=SHA_A,
        training_index_file_sha256=SHA_B,
        trainer_state_file_sha256=SHA_C,
        real_rows=6,
        synthetic_rows=8,
        total_rows=14,
        expected_optimizer_steps=4,
        actual_optimizer_steps=4,
    )
    after = [{**row, "after": row["clean_gt"]} for row in before]
    after[0]["after"] = after[0]["after"].replace("食", "色")
    result = compile_round_result(
        protocol,
        evidence,
        proposal,
        plan,
        sidecar,
        synthesis,
        training,
        before_rows=before,
        after_rows=after,
        before_predictions_file_sha256=protocol.initial_adapter_dev_predictions.sha256,
        after_predictions_file_sha256=SHA_B,
    )
    changes = {item.evidence_id: item for item in result.response.capability_changes}
    assert changes[easy.evidence_id].newly_correct_count == 2
    assert changes[easy.evidence_id].lost_count == 1
    assert (
        changes[easy.evidence_id].after_correct_count
        > changes[easy.evidence_id].before_correct_count
    )
    assert changes[easy.evidence_id].response == "regressed"
    assert changes[medium.evidence_id].response == "retained"
    outcomes = {
        item.outcome_id: item for item in result.response.expected_outcome_results
    }
    assert outcomes["retain-easy"].before_value == 0
    assert outcomes["retain-easy"].after_value == 1
    assert outcomes["retain-easy"].compiled_verdict == "not_met"
    assert outcomes["retain-medium"].compiled_verdict == "met"
    assert outcomes["safety-easy"].compiled_verdict == "met"
    pack = compile_reflection_pack(
        protocol,
        evidence,
        proposal,
        plan,
        sidecar,
        result,
        prescription_prompt_file_sha256=SHA_A,
        reflection_prompt_file_sha256=SHA_B,
        skill_version="v000",
        skill_file_sha256=SHA_C,
    )
    assert pack.prescribed_proposal == proposal
    assert pack.response.capability_changes == result.response.capability_changes
    assert all(item.citation_id in pack.valid_citation_ids for item in changes.values())
    assert pack.training.real_rows == 6
    assert pack.training.synthetic_rows == 8
    assert pack.compiled_expectation_verdict == "mixed"
