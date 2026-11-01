from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from test_therapist_outcome import (
    SHA_A,
    SHA_B,
    SHA_C,
    _assignments,
    _prediction_rows,
    _proposal,
    _protocol,
)
from therapist_harness.agent import build_reflection_request
from therapist_harness.materialize import build_materialization_sidecar
from therapist_harness.outcome import (
    METRIC_MEASURES,
    compile_evaluation_plan,
    compile_reflection_pack,
    compile_round_result,
    validate_reflection,
)
from therapist_harness.round_schema import (
    CompetingExplanation,
    ExpectationCheck,
    LessonCandidate,
    NextRoundGuidance,
    PatientHeterogeneity,
    ReflectionPack,
    ReflectionV2,
    SafetyFinding,
    SynthesisReceipt,
    SynthesizedSample,
    TargetFinding,
    TrainingReceipt,
)
from therapist_harness.schema import (
    EvidencePack,
    Proposal,
    RunProtocol,
    canonical_sha256,
)


SYSTEM_PROMPT = "Reflect on the Compiler-owned intervention result."
SKILL_TEXT = "Use only response-to-intervention evidence."


@dataclass(frozen=True)
class ReflectionArtifacts:
    protocol: RunProtocol
    evidence: EvidencePack
    proposal: Proposal
    pack: ReflectionPack
    target_id: str


@pytest.fixture()
def mixed_round(tmp_path: Path) -> ReflectionArtifacts:
    """Compile a real source-bound round with target gain and easy regression."""

    protocol = _protocol(tmp_path)
    evidence, proposal, target_id = _proposal(protocol)
    plan = compile_evaluation_plan(
        protocol,
        evidence,
        proposal,
        before_prediction_column="before",
        after_prediction_column="after",
    )
    sidecar = build_materialization_sidecar(
        proposal,
        evidence,
        protocol,
        _assignments(protocol, proposal, evidence),
    )
    synthesis = SynthesisReceipt(
        schema_version="1",
        run_id=protocol.run_id,
        round_index=1,
        status="complete",
        assignments_canonical_sha256=sidecar.assignments_canonical_sha256,
        requested=2,
        generated=2,
        failed=0,
        samples=[
            SynthesizedSample(
                sample_id=item.sample_id,
                audio_file_sha256=SHA_B,
                duration_seconds=float(index + 1),
            )
            for index, item in enumerate(sidecar.samples)
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
        synthetic_rows=2,
        total_rows=8,
        expected_optimizer_steps=2,
        actual_optimizer_steps=2,
    )
    before_rows = _prediction_rows(after=False)
    after_rows = _prediction_rows(after=True)
    # Restore one of the two target deletions, retain the other, and add two
    # easy-group substitutions outside the target. Primary recovery is genuine
    # while easy safety fails; insertion migration has its own regression test.
    next(item for item in after_rows if item["utt_id"] == "u2")["after"] = (
        "阿聽日錯錯"
    )
    result = compile_round_result(
        protocol,
        evidence,
        proposal,
        plan,
        sidecar,
        synthesis,
        training,
        before_rows=before_rows,
        after_rows=after_rows,
        before_predictions_file_sha256=(
            protocol.initial_adapter_dev_predictions.sha256
        ),
        after_predictions_file_sha256=SHA_B,
    )
    pack = compile_reflection_pack(
        protocol,
        evidence,
        proposal,
        plan,
        sidecar,
        result,
        prescription_prompt_file_sha256=SHA_A,
        reflection_prompt_file_sha256=hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        skill_version="v000",
        skill_file_sha256=hashlib.sha256(SKILL_TEXT.encode("utf-8")).hexdigest(),
    )
    assert pack.compiled_expectation_verdict == "mixed"
    outcomes = {
        item.outcome_id: item for item in pack.response.expected_outcome_results
    }
    assert outcomes["primary-target"].compiled_verdict == "met"
    assert outcomes["safety-easy"].compiled_verdict == "not_met"
    return ReflectionArtifacts(
        protocol=protocol,
        evidence=evidence,
        proposal=proposal,
        pack=pack,
        target_id=target_id,
    )


def _reflection(item: ReflectionArtifacts) -> ReflectionV2:
    pack = item.pack
    outcomes = pack.response.expected_outcome_results
    outcome_by_id = {outcome.outcome_id: outcome for outcome in outcomes}
    primary = next(outcome for outcome in outcomes if outcome.category == "primary")
    target_change = next(
        change
        for change in pack.response.monitored_cluster_changes
        if change.descriptor.evidence_id == primary.subject_evidence_id
    )
    metric_changes = {change.measure: change for change in pack.response.metric_changes}
    confound = pack.activated_confounds[0]
    return ReflectionV2(
        schema_version="4",
        run_id=pack.run_id,
        round_index=pack.round_index,
        reflection_pack_canonical_sha256=canonical_sha256(pack),
        compiled_expectation_verdict=pack.compiled_expectation_verdict,
        hypothesis_verdict="mixed",
        hypothesis_rationale=(
            "The primary target improved, but frozen-easy pooled CER worsened."
        ),
        hypothesis_citations=[outcome.citation_id for outcome in outcomes],
        expectation_checks=[
            ExpectationCheck(
                outcome_id=planned.outcome_id,
                verdict=outcome_by_id[planned.outcome_id].compiled_verdict,
                interpretation=(f"Compiler-reported result for {planned.outcome_id}."),
                result_citations=[outcome_by_id[planned.outcome_id].citation_id],
            )
            for planned in pack.evaluation_plan.planned_outcomes
        ],
        target_findings=[
            TargetFinding(
                outcome_id=primary.outcome_id,
                evidence_id=primary.subject_evidence_id,
                response=target_change.target_positions.response,
                interpretation="The monitored target count fell on real speech.",
                result_citations=[primary.citation_id, target_change.citation_id],
            )
        ],
        safety_findings=[
            SafetyFinding(
                measure=measure,
                response=metric_changes[measure].response,
                interpretation=f"Compiler-reported response for {measure}.",
                result_citations=[metric_changes[measure].citation_id],
            )
            for measure in METRIC_MEASURES
        ],
        capability_findings=[],
        patient_heterogeneity=PatientHeterogeneity(
            pattern=pack.response.patient_response.pattern,
            interpretation="Some patients improved and some worsened.",
            result_citations=[pack.response.patient_response.citation_id],
        ),
        competing_explanations=[
            CompetingExplanation(
                explanation_id=confound.confound_id,
                explanation=confound.description,
                predicted_observation="A different target mix could change regression while replay stays fixed.",
                result_citations=[confound.citation_id],
            )
        ],
        lesson_candidates=[
            LessonCandidate(
                action="adjust",
                condition="A target improves while easy speech regresses.",
                guidance="Retain target evidence but strengthen safety protection.",
                caveat="This is one combined intervention round.",
                predicted_observation="The next easy-group loss count should fall without reversing target recovery.",
                result_citations=[
                    outcome_by_id["primary-target"].citation_id,
                    outcome_by_id["safety-easy"].citation_id,
                ],
            )
        ],
        next_round_guidance=NextRoundGuidance(
            primary_strategy="retain",
            counterbalance_strategy="not_applicable",
            protection_strategy="increase_within_budget",
            maintenance_strategy="increase_within_budget",
            primary_practice_dose_direction="maintain",
            rationale="Keep the target direction while responding to easy regression.",
            predicted_observation="Fewer formerly correct easy carriers should be lost in the next round.",
            result_citations=[
                outcome_by_id["primary-target"].citation_id,
                outcome_by_id["safety-easy"].citation_id,
            ],
        ),
    )


def _mutated(reflection: ReflectionV2, mutate) -> ReflectionV2:
    payload = reflection.model_dump(mode="json")
    mutate(payload)
    return ReflectionV2.model_validate(payload)


def test_build_reflection_request_and_validate_mixed(
    mixed_round: ReflectionArtifacts,
) -> None:
    item = mixed_round
    request = build_reflection_request(
        item.protocol,
        item.evidence,
        item.proposal,
        item.pack,
        system_prompt=SYSTEM_PROMPT,
        skill_text=SKILL_TEXT,
    )

    assert request["model"] == "gpt-test"
    assert request["reasoning"] == {"effort": "xhigh"}
    assert request["store"] is False
    assert request["text"]["format"]["name"] == "therapist_reflection"
    assert request["text"]["format"]["strict"] is True
    user_payload = json.loads(request["input"][2]["content"])
    assert "not a new Proposal" in user_payload["task"]
    assert "previous_response_id" not in request
    assert "tools" not in request
    assert user_payload["reflection_pack_sha256"] == canonical_sha256(item.pack)
    assert user_payload["reflection_pack"] == item.pack.model_dump(mode="json")

    validate_reflection(_reflection(item), item.pack)


def test_reflection_rejects_changed_compiled_verdict(
    mixed_round: ReflectionArtifacts,
) -> None:
    invalid = _mutated(
        _reflection(mixed_round),
        lambda payload: payload.update(compiled_expectation_verdict="supported"),
    )

    with pytest.raises(ValueError, match="expectation summary contradicts"):
        validate_reflection(invalid, mixed_round.pack)


def test_production_reflection_prompt_builds_a_self_contained_request(
    mixed_round: ReflectionArtifacts,
) -> None:
    package = Path(__file__).parents[1] / "src" / "therapist_harness"
    instruction = (package / "prompts/reflection.md").read_text("utf-8")
    skill = (package / "skills/v000/SKILL.md").read_text("utf-8")
    payload = mixed_round.pack.model_dump(mode="json")
    payload["provenance"].update(
        reflection_prompt_file_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
        skill_file_sha256=hashlib.sha256(skill.encode()).hexdigest(),
    )
    pack = ReflectionPack.model_validate(payload)
    request = build_reflection_request(
        mixed_round.protocol,
        mixed_round.evidence,
        mixed_round.proposal,
        pack,
        system_prompt=instruction,
        skill_text=skill,
    )
    assert [item["role"] for item in request["input"]] == ["system", "developer", "user"]
    assert request["input"][0]["content"] == instruction
    assert skill in request["input"][1]["content"]
    delivered = json.loads(request["input"][2]["content"])["reflection_pack"]
    assert delivered["prescribed_proposal"] == mixed_round.proposal.model_dump(mode="json")
    assert delivered["round_result"]["response"] == pack.response.model_dump(mode="json")
    assert "response" not in delivered
    assert delivered["next_action_contract"] == pack.next_action_contract.model_dump(mode="json")
    assert "previous_response_id" not in request
    assert "tools" not in request


def test_reflection_can_interpret_hypothesis_as_inconclusive(
    mixed_round: ReflectionArtifacts,
) -> None:
    reflection = _mutated(
        _reflection(mixed_round),
        lambda payload: payload.update(
            hypothesis_verdict="inconclusive",
            hypothesis_rationale="The mixed checks do not isolate the hypothesized synthetic-data mechanism.",
        ),
    )
    validate_reflection(reflection, mixed_round.pack)


def test_reflection_pack_contains_bound_actual_stimuli(
    mixed_round: ReflectionArtifacts,
) -> None:
    pack = mixed_round.pack
    assert pack.prescribed_proposal == mixed_round.proposal
    for stimulus in mixed_round.proposal.stimuli:
        assert f"proposal.stimulus.{stimulus.stimulus_id}" in pack.valid_citation_ids
    assert "proposal.expected_response" in pack.valid_citation_ids
    assert pack.next_action_contract == mixed_round.evidence.action_contract
    assert pack.provenance.next_action_contract_canonical_sha256 == canonical_sha256(
        mixed_round.evidence.action_contract
    )
    assert "next.action_contract" in pack.valid_citation_ids


def test_reflection_accepts_new_testable_explanation(
    mixed_round: ReflectionArtifacts,
) -> None:
    def propose_explanation(payload: dict) -> None:
        payload["competing_explanations"] = [{
            "explanation_id": "text_context_repetition",
            "explanation": "The repeated carrier context may explain uneven transfer beyond its target.",
            "predicted_observation": "Diversifying that context should reduce easy regression if this explanation is useful.",
            "result_citations": [
                f"proposal.stimulus.{mixed_round.proposal.stimuli[0].stimulus_id}",
                "result.metric.easy_pooled_cer",
            ],
        }]
    reflection = _mutated(_reflection(mixed_round), propose_explanation)
    validate_reflection(reflection, mixed_round.pack)


def test_reflection_rejects_missing_outcome(
    mixed_round: ReflectionArtifacts,
) -> None:
    invalid = _mutated(
        _reflection(mixed_round),
        lambda payload: payload["expectation_checks"].pop(1),
    )

    with pytest.raises(ValueError, match="must address outcomes once"):
        validate_reflection(invalid, mixed_round.pack)


def test_reflection_rejects_forged_citation(
    mixed_round: ReflectionArtifacts,
) -> None:
    def forge(payload: dict) -> None:
        payload["expectation_checks"][0]["result_citations"].append(
            "result.outcome.forged"
        )

    invalid = _mutated(_reflection(mixed_round), forge)
    with pytest.raises(ValueError, match="unknown results"):
        validate_reflection(invalid, mixed_round.pack)


def test_reflection_rejects_wrong_metric_direction(
    mixed_round: ReflectionArtifacts,
) -> None:
    def reverse_easy(payload: dict) -> None:
        easy = next(
            finding
            for finding in payload["safety_findings"]
            if finding["measure"] == "easy_pooled_cer"
        )
        easy["response"] = "improved"

    invalid = _mutated(_reflection(mixed_round), reverse_easy)
    with pytest.raises(ValueError, match="changes compiled easy_pooled_cer response"):
        validate_reflection(invalid, mixed_round.pack)


def test_reflection_rejects_wrong_patient_pattern(
    mixed_round: ReflectionArtifacts,
) -> None:
    current = mixed_round.pack.response.patient_response.pattern
    replacement = "all_tied" if current != "all_tied" else "all_improved"

    def replace_pattern(payload: dict) -> None:
        payload["patient_heterogeneity"]["pattern"] = replacement

    invalid = _mutated(_reflection(mixed_round), replace_pattern)
    with pytest.raises(ValueError, match="changes the compiled pattern"):
        validate_reflection(invalid, mixed_round.pack)
