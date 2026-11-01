"""Deterministic post-intervention compilation for the Therapist loop."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence

from .capabilities import evaluate_capability_anchors
from .evidence import compare_prediction_snapshots
from .materialize import validate_materialization_sidecar
from .round_schema import (
    BeforeSnapshotBinding,
    ClusterDescriptor,
    ClusterDoseRealization,
    CompiledConfound,
    DoseRealization,
    EvaluationBindings,
    EvaluationPlan,
    EvaluationScope,
    EvaluationSemantics,
    InterventionRealization,
    MaterializationSidecar,
    MonitoredCluster,
    OffTargetPolicy,
    OutcomeEvaluation,
    RealSpeechResponse,
    ReflectionPack,
    ReflectionProvenance,
    ReflectionV2,
    RoleDoseRealization,
    RoundBindings,
    RoundResultV2,
    SynthesisReceipt,
    TrainingReceipt,
)
from .schema import (
    ErrorCluster,
    EvidencePack,
    Proposal,
    RunProtocol,
    canonical_sha256,
    validate_evidence_pack,
    validate_proposal,
)


METRIC_MEASURES = (
    "overall_pooled_cer",
    "easy_pooled_cer",
    "medium_pooled_cer",
    "hard_pooled_cer",
    "patient_macro_cer",
)


def _descriptor(cluster: ErrorCluster) -> ClusterDescriptor:
    return ClusterDescriptor(
        evidence_id=cluster.evidence_id,
        operation=cluster.operation,
        level=cluster.level,
        reference_unit=cluster.reference_unit,
        hypothesis_unit=cluster.hypothesis_unit,
        left_context=cluster.left_context,
        right_context=cluster.right_context,
        rate_semantics=cluster.rate_semantics,
    )


def compile_evaluation_plan(
    protocol: RunProtocol,
    evidence: EvidencePack,
    proposal: Proposal,
    *,
    before_prediction_column: str,
    after_prediction_column: str,
    off_target_max_per_level_per_direction: int = 3,
) -> EvaluationPlan:
    """Freeze every post-round measurement before synthesis or training."""

    validate_evidence_pack(evidence, protocol)
    validate_proposal(proposal, evidence, protocol)
    if not before_prediction_column or not after_prediction_column:
        raise ValueError("before and after prediction columns must be explicit")
    clusters = {item.evidence_id: item for item in evidence.error_clusters}
    capabilities = {item.evidence_id: item for item in evidence.capability_anchors}
    monitored_ids: list[str] = []
    for outcome in proposal.expected_outcomes:
        if outcome.measure in {"cluster_error_count", "cluster_rate"}:
            if outcome.subject_evidence_id not in clusters:
                raise ValueError("cluster outcome cites no supplied error cluster")
            if outcome.subject_evidence_id not in monitored_ids:
                monitored_ids.append(outcome.subject_evidence_id)

    return EvaluationPlan(
        schema_version="1",
        run_id=protocol.run_id,
        round_index=proposal.round_index,
        bindings=EvaluationBindings(
            protocol_canonical_sha256=canonical_sha256(protocol),
            evidence_canonical_sha256=canonical_sha256(evidence),
            proposal_canonical_sha256=canonical_sha256(proposal),
            feedback_split_file_sha256=protocol.feedback_split.sha256,
            sealed_test_used=False,
        ),
        before_snapshot=BeforeSnapshotBinding(
            checkpoint_directory_sha256=evidence.snapshots.current.checkpoint_sha256,
            predictions_file_sha256=evidence.snapshots.current.predictions_sha256,
            prediction_column=before_prediction_column,
        ),
        semantics=EvaluationSemantics(
            after_prediction_column=after_prediction_column,
            critical_cer_threshold=evidence.data_scope.critical_cer_threshold,
            normalizer_id=evidence.data_scope.normalizer_id,
            character_alignment_id=evidence.data_scope.character_alignment_id,
            phonology_id=evidence.data_scope.phonology_id,
            difficulty_source=evidence.data_scope.difficulty_source,
            difficulty_policy=evidence.data_scope.difficulty_policy,
            delta_convention=(
                "after_minus_before; negative CER delta means improvement"
            ),
        ),
        scope=EvaluationScope(
            sample_count=evidence.data_scope.sample_count,
            speaker_count=evidence.data_scope.speaker_count,
            content_count=evidence.data_scope.content_count,
        ),
        assessment_claims=proposal.assessment_claims,
        hypothesis=proposal.hypothesis,
        target_clusters=[
            _descriptor(clusters[evidence_id])
            for evidence_id in proposal.target_evidence_ids
        ],
        planned_outcomes=proposal.expected_outcomes,
        risks=proposal.risks,
        monitored_clusters=[
            MonitoredCluster(
                descriptor=_descriptor(clusters[evidence_id]),
                before=clusters[evidence_id].observations.current,
            )
            for evidence_id in monitored_ids
        ],
        monitored_capabilities=[
            capabilities[evidence_id]
            for evidence_id in dict.fromkeys(
                item.subject_evidence_id
                for item in proposal.expected_outcomes
                if item.measure == "capability_lost_count"
            )
        ],
        off_target_policy=OffTargetPolicy(
            universe="union_before_after_character_and_cer_supported_phone_v1",
            exclude_monitored=True,
            minimum_absolute_error_count_delta=1,
            max_per_level_per_direction=off_target_max_per_level_per_direction,
            ordering=(
                "absolute_error_delta_desc_then_max_patient_support_desc_then_"
                "max_content_support_desc_then_evidence_id"
            ),
        ),
        required_metric_groups=["overall", "easy", "medium", "hard"],
        required_transitions=["exact", "utterance_cer", "critical"],
    )


def _verify_plan(
    protocol: RunProtocol,
    evidence: EvidencePack,
    proposal: Proposal,
    plan: EvaluationPlan,
) -> None:
    expected = compile_evaluation_plan(
        protocol,
        evidence,
        proposal,
        before_prediction_column=plan.before_snapshot.prediction_column,
        after_prediction_column=plan.semantics.after_prediction_column,
        off_target_max_per_level_per_direction=(
            plan.off_target_policy.max_per_level_per_direction
        ),
    )
    if plan != expected:
        raise ValueError("EvaluationPlan differs from deterministic compilation")


def _compile_realization(
    sidecar: MaterializationSidecar,
    receipt: SynthesisReceipt,
) -> InterventionRealization:
    if receipt.run_id != sidecar.run_id or receipt.round_index != sidecar.round_index:
        raise ValueError("synthesis receipt identity differs from materialization")
    if receipt.assignments_canonical_sha256 != sidecar.assignments_canonical_sha256:
        raise ValueError("synthesis receipt is bound to different assignments")
    synthesized = {item.sample_id: item for item in receipt.samples}
    indexed = {item.sample_id: item for item in sidecar.samples}
    if set(synthesized) != set(indexed):
        raise ValueError("synthesis sample IDs differ from the materialization sidecar")

    role_requested: Counter[str] = Counter()
    role_generated: Counter[str] = Counter()
    role_seconds: Counter[str] = Counter()
    role_planned_carriers: Counter[str] = Counter()
    role_realized_carriers: Counter[str] = Counter()
    cluster_requested: Counter[str] = Counter()
    cluster_generated: Counter[str] = Counter()
    cluster_seconds: Counter[str] = Counter()
    cluster_planned_carriers: Counter[str] = Counter()
    cluster_realized_carriers: Counter[str] = Counter()
    overall_carriers = 0
    overall_seconds = 0.0
    for sample_id, index in indexed.items():
        output = synthesized[sample_id]
        overall_seconds += output.duration_seconds
        carrier_total = sum(item.count for item in index.carrier_counts)
        overall_carriers += carrier_total
        role_requested[index.role] += 1
        role_generated[index.role] += 1
        role_seconds[index.role] += output.duration_seconds
        role_planned_carriers[index.role] += carrier_total
        role_realized_carriers[index.role] += carrier_total
        for carrier in index.carrier_counts:
            cluster_requested[carrier.evidence_id] += 1
            cluster_generated[carrier.evidence_id] += 1
            cluster_seconds[carrier.evidence_id] += output.duration_seconds
            cluster_planned_carriers[carrier.evidence_id] += carrier.count
            cluster_realized_carriers[carrier.evidence_id] += carrier.count

    by_role = [
        RoleDoseRealization(
            role=role,
            requested=role_requested[role],
            generated=role_generated[role],
            failed=0,
            audio_seconds=role_seconds[role],
            planned_carrier_exposures=role_planned_carriers[role],
            realized_carrier_exposures=role_realized_carriers[role],
        )
        for role in ("target", "contrast", "protection", "maintenance")
    ]
    return InterventionRealization(
        status="complete",
        silent_replacement_count=0,
        overall=DoseRealization(
            requested=receipt.requested,
            generated=receipt.generated,
            failed=receipt.failed,
            audio_seconds=overall_seconds,
            planned_carrier_exposures=overall_carriers,
            realized_carrier_exposures=overall_carriers,
        ),
        by_role=by_role,
        by_cluster=[
            ClusterDoseRealization(
                evidence_id=evidence_id,
                requested=cluster_requested[evidence_id],
                generated=cluster_generated[evidence_id],
                failed=0,
                audio_seconds=cluster_seconds[evidence_id],
                linked_audio_seconds=cluster_seconds[evidence_id],
                planned_carrier_exposures=cluster_planned_carriers[evidence_id],
                realized_carrier_exposures=cluster_realized_carriers[evidence_id],
            )
            for evidence_id in sorted(cluster_requested)
        ],
    )


def _outcome_results(plan: EvaluationPlan, response: RealSpeechResponse):
    metrics = {item.measure: item for item in response.metric_changes}
    clusters = {
        item.descriptor.evidence_id: item for item in response.monitored_cluster_changes
    }
    capabilities = {item.evidence_id: item for item in response.capability_changes}
    results: list[OutcomeEvaluation] = []
    for outcome in plan.planned_outcomes:
        target_recovery_criterion_met = None
        if outcome.measure == "cluster_error_count":
            item = clusters[outcome.subject_evidence_id]
            before_value = float(item.before.error_count)
            after_value = float(item.after.error_count)
            delta = float(item.error_count_delta)
        elif outcome.measure == "cluster_rate":
            item = clusters[outcome.subject_evidence_id]
            before_value = item.before.rate
            after_value = item.after.rate
            delta = item.rate_delta
        elif outcome.measure == "capability_lost_count":
            # Freeze the before-correct positions: a newly acquired carrier
            # elsewhere cannot cancel a lost capability.
            capability = capabilities[outcome.subject_evidence_id]
            before_value = 0.0
            after_value = float(capability.lost_count)
            delta = after_value
        else:
            metric = metrics[outcome.measure]
            before_value = metric.before_value
            after_value = metric.after_value
            delta = metric.delta
        if outcome.measure in {"cluster_error_count", "cluster_rate"}:
            positions = item.target_positions
            if positions is None:
                raise ValueError(
                    "monitored cluster is missing paired target-position evidence"
                )
            resolved = (
                positions.after_unresolved_count <= positions.before_unresolved_count
                and positions.after_cohort_character_insertions
                <= positions.before_cohort_character_insertions
            )
            target_recovery_criterion_met = (
                positions.after_error_count < positions.before_error_count
                and positions.recovered_count > positions.regressed_count
                and resolved
                if outcome.desired_direction == "decrease"
                else positions.after_error_count <= positions.before_error_count
                and positions.regressed_count <= positions.recovered_count
                and resolved
            )
        observed = "decrease" if delta < 0 else "increase" if delta > 0 else "unchanged"
        verdict = (
            "met"
            if outcome.desired_direction == "decrease" and delta < 0
            else "met"
            if outcome.desired_direction == "non_increase" and delta <= 0
            else "not_met"
        )
        if target_recovery_criterion_met is False:
            verdict = "not_met"
        results.append(
            OutcomeEvaluation(
                outcome_id=outcome.outcome_id,
                category=outcome.category,
                subject_evidence_id=outcome.subject_evidence_id,
                measure=outcome.measure,
                desired_direction=outcome.desired_direction,
                before_value=before_value,
                after_value=after_value,
                delta=delta,
                observed_direction=observed,
                target_recovery_criterion_met=target_recovery_criterion_met,
                compiled_verdict=verdict,
                citation_id=f"result.outcome.{outcome.outcome_id}",
            )
        )
    return results


def _validate_round_facts(
    plan: EvaluationPlan,
    sidecar: MaterializationSidecar,
    result: RoundResultV2,
) -> None:
    """Check derived views against their source facts, not just their hashes."""

    response = result.response
    for change in response.metric_changes:
        group_name = change.measure.removesuffix("_pooled_cer")
        metric_field = "pooled_cer"
        if change.measure == "patient_macro_cer":
            group_name, metric_field = "overall", "patient_macro_cer"
        before_value = getattr(
            getattr(response.before_metrics, group_name), metric_field
        )
        after_value = getattr(getattr(response.after_metrics, group_name), metric_field)
        delta = getattr(response.paired_delta, change.measure)
        direction = (
            "improved" if delta < 0 else "worsened" if delta > 0 else "unchanged"
        )
        if (
            change.before_value != before_value
            or change.after_value != after_value
            or abs(change.delta - delta) > 1e-12
            or change.response != direction
        ):
            raise ValueError(
                f"metric change differs from source metrics: {change.measure}"
            )
    if result.intervention != _compile_realization(sidecar, result.synthesis):
        raise ValueError(
            "intervention realization differs from sidecar and synthesis receipt"
        )
    if response.expected_outcome_results != _outcome_results(plan, response):
        raise ValueError("compiled outcome values differ from measured response")
    if [item.evidence_id for item in response.capability_changes] != [
        item.evidence_id for item in plan.monitored_capabilities
    ]:
        raise ValueError("capability results differ from frozen EvaluationPlan")
    for measured, anchor in zip(response.capability_changes, plan.monitored_capabilities):
        if (measured.level, measured.reference_unit, measured.difficulty) != (
            anchor.level, anchor.reference_unit, anchor.difficulty
        ):
            raise ValueError("capability descriptor differs from frozen EvaluationPlan")
        if (
            measured.opportunity_count,
            measured.before_correct_count,
            measured.before_unresolved_count,
        ) != (
            anchor.opportunity_count,
            anchor.current_correct_count,
            anchor.current_unresolved_count,
        ):
            raise ValueError("capability baseline differs from frozen EvaluationPlan")
    observed = response.monitored_cluster_changes
    if len(observed) != len(plan.monitored_clusters) or any(
        measured.descriptor != planned.descriptor or measured.before != planned.before
        for measured, planned in zip(observed, plan.monitored_clusters)
    ):
        raise ValueError("monitored cluster facts differ from frozen EvaluationPlan")
    for metrics in (response.before_metrics, response.after_metrics):
        if (
            metrics.overall.sample_count,
            metrics.overall.speaker_count,
        ) != (plan.scope.sample_count, plan.scope.speaker_count):
            raise ValueError("response scope differs from frozen EvaluationPlan")


def compile_round_result(
    protocol: RunProtocol,
    evidence: EvidencePack,
    proposal: Proposal,
    plan: EvaluationPlan,
    sidecar: MaterializationSidecar,
    synthesis: SynthesisReceipt,
    training: TrainingReceipt,
    *,
    before_rows: Sequence[Mapping[str, object]],
    after_rows: Sequence[Mapping[str, object]],
    before_predictions_file_sha256: str,
    after_predictions_file_sha256: str,
) -> RoundResultV2:
    """Compile one completed round without making an accept/reject decision."""

    _verify_plan(protocol, evidence, proposal, plan)
    validate_materialization_sidecar(sidecar, proposal, evidence, protocol)
    expected_bindings = (
        canonical_sha256(protocol),
        canonical_sha256(evidence),
        canonical_sha256(proposal),
    )
    if (
        sidecar.protocol_canonical_sha256,
        sidecar.evidence_canonical_sha256,
        sidecar.proposal_canonical_sha256,
    ) != expected_bindings:
        raise ValueError("materialization sidecar bindings differ from round inputs")
    if sidecar.run_id != protocol.run_id or sidecar.round_index != proposal.round_index:
        raise ValueError("materialization sidecar identity differs from round")
    if before_predictions_file_sha256 != plan.before_snapshot.predictions_file_sha256:
        raise ValueError("before predictions hash differs from EvaluationPlan")
    if (
        training.run_id != protocol.run_id
        or training.round_index != proposal.round_index
    ):
        raise ValueError("training receipt identity differs from round")
    if synthesis.artifact_mode != training.artifact_mode:
        raise ValueError("synthesis and training artifact modes differ")
    if training.artifact_mode != evidence.artifact_mode:
        raise ValueError("execution receipts and EvidencePack artifact modes differ")
    if (
        training.parent_checkpoint_directory_sha256
        != plan.before_snapshot.checkpoint_directory_sha256
    ):
        raise ValueError("training parent differs from the planned before checkpoint")
    from .preference_contract import validate_preference_training
    validate_preference_training(protocol, training, sample_ids=[x.sample_id for x in synthesis.samples])
    if training.actual_optimizer_steps != protocol.lora.optimizer_steps_per_round:
        raise ValueError("training optimizer steps differ from protocol")

    realization = _compile_realization(sidecar, synthesis)
    paired = compare_prediction_snapshots(
        before_rows,
        after_rows,
        before_prediction_column=plan.before_snapshot.prediction_column,
        after_prediction_column=plan.semantics.after_prediction_column,
        critical_cer_threshold=plan.semantics.critical_cer_threshold,
        monitored_clusters=[item.descriptor for item in plan.monitored_clusters],
        off_target_max_per_level_per_direction=(
            plan.off_target_policy.max_per_level_per_direction
        ),
    )
    if paired.before_metrics != evidence.snapshots.current.metrics:
        raise ValueError("recomputed before metrics differ from EvidencePack current")
    response_without_outcomes = RealSpeechResponse(
        before_metrics=paired.before_metrics,
        after_metrics=paired.after_metrics,
        paired_delta=paired.paired_delta,
        metric_changes=list(paired.metric_changes),
        transitions=paired.transitions,
        patient_response=paired.patient_response,
        monitored_cluster_changes=list(paired.monitored_cluster_changes),
        capability_changes=evaluate_capability_anchors(
            plan.monitored_capabilities,
            before_rows,
            after_rows,
            before_prediction_column=plan.before_snapshot.prediction_column,
            after_prediction_column=plan.semantics.after_prediction_column,
        ),
        expected_outcome_results=[],
        off_target_changes=list(paired.off_target_changes),
        phonology_coverage_before=paired.phonology_coverage_before,
        phonology_coverage_after=paired.phonology_coverage_after,
    )
    outcomes = _outcome_results(plan, response_without_outcomes)
    response = RealSpeechResponse.model_validate(
        {
            **response_without_outcomes.model_dump(mode="python"),
            "expected_outcome_results": outcomes,
        }
    )
    result = RoundResultV2(
        schema_version="3",
        run_id=protocol.run_id,
        round_index=proposal.round_index,
        bindings=RoundBindings(
            artifact_mode=training.artifact_mode,
            protocol_canonical_sha256=canonical_sha256(protocol),
            evidence_canonical_sha256=canonical_sha256(evidence),
            proposal_canonical_sha256=canonical_sha256(proposal),
            evaluation_plan_canonical_sha256=canonical_sha256(plan),
            materialization_sidecar_canonical_sha256=canonical_sha256(sidecar),
            synthesis_receipt_canonical_sha256=canonical_sha256(synthesis),
            training_receipt_canonical_sha256=canonical_sha256(training),
            feedback_split_file_sha256=protocol.feedback_split.sha256,
            parent_checkpoint_directory_sha256=(
                training.parent_checkpoint_directory_sha256
            ),
            new_checkpoint_directory_sha256=(training.new_checkpoint_directory_sha256),
            before_predictions_file_sha256=before_predictions_file_sha256,
            after_predictions_file_sha256=after_predictions_file_sha256,
            sealed_test_used=False,
        ),
        intervention=realization,
        synthesis=synthesis,
        training=training,
        response=response,
    )
    _validate_round_facts(plan, sidecar, result)
    return result


def _citation_ids(pack: ReflectionPack) -> list[str]:
    values = ["proposal.hypothesis", "proposal.expected_response", "next.action_contract"]
    values.extend(
        f"proposal.stimulus.{item.stimulus_id}"
        for item in pack.prescribed_proposal.stimuli
    )
    values.extend(
        f"proposal.outcome.{item.outcome_id}"
        for item in pack.evaluation_plan.planned_outcomes
    )
    values.extend(
        f"proposal.risk.{index}" for index, _ in enumerate(pack.evaluation_plan.risks)
    )
    values.extend(item.citation_id for item in pack.response.expected_outcome_results)
    values.extend(item.citation_id for item in pack.response.metric_changes)
    values.extend(
        f"result.transition.{group}" for group in ("overall", "easy", "medium", "hard")
    )
    values.append(pack.response.patient_response.citation_id)
    values.extend(item.citation_id for item in pack.response.monitored_cluster_changes)
    values.extend(item.citation_id for item in pack.response.capability_changes)
    values.extend(item.citation_id for item in pack.response.off_target_changes)
    values.append("result.synthesis.overall")
    values.extend(
        f"result.synthesis.role.{item.role}" for item in pack.realization.by_role
    )
    values.extend(
        f"result.synthesis.cluster.{item.evidence_id}"
        for item in pack.realization.by_cluster
    )
    values.extend(
        [
            "result.training",
            "result.phonology_coverage.before",
            "result.phonology_coverage.after",
        ]
    )
    values.extend(item.citation_id for item in pack.activated_confounds)
    values.extend(f"intervention.{item.ledger_id}" for item in pack.history)
    if len(set(values)) != len(values):
        raise ValueError("compiled citation namespace contains duplicates")
    return values


def _privacy_check(payload: object) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    serialized = json.dumps(payload, ensure_ascii=False)
    lowered = serialized.casefold()
    forbidden = ("/data/", "audio_path", "reference_wav", "speaker_id", "patient_id")
    found = [item for item in forbidden if item in lowered]
    if re.search(r"\bvlink[0-9a-z_/-]*", lowered):
        found.append("vlink identifier")
    if found:
        raise ValueError(
            f"ReflectionPack contains private identifiers or paths: {found}"
        )


def compile_reflection_pack(
    protocol: RunProtocol,
    evidence: EvidencePack,
    proposal: Proposal,
    plan: EvaluationPlan,
    sidecar: MaterializationSidecar,
    result: RoundResultV2,
    *,
    prescription_prompt_file_sha256: str,
    reflection_prompt_file_sha256: str,
    skill_version: str,
    skill_file_sha256: str,
) -> ReflectionPack:
    """Sanitize and bind the only input accepted by Therapist.reflect."""

    _verify_plan(protocol, evidence, proposal, plan)
    validate_materialization_sidecar(sidecar, proposal, evidence, protocol)
    if result.bindings.evaluation_plan_canonical_sha256 != canonical_sha256(plan):
        raise ValueError("RoundResult is bound to a different EvaluationPlan")
    if result.bindings.materialization_sidecar_canonical_sha256 != canonical_sha256(
        sidecar
    ):
        raise ValueError("RoundResult is bound to a different sidecar")
    confounds = [
        CompiledConfound(
            confound_id="joint_round_attribution",
            description=(
                "Observed changes are the joint response to fixed real replay, "
                "realized synthetic data, and optimization; they do not isolate TTS."
                if result.training.real_rows else
                "This round used zero real replay. Observed changes reflect synthetic "
                "data and optimization together; they do not isolate the TTS mechanism."
            ),
            citation_id="result.confound.joint_round_attribution",
        )
    ]
    if plan.monitored_capabilities or any(
        item.descriptor.level != "character" for item in plan.monitored_clusters
    ):
        confounds.append(
            CompiledConfound(
                confound_id="phone_attribution_lower_bound",
                description=(
                    "Phone-component counts are conservative transcript-derived "
                    "lower bounds, not acoustic impairment measurements."
                ),
                citation_id="result.confound.phone_attribution_lower_bound",
            )
        )
    if any(
        item.before.alignment_uncertainty or item.after.alignment_uncertainty
        for item in result.response.monitored_cluster_changes
    ):
        confounds.append(
            CompiledConfound(
                confound_id="alignment_uncertainty_present",
                description="At least one monitored character cluster has ambiguous alignment mass.",
                citation_id="result.confound.alignment_uncertainty_present",
            )
        )
    pack = ReflectionPack(
        schema_version="3",
        run_id=protocol.run_id,
        round_index=proposal.round_index,
        provenance=ReflectionProvenance(
            artifact_mode=result.bindings.artifact_mode,
            protocol_canonical_sha256=canonical_sha256(protocol),
            evidence_canonical_sha256=canonical_sha256(evidence),
            proposal_canonical_sha256=canonical_sha256(proposal),
            next_action_contract_canonical_sha256=canonical_sha256(evidence.action_contract),
            evaluation_plan_canonical_sha256=canonical_sha256(plan),
            round_result_canonical_sha256=canonical_sha256(result),
            prescription_prompt_file_sha256=prescription_prompt_file_sha256,
            reflection_prompt_file_sha256=reflection_prompt_file_sha256,
            skill_version=skill_version,
            skill_file_sha256=skill_file_sha256,
            feedback_split_file_sha256=protocol.feedback_split.sha256,
            parent_checkpoint_directory_sha256=(
                result.bindings.parent_checkpoint_directory_sha256
            ),
            new_checkpoint_directory_sha256=(
                result.bindings.new_checkpoint_directory_sha256
            ),
            before_predictions_file_sha256=(
                result.bindings.before_predictions_file_sha256
            ),
            after_predictions_file_sha256=(
                result.bindings.after_predictions_file_sha256
            ),
            sealed_test_used=False,
        ),
        evaluation_plan=plan,
        prescribed_proposal=proposal,
        next_action_contract=evidence.action_contract,
        materialization_sidecar=sidecar,
        round_result=result,
        compiled_expectation_verdict=_compiled_verdict(
            result.response.expected_outcome_results
        ),
        history=evidence.intervention_history,
        activated_confounds=confounds,
        valid_citation_ids=["temporary"],
    )
    pack = ReflectionPack.model_validate(
        {
            **pack.model_dump(mode="python"),
            "valid_citation_ids": _citation_ids(pack),
        }
    )
    validate_reflection_pack(pack)
    return pack


def _compiled_verdict(outcomes: Sequence[OutcomeEvaluation]) -> str:
    if all(item.compiled_verdict == "met" for item in outcomes):
        return "supported"
    primary = [item for item in outcomes if item.category == "primary"]
    if primary and all(item.compiled_verdict == "not_met" for item in primary):
        return "refuted"
    return "mixed"


def expected_expectation_verdict(pack: ReflectionPack) -> str:
    return _compiled_verdict(pack.response.expected_outcome_results)


def validate_reflection_pack(
    pack: ReflectionPack,
    protocol: RunProtocol | None = None,
    evidence: EvidencePack | None = None,
    proposal: Proposal | None = None,
) -> None:
    if (evidence is None) != (proposal is None):
        raise ValueError("EvidencePack and Proposal must be supplied together")
    plan = pack.evaluation_plan
    sidecar = pack.materialization_sidecar
    result = pack.round_result
    if not (
        pack.run_id
        == plan.run_id
        == sidecar.run_id
        == result.run_id
        == result.training.run_id
    ) or not (
        pack.round_index
        == plan.round_index
        == sidecar.round_index
        == result.round_index
        == result.training.round_index
    ):
        raise ValueError("ReflectionPack components have different identities")
    provenance = pack.provenance
    if provenance.evaluation_plan_canonical_sha256 != canonical_sha256(plan):
        raise ValueError("ReflectionPack EvaluationPlan hash is inconsistent")
    if provenance.round_result_canonical_sha256 != canonical_sha256(result):
        raise ValueError("ReflectionPack RoundResult hash is inconsistent")
    if provenance.proposal_canonical_sha256 != canonical_sha256(pack.prescribed_proposal):
        raise ValueError("ReflectionPack prescribed Proposal hash is inconsistent")
    if provenance.next_action_contract_canonical_sha256 != canonical_sha256(pack.next_action_contract):
        raise ValueError("ReflectionPack next ActionContract hash is inconsistent")
    if (
        pack.prescribed_proposal.hypothesis != plan.hypothesis
        or pack.prescribed_proposal.expected_outcomes != plan.planned_outcomes
        or pack.prescribed_proposal.risks != plan.risks
    ):
        raise ValueError("prescribed Proposal differs from frozen EvaluationPlan")
    if result.bindings.materialization_sidecar_canonical_sha256 != canonical_sha256(
        sidecar
    ):
        raise ValueError("RoundResult sidecar hash is inconsistent")
    for field_name in (
        "protocol_canonical_sha256",
        "evidence_canonical_sha256",
        "proposal_canonical_sha256",
    ):
        values = {
            getattr(provenance, field_name),
            getattr(plan.bindings, field_name),
            getattr(sidecar, field_name),
            getattr(result.bindings, field_name),
        }
        if len(values) != 1:
            raise ValueError(f"ReflectionPack {field_name} bindings differ")
    if {
        provenance.feedback_split_file_sha256,
        plan.bindings.feedback_split_file_sha256,
        result.bindings.feedback_split_file_sha256,
    } != {provenance.feedback_split_file_sha256}:
        raise ValueError("ReflectionPack feedback split bindings differ")
    if provenance.artifact_mode != result.bindings.artifact_mode or (
        result.training.artifact_mode != result.bindings.artifact_mode
    ):
        raise ValueError("ReflectionPack artifact modes differ")
    lineage_pairs = (
        (
            provenance.parent_checkpoint_directory_sha256,
            result.bindings.parent_checkpoint_directory_sha256,
        ),
        (
            provenance.new_checkpoint_directory_sha256,
            result.bindings.new_checkpoint_directory_sha256,
        ),
        (
            provenance.before_predictions_file_sha256,
            result.bindings.before_predictions_file_sha256,
        ),
        (
            provenance.after_predictions_file_sha256,
            result.bindings.after_predictions_file_sha256,
        ),
    )
    if any(left != right for left, right in lineage_pairs):
        raise ValueError("ReflectionPack checkpoint or prediction bindings differ")
    if protocol is not None:
        if pack.run_id != protocol.run_id or pack.round_index > protocol.round_count:
            raise ValueError("ReflectionPack identity differs from protocol")
        if provenance.protocol_canonical_sha256 != canonical_sha256(protocol):
            raise ValueError("ReflectionPack is bound to a different protocol")
        if provenance.feedback_split_file_sha256 != protocol.feedback_split.sha256:
            raise ValueError("ReflectionPack is bound to a different feedback split")
        contract = pack.next_action_contract
        if (
            contract.synthetic_count != protocol.synthetic_count_per_round
            or contract.unique_stimulus_count != protocol.unique_stimulus_count_per_round
            or contract.real_replay_count != protocol.real_replay_count_per_round
            or contract.permitted_speaker_profiles != protocol.speaker_profiles
            or contract.proposal_constraints != protocol.proposal_constraints
        ):
            raise ValueError("next ActionContract differs from frozen protocol")
        from .preference_contract import validate_preference_training
        validate_preference_training(protocol, pack.training, sample_ids=[x.sample_id for x in sidecar.samples])
        if pack.training.actual_optimizer_steps != (
            protocol.lora.optimizer_steps_per_round
        ):
            raise ValueError("training optimizer steps differ from protocol")
    if evidence is not None and proposal is not None:
        if protocol is None:
            raise ValueError("source validation requires the frozen protocol")
        if pack.prescribed_proposal != proposal:
            raise ValueError("prescribed Proposal differs from source Proposal")
        if pack.next_action_contract != evidence.action_contract:
            raise ValueError("next ActionContract differs from source EvidencePack")
        if provenance.artifact_mode != evidence.artifact_mode:
            raise ValueError("ReflectionPack and EvidencePack artifact modes differ")
        _verify_plan(protocol, evidence, proposal, plan)
        validate_materialization_sidecar(sidecar, proposal, evidence, protocol)
        if pack.response.before_metrics != evidence.snapshots.current.metrics:
            raise ValueError("before metrics differ from EvidencePack current")

    _validate_round_facts(plan, sidecar, result)

    planned = plan.planned_outcomes
    observed = pack.response.expected_outcome_results
    planned_signatures = [
        (
            item.outcome_id,
            item.category,
            item.subject_evidence_id,
            item.measure,
            item.desired_direction,
        )
        for item in planned
    ]
    observed_signatures = [
        (
            item.outcome_id,
            item.category,
            item.subject_evidence_id,
            item.measure,
            item.desired_direction,
        )
        for item in observed
    ]
    if observed_signatures != planned_signatures:
        raise ValueError("compiled outcome results differ from prescribed outcomes")
    if [item.measure for item in pack.response.metric_changes] != list(METRIC_MEASURES):
        raise ValueError("metric changes are not complete and canonically ordered")
    monitored_ids = [
        item.descriptor.evidence_id for item in pack.response.monitored_cluster_changes
    ]
    required_monitored_ids = list(
        dict.fromkeys(
            item.subject_evidence_id
            for item in planned
            if item.measure in {"cluster_error_count", "cluster_rate"}
        )
    )
    if monitored_ids != required_monitored_ids:
        raise ValueError("monitored cluster results differ from cluster outcomes")
    primary_ids = list(
        dict.fromkeys(
            item.subject_evidence_id for item in planned if item.category == "primary"
        )
    )
    if [item.evidence_id for item in plan.target_clusters] != primary_ids:
        raise ValueError("target cluster digest differs from primary outcomes")
    if pack.training.parent_checkpoint_directory_sha256 != (
        pack.provenance.parent_checkpoint_directory_sha256
    ) or pack.training.new_checkpoint_directory_sha256 != (
        pack.provenance.new_checkpoint_directory_sha256
    ):
        raise ValueError("training lineage differs from ReflectionPack provenance")
    if pack.training.preference_delivery is None and pack.training.synthetic_rows != pack.realization.overall.generated:
        raise ValueError("training synthetic rows differ from realized synthesis")
    if sum(item.copies for item in sidecar.by_role) != (
        pack.realization.overall.requested
    ):
        raise ValueError("planned role dose differs from realized request count")
    if result.bindings.evaluation_plan_canonical_sha256 != canonical_sha256(plan):
        raise ValueError("RoundResult EvaluationPlan hash is inconsistent")
    if result.bindings.parent_checkpoint_directory_sha256 != (
        plan.before_snapshot.checkpoint_directory_sha256
    ):
        raise ValueError("RoundResult parent differs from EvaluationPlan")
    if result.bindings.before_predictions_file_sha256 != (
        plan.before_snapshot.predictions_file_sha256
    ):
        raise ValueError("RoundResult before predictions differ from EvaluationPlan")
    compiled_verdict = _compiled_verdict(observed)
    if pack.compiled_expectation_verdict != compiled_verdict:
        raise ValueError("compiled expectation verdict is inconsistent")
    expected = _citation_ids(pack)
    if pack.valid_citation_ids != expected:
        raise ValueError("ReflectionPack citation namespace is not canonical")
    _privacy_check(pack)


def validate_reflection(reflection: ReflectionV2, pack: ReflectionPack) -> None:
    """Reject any reflection that contradicts Compiler facts or citations."""

    validate_reflection_pack(pack)
    if reflection.run_id != pack.run_id or reflection.round_index != pack.round_index:
        raise ValueError("reflection identity differs from ReflectionPack")
    if reflection.reflection_pack_canonical_sha256 != canonical_sha256(pack):
        raise ValueError("reflection is bound to a different ReflectionPack")
    expected_verdict = expected_expectation_verdict(pack)
    if reflection.compiled_expectation_verdict != expected_verdict:
        raise ValueError("expectation summary contradicts compiled outcomes")

    allowed = set(pack.valid_citation_ids)

    def check_citations(values: Sequence[str], *, required: Sequence[str] = ()) -> None:
        if len(set(values)) != len(values):
            raise ValueError("reflection citations must be unique")
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"reflection cites unknown results: {sorted(unknown)}")
        missing = set(required) - set(values)
        if missing:
            raise ValueError(f"reflection omits required citations: {sorted(missing)}")

    outcome_results = {
        item.outcome_id: item for item in pack.response.expected_outcome_results
    }
    checks = {item.outcome_id: item for item in reflection.expectation_checks}
    if list(checks) != [
        item.outcome_id for item in pack.evaluation_plan.planned_outcomes
    ]:
        raise ValueError("reflection must address outcomes once in prescribed order")
    for outcome_id, check in checks.items():
        compiled = outcome_results[outcome_id]
        if check.verdict != compiled.compiled_verdict:
            raise ValueError(f"reflection changes compiled verdict for {outcome_id}")
        check_citations(
            check.result_citations,
            required=[f"result.outcome.{outcome_id}"],
        )
    check_citations(
        reflection.hypothesis_citations,
        required=[item.citation_id for item in outcome_results.values()],
    )

    primary = [
        item
        for item in pack.response.expected_outcome_results
        if item.category == "primary"
    ]
    findings = {item.outcome_id: item for item in reflection.target_findings}
    if list(findings) != [item.outcome_id for item in primary]:
        raise ValueError("target findings must exactly cover primary outcomes")
    cluster_changes = {
        item.descriptor.evidence_id: item
        for item in pack.response.monitored_cluster_changes
    }
    for outcome in primary:
        finding = findings[outcome.outcome_id]
        if finding.evidence_id != outcome.subject_evidence_id:
            raise ValueError("target finding uses the wrong cluster")
        expected_response = cluster_changes[finding.evidence_id].target_positions.response
        if finding.response != expected_response:
            raise ValueError("target finding changes the compiled response")
        check_citations(
            finding.result_citations,
            required=[
                outcome.citation_id,
                f"result.cluster.{finding.evidence_id}",
            ],
        )

    metric_changes = {item.measure: item for item in pack.response.metric_changes}
    safety = {item.measure: item for item in reflection.safety_findings}
    if list(safety) != list(METRIC_MEASURES):
        raise ValueError("safety findings must cover all metrics in canonical order")
    for measure, finding in safety.items():
        compiled = metric_changes[measure]
        if finding.response != compiled.response:
            raise ValueError(f"safety finding changes compiled {measure} response")
        check_citations(finding.result_citations, required=[compiled.citation_id])

    if (
        reflection.patient_heterogeneity.pattern
        != pack.response.patient_response.pattern
    ):
        raise ValueError("patient heterogeneity changes the compiled pattern")
    check_citations(
        reflection.patient_heterogeneity.result_citations,
        required=["result.patient_directions"],
    )

    capability_findings = {item.evidence_id: item for item in reflection.capability_findings}
    if list(capability_findings) != [item.evidence_id for item in pack.response.capability_changes]:
        raise ValueError("capability findings must exactly cover monitored capabilities")
    for change in pack.response.capability_changes:
        finding = capability_findings[change.evidence_id]
        if finding.response != change.response:
            raise ValueError("capability finding changes compiled retention response")
        check_citations(finding.result_citations, required=[change.citation_id])

    for item in reflection.competing_explanations:
        check_citations(item.result_citations)
        if not any(value.startswith("result.") for value in item.result_citations):
            raise ValueError("provisional explanation lacks a result citation")
    for item in reflection.lesson_candidates:
        check_citations(item.result_citations)
        if not any(value.startswith("result.") for value in item.result_citations):
            raise ValueError("lesson candidate lacks a result citation")
    check_citations(reflection.next_round_guidance.result_citations)
    if not any(
        value.startswith("result.")
        for value in reflection.next_round_guidance.result_citations
    ):
        raise ValueError("next-round guidance lacks a result citation")
    _privacy_check(reflection)


__all__ = [
    "compile_evaluation_plan",
    "compile_reflection_pack",
    "compile_round_result",
    "expected_expectation_verdict",
    "validate_reflection",
    "validate_reflection_pack",
]
