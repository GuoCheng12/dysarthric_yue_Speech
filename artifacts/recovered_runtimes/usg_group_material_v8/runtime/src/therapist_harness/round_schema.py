"""Strict contracts for post-prescription execution and reflection.

The models in this module deliberately separate deterministic Harness facts from
LLM interpretation.  Nothing here represents a Gate, a utility score, or a
checkpoint-selection decision.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .capabilities import CapabilityAnchor, CapabilityChange
from .memory_schema import CompetingExplanation, LessonCandidate, NextRoundGuidance
from .preference_contract import PreferenceDelivery
from .schema import (
    ID_PATTERN,
    SHA256_PATTERN,
    ActionContract,
    AssessmentClaim,
    ErrorObservation,
    ExpectedOutcome,
    InterventionSummary,
    MetricDelta,
    MetricSet,
    NonNegativeFloat,
    NonNegativeInt,
    PatientDirectionCounts,
    PhonologyCoverageSet,
    PositiveInt,
    Proposal,
    StrictModel,
    TransitionSet,
    canonical_sha256,
)


Role = Literal["target", "contrast", "protection", "maintenance"]
ArtifactMode = Literal["recorded_execution", "counterfactual_interface_smoke"]
Measure = Literal[
    "cluster_error_count",
    "cluster_rate",
    "capability_lost_count",
    "overall_pooled_cer",
    "easy_pooled_cer",
    "medium_pooled_cer",
    "hard_pooled_cer",
    "patient_macro_cer",
]
MetricMeasure = Literal[
    "overall_pooled_cer",
    "easy_pooled_cer",
    "medium_pooled_cer",
    "hard_pooled_cer",
    "patient_macro_cer",
]
ResponseDirection = Literal["improved", "unchanged", "worsened", "inconclusive"]


class ClusterDescriptor(StrictModel):
    evidence_id: str = Field(pattern=ID_PATTERN)
    operation: Literal["deletion", "substitution", "insertion"]
    level: Literal["character", "onset", "nucleus", "coda", "tone"]
    reference_unit: str
    hypothesis_unit: str
    left_context: str | None
    right_context: str | None
    rate_semantics: Literal[
        "deterministic_character_alignment_rate",
        "confirmed_substitution_lower_bound_rate",
    ]


class EvaluationBindings(StrictModel):
    protocol_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    proposal_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    feedback_split_file_sha256: str = Field(pattern=SHA256_PATTERN)
    sealed_test_used: Literal[False]


class BeforeSnapshotBinding(StrictModel):
    checkpoint_directory_sha256: str = Field(pattern=SHA256_PATTERN)
    predictions_file_sha256: str = Field(pattern=SHA256_PATTERN)
    prediction_column: str = Field(min_length=1)


class EvaluationSemantics(StrictModel):
    after_prediction_column: str = Field(min_length=1)
    critical_cer_threshold: float = Field(gt=0.0)
    normalizer_id: Literal["therapist_text_normalize_v1"]
    character_alignment_id: Literal[
        "levenshtein_stable_traceback_for_character_edits_and_all_optimal_paths_for_forced_substitutions_v1"
    ]
    phonology_id: Literal["pycantonese_5.0.0_surface_to_cer_forced_v2"]
    difficulty_source: Literal["frozen_zero_shot_character_cer"]
    difficulty_policy: Literal["easy:CER<0.2;medium:0.2<=CER<0.5;hard:CER>=0.5"]
    delta_convention: Literal[
        "after_minus_before; negative CER delta means improvement"
    ]


class EvaluationScope(StrictModel):
    sample_count: PositiveInt
    speaker_count: PositiveInt
    content_count: PositiveInt


class MonitoredCluster(StrictModel):
    descriptor: ClusterDescriptor
    before: ErrorObservation


class OffTargetPolicy(StrictModel):
    universe: Literal["union_before_after_character_and_cer_supported_phone_v1"]
    exclude_monitored: Literal[True]
    minimum_absolute_error_count_delta: Literal[1]
    max_per_level_per_direction: PositiveInt
    ordering: Literal[
        "absolute_error_delta_desc_then_max_patient_support_desc_then_max_content_support_desc_then_evidence_id"
    ]


class EvaluationPlan(StrictModel):
    schema_version: Literal["1"]
    run_id: str = Field(pattern=ID_PATTERN)
    round_index: PositiveInt
    bindings: EvaluationBindings
    before_snapshot: BeforeSnapshotBinding
    semantics: EvaluationSemantics
    scope: EvaluationScope
    assessment_claims: list[AssessmentClaim] = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    target_clusters: list[ClusterDescriptor] = Field(min_length=1)
    planned_outcomes: list[ExpectedOutcome] = Field(min_length=1)
    risks: list[str] = Field(min_length=1)
    monitored_clusters: list[MonitoredCluster]
    monitored_capabilities: list[CapabilityAnchor] = Field(default_factory=list)
    off_target_policy: OffTargetPolicy
    required_metric_groups: list[Literal["overall", "easy", "medium", "hard"]]
    required_transitions: list[Literal["exact", "utterance_cer", "critical"]]

    @model_validator(mode="after")
    def validate_plan(self) -> "EvaluationPlan":
        outcome_ids = [item.outcome_id for item in self.planned_outcomes]
        if len(set(outcome_ids)) != len(outcome_ids):
            raise ValueError("evaluation outcome IDs must be unique")
        cluster_ids = [item.descriptor.evidence_id for item in self.monitored_clusters]
        if len(set(cluster_ids)) != len(cluster_ids):
            raise ValueError("monitored cluster IDs must be unique")
        required_cluster_ids = {
            item.subject_evidence_id
            for item in self.planned_outcomes
            if item.measure in {"cluster_error_count", "cluster_rate"}
        }
        if set(cluster_ids) != required_cluster_ids:
            raise ValueError("monitored clusters must exactly match cluster outcomes")
        capability_ids = [item.evidence_id for item in self.monitored_capabilities]
        required_capability_ids = list(dict.fromkeys(
            item.subject_evidence_id
            for item in self.planned_outcomes
            if item.measure == "capability_lost_count"
        ))
        if capability_ids != required_capability_ids:
            raise ValueError("monitored capabilities must exactly match maintenance outcomes")
        primary_ids = list(
            dict.fromkeys(
                item.subject_evidence_id
                for item in self.planned_outcomes
                if item.category == "primary"
            )
        )
        if [item.evidence_id for item in self.target_clusters] != primary_ids:
            raise ValueError("target cluster descriptors must match primary outcomes")
        if self.required_metric_groups != ["overall", "easy", "medium", "hard"]:
            raise ValueError("all metric groups must be frozen in canonical order")
        if self.required_transitions != ["exact", "utterance_cer", "critical"]:
            raise ValueError("all transition families must be frozen")
        return self


class CarrierCount(StrictModel):
    evidence_id: str = Field(pattern=ID_PATTERN)
    count: NonNegativeInt


class MaterializedSampleIndex(StrictModel):
    sample_id: str = Field(pattern=ID_PATTERN)
    stimulus_id: str = Field(pattern=ID_PATTERN)
    copy_index: PositiveInt
    role: Role
    evidence_ids: list[str] = Field(min_length=1)
    speaker_profile_id: str = Field(pattern=ID_PATTERN)
    reference_difficulty_group: Literal["easy", "medium", "hard"]
    carrier_counts: list[CarrierCount] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sample_index(self) -> "MaterializedSampleIndex":
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("sample evidence IDs must be unique")
        carrier_ids = [item.evidence_id for item in self.carrier_counts]
        if len(set(carrier_ids)) != len(carrier_ids):
            raise ValueError("sample carrier IDs must be unique")
        if set(carrier_ids) != set(self.evidence_ids):
            raise ValueError("carrier counts must cover every cited cluster")
        return self


class PlannedRoleDose(StrictModel):
    role: Role
    unique_stimuli: NonNegativeInt
    copies: NonNegativeInt
    carrier_exposures: NonNegativeInt


class PlannedClusterDose(StrictModel):
    evidence_id: str = Field(pattern=ID_PATTERN)
    unique_stimuli: NonNegativeInt
    linked_copies: NonNegativeInt
    carrier_exposures: NonNegativeInt


class MaterializationSidecar(StrictModel):
    schema_version: Literal["1"]
    run_id: str = Field(pattern=ID_PATTERN)
    round_index: PositiveInt
    protocol_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    proposal_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    assignments_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    samples: list[MaterializedSampleIndex] = Field(min_length=1)
    by_role: list[PlannedRoleDose]
    by_cluster: list[PlannedClusterDose] = Field(min_length=1)
    whole_batch_text_carrier_exposures: list[CarrierCount] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sidecar(self) -> "MaterializationSidecar":
        sample_ids = [item.sample_id for item in self.samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("materialization sample IDs must be unique")
        if [item.role for item in self.by_role] != [
            "target",
            "contrast",
            "protection",
            "maintenance",
        ]:
            raise ValueError("role dose must use canonical order")
        if sum(item.copies for item in self.by_role) != len(self.samples):
            raise ValueError("role copies do not partition materialized samples")
        cluster_ids = [item.evidence_id for item in self.by_cluster]
        if len(set(cluster_ids)) != len(cluster_ids):
            raise ValueError("cluster dose IDs must be unique")
        role_expected: dict[str, tuple[int, int, int]] = {}
        for role in ("target", "contrast", "protection", "maintenance"):
            selected = [item for item in self.samples if item.role == role]
            role_expected[role] = (
                len({item.stimulus_id for item in selected}),
                len(selected),
                sum(
                    carrier.count
                    for item in selected
                    for carrier in item.carrier_counts
                ),
            )
        for item in self.by_role:
            if (
                item.unique_stimuli,
                item.copies,
                item.carrier_exposures,
            ) != role_expected[item.role]:
                raise ValueError("role dose differs from sample index")

        sample_cluster_ids = sorted(
            {
                carrier.evidence_id
                for item in self.samples
                for carrier in item.carrier_counts
            }
        )
        if cluster_ids != sample_cluster_ids:
            raise ValueError("cluster dose IDs differ from sample index")
        for item in self.by_cluster:
            selected = [
                sample
                for sample in self.samples
                if item.evidence_id
                in {carrier.evidence_id for carrier in sample.carrier_counts}
            ]
            exposures = sum(
                next(
                    carrier.count
                    for carrier in sample.carrier_counts
                    if carrier.evidence_id == item.evidence_id
                )
                for sample in selected
            )
            if (
                item.unique_stimuli,
                item.linked_copies,
                item.carrier_exposures,
            ) != (
                len({sample.stimulus_id for sample in selected}),
                len(selected),
                exposures,
            ):
                raise ValueError("cluster dose differs from sample index")
        return self


class SynthesizedSample(StrictModel):
    sample_id: str = Field(pattern=ID_PATTERN)
    audio_file_sha256: str = Field(pattern=SHA256_PATTERN)
    duration_seconds: NonNegativeFloat


class SynthesisReceipt(StrictModel):
    schema_version: Literal["1"]
    run_id: str = Field(pattern=ID_PATTERN)
    round_index: PositiveInt
    artifact_mode: ArtifactMode = "recorded_execution"
    status: Literal["complete"]
    assignments_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    requested: PositiveInt
    generated: PositiveInt
    failed: Literal[0]
    samples: list[SynthesizedSample] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_receipt(self) -> "SynthesisReceipt":
        if self.generated != self.requested or len(self.samples) != self.generated:
            raise ValueError("completed synthesis must realize every requested sample")
        ids = [item.sample_id for item in self.samples]
        if len(set(ids)) != len(ids):
            raise ValueError("synthesis sample IDs must be unique")
        return self


class TrainingReceipt(StrictModel):
    schema_version: Literal["1"]
    run_id: str = Field(pattern=ID_PATTERN)
    round_index: PositiveInt
    artifact_mode: ArtifactMode = "recorded_execution"
    status: Literal["complete"]
    parent_checkpoint_directory_sha256: str = Field(pattern=SHA256_PATTERN)
    new_checkpoint_directory_sha256: str = Field(pattern=SHA256_PATTERN)
    train_manifest_file_sha256: str = Field(pattern=SHA256_PATTERN)
    training_index_file_sha256: str = Field(pattern=SHA256_PATTERN)
    trainer_state_file_sha256: str = Field(pattern=SHA256_PATTERN)
    real_rows: NonNegativeInt
    synthetic_rows: NonNegativeInt
    total_rows: PositiveInt
    expected_optimizer_steps: PositiveInt
    actual_optimizer_steps: PositiveInt
    preference_delivery: PreferenceDelivery | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_training(self) -> "TrainingReceipt":
        if self.preference_delivery is None and self.synthetic_rows == 0:
            raise ValueError("legacy SFT rounds require synthetic rows")
        if self.real_rows + self.synthetic_rows != self.total_rows:
            raise ValueError("training row counts do not sum")
        if self.expected_optimizer_steps != self.actual_optimizer_steps:
            raise ValueError("training did not execute the frozen optimizer-step count")
        return self


class DoseRealization(StrictModel):
    requested: NonNegativeInt
    generated: NonNegativeInt
    failed: NonNegativeInt
    audio_seconds: NonNegativeFloat
    planned_carrier_exposures: NonNegativeInt
    realized_carrier_exposures: NonNegativeInt

    @model_validator(mode="after")
    def validate_dose(self) -> "DoseRealization":
        if self.generated + self.failed != self.requested:
            raise ValueError("dose generated and failed counts do not sum")
        return self


class RoleDoseRealization(DoseRealization):
    role: Role


class ClusterDoseRealization(DoseRealization):
    evidence_id: str = Field(pattern=ID_PATTERN)
    linked_audio_seconds: NonNegativeFloat

    @model_validator(mode="after")
    def validate_linked_duration(self) -> "ClusterDoseRealization":
        if abs(self.audio_seconds - self.linked_audio_seconds) > 1e-12:
            raise ValueError("cluster duration must be explicitly marked as linked")
        return self


class InterventionRealization(StrictModel):
    status: Literal["complete"]
    silent_replacement_count: Literal[0]
    carrier_measurement: Literal["requested_text_only_not_acoustic_verification"] = (
        "requested_text_only_not_acoustic_verification"
    )
    overall: DoseRealization
    by_role: list[RoleDoseRealization]
    by_cluster: list[ClusterDoseRealization]

    @model_validator(mode="after")
    def validate_realization(self) -> "InterventionRealization":
        if [item.role for item in self.by_role] != [
            "target",
            "contrast",
            "protection",
            "maintenance",
        ]:
            raise ValueError("realized role dose must use canonical order")
        for field in ("requested", "generated", "failed"):
            if sum(getattr(item, field) for item in self.by_role) != getattr(
                self.overall, field
            ):
                raise ValueError(f"role {field} does not partition overall")
        if (
            abs(
                sum(item.audio_seconds for item in self.by_role)
                - self.overall.audio_seconds
            )
            > 1e-9
        ):
            raise ValueError("role audio durations do not partition overall")
        for field in ("planned_carrier_exposures", "realized_carrier_exposures"):
            if sum(getattr(item, field) for item in self.by_role) != getattr(
                self.overall, field
            ):
                raise ValueError(f"role {field} does not partition overall")
        if self.overall.failed != 0:
            raise ValueError("completed intervention cannot contain failed synthesis")
        for item in [self.overall, *self.by_role, *self.by_cluster]:
            if item.failed or item.generated != item.requested:
                raise ValueError("completed intervention must realize every request")
            if item.realized_carrier_exposures != item.planned_carrier_exposures:
                raise ValueError("completed intervention must realize planned carriers")
        return self


class RoundBindings(StrictModel):
    artifact_mode: ArtifactMode = "recorded_execution"
    protocol_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    proposal_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluation_plan_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    materialization_sidecar_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    synthesis_receipt_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    training_receipt_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    feedback_split_file_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_checkpoint_directory_sha256: str = Field(pattern=SHA256_PATTERN)
    new_checkpoint_directory_sha256: str = Field(pattern=SHA256_PATTERN)
    before_predictions_file_sha256: str = Field(pattern=SHA256_PATTERN)
    after_predictions_file_sha256: str = Field(pattern=SHA256_PATTERN)
    sealed_test_used: Literal[False]


class MetricChange(StrictModel):
    measure: MetricMeasure
    before_value: NonNegativeFloat
    after_value: NonNegativeFloat
    delta: float
    response: Literal["improved", "unchanged", "worsened"]
    citation_id: str = Field(pattern=ID_PATTERN)

    @model_validator(mode="after")
    def validate_delta(self) -> "MetricChange":
        if abs((self.after_value - self.before_value) - self.delta) > 1e-12:
            raise ValueError("metric delta is inconsistent")
        response = (
            "improved"
            if self.delta < 0
            else "worsened"
            if self.delta > 0
            else "unchanged"
        )
        if self.response != response:
            raise ValueError("metric response is inconsistent")
        return self


TargetPositionState = Literal[
    "correct",
    "signature_error",
    "other_substitution",
    "whole_character_deletion",
    "unresolved",
]


class TargetPositionTransition(StrictModel):
    before: TargetPositionState
    after: TargetPositionState
    count: PositiveInt
    patient_count: PositiveInt
    content_count: PositiveInt

    @model_validator(mode="after")
    def validate_support(self) -> "TargetPositionTransition":
        if max(self.patient_count, self.content_count) > self.count:
            raise ValueError("target transition support exceeds position count")
        return self


class TargetPositionCounts(StrictModel):
    opportunity_count: NonNegativeInt
    before_error_count: NonNegativeInt
    after_error_count: NonNegativeInt
    before_unresolved_count: NonNegativeInt
    after_unresolved_count: NonNegativeInt
    recovered_count: NonNegativeInt
    regressed_count: NonNegativeInt
    signature_recovered_count: NonNegativeInt
    signature_to_other_error_count: NonNegativeInt
    signature_to_deletion_count: NonNegativeInt
    signature_unresolved_count: NonNegativeInt
    transitions: list[TargetPositionTransition]

    @model_validator(mode="after")
    def validate_counts(self) -> "TargetPositionCounts":
        keys = [(item.before, item.after) for item in self.transitions]
        if len(set(keys)) != len(keys):
            raise ValueError("target-position transitions must be unique")

        def count(predicate):
            return sum(item.count for item in self.transitions if predicate(item))

        errors = {"signature_error", "other_substitution", "whole_character_deletion"}
        expected = {
            "opportunity_count": count(lambda x: True),
            "before_error_count": count(lambda x: x.before != "correct"),
            "after_error_count": count(lambda x: x.after != "correct"),
            "before_unresolved_count": count(lambda x: x.before == "unresolved"),
            "after_unresolved_count": count(lambda x: x.after == "unresolved"),
            "recovered_count": count(
                lambda x: x.before in errors and x.after == "correct"
            ),
            "regressed_count": count(
                lambda x: x.before == "correct" and x.after in errors
            ),
            "signature_recovered_count": count(
                lambda x: x.before == "signature_error" and x.after == "correct"
            ),
            "signature_to_other_error_count": count(
                lambda x: (
                    x.before == "signature_error" and x.after == "other_substitution"
                )
            ),
            "signature_to_deletion_count": count(
                lambda x: (
                    x.before == "signature_error"
                    and x.after == "whole_character_deletion"
                )
            ),
            "signature_unresolved_count": count(
                lambda x: x.before == "signature_error" and x.after == "unresolved"
            ),
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"target-position {field} differs from transitions")
        return self


def target_position_direction(
    counts: TargetPositionCounts, *, insertion_growth: bool = False
) -> ResponseDirection:
    if insertion_growth:
        return "inconclusive"
    if counts.after_unresolved_count > counts.before_unresolved_count:
        return "inconclusive"
    net_recovered = counts.recovered_count - counts.regressed_count
    error_delta = counts.after_error_count - counts.before_error_count
    if error_delta < 0 and net_recovered > 0:
        return "improved"
    if error_delta > 0 and net_recovered < 0:
        return "worsened"
    if error_delta == 0 and net_recovered == 0 and not counts.after_unresolved_count:
        return "unchanged"
    return "inconclusive"


class TargetPositionResponse(TargetPositionCounts):
    semantics: Literal["paired_gt_carrier_character_recovery_v1"]
    response: ResponseDirection
    before_cohort_character_insertions: NonNegativeInt
    after_cohort_character_insertions: NonNegativeInt
    by_difficulty: dict[Literal["easy", "medium", "hard"], TargetPositionCounts]

    @model_validator(mode="after")
    def validate_partition(self) -> "TargetPositionResponse":
        if self.response != target_position_direction(
            self,
            insertion_growth=(
                self.after_cohort_character_insertions
                > self.before_cohort_character_insertions
            ),
        ):
            raise ValueError("target-position response differs from paired recovery")
        if set(self.by_difficulty) != {"easy", "medium", "hard"}:
            raise ValueError("target-position difficulty groups must be complete")
        for field in TargetPositionCounts.model_fields:
            if field != "transitions" and getattr(self, field) != sum(
                getattr(group, field) for group in self.by_difficulty.values()
            ):
                raise ValueError(f"target-position {field} is not partitioned")
        for transition in self.transitions:
            if transition.count != sum(
                item.count
                for group in self.by_difficulty.values()
                for item in group.transitions
                if (item.before, item.after) == (transition.before, transition.after)
            ):
                raise ValueError(
                    "target-position transition counts are not partitioned"
                )
        return self


class ClusterChange(StrictModel):
    descriptor: ClusterDescriptor
    before: ErrorObservation
    after: ErrorObservation
    error_count_delta: int
    rate_delta: float
    response: Literal["improved", "unchanged", "worsened"]
    response_semantics: Literal["exact_signature_frequency_only"] = (
        "exact_signature_frequency_only"
    )
    target_positions: TargetPositionResponse | None = None
    citation_id: str = Field(pattern=ID_PATTERN)

    @model_validator(mode="after")
    def validate_change(self) -> "ClusterChange":
        if self.target_positions is not None and (
            self.target_positions.opportunity_count != self.before.opportunity_count
        ):
            raise ValueError("target positions differ from frozen GT opportunities")
        if self.after.error_count - self.before.error_count != self.error_count_delta:
            raise ValueError("cluster error-count delta is inconsistent")
        if abs((self.after.rate - self.before.rate) - self.rate_delta) > 1e-12:
            raise ValueError("cluster rate delta is inconsistent")
        if self.before.opportunity_count != self.after.opportunity_count:
            raise ValueError("paired cluster opportunity counts differ")
        response = (
            "improved"
            if self.error_count_delta < 0
            else "worsened"
            if self.error_count_delta > 0
            else "unchanged"
        )
        if self.response != response:
            raise ValueError("cluster response is inconsistent")
        return self


class OutcomeEvaluation(StrictModel):
    outcome_id: str = Field(pattern=ID_PATTERN)
    category: Literal["primary", "counterbalance", "protection", "maintenance", "safety"]
    subject_evidence_id: str = Field(min_length=1)
    measure: Measure
    desired_direction: Literal["decrease", "non_increase"]
    before_value: NonNegativeFloat
    after_value: NonNegativeFloat
    delta: float
    observed_direction: Literal["decrease", "unchanged", "increase"]
    compiled_verdict: Literal["met", "not_met"]
    target_recovery_criterion_met: bool | None
    citation_id: str = Field(pattern=ID_PATTERN)

    @model_validator(mode="after")
    def validate_outcome(self) -> "OutcomeEvaluation":
        if abs((self.after_value - self.before_value) - self.delta) > 1e-12:
            raise ValueError("outcome delta is inconsistent")
        direction = (
            "decrease"
            if self.delta < 0
            else "increase"
            if self.delta > 0
            else "unchanged"
        )
        if self.observed_direction != direction:
            raise ValueError("outcome direction is inconsistent")
        expected = (
            "met"
            if self.desired_direction == "decrease" and self.delta < 0
            else "met"
            if self.desired_direction == "non_increase" and self.delta <= 0
            else "not_met"
        )
        if self.measure in {"cluster_error_count", "cluster_rate"}:
            if self.target_recovery_criterion_met is None:
                raise ValueError("cluster outcome requires paired target recovery")
            if not self.target_recovery_criterion_met:
                expected = "not_met"
        elif self.target_recovery_criterion_met is not None:
            raise ValueError("non-cluster outcome has no target-position criterion")
        if self.compiled_verdict != expected:
            raise ValueError("outcome verdict is inconsistent")
        return self


class PatientResponse(StrictModel):
    counts: PatientDirectionCounts
    pattern: Literal[
        "all_improved",
        "improvement_with_ties",
        "bidirectional",
        "worsening_with_ties",
        "all_worsened",
        "all_tied",
    ]
    citation_id: Literal["result.patient_directions"]

    @model_validator(mode="after")
    def validate_pattern(self) -> "PatientResponse":
        counts = self.counts
        if counts.improved and counts.worsened:
            expected = "bidirectional"
        elif counts.improved:
            expected = "all_improved" if not counts.tied else "improvement_with_ties"
        elif counts.worsened:
            expected = "all_worsened" if not counts.tied else "worsening_with_ties"
        else:
            expected = "all_tied"
        if self.pattern != expected:
            raise ValueError("patient response pattern is inconsistent")
        return self


class RealSpeechResponse(StrictModel):
    before_metrics: MetricSet
    after_metrics: MetricSet
    paired_delta: MetricDelta
    metric_changes: list[MetricChange]
    transitions: TransitionSet
    patient_response: PatientResponse
    monitored_cluster_changes: list[ClusterChange]
    capability_changes: list[CapabilityChange] = Field(default_factory=list)
    expected_outcome_results: list[OutcomeEvaluation]
    off_target_changes: list[ClusterChange]
    phonology_coverage_before: PhonologyCoverageSet
    phonology_coverage_after: PhonologyCoverageSet

    @model_validator(mode="after")
    def validate_response(self) -> "RealSpeechResponse":
        measures = [item.measure for item in self.metric_changes]
        expected_measures = [
            "overall_pooled_cer",
            "easy_pooled_cer",
            "medium_pooled_cer",
            "hard_pooled_cer",
            "patient_macro_cer",
        ]
        if measures != expected_measures:
            raise ValueError("metric changes must be complete and canonically ordered")
        if len({item.outcome_id for item in self.expected_outcome_results}) != len(
            self.expected_outcome_results
        ):
            raise ValueError("expected outcome results must be unique")
        monitored = [
            item.descriptor.evidence_id for item in self.monitored_cluster_changes
        ]
        off_target = [item.descriptor.evidence_id for item in self.off_target_changes]
        if len(set(monitored)) != len(monitored) or len(set(off_target)) != len(
            off_target
        ):
            raise ValueError("cluster result IDs must be unique")
        if set(monitored).intersection(off_target):
            raise ValueError("monitored and off-target clusters must be disjoint")
        capability_ids = [item.evidence_id for item in self.capability_changes]
        if len(set(capability_ids)) != len(capability_ids):
            raise ValueError("capability results must be unique")
        if (
            self.before_metrics.overall.sample_count
            != self.after_metrics.overall.sample_count
        ):
            raise ValueError("before and after sample counts differ")
        if (
            self.transitions.overall.sample_count
            != self.after_metrics.overall.sample_count
        ):
            raise ValueError("transition sample count differs from metrics")
        if (
            self.patient_response.counts.patient_count
            != self.after_metrics.overall.speaker_count
        ):
            raise ValueError("patient response count differs from metrics")
        if (
            self.phonology_coverage_before.overall.sample_count
            != self.before_metrics.overall.sample_count
            or self.phonology_coverage_after.overall.sample_count
            != self.after_metrics.overall.sample_count
        ):
            raise ValueError("phonology coverage sample count differs from metrics")
        expected_deltas = {
            "overall_pooled_cer": (
                self.after_metrics.overall.pooled_cer
                - self.before_metrics.overall.pooled_cer
            ),
            "easy_pooled_cer": (
                self.after_metrics.easy.pooled_cer - self.before_metrics.easy.pooled_cer
            ),
            "medium_pooled_cer": (
                self.after_metrics.medium.pooled_cer
                - self.before_metrics.medium.pooled_cer
            ),
            "hard_pooled_cer": (
                self.after_metrics.hard.pooled_cer - self.before_metrics.hard.pooled_cer
            ),
            "patient_macro_cer": (
                self.after_metrics.overall.patient_macro_cer
                - self.before_metrics.overall.patient_macro_cer
            ),
        }
        for field_name, expected in expected_deltas.items():
            if abs(getattr(self.paired_delta, field_name) - expected) > 1e-12:
                raise ValueError(f"paired metric delta is inconsistent: {field_name}")
        return self


class RoundResultV2(StrictModel):
    schema_version: Literal["3"]
    run_id: str = Field(pattern=ID_PATTERN)
    round_index: PositiveInt
    bindings: RoundBindings
    intervention: InterventionRealization
    synthesis: SynthesisReceipt
    training: TrainingReceipt
    response: RealSpeechResponse

    @model_validator(mode="after")
    def validate_round(self) -> "RoundResultV2":
        if self.bindings.synthesis_receipt_canonical_sha256 != canonical_sha256(
            self.synthesis
        ):
            raise ValueError("synthesis receipt hash binding is inconsistent")
        if (self.synthesis.run_id, self.synthesis.round_index) != (
            self.run_id,
            self.round_index,
        ):
            raise ValueError("synthesis receipt identity differs from round result")
        if not (
            self.bindings.artifact_mode
            == self.synthesis.artifact_mode
            == self.training.artifact_mode
        ):
            raise ValueError("round artifact modes differ")
        if (
            self.training.run_id != self.run_id
            or self.training.round_index != self.round_index
        ):
            raise ValueError("training receipt identity differs from round result")
        if self.bindings.training_receipt_canonical_sha256 != canonical_sha256(
            self.training
        ):
            raise ValueError("training receipt hash binding is inconsistent")
        if self.bindings.parent_checkpoint_directory_sha256 != (
            self.training.parent_checkpoint_directory_sha256
        ) or self.bindings.new_checkpoint_directory_sha256 != (
            self.training.new_checkpoint_directory_sha256
        ):
            raise ValueError("round checkpoint lineage differs from training receipt")
        if self.training.preference_delivery is None and self.intervention.overall.generated != self.training.synthetic_rows:
            raise ValueError("realized synthesis differs from synthetic training rows")
        return self


class ReflectionProvenance(StrictModel):
    artifact_mode: ArtifactMode = "recorded_execution"
    protocol_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    proposal_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    next_action_contract_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluation_plan_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    round_result_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    prescription_prompt_file_sha256: str = Field(pattern=SHA256_PATTERN)
    reflection_prompt_file_sha256: str = Field(pattern=SHA256_PATTERN)
    skill_version: str = Field(pattern=r"^v[0-9]{3}$")
    skill_file_sha256: str = Field(pattern=SHA256_PATTERN)
    feedback_split_file_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_checkpoint_directory_sha256: str = Field(pattern=SHA256_PATTERN)
    new_checkpoint_directory_sha256: str = Field(pattern=SHA256_PATTERN)
    before_predictions_file_sha256: str = Field(pattern=SHA256_PATTERN)
    after_predictions_file_sha256: str = Field(pattern=SHA256_PATTERN)
    sealed_test_used: Literal[False]


class CompiledConfound(StrictModel):
    confound_id: str = Field(pattern=ID_PATTERN)
    description: str = Field(min_length=1)
    citation_id: str = Field(pattern=ID_PATTERN)


class ReflectionPack(StrictModel):
    schema_version: Literal["3"]
    run_id: str = Field(pattern=ID_PATTERN)
    round_index: PositiveInt
    provenance: ReflectionProvenance
    evaluation_plan: EvaluationPlan
    prescribed_proposal: Proposal
    next_action_contract: ActionContract
    materialization_sidecar: MaterializationSidecar
    round_result: RoundResultV2
    compiled_expectation_verdict: Literal["supported", "mixed", "refuted"]
    history: list[InterventionSummary]
    activated_confounds: list[CompiledConfound]
    valid_citation_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_citations(self) -> "ReflectionPack":
        if len(set(self.valid_citation_ids)) != len(self.valid_citation_ids):
            raise ValueError("valid citation IDs must be unique")
        return self

    @property
    def realization(self) -> InterventionRealization:
        return self.round_result.intervention

    @property
    def training(self) -> TrainingReceipt:
        return self.round_result.training

    @property
    def response(self) -> RealSpeechResponse:
        return self.round_result.response


class ExpectationCheck(StrictModel):
    outcome_id: str = Field(pattern=ID_PATTERN)
    verdict: Literal["met", "not_met"]
    interpretation: str = Field(min_length=1)
    result_citations: list[str] = Field(min_length=1)


class TargetFinding(StrictModel):
    outcome_id: str = Field(pattern=ID_PATTERN)
    evidence_id: str = Field(pattern=ID_PATTERN)
    response: ResponseDirection
    interpretation: str = Field(min_length=1)
    result_citations: list[str] = Field(min_length=1)


class SafetyFinding(StrictModel):
    measure: MetricMeasure
    response: Literal["improved", "unchanged", "worsened"]
    interpretation: str = Field(min_length=1)
    result_citations: list[str] = Field(min_length=1)


class CapabilityFinding(StrictModel):
    evidence_id: str = Field(pattern=ID_PATTERN)
    response: Literal["retained", "regressed", "inconclusive"]
    interpretation: str = Field(min_length=1)
    result_citations: list[str] = Field(min_length=1)


class PatientHeterogeneity(StrictModel):
    pattern: Literal[
        "all_improved",
        "improvement_with_ties",
        "bidirectional",
        "worsening_with_ties",
        "all_worsened",
        "all_tied",
    ]
    interpretation: str = Field(min_length=1)
    result_citations: list[str] = Field(min_length=1)


class ReflectionV2(StrictModel):
    schema_version: Literal["4"]
    run_id: str = Field(pattern=ID_PATTERN)
    round_index: PositiveInt
    reflection_pack_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    compiled_expectation_verdict: Literal["supported", "mixed", "refuted"]
    hypothesis_verdict: Literal["supported", "mixed", "refuted", "inconclusive"]
    hypothesis_rationale: str = Field(min_length=1)
    hypothesis_citations: list[str] = Field(min_length=1)
    expectation_checks: list[ExpectationCheck] = Field(min_length=1)
    target_findings: list[TargetFinding] = Field(min_length=1)
    safety_findings: list[SafetyFinding] = Field(min_length=1)
    capability_findings: list[CapabilityFinding]
    patient_heterogeneity: PatientHeterogeneity
    competing_explanations: list[CompetingExplanation]
    lesson_candidates: list[LessonCandidate]
    next_round_guidance: NextRoundGuidance

    @model_validator(mode="after")
    def validate_unique_items(self) -> "ReflectionV2":
        for values, label in (
            ([item.outcome_id for item in self.expectation_checks], "expectations"),
            ([item.outcome_id for item in self.target_findings], "target findings"),
            ([item.measure for item in self.safety_findings], "safety findings"),
            ([item.explanation_id for item in self.competing_explanations], "explanations"),
            ([item.evidence_id for item in self.capability_findings], "capability findings"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate reflection {label}")
        return self


__all__ = [
    "ClusterDescriptor",
    "EvaluationPlan",
    "MaterializationSidecar",
    "ReflectionPack",
    "ReflectionV2",
    "RoundResultV2",
    "SynthesisReceipt",
    "TrainingReceipt",
]
