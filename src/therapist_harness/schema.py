from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from .capabilities import CapabilityAnchor
from .memory_schema import EpisodeMemory
from .preference_contract import PreferenceLoopConfig
from .text import tts_phonology_text


SHA256_PATTERN = r"^[0-9a-f]{64}$"
ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class ArtifactRef(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    kind: Literal["file", "directory"] = "file"
    expected_rows: NonNegativeInt | None = None

    @model_validator(mode="after")
    def require_absolute_path(self) -> "ArtifactRef":
        if not self.path.startswith("/"):
            raise ValueError("artifact path must be absolute")
        if self.kind == "directory" and self.expected_rows is not None:
            raise ValueError("directory artifacts cannot declare expected_rows")
        return self


class AgentConfig(StrictModel):
    provider: Literal["OpenAI"]
    base_url: HttpUrl
    model: str = Field(min_length=1)
    reasoning_effort: Literal["low", "medium", "high", "xhigh"]
    store: Literal[False]
    max_retries: int = Field(ge=0, le=8)
    timeout_seconds: PositiveInt
    max_output_tokens: PositiveInt
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")


class LoRAConfig(StrictModel):
    rank: PositiveInt
    alpha: PositiveInt
    dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: list[str] = Field(min_length=1)
    learning_rate: float = Field(gt=0.0)
    per_device_batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt
    world_size: Literal[1]
    optimizer_steps_per_round: PositiveInt
    training_passes_per_round: PositiveInt
    save_at_round_end: Literal[True]
    scheduler: Literal["linear", "constant"]
    initialization: Literal["parent_adapter_weights_only"] = (
        "parent_adapter_weights_only"
    )
    reset_optimizer_each_round: Literal[True] = True
    reset_scheduler_each_round: Literal[True]
    warmup_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    seed: NonNegativeInt

    @model_validator(mode="after")
    def unique_targets(self) -> "LoRAConfig":
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("LoRA target_modules must be unique")
        return self


class EvaluationConfig(StrictModel):
    language: Literal["Cantonese"]
    batch_size: PositiveInt
    critical_cer_threshold: float = Field(gt=0.0)
    metrics: list[
        Literal[
            "mean_utterance_cer",
            "pooled_cer",
            "patient_macro_cer",
            "critical_count",
            "exact_count",
        ],
    ] = Field(min_length=1)


class SpeakerProfileRule(StrictModel):
    difficulty_groups: list[Literal["easy", "medium", "hard"]] = Field(min_length=1)
    # Omit the historical default from serialized protocols: old artifact hashes
    # must not change when the opt-in crossed design is introduced.
    allocation: Literal["role_balanced", "crossed_16"] = Field(
        default="role_balanced", exclude_if=lambda value: value == "role_balanced"
    )

    @model_validator(mode="after")
    def unique_groups(self) -> "SpeakerProfileRule":
        if len(set(self.difficulty_groups)) != len(self.difficulty_groups):
            raise ValueError("speaker-profile difficulty groups must be unique")
        return self


class ProposalConstraints(StrictModel):
    """Frozen Agent decision bounds copied into every EvidencePack."""

    fixed_speaker_profile_id: str = Field(pattern=ID_PATTERN)
    max_primary_targets: PositiveInt
    max_copies_per_stimulus: PositiveInt
    max_surface_characters_per_stimulus: PositiveInt
    threshold_edge_max_linked_copies: PositiveInt
    threshold_edge_max_unique_stimuli: PositiveInt
    threshold_edge_max_carrier_exposures: PositiveInt
    non_targetable_cluster_stimuli_allowed: Literal[False]
    non_tts_actionable_cluster_stimuli_allowed: Literal[False]


class RunProtocol(StrictModel):
    schema_version: Literal["1"]
    protocol_status: Literal["draft", "frozen"]
    run_id: str = Field(pattern=ID_PATTERN)
    setting_id: Literal["setting_d"]
    round_count: PositiveInt
    base_model_dir: str = Field(min_length=1)
    base_model_fingerprint: ArtifactRef
    initial_adapter: ArtifactRef
    initial_adapter_dev_predictions: ArtifactRef
    initial_adapter_prediction_column: str = Field(min_length=1)
    zero_shot_dev_predictions: ArtifactRef
    zero_shot_prediction_column: str = Field(min_length=1)
    real_train: ArtifactRef
    feedback_split: ArtifactRef
    sealed_test_split: ArtifactRef | None
    speech_generator: ArtifactRef
    frozen_reference_manifest: ArtifactRef
    agent: AgentConfig
    synthetic_count_per_round: PositiveInt
    unique_stimulus_count_per_round: PositiveInt
    real_replay_count_per_round: NonNegativeInt
    speaker_profiles: dict[str, SpeakerProfileRule] = Field(min_length=1)
    proposal_constraints: ProposalConstraints
    lora: LoRAConfig
    evaluation: EvaluationConfig
    skill_update_rounds: list[PositiveInt] = Field(default_factory=list)
    preference_loop: PreferenceLoopConfig | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_protocol(self) -> "RunProtocol":
        if not self.base_model_dir.startswith("/"):
            raise ValueError("base_model_dir must be absolute")
        invalid_profiles = [
            profile_id
            for profile_id in self.speaker_profiles
            if re.fullmatch(ID_PATTERN, profile_id) is None
        ]
        if invalid_profiles:
            raise ValueError(f"invalid speaker profile IDs: {invalid_profiles}")
        constraints = self.proposal_constraints
        if constraints.fixed_speaker_profile_id not in self.speaker_profiles:
            raise ValueError("fixed speaker profile is absent from speaker_profiles")
        fixed_groups = set(
            self.speaker_profiles[
                constraints.fixed_speaker_profile_id
            ].difficulty_groups
        )
        if fixed_groups != {"easy", "medium", "hard"}:
            raise ValueError("Agent v1 fixed speaker profile must cover all patients")
        if self.unique_stimulus_count_per_round > self.synthetic_count_per_round:
            raise ValueError("unique stimulus count cannot exceed synthetic count")
        if constraints.max_primary_targets > self.unique_stimulus_count_per_round:
            raise ValueError("max primary targets exceeds unique stimulus budget")
        if constraints.max_primary_targets != 1:
            raise ValueError("Therapist Agent v1 requires exactly one primary target")
        if constraints.max_copies_per_stimulus > self.synthetic_count_per_round:
            raise ValueError("per-stimulus cap exceeds synthetic budget")
        if (
            constraints.threshold_edge_max_linked_copies
            > self.synthetic_count_per_round
        ):
            raise ValueError("threshold-edge copy cap exceeds synthetic budget")
        if (
            constraints.threshold_edge_max_unique_stimuli
            > self.unique_stimulus_count_per_round
        ):
            raise ValueError("threshold-edge stimulus cap exceeds unique budget")
        if (
            constraints.threshold_edge_max_carrier_exposures
            < constraints.threshold_edge_max_linked_copies
        ):
            raise ValueError(
                "threshold-edge carrier cap cannot be below the linked-copy cap"
            )
        if len(set(self.skill_update_rounds)) != len(self.skill_update_rounds):
            raise ValueError("skill_update_rounds must be unique")
        if any(value > self.round_count for value in self.skill_update_rounds):
            raise ValueError("skill update round exceeds round_count")
        samples_per_step = (
            self.lora.per_device_batch_size
            * self.lora.gradient_accumulation_steps
            * self.lora.world_size
        )
        expected_steps = math.ceil(
            (self.real_replay_count_per_round + self.synthetic_count_per_round)
            / samples_per_step
        ) * self.lora.training_passes_per_round
        if self.preference_loop is None and self.lora.optimizer_steps_per_round != expected_steps:
            raise ValueError(
                "optimizer_steps_per_round must cover the declared passes over the "
                f"frozen real+synthetic manifest: expected {expected_steps}"
            )
        return self

    @property
    def speaker_profile_ids(self) -> list[str]:
        return list(self.speaker_profiles)


class MetricGroup(StrictModel):
    sample_count: NonNegativeInt
    speaker_count: NonNegativeInt
    total_edits: NonNegativeInt
    total_reference_characters: NonNegativeInt
    mean_utterance_cer: NonNegativeFloat
    pooled_cer: NonNegativeFloat
    patient_macro_cer: NonNegativeFloat
    critical_count: NonNegativeInt
    exact_count: NonNegativeInt

    @model_validator(mode="after")
    def validate_counts(self) -> "MetricGroup":
        if self.critical_count > self.sample_count:
            raise ValueError("critical_count exceeds sample_count")
        if self.exact_count > self.sample_count:
            raise ValueError("exact_count exceeds sample_count")
        if self.speaker_count > self.sample_count:
            raise ValueError("speaker_count exceeds sample_count")
        if self.sample_count and not self.speaker_count:
            raise ValueError("non-empty metric group must contain a speaker")
        if self.sample_count and not self.total_reference_characters:
            raise ValueError("non-empty metric group must contain reference characters")
        if self.total_reference_characters:
            expected_pooled = self.total_edits / self.total_reference_characters
            if abs(expected_pooled - self.pooled_cer) > 1e-12:
                raise ValueError("pooled_cer does not equal edits/reference characters")
        if self.sample_count == 0 and any(
            value != 0
            for value in (
                self.speaker_count,
                self.total_edits,
                self.total_reference_characters,
                self.mean_utterance_cer,
                self.pooled_cer,
                self.patient_macro_cer,
                self.critical_count,
                self.exact_count,
            )
        ):
            raise ValueError("empty metric group must contain only zeros")
        return self


class MetricSet(StrictModel):
    overall: MetricGroup
    easy: MetricGroup
    medium: MetricGroup
    hard: MetricGroup

    @model_validator(mode="after")
    def validate_partition(self) -> "MetricSet":
        if self.overall.sample_count != sum(
            group.sample_count for group in (self.easy, self.medium, self.hard)
        ):
            raise ValueError("difficulty sample counts do not sum to overall")
        return self


class MetricDelta(StrictModel):
    overall_pooled_cer: float
    easy_pooled_cer: float
    medium_pooled_cer: float
    hard_pooled_cer: float
    patient_macro_cer: float


class TransitionCounts(StrictModel):
    recovered: NonNegativeInt
    newly_wrong: NonNegativeInt
    preserved_correct: NonNegativeInt
    persistent_wrong: NonNegativeInt


class DirectionCounts(StrictModel):
    improved: NonNegativeInt
    worsened: NonNegativeInt
    tied: NonNegativeInt


class CriticalTransitionCounts(StrictModel):
    recovered: NonNegativeInt
    newly_critical: NonNegativeInt
    preserved_noncritical: NonNegativeInt
    persistent_critical: NonNegativeInt


class TransitionGroup(StrictModel):
    sample_count: NonNegativeInt
    exact: TransitionCounts
    utterance_cer: DirectionCounts
    critical: CriticalTransitionCounts

    @model_validator(mode="after")
    def validate_totals(self) -> "TransitionGroup":
        exact_total = sum(
            (
                self.exact.recovered,
                self.exact.newly_wrong,
                self.exact.preserved_correct,
                self.exact.persistent_wrong,
            )
        )
        direction_total = sum(
            (
                self.utterance_cer.improved,
                self.utterance_cer.worsened,
                self.utterance_cer.tied,
            )
        )
        critical_total = sum(
            (
                self.critical.recovered,
                self.critical.newly_critical,
                self.critical.preserved_noncritical,
                self.critical.persistent_critical,
            )
        )
        if any(
            total != self.sample_count
            for total in (exact_total, direction_total, critical_total)
        ):
            raise ValueError("transition counts must each sum to sample_count")
        return self


class TransitionSet(StrictModel):
    overall: TransitionGroup
    easy: TransitionGroup
    medium: TransitionGroup
    hard: TransitionGroup

    @model_validator(mode="after")
    def validate_partition(self) -> "TransitionSet":
        if self.overall.sample_count != sum(
            group.sample_count for group in (self.easy, self.medium, self.hard)
        ):
            raise ValueError("difficulty transition counts do not sum to overall")
        return self


class PatientDirectionCounts(StrictModel):
    patient_count: NonNegativeInt
    improved: NonNegativeInt
    worsened: NonNegativeInt
    tied: NonNegativeInt

    @model_validator(mode="after")
    def validate_total(self) -> "PatientDirectionCounts":
        if self.improved + self.worsened + self.tied != self.patient_count:
            raise ValueError("patient direction counts must sum to patient_count")
        return self


class SnapshotEvidence(StrictModel):
    model_role: Literal[
        "current_pre_intervention",
        "real_only_closed_loop_start",
        "zero_shot_diagnostic_only",
        "previous_pre_intervention",
    ]
    closed_loop_model_index: NonNegativeInt | None
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    predictions_sha256: str = Field(pattern=SHA256_PATTERN)
    metrics: MetricSet


class SnapshotSet(StrictModel):
    current: SnapshotEvidence
    m0: SnapshotEvidence
    zero_shot: SnapshotEvidence
    previous: SnapshotEvidence | None

    @model_validator(mode="after")
    def validate_roles(self) -> "SnapshotSet":
        expected = {
            "current": "current_pre_intervention",
            "m0": "real_only_closed_loop_start",
            "zero_shot": "zero_shot_diagnostic_only",
        }
        for name, role in expected.items():
            if getattr(self, name).model_role != role:
                raise ValueError(
                    f"snapshot {name} has role {getattr(self, name).model_role}"
                )
        if self.m0.closed_loop_model_index != 0:
            raise ValueError("M0 snapshot must use closed-loop model index 0")
        if self.zero_shot.closed_loop_model_index is not None:
            raise ValueError("zero-shot diagnostic is outside the closed-loop lineage")
        if (
            self.previous is not None
            and self.previous.model_role != "previous_pre_intervention"
        ):
            raise ValueError("previous snapshot has the wrong role")
        return self


class SnapshotComparison(StrictModel):
    evaluated_snapshot: Literal["current", "m0"]
    reference_snapshot: Literal["zero_shot", "m0", "previous"]
    metric_delta: MetricDelta
    transitions: TransitionSet
    patient_directions: PatientDirectionCounts


class ComparisonSet(StrictModel):
    m0_vs_zero_shot: SnapshotComparison
    current_vs_zero_shot: SnapshotComparison
    current_vs_m0: SnapshotComparison
    current_vs_previous: SnapshotComparison | None

    @model_validator(mode="after")
    def validate_labels(self) -> "ComparisonSet":
        expected = {
            "m0_vs_zero_shot": ("m0", "zero_shot"),
            "current_vs_zero_shot": ("current", "zero_shot"),
            "current_vs_m0": ("current", "m0"),
        }
        for name, labels in expected.items():
            comparison = getattr(self, name)
            if (comparison.evaluated_snapshot, comparison.reference_snapshot) != labels:
                raise ValueError(f"comparison {name} has inconsistent labels")
        if self.current_vs_previous is not None and (
            self.current_vs_previous.evaluated_snapshot,
            self.current_vs_previous.reference_snapshot,
        ) != ("current", "previous"):
            raise ValueError("current_vs_previous has inconsistent labels")
        return self


class DifficultyErrorRate(StrictModel):
    error_count: NonNegativeInt
    opportunity_count: NonNegativeInt
    rate: NonNegativeFloat

    @model_validator(mode="after")
    def validate_rate(self) -> "DifficultyErrorRate":
        if self.opportunity_count == 0 and self.error_count:
            raise ValueError("zero opportunities cannot contain errors")
        expected = (
            self.error_count / self.opportunity_count if self.opportunity_count else 0.0
        )
        if abs(expected - self.rate) > 1e-12:
            raise ValueError("difficulty error rate has an inconsistent denominator")
        return self


class ErrorObservation(StrictModel):
    error_count: NonNegativeInt
    opportunity_count: PositiveInt
    rate: NonNegativeFloat
    patient_count: NonNegativeInt
    content_count: NonNegativeInt
    by_difficulty: dict[Literal["easy", "medium", "hard"], DifficultyErrorRate]
    alignment_uncertainty: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_observation(self) -> "ErrorObservation":
        if abs(self.error_count / self.opportunity_count - self.rate) > 1e-12:
            raise ValueError("error rate has an inconsistent denominator")
        if set(self.by_difficulty) != {"easy", "medium", "hard"}:
            raise ValueError("error observation must contain all difficulty groups")
        if (
            sum(item.error_count for item in self.by_difficulty.values())
            != self.error_count
        ):
            raise ValueError("difficulty error counts do not sum to the observation")
        if (
            sum(item.opportunity_count for item in self.by_difficulty.values())
            != self.opportunity_count
        ):
            raise ValueError("difficulty opportunities do not sum to the observation")
        if self.error_count == 0 and any(
            (self.patient_count, self.content_count, self.alignment_uncertainty)
        ):
            raise ValueError("an empty error observation cannot have coverage")
        return self


class ClusterObservations(StrictModel):
    current: ErrorObservation
    m0: ErrorObservation
    zero_shot: ErrorObservation
    previous: ErrorObservation | None


class ErrorCluster(StrictModel):
    evidence_id: str = Field(pattern=ID_PATTERN)
    operation: Literal["deletion", "substitution", "insertion"]
    level: Literal["character", "onset", "nucleus", "coda", "tone"]
    reference_unit: str
    hypothesis_unit: str
    # None only permits reading legacy packs; synthesis requires attested surface
    # units. CER-equivalent characters need not share a pronunciation.
    reference_surface_units: list[str] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    hypothesis_surface_units: list[str] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    left_context: str | None
    right_context: str | None
    rate_semantics: Literal[
        "deterministic_character_alignment_rate",
        "confirmed_substitution_lower_bound_rate",
    ]
    targetable: bool
    selection_reasons: list[str] = Field(min_length=1)
    tts_actionable: bool
    tts_actionability_reasons: list[str] = Field(min_length=1)
    observations: ClusterObservations

    @model_validator(mode="after")
    def validate_surface_units(self) -> "ErrorCluster":
        for units in (self.reference_surface_units, self.hypothesis_surface_units):
            if units is not None and (
                any(len(unit) != 1 for unit in units) or len(set(units)) != len(units)
            ):
                raise ValueError(
                    "attested surface units must be unique single characters"
                )
        return self


class PhonologyCoverageGroup(StrictModel):
    sample_count: NonNegativeInt
    reference_characters: NonNegativeInt
    hypothesis_characters: NonNegativeInt
    reference_g2p_resolved_characters: NonNegativeInt
    hypothesis_g2p_resolved_characters: NonNegativeInt
    multi_optimal_alignment_utterances: NonNegativeInt
    character_substitutions: NonNegativeInt
    character_deletions: NonNegativeInt
    character_insertions: NonNegativeInt
    forced_substitutions: NonNegativeInt
    g2p_resolved_forced_substitutions: NonNegativeInt
    g2p_unresolved_forced_substitutions: NonNegativeInt
    context_sensitive_forced_substitutions: NonNegativeInt
    context_uncheckable_forced_substitutions: NonNegativeInt
    phonologically_equivalent_forced_substitutions: NonNegativeInt
    phonologically_different_forced_substitutions: NonNegativeInt
    normalization_projection_unresolved_forced_substitutions: NonNegativeInt | None = (
        Field(default=None, exclude_if=lambda value: value is None)
    )
    not_cer_forced_phonologically_different_substitutions: NonNegativeInt | None = (
        Field(default=None, exclude_if=lambda value: value is None)
    )
    phone_attributed_forced_substitutions: NonNegativeInt

    @model_validator(mode="after")
    def validate_coverage(self) -> "PhonologyCoverageGroup":
        if self.reference_g2p_resolved_characters > self.reference_characters:
            raise ValueError(
                "resolved reference characters exceed reference characters"
            )
        if self.hypothesis_g2p_resolved_characters > self.hypothesis_characters:
            raise ValueError(
                "resolved hypothesis characters exceed hypothesis characters"
            )
        if self.multi_optimal_alignment_utterances > self.sample_count:
            raise ValueError("multi-optimal utterances exceed sample count")
        if self.forced_substitutions > self.character_substitutions:
            raise ValueError("forced substitutions exceed substitutions")
        if self.g2p_resolved_forced_substitutions > self.forced_substitutions:
            raise ValueError(
                "resolved forced substitutions exceed forced substitutions"
            )
        if (
            self.g2p_resolved_forced_substitutions
            + self.g2p_unresolved_forced_substitutions
            != self.forced_substitutions
        ):
            raise ValueError(
                "resolved and unresolved substitutions must partition forced"
            )
        if (
            self.context_sensitive_forced_substitutions
            + self.context_uncheckable_forced_substitutions
            > self.g2p_resolved_forced_substitutions
        ):
            raise ValueError("context exclusions exceed resolved substitutions")
        if (
            self.phonologically_equivalent_forced_substitutions
            + self.phonologically_different_forced_substitutions
            != self.g2p_resolved_forced_substitutions
        ):
            raise ValueError(
                "equivalent and different substitutions must partition resolved"
            )
        for field_name in (
            "context_sensitive_forced_substitutions",
            "context_uncheckable_forced_substitutions",
            "phonologically_equivalent_forced_substitutions",
            "phonologically_different_forced_substitutions",
            "phone_attributed_forced_substitutions",
        ):
            if getattr(self, field_name) > self.g2p_resolved_forced_substitutions:
                raise ValueError(
                    f"phonology coverage field {field_name} exceeds resolved substitutions"
                )
        if (
            self.phone_attributed_forced_substitutions
            > self.phonologically_different_forced_substitutions
        ):
            raise ValueError(
                "phone-attributed substitutions exceed phonological differences"
            )
        if (self.normalization_projection_unresolved_forced_substitutions or 0) + (
            self.not_cer_forced_phonologically_different_substitutions or 0
        ) > self.phonologically_different_forced_substitutions:
            raise ValueError(
                "CER-unsupported phone exclusions exceed phonological differences"
            )
        return self


class PhonologyCoverageSet(StrictModel):
    overall: PhonologyCoverageGroup
    easy: PhonologyCoverageGroup
    medium: PhonologyCoverageGroup
    hard: PhonologyCoverageGroup

    @model_validator(mode="after")
    def validate_partition(self) -> "PhonologyCoverageSet":
        fields = tuple(PhonologyCoverageGroup.model_fields)
        for field in fields:
            overall = getattr(self.overall, field)
            parts = [
                getattr(group, field) for group in (self.easy, self.medium, self.hard)
            ]
            if overall is None and all(value is None for value in parts):
                continue
            if overall is None or any(value is None for value in parts):
                raise ValueError(
                    f"phonology coverage field {field} is only partially reported"
                )
            if overall != sum(parts):
                raise ValueError(f"phonology coverage field {field} is not partitioned")
        return self


class SnapshotPhonologyCoverageSet(StrictModel):
    """Coverage diagnostics for every prediction snapshot in this round.

    These make lower-bound phone rates interpretable without pretending that a
    change in forced-alignment coverage is itself an intervention effect.
    """

    current: PhonologyCoverageSet
    m0: PhonologyCoverageSet
    zero_shot: PhonologyCoverageSet
    previous: PhonologyCoverageSet | None


class ClusterSelectionPolicy(StrictModel):
    max_clusters_per_level: PositiveInt
    historical_targets_exempt_from_limit: Literal[True]
    candidate_count: NonNegativeInt
    included_count: NonNegativeInt
    minimum_current_error_count: PositiveInt
    minimum_patient_count: PositiveInt
    minimum_content_count: PositiveInt
    maximum_alignment_uncertainty: float = Field(ge=0.0, le=1.0)
    character_insertions_targetable: Literal[False]
    ordering: Literal[
        "all_historical_targets_then_targetable_new_clusters_then_current_error_count_then_patient_content_support_then_anchor_burden"
    ]


class EvidenceDataScope(StrictModel):
    setting_id: Literal["setting_d"]
    feedback_split_sha256: str = Field(pattern=SHA256_PATTERN)
    sample_count: PositiveInt
    speaker_count: PositiveInt
    content_count: PositiveInt
    sealed_test_used: Literal[False]
    critical_cer_threshold: float = Field(gt=0.0)
    normalizer_id: Literal["therapist_text_normalize_v1"]
    character_alignment_id: Literal[
        "levenshtein_stable_traceback_for_character_edits_and_all_optimal_paths_for_forced_substitutions_v1"
    ]
    phonology_id: Literal["pycantonese_5.0.0_surface_to_cer_forced_v2"]
    tts_actionability_policy: Literal[
        "phone_components_or_pronounceable_character_deletions_or_nonhomophonic_character_substitutions_v1"
    ]
    difficulty_source: Literal["frozen_zero_shot_character_cer"]
    difficulty_policy: Literal["easy:CER<0.2;medium:0.2<=CER<0.5;hard:CER>=0.5"]
    delta_convention: Literal[
        "evaluated_minus_reference; negative CER delta means improvement"
    ]
    phonology_coverage: SnapshotPhonologyCoverageSet
    cluster_selection: ClusterSelectionPolicy


class EvidenceTargetDescriptor(StrictModel):
    evidence_id: str = Field(pattern=ID_PATTERN)
    operation: Literal["deletion", "substitution", "insertion"]
    level: Literal["character", "onset", "nucleus", "coda", "tone"]
    reference_unit: str
    hypothesis_unit: str
    left_context: str | None
    right_context: str | None


class InterventionSummary(StrictModel):
    ledger_id: str = Field(pattern=ID_PATTERN)
    ledger_sha256: str = Field(pattern=SHA256_PATTERN)
    round_index: PositiveInt
    parent_checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    new_checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    dev_predictions_sha256: str = Field(pattern=SHA256_PATTERN)
    accepted_checkpoint_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN, exclude_if=lambda value: value is None)
    accepted_predictions_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN, exclude_if=lambda value: value is None)
    target_evidence_ids: list[str] = Field(min_length=1)
    target_descriptors: list[EvidenceTargetDescriptor] = Field(min_length=1)
    target_samples: NonNegativeInt
    contrast_samples: NonNegativeInt
    protection_samples: NonNegativeInt
    maintenance_samples: NonNegativeInt = 0
    speaker_profile_samples: dict[str, NonNegativeInt]
    delta_vs_previous: MetricDelta
    patient_directions: PatientDirectionCounts
    hypothesis_verdict: Literal["supported", "mixed", "refuted", "inconclusive"]
    target_effect: Literal["improved", "unchanged", "worsened", "mixed", "inconclusive"]
    episode_memory: EpisodeMemory
    easy_medium_safety: Literal[
        "no_regression",
        "easy_regressed",
        "medium_regressed",
        "easy_and_medium_regressed",
        "mixed",
    ]

    @model_validator(mode="after")
    def validate_summary(self) -> "InterventionSummary":
        if (self.accepted_checkpoint_sha256 is None) != (self.accepted_predictions_sha256 is None):
            raise ValueError("accepted checkpoint and predictions must be recorded together")
        if self.accepted_checkpoint_sha256 is not None and self.accepted_checkpoint_sha256 not in {self.parent_checkpoint_sha256,self.new_checkpoint_sha256}:
            raise ValueError("accepted checkpoint must be the parent or the measured candidate")
        if len(set(self.target_evidence_ids)) != len(self.target_evidence_ids):
            raise ValueError("intervention target evidence IDs must be unique")
        if [
            item.evidence_id for item in self.target_descriptors
        ] != self.target_evidence_ids:
            raise ValueError(
                "target descriptors must match target evidence IDs in order"
            )
        invalid_profiles = [
            profile_id
            for profile_id in self.speaker_profile_samples
            if re.fullmatch(ID_PATTERN, profile_id) is None
        ]
        if invalid_profiles:
            raise ValueError(
                f"invalid intervention speaker profiles: {invalid_profiles}"
            )
        return self


class ActionContract(StrictModel):
    synthetic_count: PositiveInt
    unique_stimulus_count: PositiveInt
    real_replay_count: NonNegativeInt
    permitted_speaker_profiles: dict[str, SpeakerProfileRule] = Field(min_length=1)
    proposal_constraints: ProposalConstraints
    permitted_roles: list[Literal["target", "contrast", "protection", "maintenance"]] = Field(
        default_factory=lambda: ["target", "contrast", "protection", "maintenance"]
    )
    permitted_evidence_levels: list[
        Literal["character", "onset", "nucleus", "coda", "tone"]
    ] = Field(default_factory=lambda: ["character", "onset", "nucleus", "coda", "tone"])
    controllable_fields: list[
        Literal["text", "role", "evidence_ids", "speaker_profile_id", "copies"]
    ] = Field(
        default_factory=lambda: [
            "text",
            "role",
            "evidence_ids",
            "speaker_profile_id",
            "copies",
        ]
    )
    content_isolation_levels: list[
        Literal["raw_clean", "text_normalize", "jyutping", "detoned_jyutping"]
    ] = Field(
        default_factory=lambda: [
            "raw_clean",
            "text_normalize",
            "jyutping",
            "detoned_jyutping",
        ]
    )

    @property
    def permitted_speaker_profile_ids(self) -> list[str]:
        return list(self.permitted_speaker_profiles)


class EvidencePack(StrictModel):
    schema_version: Literal["3"]
    artifact_mode: Literal["recorded_execution", "counterfactual_interface_smoke"] = (
        "recorded_execution"
    )
    run_id: str = Field(pattern=ID_PATTERN)
    round_index: PositiveInt
    data_scope: EvidenceDataScope
    snapshots: SnapshotSet
    comparisons: ComparisonSet
    error_clusters: list[ErrorCluster]
    capability_anchors: list[CapabilityAnchor] = Field(default_factory=list)
    intervention_history: list[InterventionSummary]
    action_contract: ActionContract

    @model_validator(mode="after")
    def validate_pack(self) -> "EvidencePack":
        ids = [item.evidence_id for item in self.error_clusters] + [
            item.evidence_id for item in self.capability_anchors
        ]
        if len(set(ids)) != len(ids):
            raise ValueError("error cluster and capability evidence IDs must be unique")
        snapshot_samples = {
            self.snapshots.current.metrics.overall.sample_count,
            self.snapshots.m0.metrics.overall.sample_count,
            self.snapshots.zero_shot.metrics.overall.sample_count,
        }
        if self.snapshots.previous is not None:
            snapshot_samples.add(self.snapshots.previous.metrics.overall.sample_count)
        if snapshot_samples != {self.data_scope.sample_count}:
            raise ValueError("snapshot sample counts differ from the data scope")
        if (
            self.snapshots.current.metrics.overall.speaker_count
            != self.data_scope.speaker_count
        ):
            raise ValueError("current speaker count differs from the data scope")

        snapshot_metrics = {
            "current": self.snapshots.current.metrics,
            "m0": self.snapshots.m0.metrics,
            "zero_shot": self.snapshots.zero_shot.metrics,
            "previous": (
                self.snapshots.previous.metrics
                if self.snapshots.previous is not None
                else None
            ),
        }
        comparisons = (
            self.comparisons.m0_vs_zero_shot,
            self.comparisons.current_vs_zero_shot,
            self.comparisons.current_vs_m0,
            self.comparisons.current_vs_previous,
        )
        delta_fields = {
            "overall_pooled_cer": ("overall", "pooled_cer"),
            "easy_pooled_cer": ("easy", "pooled_cer"),
            "medium_pooled_cer": ("medium", "pooled_cer"),
            "hard_pooled_cer": ("hard", "pooled_cer"),
            "patient_macro_cer": ("overall", "patient_macro_cer"),
        }
        for comparison in (item for item in comparisons if item is not None):
            evaluated = snapshot_metrics[comparison.evaluated_snapshot]
            reference = snapshot_metrics[comparison.reference_snapshot]
            if evaluated is None or reference is None:
                raise ValueError("comparison references a missing snapshot")
            for delta_name, (group_name, metric_name) in delta_fields.items():
                expected_delta = getattr(
                    getattr(evaluated, group_name), metric_name
                ) - getattr(getattr(reference, group_name), metric_name)
                if (
                    abs(getattr(comparison.metric_delta, delta_name) - expected_delta)
                    > 1e-12
                ):
                    raise ValueError(
                        f"comparison {comparison.evaluated_snapshot}_vs_"
                        f"{comparison.reference_snapshot} has an inconsistent {delta_name}"
                    )
            for group_name in ("overall", "easy", "medium", "hard"):
                transition = getattr(comparison.transitions, group_name)
                evaluated_group = getattr(evaluated, group_name)
                reference_group = getattr(reference, group_name)
                if transition.sample_count != evaluated_group.sample_count:
                    raise ValueError(
                        "comparison transition sample count is inconsistent"
                    )
                if (
                    transition.exact.recovered + transition.exact.preserved_correct
                    != evaluated_group.exact_count
                    or transition.exact.newly_wrong + transition.exact.preserved_correct
                    != reference_group.exact_count
                ):
                    raise ValueError("comparison exact transitions are inconsistent")
                if (
                    transition.critical.newly_critical
                    + transition.critical.persistent_critical
                    != evaluated_group.critical_count
                    or transition.critical.recovered
                    + transition.critical.persistent_critical
                    != reference_group.critical_count
                ):
                    raise ValueError("comparison critical transitions are inconsistent")
            if (
                comparison.patient_directions.patient_count
                != evaluated.overall.speaker_count
            ):
                raise ValueError("comparison patient count is inconsistent")

        expected_history_rounds = list(range(1, self.round_index))
        if [
            item.round_index for item in self.intervention_history
        ] != expected_history_rounds:
            raise ValueError(
                "intervention history must contain every prior round in order"
            )
        ledger_ids = [item.ledger_id for item in self.intervention_history]
        if len(set(ledger_ids)) != len(ledger_ids):
            raise ValueError("intervention history ledger IDs must be unique")
        for item in self.intervention_history:
            if item.episode_memory.artifact_mode != self.artifact_mode:
                raise ValueError("intervention history artifact mode differs from evidence")
            role_total = (
                item.target_samples + item.contrast_samples + item.protection_samples
                + item.maintenance_samples
            )
            if role_total != self.action_contract.synthetic_count:
                raise ValueError(
                    "intervention role doses do not equal synthetic budget"
                )
            if (
                sum(item.speaker_profile_samples.values())
                != self.action_contract.synthetic_count
            ):
                raise ValueError(
                    "intervention profile doses do not equal synthetic budget"
                )
            if not set(item.speaker_profile_samples).issubset(
                self.action_contract.permitted_speaker_profiles
            ):
                raise ValueError(
                    "intervention history contains an unknown speaker profile"
                )
        expected_parent = self.snapshots.m0.checkpoint_sha256
        for item in self.intervention_history:
            if item.parent_checkpoint_sha256 != expected_parent:
                raise ValueError("intervention checkpoint history is not contiguous")
            expected_parent = item.accepted_checkpoint_sha256 or item.new_checkpoint_sha256
        if self.round_index == 1:
            if (
                self.snapshots.previous is not None
                or self.comparisons.current_vs_previous is not None
            ):
                raise ValueError("round 1 cannot contain a previous snapshot")
            if (
                self.snapshots.current.checkpoint_sha256
                != self.snapshots.m0.checkpoint_sha256
                or self.snapshots.current.predictions_sha256
                != self.snapshots.m0.predictions_sha256
            ):
                raise ValueError("round 1 current snapshot must be M0")
            if self.snapshots.current.metrics != self.snapshots.m0.metrics:
                raise ValueError("round 1 current metrics must equal M0 metrics")
            if (
                self.data_scope.phonology_coverage.current
                != self.data_scope.phonology_coverage.m0
            ):
                raise ValueError("round 1 current phonology coverage must equal M0")
            for cluster in self.error_clusters:
                if cluster.observations.current != cluster.observations.m0:
                    raise ValueError(
                        "round 1 current cluster observations must equal M0"
                    )
            if self.snapshots.current.closed_loop_model_index != 0:
                raise ValueError("round 1 current snapshot must use model index 0")
            if self.data_scope.phonology_coverage.previous is not None:
                raise ValueError("round 1 cannot contain previous phonology coverage")
        elif (
            self.snapshots.previous is None
            or self.comparisons.current_vs_previous is None
        ):
            raise ValueError("round 2+ requires a previous snapshot and comparison")
        elif self.data_scope.phonology_coverage.previous is None:
            raise ValueError("round 2+ requires previous phonology coverage")
        elif (
            self.snapshots.current.closed_loop_model_index != self.round_index - 1
            or self.snapshots.previous.closed_loop_model_index != self.round_index - 2
        ):
            raise ValueError("current/previous snapshots do not match round lineage")
        if self.intervention_history:
            latest = self.intervention_history[-1]
            comparison = self.comparisons.current_vs_previous
            if comparison is None:
                raise ValueError(
                    "intervention history requires current-vs-previous evidence"
                )
            candidate_accepted = latest.accepted_checkpoint_sha256 in (None, latest.new_checkpoint_sha256)
            if candidate_accepted and latest.delta_vs_previous != comparison.metric_delta:
                raise ValueError(
                    "latest history delta differs from current-vs-previous"
                )
            if candidate_accepted and latest.patient_directions != comparison.patient_directions:
                raise ValueError(
                    "latest history patient directions differ from current-vs-previous"
                )
            if (
                self.snapshots.current.checkpoint_sha256 != (latest.accepted_checkpoint_sha256 or latest.new_checkpoint_sha256)
                or self.snapshots.current.predictions_sha256
                != (latest.accepted_predictions_sha256 or latest.dev_predictions_sha256)
            ):
                raise ValueError(
                    "current snapshot differs from the latest intervention"
                )
            expected_previous_checkpoint = (
                (self.intervention_history[-2].accepted_checkpoint_sha256 or self.intervention_history[-2].new_checkpoint_sha256)
                if len(self.intervention_history) > 1
                else self.snapshots.m0.checkpoint_sha256
            )
            expected_previous_predictions = (
                (self.intervention_history[-2].accepted_predictions_sha256 or self.intervention_history[-2].dev_predictions_sha256)
                if len(self.intervention_history) > 1
                else self.snapshots.m0.predictions_sha256
            )
            if self.snapshots.previous is None or (
                self.snapshots.previous.checkpoint_sha256
                != expected_previous_checkpoint
                or self.snapshots.previous.predictions_sha256
                != expected_previous_predictions
            ):
                raise ValueError("previous snapshot differs from intervention history")
        return self


class AssessmentClaim(StrictModel):
    claim: str = Field(
        min_length=1,
        description="One falsifiable statement about the supplied ASR evidence.",
    )
    evidence_ids: list[str] = Field(
        min_length=1,
        description=(
            "Evidence references that directly support the claim: cluster IDs, "
            "snapshot.* references, comparison.* references, or intervention.* "
            "references supplied by the Harness."
        ),
    )

    @model_validator(mode="after")
    def unique_evidence_ids(self) -> "AssessmentClaim":
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("assessment evidence references must be unique")
        return self


class Stimulus(StrictModel):
    stimulus_id: str = Field(pattern=ID_PATTERN)
    text: str = Field(
        min_length=1,
        description="One complete, natural Traditional-Chinese Cantonese sentence.",
    )
    role: Literal["target", "contrast", "protection", "maintenance"]
    evidence_ids: list[str] = Field(
        min_length=1,
        description=(
            "Evidence references relevant to this text and role; at least one must "
            "be an error-cluster ID, or a capability-anchor ID for maintenance."
        ),
    )
    speaker_profile_id: str = Field(pattern=ID_PATTERN)
    copies: PositiveInt = Field(
        description="Number of distinct TTS assignments generated from this text."
    )
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_stimulus(self) -> "Stimulus":
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("stimulus evidence references must be unique")
        stripped = self.text.strip()
        if len(stripped) < 6 or len(re.findall(r"[\u3400-\u9fff]", stripped)) < 4:
            raise ValueError(
                "stimulus must contain at least four Chinese characters and six "
                "non-whitespace characters"
            )
        if stripped[-1] not in "。！？!?":
            raise ValueError("stimulus must end with sentence-final punctuation")
        return self


class ExpectedOutcome(StrictModel):
    """One directional post-intervention check compiled by the Harness."""

    outcome_id: str = Field(pattern=ID_PATTERN)
    category: Literal["primary", "counterbalance", "protection", "maintenance", "safety"]
    subject_evidence_id: str = Field(min_length=1)
    measure: Literal[
        "cluster_error_count",
        "cluster_rate",
        "capability_lost_count",
        "overall_pooled_cer",
        "easy_pooled_cer",
        "medium_pooled_cer",
        "hard_pooled_cer",
        "patient_macro_cer",
    ]
    desired_direction: Literal["decrease", "non_increase"]
    rationale: str = Field(min_length=1)


class Proposal(StrictModel):
    schema_version: Literal["3"]
    round_index: PositiveInt
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    assessment_claims: list[AssessmentClaim] = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    target_evidence_ids: list[str] = Field(
        min_length=1,
        description="Ordered unique primary targetable error-cluster IDs.",
    )
    stimuli: list[Stimulus] = Field(min_length=1)
    expected_outcomes: list[ExpectedOutcome] = Field(min_length=1)
    expected_response: str = Field(min_length=1)
    risks: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_stimuli(self) -> "Proposal":
        ids = [item.stimulus_id for item in self.stimuli]
        texts = [item.text.strip() for item in self.stimuli]
        if len(set(ids)) != len(ids):
            raise ValueError("stimulus IDs must be unique")
        if len(set(texts)) != len(texts):
            raise ValueError("stimulus texts must be unique")
        if len(set(self.target_evidence_ids)) != len(self.target_evidence_ids):
            raise ValueError("target evidence IDs must be unique")
        outcome_ids = [item.outcome_id for item in self.expected_outcomes]
        if len(set(outcome_ids)) != len(outcome_ids):
            raise ValueError("expected outcome IDs must be unique")
        outcome_signatures = [
            (item.category, item.subject_evidence_id, item.measure)
            for item in self.expected_outcomes
        ]
        if len(set(outcome_signatures)) != len(outcome_signatures):
            raise ValueError("expected outcome checks must be unique")
        return self


def canonical_sha256(value: StrictModel | dict[str, Any]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, StrictModel) else value
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_evidence_pack(evidence: EvidencePack, protocol: RunProtocol) -> None:
    """Bind an otherwise self-consistent pack to the frozen experiment protocol."""

    if evidence.run_id != protocol.run_id:
        raise ValueError("evidence run does not match protocol")
    if evidence.round_index > protocol.round_count:
        raise ValueError("evidence round exceeds protocol round_count")
    if evidence.data_scope.feedback_split_sha256 != protocol.feedback_split.sha256:
        raise ValueError("EvidencePack is not derived from the frozen feedback split")
    if (
        protocol.feedback_split.expected_rows is not None
        and evidence.data_scope.sample_count != protocol.feedback_split.expected_rows
    ):
        raise ValueError(
            "EvidencePack sample count differs from the frozen feedback split"
        )
    if evidence.snapshots.m0.checkpoint_sha256 != protocol.initial_adapter.sha256:
        raise ValueError("EvidencePack M0 checkpoint differs from the frozen protocol")
    if (
        evidence.snapshots.m0.predictions_sha256
        != protocol.initial_adapter_dev_predictions.sha256
    ):
        raise ValueError("EvidencePack M0 predictions differ from the frozen protocol")
    if (
        evidence.snapshots.zero_shot.checkpoint_sha256
        != protocol.base_model_fingerprint.sha256
    ):
        raise ValueError(
            "EvidencePack zero-shot model differs from the frozen protocol"
        )
    if (
        evidence.snapshots.zero_shot.predictions_sha256
        != protocol.zero_shot_dev_predictions.sha256
    ):
        raise ValueError(
            "EvidencePack zero-shot predictions differ from the frozen protocol"
        )

    contract = evidence.action_contract
    if contract.synthetic_count != protocol.synthetic_count_per_round:
        raise ValueError("EvidencePack synthetic budget differs from protocol")
    if contract.unique_stimulus_count != protocol.unique_stimulus_count_per_round:
        raise ValueError("EvidencePack unique-stimulus budget differs from protocol")
    if contract.real_replay_count != protocol.real_replay_count_per_round:
        raise ValueError("EvidencePack real replay budget differs from protocol")
    if contract.permitted_speaker_profiles != protocol.speaker_profiles:
        raise ValueError("EvidencePack speaker profiles differ from protocol")
    if contract.proposal_constraints != protocol.proposal_constraints:
        raise ValueError("EvidencePack proposal constraints differ from protocol")


def validate_proposal(
    proposal: Proposal, evidence: EvidencePack, protocol: RunProtocol
) -> None:
    validate_evidence_pack(evidence, protocol)
    if proposal.round_index != evidence.round_index:
        raise ValueError("proposal round does not match evidence round")
    if proposal.evidence_sha256 != canonical_sha256(evidence):
        raise ValueError("proposal does not cite the supplied EvidencePack hash")

    contract = evidence.action_contract
    if sum(item.copies for item in proposal.stimuli) != contract.synthetic_count:
        raise ValueError("proposal copies do not equal the frozen synthetic budget")
    if len(proposal.stimuli) != contract.unique_stimulus_count:
        raise ValueError("proposal stimulus count does not equal the frozen budget")
    constraints = contract.proposal_constraints
    if len(proposal.target_evidence_ids) > constraints.max_primary_targets:
        raise ValueError("proposal exceeds the frozen primary-target limit")
    excessive_stimulus_copies = [
        item.stimulus_id
        for item in proposal.stimuli
        if item.copies > constraints.max_copies_per_stimulus
    ]
    if excessive_stimulus_copies:
        raise ValueError(
            "stimulus exceeds the frozen per-stimulus copy cap: "
            f"{excessive_stimulus_copies}"
        )
    excessive_stimulus_lengths = [
        item.stimulus_id
        for item in proposal.stimuli
        if len(tts_phonology_text(item.text))
        > constraints.max_surface_characters_per_stimulus
    ]
    if excessive_stimulus_lengths:
        raise ValueError(
            "stimulus exceeds the frozen surface-character cap: "
            f"{excessive_stimulus_lengths}"
        )

    clusters = {item.evidence_id: item for item in evidence.error_clusters}
    capabilities = {item.evidence_id: item for item in evidence.capability_anchors}
    known_evidence = set(clusters) | set(capabilities)
    known_evidence.update({"snapshot.current", "snapshot.m0", "snapshot.zero_shot"})
    known_evidence.update(
        {
            "comparison.m0_vs_zero_shot",
            "comparison.current_vs_zero_shot",
            "comparison.current_vs_m0",
        }
    )
    if evidence.snapshots.previous is not None:
        known_evidence.add("snapshot.previous")
    if evidence.comparisons.current_vs_previous is not None:
        known_evidence.add("comparison.current_vs_previous")
    known_evidence.update(
        item.ledger_id for item in evidence.intervention_history
    )

    cited = {
        evidence_ref
        for claim in proposal.assessment_claims
        for evidence_ref in claim.evidence_ids
    }
    cited.update(
        evidence_ref for item in proposal.stimuli for evidence_ref in item.evidence_ids
    )
    cited.update(item.subject_evidence_id for item in proposal.expected_outcomes)
    cited.update(proposal.target_evidence_ids)
    unknown = sorted(cited - known_evidence)
    if unknown:
        raise ValueError(f"proposal cites unknown evidence references: {unknown}")

    invalid_target_refs = sorted(set(proposal.target_evidence_ids) - set(clusters))
    if invalid_target_refs:
        raise ValueError(
            f"primary targets must be error-cluster IDs: {invalid_target_refs}"
        )
    target_clusters = [clusters[item] for item in proposal.target_evidence_ids]
    non_targetable = [
        item.evidence_id for item in target_clusters if not item.targetable
    ]
    if non_targetable:
        raise ValueError(
            f"proposal uses non-targetable primary evidence: {non_targetable}"
        )
    non_actionable_targets = [
        item.evidence_id for item in target_clusters if not item.tts_actionable
    ]
    if non_actionable_targets:
        raise ValueError(
            "proposal uses non-TTS-actionable primary evidence: "
            f"{non_actionable_targets}"
        )

    permitted_levels = set(contract.permitted_evidence_levels)
    cited_cluster_ids = cited.intersection(clusters)
    forbidden_levels = sorted(
        evidence_id
        for evidence_id in cited_cluster_ids
        if clusters[evidence_id].level not in permitted_levels
    )
    if forbidden_levels:
        raise ValueError(
            f"proposal cites clusters at forbidden evidence levels: {forbidden_levels}"
        )

    stimulus_cluster_ids = {
        evidence_id
        for item in proposal.stimuli
        for evidence_id in item.evidence_ids
        if evidence_id in clusters
    }
    non_targetable_stimulus_ids = sorted(
        evidence_id
        for evidence_id in stimulus_cluster_ids
        if not clusters[evidence_id].targetable
    )
    if non_targetable_stimulus_ids:
        raise ValueError(
            f"stimuli cite non-targetable clusters: {non_targetable_stimulus_ids}"
        )
    non_actionable_stimulus_ids = sorted(
        evidence_id
        for evidence_id in stimulus_cluster_ids
        if not clusters[evidence_id].tts_actionable
    )
    if non_actionable_stimulus_ids:
        raise ValueError(
            f"stimuli cite non-TTS-actionable clusters: {non_actionable_stimulus_ids}"
        )

    target_stimuli = [item for item in proposal.stimuli if item.role == "target"]
    if not target_stimuli:
        raise ValueError("proposal must contain at least one target stimulus")
    target_ids = set(proposal.target_evidence_ids)
    invalid_target_stimuli = [
        item.stimulus_id
        for item in target_stimuli
        if not target_ids.intersection(item.evidence_ids)
        or set(item.evidence_ids).intersection(clusters) - target_ids
    ]
    if invalid_target_stimuli:
        raise ValueError(
            f"target stimuli must cite only primary target clusters: {invalid_target_stimuli}"
        )
    contrast_without_target = [
        item.stimulus_id
        for item in proposal.stimuli
        if item.role == "contrast" and not target_ids.intersection(item.evidence_ids)
    ]
    if contrast_without_target:
        raise ValueError(
            "contrast stimuli must cite the primary target they contrast: "
            f"{contrast_without_target}"
        )
    invalid_contrasts: list[str] = []
    for item in proposal.stimuli:
        if item.role != "contrast":
            continue
        item_cluster_ids = set(item.evidence_ids).intersection(clusters)
        primary_ids = item_cluster_ids.intersection(target_ids)
        comparator_ids = item_cluster_ids - target_ids
        if not primary_ids or not comparator_ids:
            invalid_contrasts.append(f"{item.stimulus_id}:missing_observed_comparator")
            continue
        primary = clusters[next(iter(primary_ids))]
        if not all(
            clusters[evidence_id].level == primary.level
            and clusters[evidence_id].reference_unit == primary.reference_unit
            for evidence_id in comparator_ids
        ):
            invalid_contrasts.append(
                f"{item.stimulus_id}:comparator_not_same_reference_family"
            )
    if invalid_contrasts:
        raise ValueError(
            "contrast role requires an observed same-reference confusion: "
            + "; ".join(invalid_contrasts)
        )

    protection_stimuli = [
        item for item in proposal.stimuli if item.role == "protection"
    ]
    protection_cluster_ids = {
        evidence_id
        for item in protection_stimuli
        for evidence_id in item.evidence_ids
        if evidence_id in clusters
    }
    protected_primary_ids = sorted(protection_cluster_ids.intersection(target_ids))
    if protected_primary_ids:
        raise ValueError(
            "protection stimuli cannot relabel a primary target: "
            f"{protected_primary_ids}"
        )
    unsupported_protection_ids: list[str] = []
    for evidence_id in sorted(protection_cluster_ids):
        observations = clusters[evidence_id].observations
        anchors = [observations.zero_shot, observations.m0]
        if observations.previous is not None:
            anchors.append(observations.previous)
        regressed = any(
            observations.current.by_difficulty[group].error_count
            > anchor.by_difficulty[group].error_count
            for group in ("easy", "medium")
            for anchor in anchors
        )
        if not regressed:
            unsupported_protection_ids.append(evidence_id)
    if unsupported_protection_ids:
        raise ValueError(
            "protection clusters lack an observed easy/medium adaptation regression: "
            f"{unsupported_protection_ids}"
        )
    uncovered_targets = sorted(
        target_ids
        - {
            evidence_ref
            for item in target_stimuli
            for evidence_ref in item.evidence_ids
        }
    )
    if uncovered_targets:
        raise ValueError(
            f"primary targets have no target stimulus: {uncovered_targets}"
        )
    maintenance_stimuli = [item for item in proposal.stimuli if item.role == "maintenance"]
    maintenance_ids = {
        evidence_id for item in maintenance_stimuli for evidence_id in item.evidence_ids
        if evidence_id in capabilities
    }
    invalid_maintenance = [
        item.stimulus_id for item in maintenance_stimuli
        if not set(item.evidence_ids).intersection(capabilities)
        or set(item.evidence_ids).intersection(clusters)
    ]
    nonmaintenance_capabilities = [
        item.stimulus_id for item in proposal.stimuli if item.role != "maintenance"
        and set(item.evidence_ids).intersection(capabilities)
    ]
    if invalid_maintenance or nonmaintenance_capabilities:
        raise ValueError(
            "maintenance must cite capability anchors, other roles error clusters: "
            f"{invalid_maintenance + nonmaintenance_capabilities}"
        )
    available_groups = {item.difficulty for item in capabilities.values()}
    maintained_groups = {capabilities[key].difficulty for key in maintenance_ids}
    if available_groups != maintained_groups:
        raise ValueError("maintenance must cover each available easy/medium capability cohort")
    stimulus_without_cluster = [
        item.stimulus_id
        for item in proposal.stimuli
        if item.role != "maintenance"
        if not set(item.evidence_ids).intersection(clusters)
    ]
    if stimulus_without_cluster:
        raise ValueError(
            "every stimulus must cite at least one error cluster: "
            f"{stimulus_without_cluster}"
        )

    # Agent v1 fixes TTS reference allocation.  Utterance-level difficulty does
    # not support selecting a patient stratum.
    restricted_profile_stimuli = [
        item.stimulus_id
        for item in proposal.stimuli
        if item.speaker_profile_id != constraints.fixed_speaker_profile_id
    ]
    if restricted_profile_stimuli:
        raise ValueError(
            "Agent v1 fixes all stimuli to an all-patient speaker profile: "
            f"{restricted_profile_stimuli}"
        )

    # A cluster that only just crosses any targetability threshold receives an
    # exploratory cap.  Count every role, not just role=target, because grounding
    # requires every cited cluster to be instantiated in the stimulus.
    selection = evidence.data_scope.cluster_selection
    threshold_edge_ids = {
        evidence_id
        for evidence_id in stimulus_cluster_ids
        for item in (clusters[evidence_id],)
        if item.targetable
        if (
            item.observations.current.error_count
            == selection.minimum_current_error_count
            or item.observations.current.patient_count
            == selection.minimum_patient_count
            or item.observations.current.content_count
            == selection.minimum_content_count
        )
    }
    max_edge_copies = constraints.threshold_edge_max_linked_copies
    max_edge_stimuli = constraints.threshold_edge_max_unique_stimuli
    excessive_edge_dose: list[str] = []
    for evidence_id in sorted(threshold_edge_ids):
        citing = [item for item in proposal.stimuli if evidence_id in item.evidence_ids]
        copies = sum(item.copies for item in citing)
        if copies > max_edge_copies or len(citing) > max_edge_stimuli:
            excessive_edge_dose.append(
                f"{evidence_id}:copies={copies}/{max_edge_copies},"
                f"stimuli={len(citing)}/{max_edge_stimuli}"
            )
    if excessive_edge_dose:
        raise ValueError(
            "threshold-edge cluster exceeds exploratory citation dose: "
            + "; ".join(excessive_edge_dose)
        )

    cluster_measures = {"cluster_error_count", "cluster_rate"}
    metric_measures = {
        "overall_pooled_cer",
        "easy_pooled_cer",
        "medium_pooled_cer",
        "hard_pooled_cer",
        "patient_macro_cer",
    }
    invalid_outcomes: list[str] = []
    for outcome in proposal.expected_outcomes:
        if outcome.measure in cluster_measures:
            if outcome.subject_evidence_id not in clusters:
                invalid_outcomes.append(
                    f"{outcome.outcome_id}:cluster_measure_requires_cluster"
                )
        elif outcome.measure == "capability_lost_count":
            if (
                outcome.subject_evidence_id not in capabilities
                or outcome.category != "maintenance"
                or outcome.desired_direction != "non_increase"
            ):
                invalid_outcomes.append(f"{outcome.outcome_id}:invalid_capability_retention_check")
        elif outcome.measure in metric_measures:
            if outcome.subject_evidence_id != "snapshot.current":
                invalid_outcomes.append(
                    f"{outcome.outcome_id}:metric_measure_requires_snapshot.current"
                )
        else:
            raise AssertionError(f"unhandled outcome measure {outcome.measure}")
    primary_outcomes = {
        item.subject_evidence_id
        for item in proposal.expected_outcomes
        if item.category == "primary"
        and item.measure in cluster_measures
        and item.desired_direction == "decrease"
    }
    if primary_outcomes != target_ids:
        invalid_outcomes.append("primary_outcomes_do_not_match_primary_targets")
    comparator_ids = {
        evidence_id
        for item in proposal.stimuli
        if item.role == "contrast"
        for evidence_id in item.evidence_ids
        if evidence_id in clusters and evidence_id not in target_ids
    }
    monitored_comparators = {
        item.subject_evidence_id
        for item in proposal.expected_outcomes
        if item.category == "counterbalance"
        and item.measure in cluster_measures
        and item.desired_direction == "non_increase"
    }
    if monitored_comparators != comparator_ids:
        invalid_outcomes.append(
            "counterbalance_outcomes_do_not_match_contrast_clusters"
        )
    monitored_protections = {
        item.subject_evidence_id
        for item in proposal.expected_outcomes
        if item.category == "protection" and item.measure in cluster_measures
    }
    if monitored_protections != protection_cluster_ids:
        invalid_outcomes.append("protection_outcomes_do_not_match_protection_clusters")
    monitored_maintenance = {
        item.subject_evidence_id for item in proposal.expected_outcomes
        if item.category == "maintenance" and item.measure == "capability_lost_count"
    }
    if monitored_maintenance != maintenance_ids:
        invalid_outcomes.append("maintenance_outcomes_do_not_match_capability_anchors")
    if any(item.category == "maintenance" and item.measure != "capability_lost_count"
           for item in proposal.expected_outcomes):
        invalid_outcomes.append("maintenance_requires_capability_lost_count")
    safety_checks = {
        (item.measure, item.desired_direction)
        for item in proposal.expected_outcomes
        if item.category == "safety"
    }
    required_safety = {
        ("easy_pooled_cer", "non_increase"),
        ("medium_pooled_cer", "non_increase"),
    }
    if not required_safety.issubset(safety_checks):
        invalid_outcomes.append("easy_and_medium_safety_outcomes_are_required")
    if any(
        item.category != "safety" and item.measure in metric_measures
        for item in proposal.expected_outcomes
    ):
        invalid_outcomes.append("aggregate_metrics_must_use_safety_category")
    if any(
        item.category == "safety" and item.measure in cluster_measures
        for item in proposal.expected_outcomes
    ):
        invalid_outcomes.append("cluster_checks_cannot_use_safety_category")
    if invalid_outcomes:
        raise ValueError(
            "proposal expected outcomes are inconsistent: "
            + "; ".join(invalid_outcomes)
        )

    permitted_profiles = set(contract.permitted_speaker_profile_ids)
    unknown_profiles = sorted(
        {item.speaker_profile_id for item in proposal.stimuli} - permitted_profiles
    )
    if unknown_profiles:
        raise ValueError(f"proposal uses unknown speaker profiles: {unknown_profiles}")

    permitted_roles = set(contract.permitted_roles)
    unknown_roles = sorted({item.role for item in proposal.stimuli} - permitted_roles)
    if unknown_roles:
        raise ValueError(f"proposal uses forbidden roles: {unknown_roles}")
