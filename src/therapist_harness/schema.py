from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


SHA256_PATTERN = r"^[0-9a-f]{64}$"
ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
AnonymousExampleId = Annotated[str, Field(pattern=r"^utt-[0-9a-f]{12}$")]


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
    max_retries: Literal[0]
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
    training_passes_per_round: Literal[1]
    save_at_round_end: Literal[True]
    scheduler: Literal["linear", "constant"]
    reset_scheduler_each_round: bool
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

    @model_validator(mode="after")
    def unique_groups(self) -> "SpeakerProfileRule":
        if len(set(self.difficulty_groups)) != len(self.difficulty_groups):
            raise ValueError("speaker-profile difficulty groups must be unique")
        return self


class RunProtocol(StrictModel):
    schema_version: Literal["1"]
    protocol_status: Literal["draft", "frozen"]
    run_id: str = Field(pattern=ID_PATTERN)
    setting_id: Literal["setting_d"]
    round_count: PositiveInt
    base_model_dir: str = Field(min_length=1)
    base_model_fingerprint: ArtifactRef
    initial_adapter: ArtifactRef | None
    real_train: ArtifactRef
    feedback_split: ArtifactRef
    sealed_test_split: ArtifactRef
    speech_generator: ArtifactRef
    frozen_reference_manifest: ArtifactRef
    agent: AgentConfig
    synthetic_count_per_round: PositiveInt
    unique_stimulus_count_per_round: PositiveInt
    real_replay_count_per_round: PositiveInt
    speaker_profiles: dict[str, SpeakerProfileRule] = Field(min_length=1)
    lora: LoRAConfig
    evaluation: EvaluationConfig
    skill_update_rounds: list[PositiveInt] = Field(default_factory=list)

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
        if self.unique_stimulus_count_per_round > self.synthetic_count_per_round:
            raise ValueError("unique stimulus count cannot exceed synthetic count")
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
        )
        if self.lora.optimizer_steps_per_round != expected_steps:
            raise ValueError(
                "optimizer_steps_per_round must cover exactly one pass over the "
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


class ErrorCluster(StrictModel):
    evidence_id: str = Field(pattern=ID_PATTERN)
    operation: Literal["deletion", "substitution", "insertion"]
    level: Literal["character", "onset", "nucleus", "coda", "tone"]
    reference_unit: str
    hypothesis_unit: str
    left_context: str | None
    right_context: str | None
    error_count: PositiveInt
    opportunity_count: PositiveInt
    error_rate: NonNegativeFloat
    patient_count: PositiveInt
    content_count: PositiveInt
    difficulty_distribution: dict[Literal["easy", "medium", "hard"], NonNegativeInt]
    previous_rate: NonNegativeFloat | None
    initial_rate: NonNegativeFloat | None
    observed_snapshot_count: PositiveInt
    bounded_examples: list[AnonymousExampleId] = Field(max_length=5)
    alignment_uncertainty: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_rate(self) -> "ErrorCluster":
        expected = self.error_count / self.opportunity_count
        if abs(expected - self.error_rate) > 1e-9:
            raise ValueError("error_rate must equal error_count / opportunity_count")
        return self


class InterventionSummary(StrictModel):
    ledger_id: str = Field(pattern=ID_PATTERN)
    round_index: PositiveInt
    hypothesis_verdict: Literal["supported", "mixed", "refuted", "inconclusive"]
    target_effect: str = Field(min_length=1)
    easy_medium_safety: str = Field(min_length=1)


class ActionContract(StrictModel):
    synthetic_count: PositiveInt
    unique_stimulus_count: PositiveInt
    permitted_speaker_profile_ids: list[str] = Field(min_length=1)
    roles: list[Literal["target", "contrast", "protection"]] = Field(
        default_factory=lambda: ["target", "contrast", "protection"]
    )


class EvidencePack(StrictModel):
    schema_version: Literal["1"]
    run_id: str = Field(pattern=ID_PATTERN)
    round_index: PositiveInt
    current_checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    feedback_split_sha256: str = Field(pattern=SHA256_PATTERN)
    current_metrics: MetricSet
    delta_vs_initial: MetricDelta
    delta_vs_previous: MetricDelta | None
    transitions_vs_initial: TransitionCounts
    transitions_vs_previous: TransitionCounts | None
    error_clusters: list[ErrorCluster]
    previous_interventions: list[InterventionSummary]
    action_contract: ActionContract
    evidence_note: Literal[
        "Phone-level fields describe ASR output errors at GT phonological positions; they are not patient articulation rules."
    ]

    @model_validator(mode="after")
    def unique_evidence_ids(self) -> "EvidencePack":
        ids = [item.evidence_id for item in self.error_clusters]
        if len(set(ids)) != len(ids):
            raise ValueError("error cluster evidence IDs must be unique")
        return self


class AssessmentClaim(StrictModel):
    claim: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class Stimulus(StrictModel):
    stimulus_id: str = Field(pattern=ID_PATTERN)
    text: str = Field(min_length=1)
    role: Literal["target", "contrast", "protection"]
    evidence_ids: list[str] = Field(min_length=1)
    speaker_profile_id: str = Field(pattern=ID_PATTERN)
    copies: PositiveInt
    rationale: str = Field(min_length=1)


class Proposal(StrictModel):
    schema_version: Literal["1"]
    round_index: PositiveInt
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    assessment_claims: list[AssessmentClaim] = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    targets: list[str] = Field(min_length=1)
    stimuli: list[Stimulus] = Field(min_length=1)
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
        return self


class ClusterResponse(StrictModel):
    evidence_id: str = Field(pattern=ID_PATTERN)
    before_rate: NonNegativeFloat
    after_rate: NonNegativeFloat
    changed_error_count: int


class SynthesisQC(StrictModel):
    requested: PositiveInt
    generated: NonNegativeInt
    failed: NonNegativeInt
    total_duration_seconds: NonNegativeFloat

    @model_validator(mode="after")
    def validate_total(self) -> "SynthesisQC":
        if self.generated + self.failed != self.requested:
            raise ValueError("generated + failed must equal requested")
        return self


class RoundResult(StrictModel):
    schema_version: Literal["1"]
    run_id: str = Field(pattern=ID_PATTERN)
    round_index: PositiveInt
    parent_checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    new_checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    before_metrics: MetricSet
    after_metrics: MetricSet
    paired_delta: MetricDelta
    transitions: TransitionCounts
    target_cluster_response: list[ClusterResponse]
    off_target_changes: list[str]
    patient_improved: NonNegativeInt
    patient_worsened: NonNegativeInt
    patient_tied: NonNegativeInt
    synthesis_qc: SynthesisQC
    artifact_sha256: dict[str, str]

    @model_validator(mode="after")
    def validate_artifact_hashes(self) -> "RoundResult":
        for name, digest in self.artifact_sha256.items():
            if not name or re.fullmatch(SHA256_PATTERN, digest) is None:
                raise ValueError("artifact_sha256 contains an invalid entry")
        return self


class Reflection(StrictModel):
    schema_version: Literal["1"]
    run_id: str = Field(pattern=ID_PATTERN)
    round_index: PositiveInt
    hypothesis_verdict: Literal["supported", "mixed", "refuted", "inconclusive"]
    evidence_citations: list[str] = Field(min_length=1)
    target_effect: str = Field(min_length=1)
    easy_medium_safety: str = Field(min_length=1)
    patient_heterogeneity: str = Field(min_length=1)
    competing_explanations: list[str]
    next_round_recommendation: str = Field(min_length=1)


class SkillPatchOperation(StrictModel):
    operation: Literal["add", "revise", "retire"]
    rule_text: str = Field(min_length=1)
    supporting_interventions: list[str] = Field(min_length=1)
    contradicting_interventions: list[str]


class SkillPatch(StrictModel):
    schema_version: Literal["1"]
    base_skill_version: str = Field(pattern=r"^v[0-9]{3}$")
    source_ledger_ids: list[str] = Field(min_length=1)
    operations: list[SkillPatchOperation] = Field(min_length=1)


def canonical_sha256(value: StrictModel | dict[str, Any]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, StrictModel) else value
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_proposal(
    proposal: Proposal, evidence: EvidencePack, protocol: RunProtocol
) -> None:
    if proposal.round_index != evidence.round_index:
        raise ValueError("proposal round does not match evidence round")
    if evidence.run_id != protocol.run_id:
        raise ValueError("evidence run does not match protocol")
    if evidence.round_index > protocol.round_count:
        raise ValueError("evidence round exceeds protocol round_count")
    if evidence.feedback_split_sha256 != protocol.feedback_split.sha256:
        raise ValueError("EvidencePack is not derived from the frozen feedback split")
    if proposal.evidence_sha256 != canonical_sha256(evidence):
        raise ValueError("proposal does not cite the supplied EvidencePack hash")

    contract = evidence.action_contract
    if contract.synthetic_count != protocol.synthetic_count_per_round:
        raise ValueError("EvidencePack synthetic budget differs from protocol")
    if contract.unique_stimulus_count != protocol.unique_stimulus_count_per_round:
        raise ValueError("EvidencePack unique-stimulus budget differs from protocol")
    if contract.permitted_speaker_profile_ids != protocol.speaker_profile_ids:
        raise ValueError("EvidencePack speaker profiles differ from protocol")

    if sum(item.copies for item in proposal.stimuli) != contract.synthetic_count:
        raise ValueError("proposal copies do not equal the frozen synthetic budget")
    if len(proposal.stimuli) != contract.unique_stimulus_count:
        raise ValueError("proposal stimulus count does not equal the frozen budget")

    known_evidence = {item.evidence_id for item in evidence.error_clusters}
    cited = {
        evidence_id
        for claim in proposal.assessment_claims
        for evidence_id in claim.evidence_ids
    }
    cited.update(
        evidence_id for item in proposal.stimuli for evidence_id in item.evidence_ids
    )
    unknown = sorted(cited - known_evidence)
    if unknown:
        raise ValueError(f"proposal cites unknown evidence IDs: {unknown}")

    permitted_profiles = set(contract.permitted_speaker_profile_ids)
    unknown_profiles = sorted(
        {item.speaker_profile_id for item in proposal.stimuli} - permitted_profiles
    )
    if unknown_profiles:
        raise ValueError(f"proposal uses unknown speaker profiles: {unknown_profiles}")

    permitted_roles = set(contract.roles)
    unknown_roles = sorted({item.role for item in proposal.stimuli} - permitted_roles)
    if unknown_roles:
        raise ValueError(f"proposal uses forbidden roles: {unknown_roles}")
