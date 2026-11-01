"""Opt-in execution facts for the ASR preference loop; no quality scores."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .retention_contract import RealKLRetention


class PreferenceContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class PreferenceLoopConfig(PreferenceContractModel):
    comparison: Literal["four_arms", "dpo_only", "synthetic_only"] = Field(default="four_arms", exclude_if=lambda value: value == "four_arms")
    execution_version: Literal["serial_v1", "persistent_dual_gpu_v2", "persistent_synthetic_only_v1", "persistent_kl_dpo_v3"] = Field(default="serial_v1", exclude_if=lambda value: value == "serial_v1")
    samples_per_audio: int = Field(default=4, ge=1)
    beta: float = Field(default=0.1, gt=0)
    real_per_step: int = Field(default=8, ge=0)
    synthetic_per_step: int = Field(default=2, ge=1)
    synthetic_weight: float = Field(default=0.05, gt=0)
    temperature: float = Field(default=0.8, gt=0)
    top_p: float = Field(default=0.95, gt=0, le=1)
    max_new_tokens: int = Field(default=128, ge=1)
    reference_policy: Literal["frozen_initial_real_only_adapter"] = "frozen_initial_real_only_adapter"
    main_branch: Literal["mas_dpo"] = "mas_dpo"
    review_mode: Literal["audio", "asr_transcript"] = Field(default="audio", exclude_if=lambda value: value == "audio")
    review_asr_adapter: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)
    review_asr_adapter_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda value: value is None)
    review_asr_base_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda value: value is None)
    audio_review_model: str = Field(min_length=1)
    audio_review_base_url: str = Field(min_length=1)
    audio_review_api_key_env: str = Field(default="OPENAI_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*$")
    review_allow_pending: bool = Field(default=False, exclude_if=lambda value: not value)
    include_greedy: bool = Field(default=False, exclude_if=lambda value: not value)
    generation_design_path: str | None = Field(default=None, exclude_if=lambda value: value is None)
    generation_design_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda value: value is None)
    checkpoint_acceptance: Literal["always", "real_dev_v1", "real_dev_initial_parent_v2"] = Field(default="always", exclude_if=lambda value: value == "always")
    training_allocation: Literal["uniform", "role_budget_v1", "mas_audio_budget_v1"] = Field(default="uniform", exclude_if=lambda value: value == "uniform")
    retention_design_path: str | None = Field(default=None, exclude_if=lambda value: value is None)
    retention_design_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def transcript_review_requires_frozen_asr(self):
        if (self.retention_design_path is None) != (self.retention_design_sha256 is None):
            raise ValueError('real-KL design requires both path and hash')
        if self.execution_version == 'persistent_kl_dpo_v3':
            if self.retention_design_path is None or self.comparison != 'synthetic_only' or self.review_mode != 'asr_transcript' or not self.include_greedy:
                raise ValueError('KL/DPO v3 requires explicit retention, synthetic preferences, fixed transcript review and greedy diagnostics')
            if self.generation_design_path is None or self.training_allocation != 'mas_audio_budget_v1' or self.checkpoint_acceptance != 'real_dev_initial_parent_v2':
                raise ValueError('KL/DPO v3 requires explicit generation decisions, MAS audio budgets and the initial/current real-development gate')
        elif self.retention_design_path is not None:
            raise ValueError('real-KL retention requires the explicit v3 execution version')
        if self.training_allocation == 'mas_audio_budget_v1' and self.execution_version != 'persistent_kl_dpo_v3':
            raise ValueError('actual MAS audio budgets require KL/DPO v3')
        if (self.generation_design_path is None) != (self.generation_design_sha256 is None):
            raise ValueError("generation design requires both path and hash")
        if self.comparison == "synthetic_only":
            if self.real_per_step != 0 or self.synthetic_weight != 1.0:
                raise ValueError("synthetic-only DPO requires zero real exposure and unit synthetic loss weight")
        elif self.real_per_step == 0:
            raise ValueError("real comparison branches require positive real_per_step")
        if self.execution_version == "persistent_synthetic_only_v1" and (self.comparison != "synthetic_only" or self.review_mode != "asr_transcript"):
            raise ValueError("persistent synthetic-only execution requires synthetic-only DPO and transcript review")
        if self.execution_version == "persistent_dual_gpu_v2" and (self.comparison != "dpo_only" or self.review_mode != "asr_transcript"):
            raise ValueError("dual GPU execution requires the two DPO branches and transcript review")
        if self.review_mode == "asr_transcript" and not all((
            self.review_asr_adapter, self.review_asr_adapter_sha256, self.review_asr_base_model_sha256,
        )):
            raise ValueError("ASR transcript review requires the frozen review adapter and model hashes")
        return self


class PreferenceSampleDelivery(PreferenceContractModel):
    sample_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    review_verdict: Literal["pass", "obvious_failure", "pending"]
    pair_count: int = Field(ge=0)
    training_exposures: int = Field(ge=0)
    role: Literal["target", "contrast", "protection", "maintenance"] | None = Field(default=None, exclude_if=lambda value: value is None)
    reference_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    generation_mode: Literal['zero_shot','instruct2'] | None = Field(default=None, exclude_if=lambda value: value is None)
    instruction_id: str | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def delivered_only_after_review(self):
        if self.review_verdict != "pass" and (self.pair_count or self.training_exposures):
            raise ValueError("obvious generation failures cannot enter preference training")
        if self.training_exposures and not self.pair_count:
            raise ValueError("pairless audio cannot have DPO exposure")
        return self


class PreferenceDelivery(PreferenceContractModel):
    objective: Literal["dpo"] = "dpo"
    current_asr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_reference_asr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviews_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pair_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exposure_log_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    samples_per_audio: int = Field(ge=1)
    real_audio_considered: int = Field(ge=0)
    real_pair_audio_count: int = Field(ge=0)
    real_pair_count: int = Field(ge=0)
    real_training_exposures: int = Field(ge=0)
    synthetic_samples: list[PreferenceSampleDelivery]
    chosen_label_semantics: Literal["target_text_after_obvious_failure_review_not_verbatim_verified"] = "target_text_after_obvious_failure_review_not_verbatim_verified"
    preference_token_mask: Literal['full', 'edit_eos'] = Field(default='full', exclude_if=lambda value: value == 'full')
    chosen_text_transform: Literal['identity', 'strip_terminal_sentence_punctuation_v2'] = Field(default='identity', exclude_if=lambda value: value == 'identity')
    real_kl_retention: RealKLRetention | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def counts(self):
        if self.real_kl_retention is not None and self.real_kl_retention.teacher_asr_sha256 != self.frozen_reference_asr_sha256:
            raise ValueError('auxiliary real KL must use the same frozen initial R-only reference')
        if len({x.sample_id for x in self.synthetic_samples}) != len(self.synthetic_samples):
            raise ValueError("duplicate synthetic delivery identity")
        if not self.real_pair_audio_count <= self.real_audio_considered:
            raise ValueError("more paired real audio than considered audio")
        if self.real_training_exposures and not self.real_pair_count:
            raise ValueError("pairless real audio cannot have DPO exposure")
        if not self.real_pair_audio_count <= self.real_pair_count <= self.real_pair_audio_count * self.samples_per_audio:
            raise ValueError("real preference pair counts are inconsistent")
        if any(x.pair_count > self.samples_per_audio for x in self.synthetic_samples):
            raise ValueError("synthetic pairs exceed actual samples")
        return self


def validate_preference_training(protocol, training, *, sample_ids=None):
    """Keep old SFT budgets strict; opt-in DPO budgets count actual pair rows."""
    config = protocol.preference_loop
    delivery = training.preference_delivery
    if config is None:
        if delivery is not None:
            raise ValueError("preference delivery requires a preference-loop protocol")
        if training.real_rows != protocol.real_replay_count_per_round:
            raise ValueError("training real-replay count differs from protocol")
        if training.synthetic_rows != protocol.synthetic_count_per_round:
            raise ValueError("training synthetic count differs from protocol")
        return
    if delivery is None:
        raise ValueError("preference-loop training requires actual delivery facts")
    if config.execution_version == 'persistent_kl_dpo_v3':
        if delivery.real_kl_retention is None or delivery.preference_token_mask != 'edit_eos' or delivery.chosen_text_transform != 'strip_terminal_sentence_punctuation_v2':
            raise ValueError('KL/DPO v3 requires explicit auxiliary retention and edit+EOS label/score semantics')
    elif delivery.real_kl_retention is not None or delivery.preference_token_mask != 'full' or delivery.chosen_text_transform != 'identity':
        raise ValueError('legacy preference runs cannot silently change their training objective')
    if delivery.frozen_reference_asr_sha256 != protocol.initial_adapter.sha256:
        raise ValueError("DPO reference must remain the frozen initial R-only ASR")
    if delivery.current_asr_sha256 != training.parent_checkpoint_directory_sha256:
        raise ValueError("rejected samples must come from the current parent ASR")
    if delivery.samples_per_audio != config.samples_per_audio:
        raise ValueError("sampling count differs from the frozen preference protocol")
    if delivery.real_audio_considered != protocol.real_replay_count_per_round:
        raise ValueError("real sampling pool differs from the declared replay selection")
    if len(delivery.synthetic_samples) != protocol.synthetic_count_per_round:
        raise ValueError("delivery must account for every requested synthetic audio")
    if sample_ids is not None and set(sample_ids) != {x.sample_id for x in delivery.synthetic_samples}:
        raise ValueError("preference delivery identities differ from synthesis")
    if training.real_rows != delivery.real_pair_count or training.synthetic_rows != sum(x.pair_count for x in delivery.synthetic_samples):
        raise ValueError("training row counts must report actual preference pairs")
    if training.train_manifest_file_sha256 != delivery.pair_manifest_sha256:
        raise ValueError("trained manifest differs from the preference manifest")
