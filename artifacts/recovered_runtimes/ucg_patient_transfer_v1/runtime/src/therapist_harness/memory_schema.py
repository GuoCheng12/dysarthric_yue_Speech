from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .preference_contract import PreferenceDelivery


class MemoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CompetingExplanation(MemoryModel):
    explanation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    explanation: str = Field(min_length=1)
    predicted_observation: str = Field(min_length=1)
    result_citations: list[str] = Field(min_length=1)


class LessonCandidate(MemoryModel):
    action: Literal["retain", "adjust", "avoid"]
    condition: str = Field(min_length=1)
    guidance: str = Field(min_length=1)
    caveat: str = Field(min_length=1)
    predicted_observation: str = Field(min_length=1)
    result_citations: list[str] = Field(min_length=1)


class NextRoundGuidance(MemoryModel):
    primary_strategy: Literal["retain", "retarget", "reassess"]
    counterbalance_strategy: Literal[
        "maintain", "increase_within_budget", "decrease", "omit", "not_applicable"
    ]
    protection_strategy: Literal[
        "maintain", "increase_within_budget", "decrease", "omit", "not_applicable"
    ]
    maintenance_strategy: Literal[
        "maintain", "increase_within_budget", "decrease", "omit", "not_applicable"
    ]
    primary_practice_dose_direction: Literal["increase", "maintain", "decrease"] = (
        Field(
            description=(
                "Direction of the combined target + contrast copy allocation in the next "
                "proposal. Redistribute copies within the fixed total synthetic count; "
                "this does not authorize a change to total synthetic copies, training "
                "steps or real replay. Audio duration is measured after synthesis, "
                "not fixed by the copy budget."
            )
        )
    )
    rationale: str = Field(min_length=1)
    predicted_observation: str = Field(min_length=1)
    result_citations: list[str] = Field(min_length=1)


class EpisodeMemory(MemoryModel):
    artifact_mode: Literal["recorded_execution", "counterfactual_interface_smoke"]
    reflection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reflection_pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    active_skill_version: str = Field(pattern=r"^v[0-9]{3}$")
    active_skill_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hypothesis_rationale: str = Field(min_length=1)
    competing_explanations: list[CompetingExplanation]
    lesson_candidates: list[LessonCandidate]
    next_round_guidance: NextRoundGuidance
    preference_delivery: PreferenceDelivery | None = Field(default=None, exclude_if=lambda value: value is None)
