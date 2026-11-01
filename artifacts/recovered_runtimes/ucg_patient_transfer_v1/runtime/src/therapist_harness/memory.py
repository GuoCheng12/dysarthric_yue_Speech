from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from .memory_schema import EpisodeMemory, MemoryModel
from .outcome import validate_reflection, validate_reflection_pack
from .round_schema import ReflectionPack, ReflectionV2
from .schema import (
    EvidencePack,
    EvidenceTargetDescriptor,
    InterventionSummary,
    Proposal,
    RunProtocol,
    canonical_sha256,
)
from .store import sha256_file, write_immutable_bytes, write_immutable_json


def compile_intervention_summary(
    protocol: RunProtocol,
    evidence: EvidencePack,
    proposal: Proposal,
    pack: ReflectionPack,
    reflection: ReflectionV2,
) -> InterventionSummary:
    """Commit measured facts and the actual Therapist reflection as one episode."""

    validate_reflection_pack(pack, protocol, evidence, proposal)
    validate_reflection(reflection, pack)
    role_counts = {item.role: item.generated for item in pack.realization.by_role}
    profiles = Counter(
        item.speaker_profile_id for item in pack.materialization_sidecar.samples
    )
    primary_outcomes = {
        item.outcome_id
        for item in pack.evaluation_plan.planned_outcomes
        if item.category == "primary"
    }
    directions = {
        item.response
        for item in reflection.target_findings
        if item.outcome_id in primary_outcomes
    }
    delta = pack.response.paired_delta
    easy_regressed = delta.easy_pooled_cer > 0
    medium_regressed = delta.medium_pooled_cer > 0
    safety = (
        "easy_and_medium_regressed"
        if easy_regressed and medium_regressed
        else "easy_regressed"
        if easy_regressed
        else "medium_regressed"
        if medium_regressed
        else "no_regression"
    )
    descriptors = [
        EvidenceTargetDescriptor.model_validate(
            item.model_dump(mode="python", exclude={"rate_semantics"})
        )
        for item in pack.evaluation_plan.target_clusters
    ]
    payload = {
        "ledger_id": f"intervention.round_{proposal.round_index:03d}",
        "round_index": proposal.round_index,
        "parent_checkpoint_sha256": pack.provenance.parent_checkpoint_directory_sha256,
        "new_checkpoint_sha256": pack.provenance.new_checkpoint_directory_sha256,
        "dev_predictions_sha256": pack.provenance.after_predictions_file_sha256,
        "target_evidence_ids": [item.evidence_id for item in descriptors],
        "target_descriptors": [item.model_dump(mode="json") for item in descriptors],
        "target_samples": role_counts["target"],
        "contrast_samples": role_counts["contrast"],
        "protection_samples": role_counts["protection"],
        "maintenance_samples": role_counts["maintenance"],
        "speaker_profile_samples": dict(profiles),
        "delta_vs_previous": delta.model_dump(mode="json"),
        "patient_directions": pack.response.patient_response.counts.model_dump(
            mode="json"
        ),
        "hypothesis_verdict": reflection.hypothesis_verdict,
        "target_effect": next(iter(directions)) if len(directions) == 1 else "mixed",
        "easy_medium_safety": safety,
        "episode_memory": EpisodeMemory(
            artifact_mode=pack.provenance.artifact_mode,
            reflection_sha256=canonical_sha256(reflection),
            reflection_pack_sha256=canonical_sha256(pack),
            active_skill_version=pack.provenance.skill_version,
            active_skill_sha256=pack.provenance.skill_file_sha256,
            hypothesis_rationale=reflection.hypothesis_rationale,
            competing_explanations=reflection.competing_explanations,
            lesson_candidates=reflection.lesson_candidates,
            next_round_guidance=reflection.next_round_guidance,
            preference_delivery=pack.training.preference_delivery,
        ).model_dump(mode="json"),
    }
    if protocol.preference_loop is not None and protocol.preference_loop.checkpoint_acceptance != "always":
        from .checkpoint_promotion import accepted_hashes
        checkpoint_hash, predictions_hash = accepted_hashes(pack, protocol=protocol)
        payload.update(accepted_checkpoint_sha256=checkpoint_hash, accepted_predictions_sha256=predictions_hash)
    return InterventionSummary.model_validate(
        {**payload, "ledger_sha256": canonical_sha256(payload)}
    )


def validate_summary_digest(summary: InterventionSummary) -> None:
    payload = summary.model_dump(mode="json", exclude={"ledger_sha256"})
    if summary.ledger_sha256 != canonical_sha256(payload):
        raise ValueError("intervention ledger digest differs from its episode content")


class SkillBinding(MemoryModel):
    version: str = Field(pattern=r"^v[0-9]{3}$")
    file_path: str = Field(min_length=1)
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def read_text(self) -> str:
        path = Path(self.file_path)
        if sha256_file(path) != self.file_sha256:
            raise ValueError("immutable Skill snapshot has changed")
        return path.read_text("utf-8")


class SkillActivation(MemoryModel):
    after_round_index: int = Field(ge=0)
    action: Literal["initial", "updated", "no_change"]
    skill: SkillBinding


class SkillMemory:
    """Immutable Skill versions and round bindings, separate from episode history.

    Updating a Skill never deletes reflection episodes. A round captures one
    version for both prescribing and reflecting; an update becomes visible only
    to the following round. SkillOptimizer authoring itself is not implemented.
    """

    def __init__(self, directory: Path):
        self.directory = directory.resolve()

    @classmethod
    def initialize(cls, directory: Path, initial_skill: Path) -> "SkillMemory":
        memory = cls(directory)
        binding = memory._snapshot("v000", initial_skill)
        memory._write_activation(
            SkillActivation(after_round_index=0, action="initial", skill=binding)
        )
        return memory

    def _snapshot(self, version: str, source: Path) -> SkillBinding:
        target = self.directory / "versions" / version / "SKILL.md"
        write_immutable_bytes(target, source.read_bytes())
        return SkillBinding(
            version=version, file_path=str(target), file_sha256=sha256_file(target)
        )

    def _write_activation(self, activation: SkillActivation) -> None:
        write_immutable_json(
            self.directory
            / "activations"
            / f"after_round_{activation.after_round_index:03d}.json",
            activation.model_dump(mode="json"),
        )

    def resolve(self, round_index: int) -> SkillBinding:
        if round_index < 1:
            raise ValueError("Skill round index must be positive")
        activations = [
            SkillActivation.model_validate_json(path.read_text("utf-8"))
            for path in sorted((self.directory / "activations").glob("*.json"))
        ]
        eligible = [
            item for item in activations if item.after_round_index < round_index
        ]
        if not eligible:
            raise ValueError("Skill memory has no initial activation")
        binding = max(eligible, key=lambda item: item.after_round_index).skill
        binding.read_text()
        return binding

    def bind_round(self, round_index: int) -> SkillBinding:
        binding = self.resolve(round_index)
        path = self.directory / "rounds" / f"round_{round_index:03d}.json"
        if path.exists():
            captured = SkillBinding.model_validate_json(path.read_text("utf-8"))
            if captured != binding:
                raise ValueError("active Skill differs from the captured round Skill")
            return captured
        write_immutable_json(path, binding.model_dump(mode="json"))
        return binding

    def activate(
        self,
        protocol: RunProtocol,
        *,
        after_round_index: int,
        updated_skill: Path | None,
    ) -> SkillActivation:
        if after_round_index not in protocol.skill_update_rounds:
            raise ValueError("Skill activation is not scheduled after this round")
        if any(
            int(path.stem.removeprefix("round_")) > after_round_index
            for path in (self.directory / "rounds").glob("round_*.json")
        ):
            raise ValueError("cannot change Skill after a later round was bound")
        activation_path = (
            self.directory / "activations" / f"after_round_{after_round_index:03d}.json"
        )
        if activation_path.exists():
            raise FileExistsError(activation_path)
        previous = self.bind_round(after_round_index)
        binding = (
            previous
            if updated_skill is None
            else self._snapshot(f"v{int(previous.version[1:]) + 1:03d}", updated_skill)
        )
        activation = SkillActivation(
            after_round_index=after_round_index,
            action="no_change" if updated_skill is None else "updated",
            skill=binding,
        )
        self._write_activation(activation)
        return activation
