from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_therapist_outcome import _prediction_rows, _protocol
from test_therapist_reflection import _reflection
from test_therapist_reflection import mixed_round as mixed_round
from therapist_harness.agent import build_therapist_request
from therapist_harness.cli import _round_skill_text, build_parser
from therapist_harness.evidence_pack import build_evidence_pack
from therapist_harness.memory import (
    SkillMemory,
    compile_intervention_summary,
    validate_summary_digest,
)
from therapist_harness.memory_schema import NextRoundGuidance
from therapist_harness.schema import EvidencePack, InterventionSummary, canonical_sha256


def _summary(item):
    return compile_intervention_summary(
        item.protocol, item.evidence, item.proposal, item.pack, _reflection(item)
    )


@pytest.mark.parametrize("reference", ["intervention.round_001", "intervention.intervention.round_001", "intervention.round_999"])
def test_generated_ledger_id_is_used_verbatim(mixed_round, tmp_path, monkeypatch, reference):
    from test_therapist_harness import protocol, evidence, proposal
    from therapist_harness import schema

    citation_root = tmp_path / "citation_case"
    citation_root.mkdir()
    spec = protocol(citation_root)
    # Isolate citation resolution from independently tested checkpoint/history
    # binding. The history ID comes from the real summary producer, not a stub.
    summary = _summary(mixed_round)
    assert summary.ledger_id == "intervention.round_001"
    pack = evidence(spec).model_copy(update={"intervention_history": [summary]})
    candidate = proposal(pack)
    candidate.assessment_claims[0].evidence_ids.append(reference)
    monkeypatch.setattr(schema, "validate_evidence_pack", lambda *_: None)
    if reference == summary.ledger_id:
        schema.validate_proposal(candidate, pack, spec)
    else:
        with pytest.raises(ValueError, match="unknown evidence references"):
            schema.validate_proposal(candidate, pack, spec)


def test_episode_preserves_real_reflection_not_only_numeric_verdict(mixed_round):
    item = mixed_round
    reflection = _reflection(item)
    summary = _summary(item)
    validate_summary_digest(summary)
    assert (
        summary.episode_memory.hypothesis_rationale == reflection.hypothesis_rationale
    )
    assert summary.episode_memory.lesson_candidates == reflection.lesson_candidates
    assert summary.episode_memory.next_round_guidance == reflection.next_round_guidance
    assert (
        summary.episode_memory.competing_explanations
        == reflection.competing_explanations
    )
    assert summary.episode_memory.reflection_sha256 == canonical_sha256(reflection)
    assert summary.episode_memory.active_skill_version == "v000"
    assert summary.easy_medium_safety == "easy_regressed"
    assert summary.target_effect == "improved"


def test_episode_content_change_breaks_digest(mixed_round):
    summary = _summary(mixed_round)
    changed = summary.model_copy(update={"hypothesis_verdict": "supported"})
    with pytest.raises(ValueError, match="ledger digest"):
        validate_summary_digest(changed)


def test_next_guidance_dose_means_primary_practice_not_total_synthetic(mixed_round):
    guidance = _reflection(mixed_round).next_round_guidance
    payload = guidance.model_dump(mode="json")
    assert payload["primary_practice_dose_direction"] == "maintain"
    assert "within_budget_dose_direction" not in payload
    description = NextRoundGuidance.model_json_schema()["properties"][
        "primary_practice_dose_direction"
    ]["description"]
    assert "combined target + contrast" in description
    assert "fixed total synthetic count" in description
    payload["within_budget_dose_direction"] = payload.pop(
        "primary_practice_dose_direction"
    )
    with pytest.raises(ValueError, match="primary_practice_dose_direction"):
        NextRoundGuidance.model_validate(payload)


def test_history_requires_explicit_reflection_memory(mixed_round):
    payload = _summary(mixed_round).model_dump(mode="json")
    payload.pop("episode_memory")
    with pytest.raises(ValueError, match="episode_memory"):
        InterventionSummary.model_validate(payload)


def test_next_agent_input_receives_immediate_reflection_with_unchanged_skill(
    mixed_round,
):
    item = mixed_round
    summary = _summary(item)
    before = _prediction_rows(after=False)
    after = _prediction_rows(after=True)
    next(row for row in after if row["utt_id"] == "u2")["after"] = "阿聽日錯錯"
    protocol = item.protocol
    pack = build_evidence_pack(
        protocol,
        round_index=2,
        current_rows=after,
        current_prediction_column="after",
        current_checkpoint_sha256=summary.new_checkpoint_sha256,
        current_predictions_sha256=summary.dev_predictions_sha256,
        m0_rows=before,
        m0_prediction_column="before",
        m0_predictions_sha256=protocol.initial_adapter_dev_predictions.sha256,
        zero_shot_rows=before,
        zero_shot_prediction_column="before",
        zero_shot_predictions_sha256=protocol.zero_shot_dev_predictions.sha256,
        previous_rows=before,
        previous_prediction_column="before",
        previous_checkpoint_sha256=protocol.initial_adapter.sha256,
        previous_predictions_sha256=protocol.initial_adapter_dev_predictions.sha256,
        intervention_history=[summary],
    )
    request = build_therapist_request(
        protocol,
        pack,
        system_prompt="Propose the next intervention.",
        skill_text="v000 unchanged",
    )
    episode = json.loads(request["input"][2]["content"])["evidence"][
        "intervention_history"
    ][0]
    assert (
        episode["episode_memory"]["next_round_guidance"]["maintenance_strategy"]
        == "increase_within_budget"
    )
    assert episode["episode_memory"]["lesson_candidates"][0]["predicted_observation"]
    assert request["input"][1]["content"].endswith("v000 unchanged")
    assert pack.action_contract.real_replay_count == 6
    assert pack.action_contract.synthetic_count == 2
    # Episode memory is not a cumulative synthetic train manifest.
    assert "synthesis_manifest" not in episode
    contaminated = pack.model_dump(mode="json")
    contaminated["artifact_mode"] = "counterfactual_interface_smoke"
    with pytest.raises(ValueError, match="artifact mode differs"):
        EvidencePack.model_validate(contaminated)


def test_active_skill_switches_after_round_two_not_during_it(tmp_path: Path):
    protocol = _protocol(tmp_path)
    initial = tmp_path / "initial.md"
    initial.write_text("Initial response-to-intervention skill.", encoding="utf-8")
    revised = tmp_path / "revised.md"
    revised.write_text(
        "Balance target repairs with acquired ability maintenance.", encoding="utf-8"
    )
    memory = SkillMemory.initialize(tmp_path / "skills", initial)
    first = memory.bind_round(1)
    second = memory.bind_round(2)
    activation = memory.activate(protocol, after_round_index=2, updated_skill=revised)
    third = memory.bind_round(3)
    assert first.version == second.version == "v000"
    assert memory.bind_round(2) == second
    assert activation.skill == third
    assert third.version == "v001"
    assert third.read_text() == revised.read_text()
    assert first.read_text() == initial.read_text()
    for command in ("propose", "reflect"):
        positional = (
            ["protocol", "evidence", "output"]
            if command == "propose"
            else ["protocol", "evidence", "proposal", "pack", "output"]
        )
        args = build_parser().parse_args(
            [command, *positional, "--skills-dir", str(memory.directory)]
        )
        assert _round_skill_text(args, 2) == initial.read_text()
        assert _round_skill_text(args, 3) == revised.read_text()


def test_skill_update_no_change_is_explicit_and_does_not_erase_history(tmp_path: Path):
    protocol = _protocol(tmp_path)
    initial = tmp_path / "initial.md"
    initial.write_text("Keep prior skill; one episode is not a general rule.")
    memory = SkillMemory.initialize(tmp_path / "skills", initial)
    activation = memory.activate(protocol, after_round_index=2, updated_skill=None)
    assert activation.action == "no_change"
    assert memory.bind_round(3).version == "v000"
    assert (memory.directory / "activations" / "after_round_002.json").exists()


def test_skill_cannot_be_rewritten_or_activated_after_future_round_is_bound(
    tmp_path: Path,
):
    protocol = _protocol(tmp_path)
    initial = tmp_path / "initial.md"
    initial.write_text("Initial Skill")
    memory = SkillMemory.initialize(tmp_path / "skills", initial)
    with pytest.raises(ValueError, match="not scheduled"):
        memory.activate(protocol, after_round_index=1, updated_skill=None)
    memory.bind_round(3)
    with pytest.raises(ValueError, match="later round was bound"):
        memory.activate(protocol, after_round_index=2, updated_skill=None)
    Path(memory.resolve(1).file_path).write_text("Tampered Skill")
    with pytest.raises(ValueError, match="snapshot has changed"):
        memory.resolve(1)


def test_propose_never_implicitly_reads_v000():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["propose", "protocol", "evidence", "output"])
