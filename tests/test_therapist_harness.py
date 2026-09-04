from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from therapist_harness.agent import build_therapist_request
from therapist_harness.schema import (
    ActionContract,
    AgentConfig,
    ArtifactRef,
    AssessmentClaim,
    ErrorCluster,
    EvaluationConfig,
    EvidencePack,
    LoRAConfig,
    MetricDelta,
    MetricGroup,
    MetricSet,
    Proposal,
    RunProtocol,
    Stimulus,
    TransitionCounts,
    canonical_sha256,
    validate_proposal,
)
from therapist_harness.store import (
    REQUIRED_ARTIFACTS,
    ROUND_PHASES,
    RunStore,
    sha256_file,
    write_immutable_json,
)


EVIDENCE_NOTE = (
    "Phone-level fields describe ASR output errors at GT phonological positions; "
    "they are not patient articulation rules."
)


def artifact(path: Path, content: str = "one\n") -> ArtifactRef:
    path.write_text(content, encoding="utf-8")
    return ArtifactRef(path=str(path), sha256=sha256_file(path), expected_rows=None)


def protocol(tmp_path: Path, *, status: str = "frozen", rounds: int = 1) -> RunProtocol:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    base = artifact(model_dir / "model.index.json", "{}\n")
    return RunProtocol(
        schema_version="1",
        protocol_status=status,
        run_id="toy_run",
        setting_id="setting_d",
        round_count=rounds,
        base_model_dir=str(model_dir),
        base_model_fingerprint=base,
        initial_adapter=None,
        real_train=artifact(tmp_path / "train.jsonl"),
        feedback_split=artifact(tmp_path / "dev.jsonl"),
        sealed_test_split=artifact(tmp_path / "test.jsonl"),
        speech_generator=artifact(tmp_path / "speech_generator.pt"),
        frozen_reference_manifest=artifact(tmp_path / "references.json"),
        agent=AgentConfig(
            provider="OpenAI",
            base_url="https://api.example.test",
            model="gpt-test",
            reasoning_effort="xhigh",
            store=False,
            max_retries=0,
            api_key_env="OPENAI_API_KEY",
        ),
        synthetic_count_per_round=2,
        unique_stimulus_count_per_round=2,
        real_replay_count_per_round=4,
        speaker_profiles={
            "all_balanced": {"difficulty_groups": ["easy", "medium", "hard"]},
            "hard_balanced": {"difficulty_groups": ["hard"]},
        },
        lora=LoRAConfig(
            rank=16,
            alpha=4,
            dropout=0.05,
            target_modules=["q_proj", "k_proj"],
            learning_rate=0.0002,
            per_device_batch_size=1,
            gradient_accumulation_steps=4,
            world_size=1,
            optimizer_steps_per_round=2,
            training_passes_per_round=1,
            save_at_round_end=True,
            scheduler="linear",
            reset_scheduler_each_round=True,
            seed=42,
        ),
        evaluation=EvaluationConfig(
            language="Cantonese",
            batch_size=2,
            critical_cer_threshold=0.5,
            metrics=["pooled_cer", "patient_macro_cer"],
        ),
        skill_update_rounds=[],
    )


def group(samples: int) -> MetricGroup:
    return MetricGroup(
        sample_count=samples,
        speaker_count=samples,
        total_edits=samples,
        total_reference_characters=samples * 2,
        mean_utterance_cer=0.5 if samples else 0.0,
        pooled_cer=0.5 if samples else 0.0,
        patient_macro_cer=0.5 if samples else 0.0,
        critical_count=samples,
        exact_count=0,
    )


def metrics() -> MetricSet:
    return MetricSet(overall=group(3), easy=group(1), medium=group(1), hard=group(1))


def evidence(feedback_split_sha256: str = "b" * 64) -> EvidencePack:
    return EvidencePack(
        schema_version="1",
        run_id="toy_run",
        round_index=1,
        current_checkpoint_sha256="a" * 64,
        feedback_split_sha256=feedback_split_sha256,
        current_metrics=metrics(),
        delta_vs_initial=MetricDelta(
            overall_pooled_cer=0.0,
            easy_pooled_cer=0.0,
            medium_pooled_cer=0.0,
            hard_pooled_cer=0.0,
            patient_macro_cer=0.0,
        ),
        delta_vs_previous=None,
        transitions_vs_initial=TransitionCounts(
            recovered=0,
            newly_wrong=0,
            preserved_correct=0,
            persistent_wrong=3,
        ),
        transitions_vs_previous=None,
        error_clusters=[
            ErrorCluster(
                evidence_id="e1",
                operation="deletion",
                level="coda",
                reference_unit="k",
                hypothesis_unit="",
                left_context=None,
                right_context=None,
                error_count=2,
                opportunity_count=4,
                error_rate=0.5,
                patient_count=2,
                content_count=2,
                difficulty_distribution={"easy": 0, "medium": 1, "hard": 1},
                previous_rate=None,
                initial_rate=0.5,
                observed_snapshot_count=1,
                bounded_examples=["utt-0123456789ab"],
                alignment_uncertainty=0.0,
            )
        ],
        previous_interventions=[],
        action_contract=ActionContract(
            synthetic_count=2,
            unique_stimulus_count=2,
            permitted_speaker_profile_ids=["all_balanced", "hard_balanced"],
        ),
        evidence_note=EVIDENCE_NOTE,
    )


def proposal(pack: EvidencePack) -> Proposal:
    return Proposal(
        schema_version="1",
        round_index=1,
        evidence_sha256=canonical_sha256(pack),
        assessment_claims=[
            AssessmentClaim(claim="Coda deletion persists", evidence_ids=["e1"])
        ],
        hypothesis="Target and contrast practice will reduce this ASR error.",
        targets=["e1"],
        stimuli=[
            Stimulus(
                stimulus_id="s1",
                text="第一句",
                role="target",
                evidence_ids=["e1"],
                speaker_profile_id="hard_balanced",
                copies=1,
                rationale="Target the cited error.",
            ),
            Stimulus(
                stimulus_id="s2",
                text="第二句",
                role="protection",
                evidence_ids=["e1"],
                speaker_profile_id="all_balanced",
                copies=1,
                rationale="Protect non-hard speech.",
            ),
        ],
        expected_response="Lower coda deletion without easy regression.",
        risks=["Patient response may vary."],
    )


def test_proposal_contract_rejects_budget_unknown_fields_and_unknown_evidence(
    tmp_path: Path,
) -> None:
    spec = protocol(tmp_path)
    pack = evidence(spec.feedback_split.sha256)
    validate_proposal(proposal(pack), pack, spec)

    payload = proposal(pack).model_dump(mode="json")
    payload["learning_rate"] = 0.1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Proposal.model_validate(payload)

    payload.pop("learning_rate")
    payload["stimuli"][0]["evidence_ids"] = ["invented"]
    invalid = Proposal.model_validate(payload)
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_proposal(invalid, pack, spec)


def test_agent_request_is_single_strict_xhigh_call_without_private_paths(
    tmp_path: Path,
) -> None:
    spec = protocol(tmp_path)
    pack = evidence(spec.feedback_split.sha256)
    request = build_therapist_request(
        spec, pack, system_prompt="system", skill_text="skill"
    )
    assert request["reasoning"] == {"effort": "xhigh"}
    assert request["store"] is False
    assert request["text"]["format"]["strict"] is True
    serialized = str(request)
    assert "/data/" not in serialized
    assert "speaker_id" not in serialized
    assert "patient_id" not in serialized


def test_store_rejects_draft_skip_and_overwrite(tmp_path: Path) -> None:
    draft_root = tmp_path / "draft"
    draft_root.mkdir()
    with pytest.raises(ValueError, match="frozen"):
        RunStore.initialize(protocol(draft_root, status="draft"), tmp_path / "bad-run")

    frozen_root = tmp_path / "frozen"
    frozen_root.mkdir()
    store = RunStore.initialize(protocol(frozen_root), tmp_path / "run")
    evidence_file = tmp_path / "evidence.json"
    evidence_file.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires assessed"):
        store.advance("proposed", {"proposal": evidence_file})

    checkpoint_dir = tmp_path / "checkpoint-2"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    for phase in ROUND_PHASES:
        artifacts = {name: evidence_file for name in REQUIRED_ARTIFACTS[phase]}
        if phase == "trained":
            artifacts["checkpoint"] = checkpoint_dir
        event = store.advance(phase, artifacts)
    assert event.phase == "committed"
    assert (
        store.advance(
            "completed",
            {name: evidence_file for name in REQUIRED_ARTIFACTS["completed"]},
        ).phase
        == "completed"
    )
    store.verify_integrity()
    (checkpoint_dir / "adapter_config.json").write_text(
        '{"changed": true}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="artifact changed"):
        store.verify_integrity()
    with pytest.raises(ValueError, match="cannot advance"):
        store.advance("assessed", {"evidence": evidence_file})

    immutable = tmp_path / "immutable.json"
    write_immutable_json(immutable, {"value": 1})
    with pytest.raises(FileExistsError):
        write_immutable_json(immutable, {"value": 2})
