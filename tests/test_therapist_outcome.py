from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from therapist_harness.agent import build_reflection_request
from therapist_harness.evidence_pack import build_evidence_pack
from therapist_harness.materialize import (
    build_materialization_sidecar,
    materialize_proposal,
)
from therapist_harness.outcome import (
    compile_evaluation_plan,
    compile_reflection_pack,
    compile_round_result,
    validate_reflection_pack,
)
from therapist_harness.round_schema import (
    ClusterChange,
    MaterializationSidecar,
    MetricChange,
    ReflectionPack,
    RoundResultV2,
    SynthesisReceipt,
    SynthesizedSample,
    TrainingReceipt,
)
from therapist_harness.schema import (
    AgentConfig,
    ArtifactRef,
    AssessmentClaim,
    EvaluationConfig,
    ExpectedOutcome,
    LoRAConfig,
    Proposal,
    ProposalConstraints,
    RunProtocol,
    Stimulus,
    canonical_sha256,
)
from therapist_harness.store import sha256_directory, sha256_file


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _write_file(path: Path, content: str, *, rows: int | None = None) -> ArtifactRef:
    path.write_text(content, encoding="utf-8")
    return ArtifactRef(
        path=str(path),
        sha256=sha256_file(path),
        expected_rows=rows,
    )


def _feedback_rows() -> list[dict[str, str]]:
    specifications = (
        ("u1", "p1", "c1", "easy", "阿叔今日食粥"),
        ("u2", "p2", "c2", "easy", "阿叔聽日覆診"),
        ("u3", "p1", "c3", "medium", "姑娘幫我開門"),
        ("u4", "p2", "c4", "medium", "記得夜晚食藥"),
        ("u5", "p3", "c5", "hard", "我今日返屋企"),
        ("u6", "p4", "c6", "hard", "我聽日去覆診"),
    )
    return [
        {
            "utt_id": utterance_id,
            "speaker_id": speaker_id,
            "prompt_id": content_id,
            "zero_shot_bucket": difficulty,
            "clean_gt": gt,
        }
        for utterance_id, speaker_id, content_id, difficulty, gt in specifications
    ]


def _prediction_rows(*, after: bool) -> list[dict[str, str]]:
    rows = _feedback_rows()
    before_predictions = {
        "u1": "阿今日食粥",
        "u2": "阿聽日覆診",
        "u3": "姑娘幫我開門",
        "u4": "記得夜晚食藥",
        "u5": "我今日返屋企",
        "u6": "我聽日去覆診",
    }
    after_predictions = {
        "u1": "阿叔今日食粥",
        "u2": "阿叔聽日覆診",
        "u3": "姑娘幫我開門",
        "u4": "記得夜晚食藥",
        "u5": "今日返屋企",
        "u6": "聽日去覆診",
    }
    prediction_column = "after" if after else "before"
    predictions = after_predictions if after else before_predictions
    return [{**row, prediction_column: predictions[row["utt_id"]]} for row in rows]


def _jsonl(rows: list[dict[str, str]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _protocol(tmp_path: Path) -> RunProtocol:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")

    feedback_rows = _feedback_rows()
    feedback = _write_file(
        tmp_path / "feedback.jsonl",
        _jsonl(feedback_rows),
        rows=len(feedback_rows),
    )
    train_rows = []
    references = []
    reference_speakers = set()
    for row in feedback_rows:
        wav = tmp_path / f"reference-{row['utt_id']}.wav"
        wav.write_bytes(f"reference-{row['utt_id']}".encode())
        train_rows.append({**row, "audio": str(wav)})
        if row["speaker_id"] not in reference_speakers:
            reference_speakers.add(row["speaker_id"])
            references.append(
                {
                    "utt_id": row["utt_id"],
                    "speaker_id": row["speaker_id"],
                    "difficulty_group": row["zero_shot_bucket"],
                    "reference_text": row["clean_gt"],
                    "reference_wav": str(wav),
                    "reference_wav_sha256": sha256_file(wav),
                }
            )
    real_train = _write_file(
        tmp_path / "train.jsonl",
        _jsonl(train_rows),
        rows=len(train_rows),
    )
    reference_manifest = _write_file(
        tmp_path / "references.json",
        json.dumps(
            {
                "schema_version": "1",
                "source_manifest_path": real_train.path,
                "source_manifest_sha256": real_train.sha256,
                "profiles": {
                    "all_balanced": {"difficulty_groups": ["easy", "medium", "hard"]}
                },
                "references": references,
            },
            ensure_ascii=False,
        ),
    )
    sealed_test = _write_file(
        tmp_path / "test.jsonl",
        _jsonl([feedback_rows[0]]),
        rows=1,
    )
    m0_predictions = _write_file(
        tmp_path / "m0.csv",
        "prediction\n" + "\n".join("x" for _ in feedback_rows) + "\n",
        rows=len(feedback_rows),
    )
    zero_predictions = _write_file(
        tmp_path / "zero.csv",
        "prediction\n" + "\n".join("z" for _ in feedback_rows) + "\n",
        rows=len(feedback_rows),
    )

    return RunProtocol(
        schema_version="1",
        protocol_status="frozen",
        run_id="toy_outcome",
        setting_id="setting_d",
        round_count=4,
        base_model_dir=str(model_dir),
        base_model_fingerprint=_write_file(model_dir / "model.index.json", "{}\n"),
        initial_adapter=ArtifactRef(
            path=str(adapter_dir),
            sha256=sha256_directory(adapter_dir),
            kind="directory",
        ),
        initial_adapter_dev_predictions=m0_predictions,
        initial_adapter_prediction_column="before",
        zero_shot_dev_predictions=zero_predictions,
        zero_shot_prediction_column="before",
        real_train=real_train,
        feedback_split=feedback,
        sealed_test_split=sealed_test,
        speech_generator=_write_file(tmp_path / "speech_generator.pt", "tts"),
        frozen_reference_manifest=reference_manifest,
        agent=AgentConfig(
            provider="OpenAI",
            base_url="https://api.example.test",
            model="gpt-test",
            reasoning_effort="xhigh",
            store=False,
            max_retries=0,
            timeout_seconds=120,
            max_output_tokens=8192,
            api_key_env="OPENAI_API_KEY",
        ),
        synthetic_count_per_round=2,
        unique_stimulus_count_per_round=1,
        real_replay_count_per_round=6,
        speaker_profiles={
            "all_balanced": {"difficulty_groups": ["easy", "medium", "hard"]}
        },
        proposal_constraints=ProposalConstraints(
            fixed_speaker_profile_id="all_balanced",
            max_primary_targets=1,
            max_copies_per_stimulus=2,
            max_surface_characters_per_stimulus=32,
            threshold_edge_max_linked_copies=2,
            threshold_edge_max_unique_stimuli=1,
            threshold_edge_max_carrier_exposures=2,
            non_targetable_cluster_stimuli_allowed=False,
            non_tts_actionable_cluster_stimuli_allowed=False,
        ),
        lora=LoRAConfig(
            rank=8,
            alpha=16,
            dropout=0.05,
            target_modules=["q_proj"],
            learning_rate=0.0001,
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
        skill_update_rounds=[2],
    )


def _proposal(protocol: RunProtocol):
    before = _prediction_rows(after=False)
    evidence = build_evidence_pack(
        protocol,
        round_index=1,
        current_rows=before,
        current_prediction_column="before",
        current_checkpoint_sha256=protocol.initial_adapter.sha256,
        current_predictions_sha256=protocol.initial_adapter_dev_predictions.sha256,
        m0_rows=before,
        m0_prediction_column="before",
        m0_predictions_sha256=protocol.initial_adapter_dev_predictions.sha256,
        zero_shot_rows=before,
        zero_shot_prediction_column="before",
        zero_shot_predictions_sha256=protocol.zero_shot_dev_predictions.sha256,
    )
    # This fixture isolates legacy target-recovery checks at a two-copy budget.
    # Dedicated maintenance tests exercise populated capability anchors.
    evidence = evidence.model_copy(update={"capability_anchors": []})
    target = next(
        item
        for item in evidence.error_clusters
        if item.level == "character"
        and item.operation == "deletion"
        and item.reference_unit == "叔"
    )
    assert target.targetable and target.tts_actionable
    proposal = Proposal(
        schema_version="3",
        round_index=1,
        evidence_sha256=canonical_sha256(evidence),
        assessment_claims=[
            AssessmentClaim(
                claim="The same deletion occurs across patients and contents.",
                evidence_ids=[target.evidence_id],
            )
        ],
        hypothesis="Rehearsing the target should reduce its deletion without easy or medium regression.",
        target_evidence_ids=[target.evidence_id],
        stimuli=[
            Stimulus(
                stimulus_id="s1",
                text="阿叔今日記得飲暖水。",
                role="target",
                evidence_ids=[target.evidence_id],
                speaker_profile_id="all_balanced",
                copies=2,
                rationale="Exercise the observed deletion carrier.",
            )
        ],
        expected_outcomes=[
            ExpectedOutcome(
                outcome_id="primary-target",
                category="primary",
                subject_evidence_id=target.evidence_id,
                measure="cluster_error_count",
                desired_direction="decrease",
                rationale="The target deletion count should fall.",
            ),
            ExpectedOutcome(
                outcome_id="safety-easy",
                category="safety",
                subject_evidence_id="snapshot.current",
                measure="easy_pooled_cer",
                desired_direction="non_increase",
                rationale="Easy real speech should not regress.",
            ),
            ExpectedOutcome(
                outcome_id="safety-medium",
                category="safety",
                subject_evidence_id="snapshot.current",
                measure="medium_pooled_cer",
                desired_direction="non_increase",
                rationale="Medium real speech should not regress.",
            ),
        ],
        expected_response="Fewer target deletions with preserved easy and medium CER.",
        risks=["The intervention may move errors into hard utterances."],
    )
    return evidence, proposal, target.evidence_id


def _assignments(
    protocol: RunProtocol, proposal: Proposal, evidence
) -> list[dict[str, str | int]]:
    return materialize_proposal(
        proposal, evidence, protocol, output_root=Path(protocol.real_train.path).parent
    )


@dataclass(frozen=True)
class RoundArtifacts:
    protocol: RunProtocol
    evidence: object
    proposal: Proposal
    target_id: str
    plan: object
    sidecar: MaterializationSidecar
    synthesis: SynthesisReceipt
    training: TrainingReceipt
    result: object
    pack: ReflectionPack
    before_rows: list[dict[str, str]]
    after_rows: list[dict[str, str]]


@pytest.fixture()
def round_artifacts(tmp_path: Path) -> RoundArtifacts:
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
        reflection_prompt_file_sha256=SHA_B,
        skill_version="v000",
        skill_file_sha256=SHA_C,
    )
    return RoundArtifacts(
        protocol=protocol,
        evidence=evidence,
        proposal=proposal,
        target_id=target_id,
        plan=plan,
        sidecar=sidecar,
        synthesis=synthesis,
        training=training,
        result=result,
        pack=pack,
        before_rows=before_rows,
        after_rows=after_rows,
    )


def test_evaluation_plan_is_deterministic_complete_and_frozen(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    repeated = compile_evaluation_plan(
        item.protocol,
        item.evidence,
        item.proposal,
        before_prediction_column="before",
        after_prediction_column="after",
    )

    assert repeated == item.plan
    assert canonical_sha256(repeated) == canonical_sha256(item.plan)
    assert repeated.bindings.sealed_test_used is False
    assert repeated.before_snapshot.predictions_file_sha256 == (
        item.protocol.initial_adapter_dev_predictions.sha256
    )
    assert [outcome.outcome_id for outcome in repeated.planned_outcomes] == [
        "primary-target",
        "safety-easy",
        "safety-medium",
    ]
    assert [
        cluster.descriptor.evidence_id for cluster in repeated.monitored_clusters
    ] == [item.target_id]
    assert repeated.assessment_claims == item.proposal.assessment_claims
    assert repeated.hypothesis == item.proposal.hypothesis
    assert repeated.risks == item.proposal.risks
    assert [cluster.evidence_id for cluster in repeated.target_clusters] == [
        item.target_id
    ]
    assert repeated.required_metric_groups == ["overall", "easy", "medium", "hard"]
    with pytest.raises(ValidationError, match="frozen"):
        repeated.scope.sample_count = 99


def test_sidecar_and_completed_receipts_preserve_exact_dose(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    sidecar = item.sidecar
    assert len(sidecar.samples) == 2
    assert [sample.copy_index for sample in sidecar.samples] == [1, 2]
    assert all(sample.role == "target" for sample in sidecar.samples)
    assert all(sample.evidence_ids == [item.target_id] for sample in sidecar.samples)
    assert all(sample.carrier_counts[0].count == 1 for sample in sidecar.samples)
    assert sidecar.by_role[0].copies == 2
    assert sidecar.by_role[0].carrier_exposures == 2
    assert sidecar.by_cluster[0].linked_copies == 2
    assert sidecar.by_cluster[0].carrier_exposures == 2

    assert item.synthesis.status == "complete"
    assert item.synthesis.requested == item.synthesis.generated == 2
    assert item.synthesis.failed == 0
    assert item.training.status == "complete"
    assert (item.training.real_rows, item.training.synthetic_rows) == (6, 2)
    assert item.training.actual_optimizer_steps == 2


def test_round_result_compiles_directions_off_target_and_realization(
    round_artifacts: RoundArtifacts,
) -> None:
    result = round_artifacts.result
    outcomes = {
        item.outcome_id: item for item in result.response.expected_outcome_results
    }
    assert outcomes["primary-target"].before_value == 2
    assert outcomes["primary-target"].after_value == 0
    assert outcomes["primary-target"].observed_direction == "decrease"
    assert outcomes["primary-target"].compiled_verdict == "met"
    assert outcomes["safety-easy"].compiled_verdict == "met"
    assert outcomes["safety-medium"].observed_direction == "unchanged"
    assert outcomes["safety-medium"].compiled_verdict == "met"

    metric_changes = {item.measure: item for item in result.response.metric_changes}
    assert metric_changes["easy_pooled_cer"].response == "improved"
    assert metric_changes["medium_pooled_cer"].response == "unchanged"
    assert metric_changes["hard_pooled_cer"].response == "worsened"
    assert result.response.patient_response.pattern == "bidirectional"
    assert result.response.transitions.easy.utterance_cer.improved == 2
    assert result.response.transitions.hard.utterance_cer.worsened == 2

    target = result.response.monitored_cluster_changes[0]
    assert target.descriptor.evidence_id == round_artifacts.target_id
    assert target.error_count_delta == -2
    assert target.response == "improved"
    hard_regression = next(
        item
        for item in result.response.off_target_changes
        if item.descriptor.level == "character"
        and item.descriptor.operation == "deletion"
        and item.descriptor.reference_unit == "我"
    )
    assert hard_regression.error_count_delta == 2
    assert hard_regression.response == "worsened"
    assert hard_regression.citation_id.startswith("result.off_target.")

    assert result.intervention.overall.audio_seconds == 3.0
    assert result.intervention.overall.realized_carrier_exposures == 2
    assert result.intervention.by_role[0].generated == 2
    assert result.bindings.sealed_test_used is False


def test_changed_error_form_is_not_a_met_recovery_outcome(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    after = [
        {**row, "after": row["before"]}
        for row in item.before_rows
    ]
    for row in after[:2]:
        row["after"] = row["clean_gt"].replace("叔", "淑")
    result = compile_round_result(
        item.protocol,
        item.evidence,
        item.proposal,
        item.plan,
        item.sidecar,
        item.synthesis,
        item.training,
        before_rows=item.before_rows,
        after_rows=after,
        before_predictions_file_sha256=item.protocol.initial_adapter_dev_predictions.sha256,
        after_predictions_file_sha256=SHA_B,
    )
    change = result.response.monitored_cluster_changes[0]
    outcome = result.response.expected_outcome_results[0]
    assert change.error_count_delta == -2  # 叔 deletion disappears, but becomes 淑.
    assert change.target_positions.before_error_count == 2
    assert change.target_positions.after_error_count == 2
    assert change.target_positions.recovered_count == 0
    assert outcome.delta == -2
    assert outcome.target_recovery_criterion_met is False
    assert outcome.compiled_verdict == "not_met"
    assert result.response.paired_delta.overall_pooled_cer == 0


def test_reflection_pack_has_canonical_citations_and_no_private_material(
    round_artifacts: RoundArtifacts,
) -> None:
    pack = round_artifacts.pack
    validate_reflection_pack(
        pack,
        round_artifacts.protocol,
        round_artifacts.evidence,
        round_artifacts.proposal,
    )
    assert pack.evaluation_plan == round_artifacts.plan
    assert pack.materialization_sidecar == round_artifacts.sidecar
    assert pack.round_result == round_artifacts.result
    assert pack.compiled_expectation_verdict == "supported"
    citations = set(pack.valid_citation_ids)
    assert "proposal.hypothesis" in citations
    assert "result.outcome.primary-target" in citations
    assert f"result.cluster.{round_artifacts.target_id}" in citations
    assert "result.metric.easy_pooled_cer" in citations
    assert "result.transition.hard" in citations
    assert "result.patient_directions" in citations
    assert "result.synthesis.role.target" in citations
    assert "result.training" in citations
    assert "result.confound.joint_round_attribution" in citations

    serialized = json.dumps(pack.model_dump(mode="json"), ensure_ascii=False)
    assert "/private/" not in serialized
    assert "reference_wav" not in serialized
    assert "speaker_id" not in serialized
    assert "阿叔今日記得飲暖水" in serialized  # Synthetic prescription, not private GT.
    assert pack.provenance.sealed_test_used is False


def test_reflection_rejects_changed_prescribed_text(
    round_artifacts: RoundArtifacts,
) -> None:
    payload = round_artifacts.pack.model_dump(mode="json")
    payload["prescribed_proposal"]["stimuli"][0]["text"] = "阿叔記得食藥先出門。"
    changed = ReflectionPack.model_validate(payload)
    with pytest.raises(ValueError, match="prescribed Proposal hash"):
        validate_reflection_pack(changed)


def test_reflection_rejects_changed_next_action_contract_hash(
    round_artifacts: RoundArtifacts,
) -> None:
    payload = round_artifacts.pack.model_dump(mode="json")
    payload["next_action_contract"]["synthetic_count"] += 1
    changed = ReflectionPack.model_validate(payload)
    with pytest.raises(ValueError, match="next ActionContract hash"):
        validate_reflection_pack(changed)


def test_reflection_rejects_rehashed_next_contract_that_differs_from_source(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    payload = item.pack.model_dump(mode="json")
    payload["next_action_contract"]["permitted_roles"].remove("maintenance")
    payload["provenance"]["next_action_contract_canonical_sha256"] = canonical_sha256(
        payload["next_action_contract"]
    )
    changed = ReflectionPack.model_validate(payload)
    with pytest.raises(ValueError, match="next ActionContract differs from source"):
        validate_reflection_pack(changed, item.protocol, item.evidence, item.proposal)


def test_reflection_rejects_next_total_budget_change_even_after_rehash(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    payload = item.pack.model_dump(mode="json")
    payload["next_action_contract"]["synthetic_count"] -= 1
    payload["provenance"]["next_action_contract_canonical_sha256"] = canonical_sha256(
        payload["next_action_contract"]
    )
    changed = ReflectionPack.model_validate(payload)
    with pytest.raises(ValueError, match="next ActionContract differs from frozen protocol"):
        validate_reflection_pack(changed, item.protocol)


def test_rejects_mutated_evaluation_plan_binding(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    bad_bindings = item.plan.bindings.model_copy(
        update={"proposal_canonical_sha256": SHA_A}
    )
    bad_plan = item.plan.model_copy(update={"bindings": bad_bindings})
    with pytest.raises(ValueError, match="EvaluationPlan differs"):
        compile_round_result(
            item.protocol,
            item.evidence,
            item.proposal,
            bad_plan,
            item.sidecar,
            item.synthesis,
            item.training,
            before_rows=item.before_rows,
            after_rows=item.after_rows,
            before_predictions_file_sha256=(
                item.protocol.initial_adapter_dev_predictions.sha256
            ),
            after_predictions_file_sha256=SHA_B,
        )


def test_rejects_sidecar_dose_that_disagrees_with_indexed_samples(
    round_artifacts: RoundArtifacts,
) -> None:
    payload = round_artifacts.sidecar.model_dump(mode="json")
    payload["by_role"][0]["carrier_exposures"] += 1
    with pytest.raises(ValidationError, match="role dose"):
        MaterializationSidecar.model_validate(payload)


def test_rejects_synchronized_but_fabricated_carrier_counts(
    round_artifacts: RoundArtifacts,
) -> None:
    """Self-consistent aggregates still cannot rewrite Proposal grounding."""

    item = round_artifacts
    payload = item.sidecar.model_dump(mode="json")
    for sample in payload["samples"]:
        sample["carrier_counts"][0]["count"] = 100
    payload["by_role"][0]["carrier_exposures"] = 200
    payload["by_cluster"][0]["carrier_exposures"] = 200
    fabricated = MaterializationSidecar.model_validate(payload)

    with pytest.raises(ValueError, match="sidecar rows differ from Proposal grounding"):
        compile_round_result(
            item.protocol,
            item.evidence,
            item.proposal,
            item.plan,
            fabricated,
            item.synthesis,
            item.training,
            before_rows=item.before_rows,
            after_rows=item.after_rows,
            before_predictions_file_sha256=(
                item.protocol.initial_adapter_dev_predictions.sha256
            ),
            after_predictions_file_sha256=SHA_B,
        )


@pytest.mark.parametrize(
    ("model", "field_path"),
    [(MetricChange, "metric_changes"), (ClusterChange, "monitored_cluster_changes")],
)
def test_rejects_response_label_that_contradicts_delta(
    round_artifacts: RoundArtifacts, model: type, field_path: str
) -> None:
    source = getattr(round_artifacts.result.response, field_path)[0]
    payload = source.model_dump(mode="json")
    payload["response"] = "worsened" if source.response != "worsened" else "improved"
    with pytest.raises(ValidationError, match="response"):
        model.model_validate(payload)


def test_rejects_incomplete_or_substituted_synthesis_receipt(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    with pytest.raises(ValidationError, match="completed synthesis"):
        SynthesisReceipt(
            schema_version="1",
            run_id=item.protocol.run_id,
            round_index=1,
            status="complete",
            assignments_canonical_sha256=item.sidecar.assignments_canonical_sha256,
            requested=2,
            generated=1,
            failed=0,
            samples=[item.synthesis.samples[0]],
        )

    substituted = item.synthesis.model_copy(
        update={
            "samples": [
                item.synthesis.samples[0],
                item.synthesis.samples[1].model_copy(
                    update={"sample_id": "unexpected.sample"}
                ),
            ]
        }
    )
    with pytest.raises(ValueError, match="sample IDs differ"):
        compile_round_result(
            item.protocol,
            item.evidence,
            item.proposal,
            item.plan,
            item.sidecar,
            substituted,
            item.training,
            before_rows=item.before_rows,
            after_rows=item.after_rows,
            before_predictions_file_sha256=(
                item.protocol.initial_adapter_dev_predictions.sha256
            ),
            after_predictions_file_sha256=SHA_B,
        )


def test_rejects_training_receipt_outside_frozen_protocol(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    bad_training = item.training.model_copy(
        update={"expected_optimizer_steps": 1, "actual_optimizer_steps": 1}
    )
    with pytest.raises(ValueError, match="optimizer steps differ"):
        compile_round_result(
            item.protocol,
            item.evidence,
            item.proposal,
            item.plan,
            item.sidecar,
            item.synthesis,
            bad_training,
            before_rows=item.before_rows,
            after_rows=item.after_rows,
            before_predictions_file_sha256=(
                item.protocol.initial_adapter_dev_predictions.sha256
            ),
            after_predictions_file_sha256=SHA_B,
        )


def test_rejects_unpaired_feedback_metadata(round_artifacts: RoundArtifacts) -> None:
    item = round_artifacts
    bad_after = [dict(row) for row in item.after_rows]
    bad_after[0]["prompt_id"] = "changed-content"
    with pytest.raises(ValueError, match="paired metadata mismatch"):
        compile_round_result(
            item.protocol,
            item.evidence,
            item.proposal,
            item.plan,
            item.sidecar,
            item.synthesis,
            item.training,
            before_rows=item.before_rows,
            after_rows=bad_after,
            before_predictions_file_sha256=(
                item.protocol.initial_adapter_dev_predictions.sha256
            ),
            after_predictions_file_sha256=SHA_B,
        )


def test_rejects_noncanonical_citations_and_private_reflection_data(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    payload = item.pack.model_dump(mode="json")
    payload["valid_citation_ids"] = payload["valid_citation_ids"][:-1]
    bad_citations = ReflectionPack.model_validate(payload)
    with pytest.raises(ValueError, match="citation namespace"):
        validate_reflection_pack(
            bad_citations, item.protocol, item.evidence, item.proposal
        )

    payload = item.pack.model_dump(mode="json")
    payload["activated_confounds"][0]["description"] = (
        "Inspect /data/private/vlink228 first."
    )
    private_pack = ReflectionPack.model_validate(payload)
    with pytest.raises(ValueError, match="private identifiers or paths"):
        validate_reflection_pack(
            private_pack, item.protocol, item.evidence, item.proposal
        )


def test_formal_reflection_entry_rejects_source_consistent_carrier_forgery(
    round_artifacts: RoundArtifacts,
) -> None:
    """Rehashing a self-consistent Pack cannot replace source grounding."""

    item = round_artifacts
    payload = item.pack.model_dump(mode="json")
    sidecar_payload = payload["materialization_sidecar"]
    sample = sidecar_payload["samples"][0]
    evidence_id = sample["carrier_counts"][0]["evidence_id"]
    sample["carrier_counts"][0]["count"] += 99
    next(dose for dose in sidecar_payload["by_role"] if dose["role"] == sample["role"])[
        "carrier_exposures"
    ] += 99
    next(
        dose
        for dose in sidecar_payload["by_cluster"]
        if dose["evidence_id"] == evidence_id
    )["carrier_exposures"] += 99
    fabricated_sidecar = MaterializationSidecar.model_validate(sidecar_payload)
    payload["materialization_sidecar"] = fabricated_sidecar.model_dump(mode="json")

    result_payload = payload["round_result"]
    result_payload["bindings"]["materialization_sidecar_canonical_sha256"] = (
        canonical_sha256(fabricated_sidecar)
    )
    fabricated_result = RoundResultV2.model_validate(result_payload)
    payload["round_result"] = fabricated_result.model_dump(mode="json")
    payload["provenance"]["round_result_canonical_sha256"] = canonical_sha256(
        fabricated_result
    )
    fabricated_pack = ReflectionPack.model_validate(payload)

    # Internal hashes are deliberately coherent. The formal source-aware boundary
    # must still reconstruct the carrier count from the original Proposal text.
    with pytest.raises(ValueError, match="realization differs from sidecar"):
        validate_reflection_pack(fabricated_pack, item.protocol)
    with pytest.raises(ValueError, match="rows differ from Proposal grounding"):
        validate_reflection_pack(
            fabricated_pack, item.protocol, item.evidence, item.proposal
        )


def test_rejects_round_result_protocol_binding_tamper_in_pack_and_request(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    result_payload = item.result.model_dump(mode="json")
    assert result_payload["bindings"]["protocol_canonical_sha256"] != SHA_A
    result_payload["bindings"]["protocol_canonical_sha256"] = SHA_A
    tampered_result = RoundResultV2.model_validate(result_payload)

    pack_payload = item.pack.model_dump(mode="json")
    pack_payload["round_result"] = tampered_result.model_dump(mode="json")
    pack_payload["provenance"]["round_result_canonical_sha256"] = canonical_sha256(
        tampered_result
    )
    tampered_pack = ReflectionPack.model_validate(pack_payload)

    with pytest.raises(ValueError, match="protocol_canonical_sha256 bindings differ"):
        validate_reflection_pack(
            tampered_pack, item.protocol, item.evidence, item.proposal
        )
    with pytest.raises(ValueError, match="protocol_canonical_sha256 bindings differ"):
        build_reflection_request(
            item.protocol,
            item.evidence,
            item.proposal,
            tampered_pack,
            system_prompt="not reached",
            skill_text="not reached",
        )


def _rehash_result_in_pack(payload: dict) -> ReflectionPack:
    """Rehash edited facts so tests exercise semantics rather than stale hashes."""

    result = RoundResultV2.model_validate(payload["round_result"])
    payload["provenance"]["round_result_canonical_sha256"] = canonical_sha256(result)
    return ReflectionPack.model_validate(payload)


def test_rejects_metric_view_that_disagrees_with_source_metrics(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    payload = item.pack.model_dump(mode="json")
    metric = next(
        change
        for change in payload["round_result"]["response"]["metric_changes"]
        if change["measure"] == "easy_pooled_cer"
    )
    assert metric["after_value"] == 0
    metric["after_value"] = 0.5
    metric["delta"] = metric["after_value"] - metric["before_value"]
    metric["response"] = "worsened"
    with pytest.raises(ValueError, match="metric change differs from source metrics"):
        validate_reflection_pack(
            _rehash_result_in_pack(payload), item.protocol, item.evidence, item.proposal
        )


def test_rejects_outcome_view_that_disagrees_with_measured_cluster(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    payload = item.pack.model_dump(mode="json")
    outcome = payload["round_result"]["response"]["expected_outcome_results"][0]
    assert outcome["after_value"] == 0
    outcome["after_value"] = 1
    outcome["delta"] = -1
    # Both copies still say "met": the contradiction is in the factual count.
    with pytest.raises(ValueError, match="outcome values differ from measured response"):
        validate_reflection_pack(
            _rehash_result_in_pack(payload), item.protocol, item.evidence, item.proposal
        )


def test_rejects_realization_carriers_that_disagree_with_source_sidecar(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    payload = item.pack.model_dump(mode="json")
    intervention = payload["round_result"]["intervention"]
    for dose in (
        intervention["overall"], intervention["by_role"][0], intervention["by_cluster"][0]
    ):
        dose["planned_carrier_exposures"] += 99
        dose["realized_carrier_exposures"] += 99
    with pytest.raises(ValueError, match="realization differs from sidecar"):
        validate_reflection_pack(
            _rehash_result_in_pack(payload), item.protocol, item.evidence, item.proposal
        )


def test_rejects_monitored_before_coverage_that_disagrees_with_frozen_plan(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    payload = item.pack.model_dump(mode="json")
    change = payload["round_result"]["response"]["monitored_cluster_changes"][0]
    assert change["before"]["patient_count"] == 2
    change["before"]["patient_count"] = 1
    with pytest.raises(ValueError, match="cluster facts differ from frozen EvaluationPlan"):
        validate_reflection_pack(
            _rehash_result_in_pack(payload), item.protocol, item.evidence, item.proposal
        )


def test_rejects_realization_duration_that_disagrees_with_synthesis_receipt(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    payload = item.pack.model_dump(mode="json")
    result = payload["round_result"]
    result["synthesis"]["samples"][0]["duration_seconds"] += 1
    result["bindings"]["synthesis_receipt_canonical_sha256"] = canonical_sha256(
        SynthesisReceipt.model_validate(result["synthesis"])
    )
    with pytest.raises(ValueError, match="realization differs from sidecar"):
        validate_reflection_pack(
            _rehash_result_in_pack(payload), item.protocol, item.evidence, item.proposal
        )


def test_reflection_entry_rechecks_training_steps_against_protocol(
    round_artifacts: RoundArtifacts,
) -> None:
    item = round_artifacts
    payload = item.pack.model_dump(mode="json")
    result = payload["round_result"]
    result["training"]["expected_optimizer_steps"] += 1
    result["training"]["actual_optimizer_steps"] += 1
    result["bindings"]["training_receipt_canonical_sha256"] = canonical_sha256(
        TrainingReceipt.model_validate(result["training"])
    )
    with pytest.raises(ValueError, match="optimizer steps differ from protocol"):
        validate_reflection_pack(
            _rehash_result_in_pack(payload), item.protocol, item.evidence, item.proposal
        )
