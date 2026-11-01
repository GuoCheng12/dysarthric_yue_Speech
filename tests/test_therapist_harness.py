from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from therapist_harness.agent import (
    build_therapist_request,
    validate_stimulus_grounding,
)
from therapist_harness.schema import (
    ActionContract,
    AgentConfig,
    ArtifactRef,
    AssessmentClaim,
    ClusterObservations,
    ClusterSelectionPolicy,
    ComparisonSet,
    CriticalTransitionCounts,
    DifficultyErrorRate,
    DirectionCounts,
    ErrorCluster,
    ErrorObservation,
    EvidenceDataScope,
    EvaluationConfig,
    EvidencePack,
    ExpectedOutcome,
    LoRAConfig,
    MetricDelta,
    MetricGroup,
    MetricSet,
    PatientDirectionCounts,
    PhonologyCoverageGroup,
    PhonologyCoverageSet,
    Proposal,
    ProposalConstraints,
    RunProtocol,
    SnapshotComparison,
    SnapshotEvidence,
    SnapshotPhonologyCoverageSet,
    SnapshotSet,
    Stimulus,
    TransitionCounts,
    TransitionGroup,
    TransitionSet,
    canonical_sha256,
    validate_proposal,
)
from therapist_harness.store import (
    REQUIRED_ARTIFACTS,
    RunStore,
    sha256_directory,
    sha256_file,
    write_immutable_json,
)


def artifact(path: Path, content: str = "one\n") -> ArtifactRef:
    path.write_text(content, encoding="utf-8")
    return ArtifactRef(path=str(path), sha256=sha256_file(path), expected_rows=None)


def protocol(tmp_path: Path, *, status: str = "frozen", rounds: int = 1) -> RunProtocol:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    base = artifact(model_dir / "model.index.json", "{}\n")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    artifact(adapter_dir / "adapter_model.safetensors", "adapter\n")
    m0_predictions = artifact(tmp_path / "m0_predictions.csv")
    zero_shot_predictions = artifact(tmp_path / "zero_shot_predictions.csv")
    return RunProtocol(
        schema_version="1",
        protocol_status=status,
        run_id="toy_run",
        setting_id="setting_d",
        round_count=rounds,
        base_model_dir=str(model_dir),
        base_model_fingerprint=base,
        initial_adapter=ArtifactRef(
            path=str(adapter_dir),
            sha256=sha256_directory(adapter_dir),
            kind="directory",
        ),
        initial_adapter_dev_predictions=m0_predictions,
        initial_adapter_prediction_column="prediction",
        zero_shot_dev_predictions=zero_shot_predictions,
        zero_shot_prediction_column="prediction",
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
            timeout_seconds=120,
            max_output_tokens=8192,
            api_key_env="OPENAI_API_KEY",
        ),
        synthetic_count_per_round=2,
        unique_stimulus_count_per_round=2,
        real_replay_count_per_round=4,
        speaker_profiles={
            "all_balanced": {"difficulty_groups": ["easy", "medium", "hard"]},
            "hard_balanced": {"difficulty_groups": ["hard"]},
        },
        proposal_constraints=ProposalConstraints(
            fixed_speaker_profile_id="all_balanced",
            max_primary_targets=1,
            max_copies_per_stimulus=2,
            max_surface_characters_per_stimulus=24,
            threshold_edge_max_linked_copies=1,
            threshold_edge_max_unique_stimuli=1,
            threshold_edge_max_carrier_exposures=3,
            non_targetable_cluster_stimuli_allowed=False,
            non_tts_actionable_cluster_stimuli_allowed=False,
        ),
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


def evidence(spec: RunProtocol) -> EvidencePack:
    zero_delta = MetricDelta(
        overall_pooled_cer=0.0,
        easy_pooled_cer=0.0,
        medium_pooled_cer=0.0,
        hard_pooled_cer=0.0,
        patient_macro_cer=0.0,
    )

    def transition_group(samples: int) -> TransitionGroup:
        return TransitionGroup(
            sample_count=samples,
            exact=TransitionCounts(
                recovered=0,
                newly_wrong=0,
                preserved_correct=0,
                persistent_wrong=samples,
            ),
            utterance_cer=DirectionCounts(improved=0, worsened=0, tied=samples),
            critical=CriticalTransitionCounts(
                recovered=0,
                newly_critical=0,
                preserved_noncritical=0,
                persistent_critical=samples,
            ),
        )

    transitions = TransitionSet(
        overall=transition_group(3),
        easy=transition_group(1),
        medium=transition_group(1),
        hard=transition_group(1),
    )
    patients = PatientDirectionCounts(patient_count=3, improved=0, worsened=0, tied=3)

    def comparison(evaluated: str, reference: str) -> SnapshotComparison:
        return SnapshotComparison(
            evaluated_snapshot=evaluated,
            reference_snapshot=reference,
            metric_delta=zero_delta,
            transitions=transitions,
            patient_directions=patients,
        )

    difficulty = {
        "easy": DifficultyErrorRate(error_count=0, opportunity_count=1, rate=0.0),
        "medium": DifficultyErrorRate(error_count=1, opportunity_count=1, rate=1.0),
        "hard": DifficultyErrorRate(error_count=1, opportunity_count=2, rate=0.5),
    }
    observation = ErrorObservation(
        error_count=2,
        opportunity_count=4,
        rate=0.5,
        patient_count=2,
        content_count=2,
        by_difficulty=difficulty,
        alignment_uncertainty=0.0,
    )

    def coverage(samples: int) -> PhonologyCoverageGroup:
        return PhonologyCoverageGroup(
            sample_count=samples,
            reference_characters=0,
            hypothesis_characters=0,
            reference_g2p_resolved_characters=0,
            hypothesis_g2p_resolved_characters=0,
            multi_optimal_alignment_utterances=0,
            character_substitutions=0,
            character_deletions=0,
            character_insertions=0,
            forced_substitutions=0,
            g2p_resolved_forced_substitutions=0,
            g2p_unresolved_forced_substitutions=0,
            context_sensitive_forced_substitutions=0,
            context_uncheckable_forced_substitutions=0,
            phonologically_equivalent_forced_substitutions=0,
            phonologically_different_forced_substitutions=0,
            phone_attributed_forced_substitutions=0,
        )

    return EvidencePack(
        schema_version="3",
        run_id="toy_run",
        round_index=1,
        data_scope=EvidenceDataScope(
            setting_id="setting_d",
            feedback_split_sha256=spec.feedback_split.sha256,
            sample_count=3,
            speaker_count=3,
            content_count=3,
            sealed_test_used=False,
            critical_cer_threshold=0.5,
            normalizer_id="therapist_text_normalize_v1",
            character_alignment_id=(
                "levenshtein_stable_traceback_for_character_edits_and_"
                "all_optimal_paths_for_forced_substitutions_v1"
            ),
            phonology_id="pycantonese_5.0.0_surface_to_cer_forced_v2",
            tts_actionability_policy=(
                "phone_components_or_pronounceable_character_deletions_or_"
                "nonhomophonic_character_substitutions_v1"
            ),
            difficulty_source="frozen_zero_shot_character_cer",
            difficulty_policy=("easy:CER<0.2;medium:0.2<=CER<0.5;hard:CER>=0.5"),
            delta_convention=(
                "evaluated_minus_reference; negative CER delta means improvement"
            ),
            phonology_coverage=SnapshotPhonologyCoverageSet(
                current=PhonologyCoverageSet(
                    overall=coverage(3),
                    easy=coverage(1),
                    medium=coverage(1),
                    hard=coverage(1),
                ),
                m0=PhonologyCoverageSet(
                    overall=coverage(3),
                    easy=coverage(1),
                    medium=coverage(1),
                    hard=coverage(1),
                ),
                zero_shot=PhonologyCoverageSet(
                    overall=coverage(3),
                    easy=coverage(1),
                    medium=coverage(1),
                    hard=coverage(1),
                ),
                previous=None,
            ),
            cluster_selection=ClusterSelectionPolicy(
                max_clusters_per_level=10,
                historical_targets_exempt_from_limit=True,
                candidate_count=1,
                included_count=1,
                minimum_current_error_count=1,
                minimum_patient_count=1,
                minimum_content_count=1,
                maximum_alignment_uncertainty=0.5,
                character_insertions_targetable=False,
                ordering=(
                    "all_historical_targets_then_targetable_new_clusters_then_"
                    "current_error_count_then_patient_content_support_then_anchor_burden"
                ),
            ),
        ),
        snapshots=SnapshotSet(
            current=SnapshotEvidence(
                model_role="current_pre_intervention",
                closed_loop_model_index=0,
                checkpoint_sha256=spec.initial_adapter.sha256,
                predictions_sha256=spec.initial_adapter_dev_predictions.sha256,
                metrics=metrics(),
            ),
            m0=SnapshotEvidence(
                model_role="real_only_closed_loop_start",
                closed_loop_model_index=0,
                checkpoint_sha256=spec.initial_adapter.sha256,
                predictions_sha256=spec.initial_adapter_dev_predictions.sha256,
                metrics=metrics(),
            ),
            zero_shot=SnapshotEvidence(
                model_role="zero_shot_diagnostic_only",
                closed_loop_model_index=None,
                checkpoint_sha256=spec.base_model_fingerprint.sha256,
                predictions_sha256=spec.zero_shot_dev_predictions.sha256,
                metrics=metrics(),
            ),
            previous=None,
        ),
        comparisons=ComparisonSet(
            m0_vs_zero_shot=comparison("m0", "zero_shot"),
            current_vs_zero_shot=comparison("current", "zero_shot"),
            current_vs_m0=comparison("current", "m0"),
            current_vs_previous=None,
        ),
        error_clusters=[
            ErrorCluster(
                evidence_id="e1",
                operation="deletion",
                level="coda",
                reference_unit="k",
                hypothesis_unit="∅",
                left_context=None,
                right_context=None,
                rate_semantics="confirmed_substitution_lower_bound_rate",
                targetable=True,
                selection_reasons=[
                    "meets_minimum_cross_patient_and_cross_content_support"
                ],
                tts_actionable=True,
                tts_actionability_reasons=[
                    "phone_component_has_pronunciation_facing_reference_carrier"
                ],
                observations=ClusterObservations(
                    current=observation,
                    m0=observation,
                    zero_shot=observation,
                    previous=None,
                ),
            )
        ],
        intervention_history=[],
        action_contract=ActionContract(
            synthetic_count=2,
            unique_stimulus_count=2,
            real_replay_count=4,
            permitted_speaker_profiles={
                "all_balanced": {"difficulty_groups": ["easy", "medium", "hard"]},
                "hard_balanced": {"difficulty_groups": ["hard"]},
            },
            proposal_constraints=spec.proposal_constraints,
        ),
    )


def proposal(pack: EvidencePack) -> Proposal:
    return Proposal(
        schema_version="3",
        round_index=1,
        evidence_sha256=canonical_sha256(pack),
        assessment_claims=[
            AssessmentClaim(claim="Coda deletion persists", evidence_ids=["e1"])
        ],
        hypothesis="Target and contrast practice will reduce this ASR error.",
        target_evidence_ids=["e1"],
        stimuli=[
            Stimulus(
                stimulus_id="s1",
                text="阿叔今日食白粥。",
                role="target",
                evidence_ids=["e1"],
                speaker_profile_id="all_balanced",
                copies=1,
                rationale="Target the cited error.",
            ),
            Stimulus(
                stimulus_id="s2",
                text="叔叔執好藥盒返屋企。",
                role="target",
                evidence_ids=["e1"],
                speaker_profile_id="all_balanced",
                copies=1,
                rationale="Exercise the cited output position in another context.",
            ),
        ],
        expected_outcomes=[
            ExpectedOutcome(
                outcome_id="primary-e1",
                category="primary",
                subject_evidence_id="e1",
                measure="cluster_error_count",
                desired_direction="decrease",
                rationale="The prescribed cluster should decrease.",
            ),
            ExpectedOutcome(
                outcome_id="safety-easy",
                category="safety",
                subject_evidence_id="snapshot.current",
                measure="easy_pooled_cer",
                desired_direction="non_increase",
                rationale="Protect frozen easy utterances.",
            ),
            ExpectedOutcome(
                outcome_id="safety-medium",
                category="safety",
                subject_evidence_id="snapshot.current",
                measure="medium_pooled_cer",
                desired_direction="non_increase",
                rationale="Protect frozen medium utterances.",
            ),
        ],
        expected_response="Lower coda deletion without easy regression.",
        risks=["Patient response may vary."],
    )


def add_cluster(
    pack: EvidencePack,
    *,
    evidence_id: str,
    reference_unit: str = "k",
    hypothesis_unit: str = "t",
    targetable: bool = True,
    tts_actionable: bool = True,
    zero_shot_error_count: int | None = None,
) -> EvidencePack:
    """Clone the toy coda cluster while preserving EvidencePack invariants."""

    payload = pack.model_dump(mode="json")
    cluster = pack.error_clusters[0].model_dump(mode="json")
    cluster.update(
        {
            "evidence_id": evidence_id,
            "operation": "substitution",
            "reference_unit": reference_unit,
            "hypothesis_unit": hypothesis_unit,
            "targetable": targetable,
            "selection_reasons": [
                (
                    "meets_minimum_cross_patient_and_cross_content_support"
                    if targetable
                    else "current_content_coverage_below_2"
                )
            ],
            "tts_actionable": tts_actionable,
            "tts_actionability_reasons": [
                (
                    "phone_component_has_pronunciation_facing_reference_carrier"
                    if tts_actionable
                    else "character_substitution_is_homophonic_in_isolated_jyutping"
                )
            ],
        }
    )
    if zero_shot_error_count is not None:
        if zero_shot_error_count != 1:
            raise ValueError("toy helper only supports a one-error zero-shot cluster")
        zero_shot = cluster["observations"]["zero_shot"]
        zero_shot.update(
            {
                "error_count": 1,
                "rate": 0.25,
                "patient_count": 1,
                "content_count": 1,
            }
        )
        zero_shot["by_difficulty"]["medium"].update({"error_count": 0, "rate": 0.0})
    payload["error_clusters"].append(cluster)
    selection = payload["data_scope"]["cluster_selection"]
    selection["candidate_count"] += 1
    selection["included_count"] += 1
    return EvidencePack.model_validate(payload)


def counterbalance_outcome(evidence_id: str) -> dict[str, str]:
    return {
        "outcome_id": f"counterbalance-{evidence_id}",
        "category": "counterbalance",
        "subject_evidence_id": evidence_id,
        "measure": "cluster_error_count",
        "desired_direction": "non_increase",
        "rationale": "The observed competing realization should not increase.",
    }


def protection_outcome(evidence_id: str) -> dict[str, str]:
    return {
        "outcome_id": f"protection-{evidence_id}",
        "category": "protection",
        "subject_evidence_id": evidence_id,
        "measure": "cluster_error_count",
        "desired_direction": "decrease",
        "rationale": "The observed adaptation regression should recover.",
    }


def test_proposal_contract_rejects_budget_unknown_fields_and_unknown_evidence(
    tmp_path: Path,
) -> None:
    spec = protocol(tmp_path)
    pack = evidence(spec)
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
    pack = evidence(spec)
    request = build_therapist_request(
        spec, pack, system_prompt="system", skill_text="skill"
    )
    assert request["reasoning"] == {"effort": "xhigh"}
    assert request["store"] is False
    assert request["stream"] is True
    assert request["max_output_tokens"] == 8192
    assert request["text"]["format"]["strict"] is True
    user_payload = json.loads(request["input"][2]["content"])
    assert "Proposal JSON" in user_payload["task"]
    assert user_payload["evidence"] == pack.model_dump(mode="json")
    assert user_payload["evidence_sha256"] == canonical_sha256(pack)
    assert "previous_response_id" not in request
    assert "tools" not in request
    serialized = str(request)
    assert "/data/" not in serialized
    assert "speaker_id" not in serialized
    assert "patient_id" not in serialized


def test_therapist_prompt_defines_evidence_semantics() -> None:
    prompt = (
        Path(__file__).parents[1]
        / "src"
        / "therapist_harness"
        / "prompts"
        / "therapist.md"
    ).read_text("utf-8")
    required_terms = (
        "mean_utterance_cer",
        "pooled_cer",
        "patient_macro_cer",
        "alignment_uncertainty",
        "confirmed_substitution_lower_bound_rate",
        "comparison.current_vs_previous",
        "target_evidence_ids",
        "speaker_profile_id",
        "detoned-Jyutping",
        "threshold-edge cluster",
        "actual text sent to TTS",
        "Harness-computed carrier sidecar",
    )
    assert all(term in prompt for term in required_terms)


def test_proposal_contract_requires_targetable_covered_primary_target(
    tmp_path: Path,
) -> None:
    spec = protocol(tmp_path)
    pack = evidence(spec)

    payload = proposal(pack).model_dump(mode="json")
    payload["target_evidence_ids"] = ["invented"]
    invalid = Proposal.model_validate(payload)
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_proposal(invalid, pack, spec)

    pack_payload = pack.model_dump(mode="json")
    pack_payload["error_clusters"][0]["targetable"] = False
    pack_payload["error_clusters"][0]["selection_reasons"] = [
        "current_content_coverage_below_2"
    ]
    unsupported_pack = EvidencePack.model_validate(pack_payload)
    unsupported_payload = proposal(unsupported_pack).model_dump(mode="json")
    unsupported = Proposal.model_validate(unsupported_payload)
    with pytest.raises(ValueError, match="non-targetable primary evidence"):
        validate_proposal(unsupported, unsupported_pack, spec)

    uncovered_payload = proposal(pack).model_dump(mode="json")
    for item in uncovered_payload["stimuli"]:
        item["role"] = "protection"
    uncovered = Proposal.model_validate(uncovered_payload)
    with pytest.raises(ValueError, match="at least one target stimulus"):
        validate_proposal(uncovered, pack, spec)


def test_agent_v1_rejects_patient_stratum_inference_and_threshold_edge_overdose(
    tmp_path: Path,
) -> None:
    spec = protocol(tmp_path)
    pack = evidence(spec)

    restricted_payload = proposal(pack).model_dump(mode="json")
    restricted_payload["stimuli"][0]["speaker_profile_id"] = "hard_balanced"
    restricted = Proposal.model_validate(restricted_payload)
    with pytest.raises(ValueError, match="fixes all stimuli"):
        validate_proposal(restricted, pack, spec)

    edge_payload = pack.model_dump(mode="json")
    policy = edge_payload["data_scope"]["cluster_selection"]
    policy["minimum_current_error_count"] = 2
    policy["minimum_patient_count"] = 2
    policy["minimum_content_count"] = 2
    edge_pack = EvidencePack.model_validate(edge_payload)
    overdosed_payload = proposal(edge_pack).model_dump(mode="json")
    overdosed = Proposal.model_validate(overdosed_payload)
    with pytest.raises(ValueError, match="threshold-edge cluster"):
        validate_proposal(overdosed, edge_pack, spec)


def test_stimulus_grounding_uses_tts_phone_units(tmp_path: Path) -> None:
    spec = protocol(tmp_path)
    pack = evidence(spec)
    payload = proposal(pack).model_dump(mode="json")
    payload["stimuli"][0]["text"] = "阿叔今日食白粥。"
    payload["stimuli"][1]["text"] = "叔叔執好藥盒返屋企。"
    grounded = Proposal.model_validate(payload)
    validate_stimulus_grounding(grounded, pack)

    payload["stimuli"][1]["role"] = "contrast"
    payload["stimuli"][1]["text"] = "店員將八盒茶同一百盒餅分開擺好。"
    number_word_carrier = Proposal.model_validate(payload)
    validate_stimulus_grounding(number_word_carrier, pack)

    payload["stimuli"][1]["text"] = "店員將兩盒茶同三盒餅分開擺好。"
    ungrounded = Proposal.model_validate(payload)
    with pytest.raises(ValueError, match="reference_unit_missing_from_tts_text"):
        validate_stimulus_grounding(ungrounded, pack)


def test_stimulus_grounding_uses_tts_text_not_cer_equivalence_rewrites(
    tmp_path: Path,
) -> None:
    spec = protocol(tmp_path)
    pack_payload = evidence(spec).model_dump(mode="json")
    cluster = pack_payload["error_clusters"][0]
    cluster["level"] = "tone"
    cluster["operation"] = "substitution"
    cluster["reference_unit"] = "6"
    cluster["hypothesis_unit"] = "4"
    tone_pack = EvidencePack.model_validate(pack_payload)

    proposal_payload = proposal(tone_pack).model_dump(mode="json")
    # Grounding treats hypothesis units as comparator carriers, not as desired
    # carriers for the primary error itself.  This isolated unit test therefore
    # marks the supplied tone cluster as the non-primary comparator.
    proposal_payload["target_evidence_ids"] = ["test-primary"]
    proposal_payload["stimuli"][0]["role"] = "contrast"
    proposal_payload["stimuli"][0]["text"] = "今日有事冇事。"
    proposal_payload["stimuli"][1]["text"] = "件事今日處理好。"
    grounded = Proposal.model_validate(proposal_payload)
    with pytest.raises(ValueError, match="comparator_unit_missing_from_tts_text"):
        validate_stimulus_grounding(grounded, tone_pack)


def test_proposal_rejects_non_tts_actionable_primary_and_stimulus(
    tmp_path: Path,
) -> None:
    spec = protocol(tmp_path)
    pack_payload = evidence(spec).model_dump(mode="json")
    pack_payload["error_clusters"][0].update(
        {
            "tts_actionable": False,
            "tts_actionability_reasons": [
                "character_substitution_is_homophonic_in_isolated_jyutping"
            ],
        }
    )
    non_actionable_primary_pack = EvidencePack.model_validate(pack_payload)
    with pytest.raises(ValueError, match="non-TTS-actionable primary evidence"):
        validate_proposal(
            proposal(non_actionable_primary_pack), non_actionable_primary_pack, spec
        )

    actionable_pack = evidence(spec)
    non_actionable_stimulus_pack = add_cluster(
        actionable_pack,
        evidence_id="e2",
        targetable=True,
        tts_actionable=False,
    )
    payload = proposal(non_actionable_stimulus_pack).model_dump(mode="json")
    payload["stimuli"][0]["evidence_ids"].append("e2")
    invalid = Proposal.model_validate(payload)
    with pytest.raises(ValueError, match="stimuli cite non-TTS-actionable clusters"):
        validate_proposal(invalid, non_actionable_stimulus_pack, spec)


def test_non_targetable_cluster_is_rejected_even_as_protection(
    tmp_path: Path,
) -> None:
    spec = protocol(tmp_path)
    pack = add_cluster(
        evidence(spec),
        evidence_id="e2",
        reference_unit="t",
        hypothesis_unit="p",
        targetable=False,
        tts_actionable=True,
    )
    payload = proposal(pack).model_dump(mode="json")
    payload["stimuli"][1].update({"role": "protection", "evidence_ids": ["e2"]})
    payload["expected_outcomes"].append(protection_outcome("e2"))
    invalid = Proposal.model_validate(payload)
    with pytest.raises(ValueError, match="stimuli cite non-targetable clusters"):
        validate_proposal(invalid, pack, spec)


def test_contrast_requires_observed_same_reference_comparator_and_outcome(
    tmp_path: Path,
) -> None:
    spec = protocol(tmp_path)
    base_pack = evidence(spec)
    missing_payload = proposal(base_pack).model_dump(mode="json")
    missing_payload["stimuli"][1]["role"] = "contrast"
    missing = Proposal.model_validate(missing_payload)
    with pytest.raises(ValueError, match="missing_observed_comparator"):
        validate_proposal(missing, base_pack, spec)

    wrong_family_pack = add_cluster(
        base_pack,
        evidence_id="e2",
        reference_unit="t",
        hypothesis_unit="p",
    )
    wrong_family_payload = proposal(wrong_family_pack).model_dump(mode="json")
    wrong_family_payload["stimuli"][1].update(
        {"role": "contrast", "evidence_ids": ["e1", "e2"]}
    )
    wrong_family_payload["expected_outcomes"].append(counterbalance_outcome("e2"))
    wrong_family = Proposal.model_validate(wrong_family_payload)
    with pytest.raises(ValueError, match="comparator_not_same_reference_family"):
        validate_proposal(wrong_family, wrong_family_pack, spec)

    valid_family_pack = add_cluster(base_pack, evidence_id="e2")
    missing_outcome_payload = proposal(valid_family_pack).model_dump(mode="json")
    missing_outcome_payload["stimuli"][1].update(
        {"role": "contrast", "evidence_ids": ["e1", "e2"]}
    )
    missing_outcome = Proposal.model_validate(missing_outcome_payload)
    with pytest.raises(
        ValueError,
        match="counterbalance_outcomes_do_not_match_contrast_clusters",
    ):
        validate_proposal(missing_outcome, valid_family_pack, spec)

    valid_payload = missing_outcome_payload
    valid_payload["expected_outcomes"].append(counterbalance_outcome("e2"))
    valid = Proposal.model_validate(valid_payload)
    validate_proposal(valid, valid_family_pack, spec)


def test_protection_cannot_relabel_primary_and_requires_observed_regression(
    tmp_path: Path,
) -> None:
    spec = protocol(tmp_path)
    base_pack = evidence(spec)
    primary_payload = proposal(base_pack).model_dump(mode="json")
    primary_payload["stimuli"][1]["role"] = "protection"
    primary = Proposal.model_validate(primary_payload)
    with pytest.raises(ValueError, match="cannot relabel a primary target"):
        validate_proposal(primary, base_pack, spec)

    stable_pack = add_cluster(
        base_pack,
        evidence_id="e2",
        reference_unit="t",
        hypothesis_unit="p",
    )
    stable_payload = proposal(stable_pack).model_dump(mode="json")
    stable_payload["stimuli"][1].update({"role": "protection", "evidence_ids": ["e2"]})
    stable_payload["expected_outcomes"].append(protection_outcome("e2"))
    stable = Proposal.model_validate(stable_payload)
    with pytest.raises(ValueError, match="lack an observed easy/medium adaptation regression"):
        validate_proposal(stable, stable_pack, spec)

    regressed_pack = add_cluster(
        base_pack,
        evidence_id="e2",
        reference_unit="t",
        hypothesis_unit="p",
        zero_shot_error_count=1,
    )
    regressed_payload = proposal(regressed_pack).model_dump(mode="json")
    regressed_payload["stimuli"][1].update(
        {"role": "protection", "evidence_ids": ["e2"]}
    )
    regressed_payload["expected_outcomes"].append(protection_outcome("e2"))
    regressed = Proposal.model_validate(regressed_payload)
    validate_proposal(regressed, regressed_pack, spec)


def test_grounding_enforces_threshold_edge_exposure(
    tmp_path: Path,
) -> None:
    spec = protocol(tmp_path)

    edge_payload = evidence(spec).model_dump(mode="json")
    selection = edge_payload["data_scope"]["cluster_selection"]
    selection.update(
        {
            "minimum_current_error_count": 2,
            "minimum_patient_count": 2,
            "minimum_content_count": 2,
        }
    )
    edge_payload["action_contract"]["proposal_constraints"][
        "threshold_edge_max_carrier_exposures"
    ] = 7
    edge_pack = EvidencePack.model_validate(edge_payload)
    edge_proposal = proposal(edge_pack)
    with pytest.raises(ValueError, match=r"threshold_edge_carrier_exposures=8/7"):
        validate_stimulus_grounding(edge_proposal, edge_pack)


@pytest.mark.parametrize("missing_measure", ["easy_pooled_cer", "medium_pooled_cer"])
def test_expected_outcomes_require_easy_and_medium_safety(
    tmp_path: Path,
    missing_measure: str,
) -> None:
    spec = protocol(tmp_path)
    pack = evidence(spec)
    payload = proposal(pack).model_dump(mode="json")
    payload["expected_outcomes"] = [
        item
        for item in payload["expected_outcomes"]
        if item["measure"] != missing_measure
    ]
    invalid = Proposal.model_validate(payload)
    with pytest.raises(
        ValueError, match="easy_and_medium_safety_outcomes_are_required"
    ):
        validate_proposal(invalid, pack, spec)


def test_store_rejects_draft_skip_and_overwrite(tmp_path: Path) -> None:
    draft_root = tmp_path / "draft"
    draft_root.mkdir()
    with pytest.raises(ValueError, match="frozen"):
        RunStore.initialize(protocol(draft_root, status="draft"), tmp_path / "bad-run")

    frozen_root = tmp_path / "frozen"
    frozen_root.mkdir()
    spec = protocol(frozen_root)
    store = RunStore.initialize(spec, tmp_path / "run")
    evidence_file = tmp_path / "evidence.json"
    evidence_file.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires assessed"):
        store.advance("proposed", {"proposal": evidence_file})
    with pytest.raises(ValidationError):
        store.advance("assessed", {"evidence": evidence_file})

    checkpoint_dir = tmp_path / "checkpoint-2"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    evidence_file.write_text(evidence(spec).model_dump_json(), encoding="utf-8")
    event = store.advance(
        "assessed", {"evidence": evidence_file, "diagnostic_checkpoint": checkpoint_dir}
    )
    assert event.phase == "assessed"
    bad_proposal = tmp_path / "bad_proposal.json"
    bad_proposal.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        store.advance(
            "proposed", {name: bad_proposal for name in REQUIRED_ARTIFACTS["proposed"]}
        )
    assert store.current() == event
    store.verify_integrity()
    (checkpoint_dir / "adapter_config.json").write_text(
        '{"changed": true}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="artifact changed"):
        store.verify_integrity()
    with pytest.raises(ValueError, match="cannot advance"):
        RunStore._next(event.model_copy(update={"phase": "completed"}), 1)

    immutable = tmp_path / "immutable.json"
    write_immutable_json(immutable, {"value": 1})
    with pytest.raises(FileExistsError):
        write_immutable_json(immutable, {"value": 2})
