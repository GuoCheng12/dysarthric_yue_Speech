from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from therapist_harness.evidence_pack import build_evidence_pack
from therapist_harness.materialize import (
    REQUIRED_ASSIGNMENT_FIELDS,
    build_materialization_sidecar,
    materialize_proposal,
)
from therapist_harness.schema import (
    AgentConfig,
    ArtifactRef,
    AssessmentClaim,
    EvaluationConfig,
    EvidencePack,
    ExpectedOutcome,
    LoRAConfig,
    Proposal,
    ProposalConstraints,
    RunProtocol,
    Stimulus,
    canonical_sha256,
)
from therapist_harness.store import sha256_directory, sha256_file
from therapist_harness.text import identity_keys


def _write_artifact(
    path: Path, content: str, *, rows: int | None = None
) -> ArtifactRef:
    path.write_text(content, encoding="utf-8")
    return ArtifactRef(path=str(path), sha256=sha256_file(path), expected_rows=rows)


def _jsonl_rows(texts: list[str], speaker_ids: list[str], split: str) -> list[dict]:
    count = max(len(texts), len(speaker_ids))
    return [
        {
            "utt_id": f"{split}-u{index}",
            "speaker_id": speaker_ids[index % len(speaker_ids)],
            "prompt_id": f"{split}-p{index % len(texts)}",
            "clean_gt": texts[index % len(texts)],
            "zero_shot_bucket": "hard",
        }
        for index in range(count)
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> ArtifactRef:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return _write_artifact(path, content, rows=len(rows))


def _protocol(
    tmp_path: Path,
    *,
    synthetic_count: int = 1,
    unique_count: int = 1,
    seed: int = 4,
    train_texts: list[str] | None = None,
    dev_texts: list[str] | None = None,
    test_texts: list[str] | None = None,
    speaker_ids: list[str] | None = None,
) -> RunProtocol:
    train_texts = train_texts or ["火車到站"]
    dev_texts = dev_texts or ["記得食藥"]
    test_texts = test_texts or ["聽日覆診"]
    speaker_ids = speaker_ids or ["spk-a"]
    real_replay_count = len(_jsonl_rows(train_texts, speaker_ids, "train"))
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(exist_ok=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")

    train_rows = _jsonl_rows(train_texts, speaker_ids, "train")
    for row in train_rows:
        row["audio"] = str(tmp_path / f"{row['speaker_id']}.wav")
    real_train = _write_jsonl(
        tmp_path / "train.jsonl",
        train_rows,
    )
    feedback = _write_jsonl(
        tmp_path / "dev.jsonl",
        _jsonl_rows(dev_texts, [speaker_ids[0]], "dev"),
    )
    sealed_test = _write_jsonl(
        tmp_path / "test.jsonl",
        _jsonl_rows(test_texts, [speaker_ids[0]], "test"),
    )

    references = []
    for speaker_id in speaker_ids:
        wav = tmp_path / f"{speaker_id}.wav"
        wav.write_bytes(f"reference-{speaker_id}".encode())
        references.append(
            {
                "speaker_id": speaker_id,
                "difficulty_group": "hard",
                "reference_text": next(
                    row["clean_gt"]
                    for row in train_rows
                    if row["speaker_id"] == speaker_id
                ),
                "reference_wav": str(wav),
                "reference_wav_sha256": sha256_file(wav),
            }
        )
    profile_payload = {
        "all_balanced": {"difficulty_groups": ["easy", "medium", "hard"]}
    }
    reference_payload = {
        "schema_version": "1",
        "source_manifest_path": str(tmp_path / "source.json"),
        "source_manifest_sha256": "d" * 64,
        "profiles": profile_payload,
        "references": references,
    }
    reference_manifest = _write_artifact(
        tmp_path / "references.json",
        json.dumps(reference_payload, ensure_ascii=False, sort_keys=True) + "\n",
    )

    return RunProtocol(
        schema_version="1",
        protocol_status="frozen",
        run_id="toy_materialize",
        setting_id="setting_d",
        round_count=4,
        base_model_dir=str(model_dir),
        base_model_fingerprint=_write_artifact(model_dir / "model.index.json", "{}\n"),
        initial_adapter=ArtifactRef(
            path=str(adapter_dir),
            sha256=sha256_directory(adapter_dir),
            kind="directory",
        ),
        initial_adapter_dev_predictions=_write_artifact(
            tmp_path / "m0_predictions.csv", "m0\n"
        ),
        initial_adapter_prediction_column="prediction",
        zero_shot_dev_predictions=_write_artifact(
            tmp_path / "zero_shot_predictions.csv", "zero\n"
        ),
        zero_shot_prediction_column="prediction",
        real_train=real_train,
        feedback_split=feedback,
        sealed_test_split=sealed_test,
        speech_generator=_write_artifact(tmp_path / "speech_generator.pt", "x"),
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
        synthetic_count_per_round=synthetic_count,
        unique_stimulus_count_per_round=unique_count,
        real_replay_count_per_round=real_replay_count,
        speaker_profiles=profile_payload,
        proposal_constraints=ProposalConstraints(
            fixed_speaker_profile_id="all_balanced",
            max_primary_targets=1,
            max_copies_per_stimulus=synthetic_count,
            max_surface_characters_per_stimulus=32,
            threshold_edge_max_linked_copies=synthetic_count,
            threshold_edge_max_unique_stimuli=unique_count,
            threshold_edge_max_carrier_exposures=synthetic_count * 8,
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
            optimizer_steps_per_round=(real_replay_count + synthetic_count + 3) // 4,
            training_passes_per_round=1,
            save_at_round_end=True,
            scheduler="linear",
            reset_scheduler_each_round=True,
            seed=seed,
        ),
        evaluation=EvaluationConfig(
            language="Cantonese",
            batch_size=2,
            critical_cer_threshold=0.5,
            metrics=["pooled_cer"],
        ),
        skill_update_rounds=[],
    )


def _evidence(protocol: RunProtocol) -> EvidencePack:
    rows = [
        {
            **json.loads(line),
            "prediction": "錯",
        }
        for line in Path(protocol.feedback_split.path).read_text("utf-8").splitlines()
    ]
    evidence = build_evidence_pack(
        protocol,
        round_index=1,
        current_rows=rows,
        current_prediction_column="prediction",
        current_checkpoint_sha256=protocol.initial_adapter.sha256,
        current_predictions_sha256=protocol.initial_adapter_dev_predictions.sha256,
        m0_rows=rows,
        m0_prediction_column="prediction",
        m0_predictions_sha256=protocol.initial_adapter_dev_predictions.sha256,
        zero_shot_rows=rows,
        zero_shot_prediction_column="prediction",
        zero_shot_predictions_sha256=protocol.zero_shot_dev_predictions.sha256,
    )
    payload = evidence.model_dump(mode="json")
    payload["error_clusters"][0]["targetable"] = True
    payload["error_clusters"][0]["selection_reasons"] = [
        "materialization_fixture_target"
    ]
    payload["error_clusters"][0]["tts_actionable"] = True
    payload["error_clusters"][0]["tts_actionability_reasons"] = [
        "pronounceable_reference_character_can_be_rehearsed"
    ]
    payload["error_clusters"][0]["operation"] = "deletion"
    payload["error_clusters"][0]["level"] = "character"
    payload["error_clusters"][0]["reference_unit"] = "叔"
    payload["error_clusters"][0]["hypothesis_unit"] = ""
    payload["error_clusters"][0]["reference_surface_units"] = ["叔"]
    payload["error_clusters"][0]["hypothesis_surface_units"] = []
    return EvidencePack.model_validate(payload)


def _proposal(
    evidence: EvidencePack, texts_and_copies: list[tuple[str, int]]
) -> Proposal:
    evidence_id = evidence.error_clusters[0].evidence_id
    return Proposal(
        schema_version="3",
        round_index=evidence.round_index,
        evidence_sha256=canonical_sha256(evidence),
        assessment_claims=[
            AssessmentClaim(claim="Persistent error", evidence_ids=[evidence_id])
        ],
        hypothesis="The fixed intervention should reduce this output error.",
        target_evidence_ids=[evidence_id],
        stimuli=[
            Stimulus(
                stimulus_id=f"s{index}",
                text=text,
                role="target",
                evidence_ids=[evidence_id],
                speaker_profile_id="all_balanced",
                copies=copies,
                rationale="Exercise the cited output position.",
            )
            for index, (text, copies) in enumerate(texts_and_copies, start=1)
        ],
        expected_outcomes=[
            ExpectedOutcome(
                outcome_id="primary-target",
                category="primary",
                subject_evidence_id=evidence_id,
                measure="cluster_error_count",
                desired_direction="decrease",
                rationale="The selected error should decrease.",
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
        expected_response="Lower target errors.",
        risks=["Heterogeneous response."],
    )


def _materialize(
    tmp_path: Path, proposal: Proposal, evidence: EvidencePack, protocol: RunProtocol
) -> list[dict[str, str | int]]:
    output_root = tmp_path / "outputs"
    output_root.mkdir(exist_ok=True)
    return materialize_proposal(proposal, evidence, protocol, output_root=output_root)


@pytest.mark.parametrize(
    ("target", "existing", "layer"),
    [
        ("阿叔你好食白粥。", "阿叔你好食白粥。", "raw_clean"),
        ("阿叔行到后門食粥。", "阿叔行到後門食粥。", "text_normalize"),
        ("阿叔話件事要食粥。", "阿叔話件是要食粥。", "jyutping"),
        ("阿叔講詩食白粥。", "阿叔講試食白粥。", "detoned_jyutping"),
    ],
)
def test_rejects_overlap_at_every_identity_layer(
    tmp_path: Path, target: str, existing: str, layer: str
) -> None:
    protocol = _protocol(tmp_path, train_texts=[existing])
    evidence = _evidence(protocol)
    proposal = _proposal(evidence, [(target, 1)])
    with pytest.raises(ValueError) as caught:
        _materialize(tmp_path, proposal, evidence, protocol)
    assert f"s1:{layer}=train" in str(caught.value)


def test_materialize_without_separate_sealed_test(tmp_path: Path) -> None:
    from therapist_harness.store import RunStore

    original = _protocol(tmp_path)
    protocol = original.model_copy(update={"sealed_test_split": None})
    Path(original.sealed_test_split.path).unlink()
    RunStore.initialize(protocol, tmp_path / "no_sealed_test_run")
    evidence = _evidence(protocol)
    proposal = _proposal(evidence, [("阿叔你好食白粥。", 1)])
    assignments = _materialize(tmp_path, proposal, evidence, protocol)
    assert len(assignments) == 1


def test_rejects_duplicate_proposal_content_after_normalization(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path, synthetic_count=2, unique_count=2)
    evidence = _evidence(protocol)
    proposal = _proposal(
        evidence,
        [("阿叔行到後門食粥。", 1), ("阿叔行到后門食粥。", 1)],
    )
    with pytest.raises(ValueError, match="stimulus:s1"):
        _materialize(tmp_path, proposal, evidence, protocol)


def test_unknown_jyutping_keeps_surface_identity(tmp_path: Path) -> None:
    assert identity_keys("𠮷")["jyutping"] == "UNK:𠮷"
    assert identity_keys("🦆")["jyutping"] == "UNK:🦆"
    protocol = _protocol(tmp_path, train_texts=["阿叔見到🦆字食粥。"])
    evidence = _evidence(protocol)
    assignments = _materialize(
        tmp_path,
        _proposal(evidence, [("阿叔見到𠮷字食粥。", 1)]),
        evidence,
        protocol,
    )
    assert len(assignments) == 1


def test_reference_audio_is_bound_by_hash(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    evidence = _evidence(protocol)
    (tmp_path / "spk-a.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="reference_wav SHA-256 mismatch"):
        _materialize(
            tmp_path,
            _proposal(evidence, [("阿叔請幫我開門。", 1)]),
            evidence,
            protocol,
        )


def test_deterministic_rotation_exact_count_and_no_wav_writes(tmp_path: Path) -> None:
    speaker_ids = ["spk-a", "spk-b", "spk-c"]
    protocol = _protocol(
        tmp_path,
        synthetic_count=5,
        unique_count=2,
        seed=4,
        speaker_ids=speaker_ids,
    )
    evidence = _evidence(protocol)
    proposal = _proposal(
        evidence,
        [("阿叔我想飲暖水。", 3), ("請阿叔幫我開窗。", 2)],
    )
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    random.seed(923)
    random_state = random.getstate()

    first = materialize_proposal(proposal, evidence, protocol, output_root=output_root)
    second = materialize_proposal(proposal, evidence, protocol, output_root=output_root)
    assert first == second
    assert len(first) == 5
    assert all(set(row) == REQUIRED_ASSIGNMENT_FIELDS for row in first)
    speakers = [Path(row["reference_wav"]).stem for row in first]
    assert set(speakers[:3]) == set(speaker_ids)
    assert speakers[3:] == speakers[:2]
    assert len({row["sample_id"] for row in first}) == 5
    assert len({row["output_wav"] for row in first}) == 5
    assert len({row["seed"] for row in first}) == 5
    assert all(0 <= int(row["seed"]) < 2**63 for row in first)
    assert list(output_root.iterdir()) == []
    assert random.getstate() == random_state
    build_materialization_sidecar(proposal, evidence, protocol, first)
    reassigned = [dict(row) for row in first]
    for field in ("reference_text", "reference_wav", "reference_wav_sha256"):
        reassigned[0][field] = first[1][field]
    with pytest.raises(ValueError, match="reference differs from frozen allocation"):
        build_materialization_sidecar(proposal, evidence, protocol, reassigned)


@pytest.mark.parametrize("mutation", ["audio", "gt", "speaker", "utt_id"])
def test_reference_must_match_one_exact_train_row(
    tmp_path: Path, mutation: str
) -> None:
    protocol = _protocol(tmp_path, speaker_ids=["spk-a", "spk-b"])
    path = Path(protocol.frozen_reference_manifest.path)
    payload = json.loads(path.read_text("utf-8"))
    record = payload["references"][0]
    if mutation == "audio":
        wav = tmp_path / "heldout.wav"
        wav.write_bytes(b"heldout reference")
        record["reference_wav"] = str(wav)
        record["reference_wav_sha256"] = sha256_file(wav)
    elif mutation == "gt":
        record["reference_text"] = "另一句未對齊嘅文字"
    elif mutation == "speaker":
        other = payload["references"][1]
        record["reference_wav"] = other["reference_wav"]
        record["reference_wav_sha256"] = other["reference_wav_sha256"]
    else:
        record["utt_id"] = "train-u1"
    updated = _write_artifact(path, json.dumps(payload, ensure_ascii=False))
    protocol = protocol.model_copy(update={"frozen_reference_manifest": updated})
    evidence = _evidence(protocol)
    with pytest.raises(ValueError, match="reference .*frozen train"):
        _materialize(
            tmp_path, _proposal(evidence, [("阿叔請幫我開門。", 1)]), evidence, protocol
        )


def test_reference_accepts_matching_optional_utterance_id(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    path = Path(protocol.frozen_reference_manifest.path)
    payload = json.loads(path.read_text("utf-8"))
    payload["references"][0]["utt_id"] = "train-u0"
    updated = _write_artifact(path, json.dumps(payload, ensure_ascii=False))
    protocol = protocol.model_copy(update={"frozen_reference_manifest": updated})
    evidence = _evidence(protocol)
    assignments = _materialize(
        tmp_path, _proposal(evidence, [("阿叔請幫我開門。", 1)]), evidence, protocol
    )
    assert len(assignments) == 1


def test_roles_balance_reference_strata_independent_of_stimulus_order(
    tmp_path: Path,
) -> None:
    protocol = _protocol(
        tmp_path,
        synthetic_count=24,
        unique_count=3,
        seed=42,
        speaker_ids=["easy-a", "easy-b", "medium-a", "medium-b", "hard-a", "hard-b"],
    )
    manifest_path = Path(protocol.frozen_reference_manifest.path)
    payload = json.loads(manifest_path.read_text("utf-8"))
    for record in payload["references"]:
        record["difficulty_group"] = record["speaker_id"].split("-")[0]
    updated = _write_artifact(manifest_path, json.dumps(payload, ensure_ascii=False))
    protocol = protocol.model_copy(update={"frozen_reference_manifest": updated})
    evidence = _evidence(protocol)
    evidence_payload = evidence.model_dump(mode="json")
    cluster = json.loads(json.dumps(evidence_payload["error_clusters"][0]))
    cluster["evidence_id"] = "protection-fixture"
    cluster["reference_unit"] = "門"
    cluster["reference_surface_units"] = ["門"]
    cluster["selection_reasons"] = ["regression_vs_zero_shot"]
    for snapshot in ("current", "m0"):
        observation = cluster["observations"][snapshot]
        observation["by_difficulty"]["easy"], observation["by_difficulty"]["hard"] = (
            observation["by_difficulty"]["hard"], observation["by_difficulty"]["easy"]
        )
    zero_shot = cluster["observations"]["zero_shot"]
    zero_shot.update(
        error_count=0,
        rate=0.0,
        patient_count=0,
        content_count=0,
        alignment_uncertainty=0.0,
    )
    for group in zero_shot["by_difficulty"].values():
        group.update(error_count=0, rate=0.0)
    evidence_payload["error_clusters"].append(cluster)
    evidence = EvidencePack.model_validate(evidence_payload)
    proposal = _proposal(
        evidence,
        [("阿叔我想飲暖水。", 8), ("請阿叔幫我開窗。", 8), ("請你記得閂大門。", 8)],
    )
    proposal.stimuli[-1] = proposal.stimuli[-1].model_copy(
        update={"role": "protection", "evidence_ids": ["protection-fixture"]}
    )
    proposal.expected_outcomes.append(
        ExpectedOutcome(
            outcome_id="protect-door",
            category="protection",
            subject_evidence_id="protection-fixture",
            measure="cluster_error_count",
            desired_direction="decrease",
            rationale="Rehearse the separate protection carrier.",
        )
    )
    first = _materialize(tmp_path, proposal, evidence, protocol)
    counts: dict[str, Counter[str]] = {"target": Counter(), "protection": Counter()}
    for index, row in enumerate(first):
        role = "target" if index < 16 else "protection"
        counts[role][Path(row["reference_wav"]).stem.split("-")[0]] += 1
    assert counts["target"] == {"easy": 6, "medium": 5, "hard": 5}
    assert counts["protection"] == {"easy": 3, "medium": 3, "hard": 2}

    sidecar = build_materialization_sidecar(proposal, evidence, protocol, first)
    assert (
        Counter(
            sample.reference_difficulty_group
            for sample in sidecar.samples
            if sample.role == "protection"
        )
        == counts["protection"]
    )

    reversed_proposal = proposal.model_copy(
        update={"stimuli": list(reversed(proposal.stimuli))}
    )
    second = _materialize(tmp_path, reversed_proposal, evidence, protocol)
    assert {row["sample_id"]: row["reference_wav"] for row in first} == {
        row["sample_id"]: row["reference_wav"] for row in second
    }


def test_output_root_must_be_absolute_and_preexisting(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    evidence = _evidence(protocol)
    proposal = _proposal(evidence, [("阿叔請幫我開門。", 1)])
    with pytest.raises(ValueError, match="output_root must be absolute"):
        materialize_proposal(proposal, evidence, protocol, output_root=Path("relative"))
    with pytest.raises(FileNotFoundError, match="output_root does not exist"):
        materialize_proposal(
            proposal, evidence, protocol, output_root=tmp_path / "missing"
        )
