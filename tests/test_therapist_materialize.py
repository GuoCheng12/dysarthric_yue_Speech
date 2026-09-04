from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from therapist_harness.materialize import (
    REQUIRED_ASSIGNMENT_FIELDS,
    materialize_proposal,
)
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
)
from therapist_harness.store import sha256_file
from therapist_harness.text import identity_keys


EVIDENCE_NOTE = (
    "Phone-level fields describe ASR output errors at GT phonological positions; "
    "they are not patient articulation rules."
)


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

    real_train = _write_jsonl(
        tmp_path / "train.jsonl",
        _jsonl_rows(train_texts, speaker_ids, "train"),
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
                "reference_text": f"參考句子{speaker_id}",
                "reference_wav": str(wav),
                "reference_wav_sha256": sha256_file(wav),
            }
        )
    profile_payload = {"hard_balanced": {"difficulty_groups": ["hard"]}}
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
        initial_adapter=None,
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
            api_key_env="OPENAI_API_KEY",
        ),
        synthetic_count_per_round=synthetic_count,
        unique_stimulus_count_per_round=unique_count,
        real_replay_count_per_round=real_replay_count,
        speaker_profiles=profile_payload,
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


def _empty_group() -> MetricGroup:
    return MetricGroup(
        sample_count=0,
        speaker_count=0,
        total_edits=0,
        total_reference_characters=0,
        mean_utterance_cer=0.0,
        pooled_cer=0.0,
        patient_macro_cer=0.0,
        critical_count=0,
        exact_count=0,
    )


def _evidence(protocol: RunProtocol) -> EvidencePack:
    zero = MetricDelta(
        overall_pooled_cer=0.0,
        easy_pooled_cer=0.0,
        medium_pooled_cer=0.0,
        hard_pooled_cer=0.0,
        patient_macro_cer=0.0,
    )
    transitions = TransitionCounts(
        recovered=0,
        newly_wrong=0,
        preserved_correct=0,
        persistent_wrong=0,
    )
    return EvidencePack(
        schema_version="1",
        run_id=protocol.run_id,
        round_index=1,
        current_checkpoint_sha256="b" * 64,
        feedback_split_sha256=protocol.feedback_split.sha256,
        current_metrics=MetricSet(
            overall=_empty_group(),
            easy=_empty_group(),
            medium=_empty_group(),
            hard=_empty_group(),
        ),
        delta_vs_initial=zero,
        delta_vs_previous=None,
        transitions_vs_initial=transitions,
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
                error_count=1,
                opportunity_count=1,
                error_rate=1.0,
                patient_count=1,
                content_count=1,
                difficulty_distribution={"easy": 0, "medium": 0, "hard": 1},
                previous_rate=None,
                initial_rate=1.0,
                observed_snapshot_count=1,
                bounded_examples=[],
                alignment_uncertainty=0.0,
            )
        ],
        previous_interventions=[],
        action_contract=ActionContract(
            synthetic_count=protocol.synthetic_count_per_round,
            unique_stimulus_count=protocol.unique_stimulus_count_per_round,
            permitted_speaker_profile_ids=protocol.speaker_profile_ids,
        ),
        evidence_note=EVIDENCE_NOTE,
    )


def _proposal(
    evidence: EvidencePack, texts_and_copies: list[tuple[str, int]]
) -> Proposal:
    return Proposal(
        schema_version="1",
        round_index=evidence.round_index,
        evidence_sha256=canonical_sha256(evidence),
        assessment_claims=[
            AssessmentClaim(claim="Persistent error", evidence_ids=["e1"])
        ],
        hypothesis="The fixed intervention should reduce this output error.",
        targets=["e1"],
        stimuli=[
            Stimulus(
                stimulus_id=f"s{index}",
                text=text,
                role="target",
                evidence_ids=["e1"],
                speaker_profile_id="hard_balanced",
                copies=copies,
                rationale="Exercise the cited output position.",
            )
            for index, (text, copies) in enumerate(texts_and_copies, start=1)
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
        ("你好", "你好", "raw_clean"),
        ("后", "後", "text_normalize"),
        ("事", "是", "jyutping"),
        ("詩", "試", "detoned_jyutping"),
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


def test_rejects_duplicate_proposal_content_after_normalization(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path, synthetic_count=2, unique_count=2)
    evidence = _evidence(protocol)
    proposal = _proposal(evidence, [("後", 1), ("后", 1)])
    with pytest.raises(ValueError, match="stimulus:s1"):
        _materialize(tmp_path, proposal, evidence, protocol)


def test_unknown_jyutping_keeps_surface_identity(tmp_path: Path) -> None:
    assert identity_keys("𠮷")["jyutping"] == "UNK:𠮷"
    assert identity_keys("🦆")["jyutping"] == "UNK:🦆"
    protocol = _protocol(tmp_path, train_texts=["🦆"])
    evidence = _evidence(protocol)
    assignments = _materialize(
        tmp_path, _proposal(evidence, [("𠮷", 1)]), evidence, protocol
    )
    assert len(assignments) == 1


def test_reference_audio_is_bound_by_hash(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    evidence = _evidence(protocol)
    (tmp_path / "spk-a.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="reference_wav SHA-256 mismatch"):
        _materialize(
            tmp_path,
            _proposal(evidence, [("請幫我開門", 1)]),
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
    proposal = _proposal(evidence, [("我想飲暖水", 3), ("請幫我開窗", 2)])
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    random.seed(923)
    random_state = random.getstate()

    first = materialize_proposal(proposal, evidence, protocol, output_root=output_root)
    second = materialize_proposal(proposal, evidence, protocol, output_root=output_root)
    assert first == second
    assert len(first) == 5
    assert all(set(row) == REQUIRED_ASSIGNMENT_FIELDS for row in first)
    assert [Path(row["reference_wav"]).stem for row in first] == [
        "spk-b",
        "spk-c",
        "spk-a",
        "spk-b",
        "spk-c",
    ]
    assert len({row["sample_id"] for row in first}) == 5
    assert len({row["output_wav"] for row in first}) == 5
    assert len({row["seed"] for row in first}) == 5
    assert all(0 <= int(row["seed"]) < 2**63 for row in first)
    assert list(output_root.iterdir()) == []
    assert random.getstate() == random_state


def test_output_root_must_be_absolute_and_preexisting(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    evidence = _evidence(protocol)
    proposal = _proposal(evidence, [("請幫我開門", 1)])
    with pytest.raises(ValueError, match="output_root must be absolute"):
        materialize_proposal(proposal, evidence, protocol, output_root=Path("relative"))
    with pytest.raises(FileNotFoundError, match="output_root does not exist"):
        materialize_proposal(
            proposal, evidence, protocol, output_root=tmp_path / "missing"
        )
