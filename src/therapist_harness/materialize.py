"""Deterministically turn one validated Therapist proposal into TTS assignments.

This module is the private boundary between the aggregate, identifier-free agent
contract and the patient reference manifest.  It does not generate audio, choose a
different reference, refill a rejected row, or change the frozen synthetic budget.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .schema import ArtifactRef, EvidencePack, Proposal, RunProtocol, validate_proposal
from .store import sha256_file, verify_artifact
from .text import identity_keys


REQUIRED_ASSIGNMENT_FIELDS = frozenset(
    {
        "sample_id",
        "target_text",
        "reference_text",
        "reference_wav",
        "reference_wav_sha256",
        "output_wav",
        "seed",
    }
)


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _load_split(reference: ArtifactRef, split: str) -> tuple[tuple[str, ...], set[str]]:
    verify_artifact(reference)
    texts: list[str] = []
    speakers: set[str] = set()
    utterance_ids: set[str] = set()
    with Path(reference.path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"{split} line {line_number}: blank rows are forbidden"
                )
            try:
                row = json.loads(line, object_pairs_hook=_json_object)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"{split} line {line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(f"{split} line {line_number}: row must be an object")
            required = ("utt_id", "speaker_id", "prompt_id", "clean_gt")
            missing = [field for field in required if field not in row]
            if missing:
                raise ValueError(
                    f"{split} line {line_number}: missing fields {missing}"
                )
            values = {field: row[field] for field in required}
            if any(
                not isinstance(value, str) or not value.strip()
                for value in values.values()
            ):
                raise ValueError(
                    f"{split} line {line_number}: required fields must be "
                    "non-empty strings"
                )
            utterance_id = values["utt_id"].strip()
            if utterance_id in utterance_ids:
                raise ValueError(f"{split}: duplicate utt_id {utterance_id}")
            utterance_ids.add(utterance_id)
            speakers.add(values["speaker_id"].strip())
            texts.append(values["clean_gt"])
    return tuple(texts), speakers


def _load_frozen_references(
    protocol: RunProtocol, train_speakers: set[str]
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, Path, str]]]:
    manifest_ref = protocol.frozen_reference_manifest
    verify_artifact(manifest_ref)
    try:
        payload = json.loads(
            Path(manifest_ref.path).read_text("utf-8"),
            object_pairs_hook=_json_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid frozen reference manifest: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError("frozen reference manifest must be an object")
    required_root = {
        "schema_version",
        "source_manifest_path",
        "source_manifest_sha256",
        "profiles",
        "references",
    }
    if set(payload) != required_root or payload["schema_version"] != "1":
        raise ValueError("frozen reference manifest has an unexpected schema")
    declared_profiles = {
        profile_id: rule.model_dump(mode="json")
        for profile_id, rule in protocol.speaker_profiles.items()
    }
    if payload["profiles"] != declared_profiles:
        raise ValueError("reference-manifest profiles differ from the frozen protocol")
    records = payload["references"]
    if not isinstance(records, list) or not records:
        raise ValueError("reference manifest must contain references")

    references: dict[str, tuple[str, Path, str]] = {}
    speaker_groups: dict[str, str] = {}
    seen_paths: set[Path] = set()
    required_record = {
        "speaker_id",
        "difficulty_group",
        "reference_text",
        "reference_wav",
        "reference_wav_sha256",
    }
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or set(record) != required_record:
            raise ValueError(f"reference record {index} has an unexpected schema")
        speaker_id = record["speaker_id"]
        difficulty = record["difficulty_group"]
        reference_text = record["reference_text"]
        raw_path = record["reference_wav"]
        expected_hash = record["reference_wav_sha256"]
        if (
            not isinstance(speaker_id, str)
            or not speaker_id.strip()
            or speaker_id != speaker_id.strip()
        ):
            raise ValueError(f"reference record {index} has an invalid speaker_id")
        if speaker_id in references:
            raise ValueError(f"duplicate reference speaker: {speaker_id}")
        if speaker_id not in train_speakers:
            raise ValueError(
                f"reference speaker is absent from frozen train: {speaker_id}"
            )
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(
                f"reference record {index} has an invalid difficulty_group"
            )
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise ValueError(f"reference record {index} has invalid reference_text")
        if not isinstance(raw_path, str):
            raise TypeError(f"reference record {index} reference_wav must be a string")
        reference_wav = Path(raw_path)
        if not reference_wav.is_absolute() or ".." in reference_wav.parts:
            raise ValueError(
                f"reference_wav for {speaker_id} must be absolute and traversal-free"
            )
        if reference_wav in seen_paths:
            raise ValueError(f"duplicate reference_wav: {reference_wav}")
        if not reference_wav.is_file():
            raise FileNotFoundError(reference_wav)
        if (
            not isinstance(expected_hash, str)
            or sha256_file(reference_wav) != expected_hash
        ):
            raise ValueError(f"reference_wav SHA-256 mismatch for {speaker_id}")
        references[speaker_id] = (reference_text, reference_wav, expected_hash)
        speaker_groups[speaker_id] = difficulty
        seen_paths.add(reference_wav)

    profiles: dict[str, tuple[str, ...]] = {}
    for profile_id, rule in protocol.speaker_profiles.items():
        allowed = set(rule.difficulty_groups)
        members = tuple(
            sorted(
                speaker_id
                for speaker_id, difficulty in speaker_groups.items()
                if difficulty in allowed
            )
        )
        if not members:
            raise ValueError(f"frozen speaker profile is empty: {profile_id}")
        profiles[profile_id] = members
    return profiles, references


def _validate_content_isolation(
    proposal: Proposal,
    *,
    train_gt_texts: Sequence[str],
    dev_gt_texts: Sequence[str],
    test_gt_texts: Sequence[str],
) -> None:
    split_texts = {
        "train": train_gt_texts,
        "dev": dev_gt_texts,
        "test": test_gt_texts,
    }
    index: dict[str, dict[str, set[str]]] = {
        layer: defaultdict(set)
        for layer in (
            "raw_clean",
            "text_normalize",
            "jyutping",
            "detoned_jyutping",
        )
    }
    for split, texts in split_texts.items():
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError(f"{split}_gt_texts must be a sequence of strings")
        for row_index, text in enumerate(dict.fromkeys(texts)):
            keys = identity_keys(text, source_label=f"{split}[{row_index}]")
            for layer, key in keys.items():
                index[layer][key].add(split)

    collisions: list[str] = []
    proposal_index: dict[str, dict[str, str]] = {layer: {} for layer in index}
    for stimulus in proposal.stimuli:
        keys = identity_keys(
            stimulus.text, source_label=f"stimulus {stimulus.stimulus_id}"
        )
        for layer, key in keys.items():
            matching_splits = index[layer].get(key, set())
            if matching_splits:
                collisions.append(
                    f"{stimulus.stimulus_id}:{layer}="
                    + ",".join(sorted(matching_splits))
                )
            previous_stimulus = proposal_index[layer].get(key)
            if previous_stimulus is not None:
                collisions.append(
                    f"{stimulus.stimulus_id}:{layer}=stimulus:{previous_stimulus}"
                )
            proposal_index[layer][key] = stimulus.stimulus_id
    if collisions:
        raise ValueError(
            "proposal content overlaps frozen GT identities: " + "; ".join(collisions)
        )


def _assignment_seed(
    protocol_seed: int, round_index: int, assignment_index: int, sample_id: str
) -> int:
    payload = (
        f"{protocol_seed}\0{round_index}\0{assignment_index}\0{sample_id}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def materialize_proposal(
    proposal: Proposal,
    evidence: EvidencePack,
    protocol: RunProtocol,
    *,
    output_root: str | Path,
) -> list[dict[str, str | int]]:
    """Return one exact, deterministic Speech Generator assignment list.

    The result contains exactly the seven fields consumed by
    ``synthesis/cosyvoice3_patient_sft/synthesize_manifest.py``.  No filesystem
    content is created or modified.
    """

    validate_proposal(proposal, evidence, protocol)
    if protocol.protocol_status != "frozen":
        raise ValueError("only a frozen protocol can be materialized")
    if proposal.round_index > protocol.round_count:
        raise ValueError("proposal round exceeds the frozen protocol")

    output_dir = Path(output_root)
    if not output_dir.is_absolute() or ".." in output_dir.parts:
        raise ValueError("output_root must be absolute and traversal-free")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"output_root does not exist: {output_dir}")

    train_gt_texts, train_speakers = _load_split(protocol.real_train, "train")
    dev_gt_texts, _ = _load_split(protocol.feedback_split, "dev")
    test_gt_texts, _ = _load_split(protocol.sealed_test_split, "test")
    validated_profiles, validated_references = _load_frozen_references(
        protocol, train_speakers
    )
    _validate_content_isolation(
        proposal,
        train_gt_texts=train_gt_texts,
        dev_gt_texts=dev_gt_texts,
        test_gt_texts=test_gt_texts,
    )

    profile_uses: dict[str, int] = defaultdict(int)
    assignments: list[dict[str, str | int]] = []
    for stimulus in proposal.stimuli:
        speaker_ids = validated_profiles[stimulus.speaker_profile_id]
        initial_offset = (protocol.lora.seed + proposal.round_index - 1) % len(
            speaker_ids
        )
        for copy_index in range(1, stimulus.copies + 1):
            profile_use = profile_uses[stimulus.speaker_profile_id]
            speaker_id = speaker_ids[(initial_offset + profile_use) % len(speaker_ids)]
            profile_uses[stimulus.speaker_profile_id] += 1
            reference_text, reference_wav, reference_wav_sha256 = validated_references[
                speaker_id
            ]

            sample_id = (
                f"{protocol.run_id}.r{proposal.round_index:03d}."
                f"{stimulus.stimulus_id}.c{copy_index:04d}"
            )
            output_wav = output_dir / f"{sample_id}.wav"
            if os.path.lexists(output_wav):
                raise FileExistsError(f"output_wav already exists: {output_wav}")
            assignment_index = len(assignments)
            assignments.append(
                {
                    "sample_id": sample_id,
                    "target_text": stimulus.text,
                    "reference_text": reference_text,
                    "reference_wav": str(reference_wav),
                    "reference_wav_sha256": reference_wav_sha256,
                    "output_wav": str(output_wav),
                    "seed": _assignment_seed(
                        protocol.lora.seed,
                        proposal.round_index,
                        assignment_index,
                        sample_id,
                    ),
                }
            )

    if len(assignments) != protocol.synthetic_count_per_round:
        raise RuntimeError("materialized assignment count differs from frozen budget")
    if any(set(item) != REQUIRED_ASSIGNMENT_FIELDS for item in assignments):
        raise RuntimeError("materialized assignment does not match synthesis contract")
    return assignments


__all__ = [
    "REQUIRED_ASSIGNMENT_FIELDS",
    "materialize_proposal",
]
