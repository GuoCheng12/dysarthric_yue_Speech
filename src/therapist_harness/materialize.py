"""Deterministically turn one validated Therapist proposal into TTS assignments.

This module is the private boundary between the aggregate, identifier-free agent
contract and the patient reference manifest.  It does not generate audio, choose a
different reference, refill a rejected row, or change the frozen synthetic budget.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .grounding import (
    stimulus_cluster_grounding,
    validate_stimulus_grounding,
    whole_proposal_carrier_exposures,
)
from .round_schema import (
    CarrierCount,
    MaterializationSidecar,
    MaterializedSampleIndex,
    PlannedClusterDose,
    PlannedRoleDose,
)
from .schema import ArtifactRef, EvidencePack, Proposal, RunProtocol, validate_proposal
from .schema import canonical_sha256
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


def _load_split(
    reference: ArtifactRef, split: str
) -> tuple[tuple[str, ...], dict[str, dict[str, str]]]:
    verify_artifact(reference)
    texts: list[str] = []
    utterances: dict[str, dict[str, str]] = {}
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
            if split == "train":
                required += ("audio",)
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
            if utterance_id in utterances:
                raise ValueError(f"{split}: duplicate utt_id {utterance_id}")
            utterances[utterance_id] = values
            texts.append(values["clean_gt"])
    return tuple(texts), utterances


def _load_frozen_references(
    protocol: RunProtocol, train_utterances: dict[str, dict[str, str]]
) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, tuple[str, Path, str]]]:
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
    manifest_profiles = payload["profiles"]
    if not isinstance(manifest_profiles, dict) or any(
        manifest_profiles.get(profile_id) != rule
        for profile_id, rule in declared_profiles.items()
    ):
        raise ValueError(
            "reference manifest does not contain the frozen protocol profiles"
        )
    records = payload["references"]
    if not isinstance(records, list) or not records:
        raise ValueError("reference manifest must contain references")

    references: dict[str, tuple[str, Path, str]] = {}
    speaker_groups: dict[str, str] = {}
    train_index: dict[tuple[str, Path, str], set[str]] = defaultdict(set)
    for utt_id, row in train_utterances.items():
        train_index[
            (row["speaker_id"], Path(row["audio"]).resolve(), row["clean_gt"])
        ].add(utt_id)
    train_speakers = {row["speaker_id"] for row in train_utterances.values()}
    seen_paths: set[Path] = set()
    required_record = {
        "speaker_id",
        "difficulty_group",
        "reference_text",
        "reference_wav",
        "reference_wav_sha256",
    }
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or not (
            required_record <= set(record) <= required_record | {"utt_id"}
        ):
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
        matching_utterances = train_index.get(
            (speaker_id, reference_wav.resolve(), reference_text), set()
        )
        if not matching_utterances:
            raise ValueError(
                "reference speaker/audio/GT does not match a frozen train row: "
                f"{speaker_id}"
            )
        if "utt_id" in record and record["utt_id"] not in matching_utterances:
            raise ValueError(
                f"reference utt_id differs from frozen train: {speaker_id}"
            )
        references[speaker_id] = (reference_text, reference_wav, expected_hash)
        speaker_groups[speaker_id] = difficulty
        seen_paths.add(reference_wav)

    profiles: dict[str, dict[str, tuple[str, ...]]] = {}
    for profile_id, rule in protocol.speaker_profiles.items():
        groups = {
            difficulty: tuple(
                sorted(
                    speaker_id
                    for speaker_id, speaker_difficulty in speaker_groups.items()
                    if speaker_difficulty == difficulty
                )
            )
            for difficulty in rule.difficulty_groups
            if difficulty in speaker_groups.values()
        }
        if not groups:
            raise ValueError(f"frozen speaker profile is empty: {profile_id}")
        profiles[profile_id] = groups
    return profiles, references


def _allocate_speakers(
    proposal: Proposal,
    protocol: RunProtocol,
    profiles: dict[str, dict[str, tuple[str, ...]]],
) -> dict[str, list[str]]:
    """Balance each exercise role across its eligible reference strata.

    The reference pool, not all train speakers, defines eligibility.  Within a
    stratum every patient is used once before repeating.  Canonical stimulus IDs
    make assignments independent of the Agent's presentation order.
    """

    role_uses: Counter[tuple[str, str]] = Counter()
    group_uses: Counter[tuple[str, str, str]] = Counter()
    allocation: dict[str, list[str]] = {}
    for stimulus in sorted(proposal.stimuli, key=lambda item: item.stimulus_id):
        profile_id = stimulus.speaker_profile_id
        groups = profiles[profile_id]
        group_names = list(groups)
        group_offset = (protocol.lora.seed + proposal.round_index - 1) % len(groups)
        if protocol.speaker_profiles[profile_id].allocation == "crossed_16":
            if stimulus.copies != 16 or len(groups) != 3:
                raise ValueError("crossed_16 requires 16 copies and all three strata")
            selected = []
            for position, group in enumerate(group_names):
                count = 6 if position == group_offset else 5
                members = sorted(groups[group], key=lambda speaker_id: hashlib.sha256(
                    f"{protocol.lora.seed}\0{proposal.round_index}\0{profile_id}\0"
                    f"crossed_16\0{speaker_id}".encode("utf-8")
                ).digest())
                if len(members) < count:
                    raise ValueError("crossed_16 reference stratum is too small")
                selected.extend(members[:count])
            allocation[stimulus.stimulus_id] = selected
            continue
        allocation[stimulus.stimulus_id] = []
        for _ in range(stimulus.copies):
            role_key = (profile_id, stimulus.role)
            group = group_names[(group_offset + role_uses[role_key]) % len(groups)]
            role_uses[role_key] += 1
            group_key = (*role_key, group)
            members = sorted(
                groups[group],
                key=lambda speaker_id: hashlib.sha256(
                    f"{protocol.lora.seed}\0{proposal.round_index}\0{profile_id}\0"
                    f"{stimulus.role}\0{speaker_id}".encode("utf-8")
                ).digest(),
            )
            speaker_id = members[group_uses[group_key] % len(members)]
            group_uses[group_key] += 1
            allocation[stimulus.stimulus_id].append(speaker_id)
    return allocation


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
    validate_stimulus_grounding(proposal, evidence)
    if protocol.protocol_status != "frozen":
        raise ValueError("only a frozen protocol can be materialized")
    if proposal.round_index > protocol.round_count:
        raise ValueError("proposal round exceeds the frozen protocol")

    output_dir = Path(output_root)
    if not output_dir.is_absolute() or ".." in output_dir.parts:
        raise ValueError("output_root must be absolute and traversal-free")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"output_root does not exist: {output_dir}")

    train_gt_texts, train_utterances = _load_split(protocol.real_train, "train")
    dev_gt_texts, _ = _load_split(protocol.feedback_split, "dev")
    test_gt_texts = ()
    if protocol.sealed_test_split is not None:
        test_gt_texts, _ = _load_split(protocol.sealed_test_split, "test")
    validated_profiles, validated_references = _load_frozen_references(
        protocol, train_utterances
    )
    _validate_content_isolation(
        proposal,
        train_gt_texts=train_gt_texts,
        dev_gt_texts=dev_gt_texts,
        test_gt_texts=test_gt_texts,
    )

    speaker_allocation = _allocate_speakers(proposal, protocol, validated_profiles)
    assignments: list[dict[str, str | int]] = []
    for stimulus in proposal.stimuli:
        for copy_index in range(1, stimulus.copies + 1):
            speaker_id = speaker_allocation[stimulus.stimulus_id][copy_index - 1]
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


def build_materialization_sidecar(
    proposal: Proposal,
    evidence: EvidencePack,
    protocol: RunProtocol,
    assignments: Sequence[dict[str, str | int]],
    *, generation_plan_path: Path | None = None,
) -> MaterializationSidecar:
    """Bind private TTS rows to the Proposal semantics without exposing patients.

    This function consumes assignments in the exact order returned by
    :func:`materialize_proposal`.  It never recovers semantics by parsing an audio
    path or sample ID.
    """

    validate_proposal(proposal, evidence, protocol)
    validate_stimulus_grounding(proposal, evidence)
    if len(assignments) != protocol.synthetic_count_per_round:
        raise ValueError("assignment count differs from frozen synthetic budget")
    if any(set(item) != REQUIRED_ASSIGNMENT_FIELDS for item in assignments):
        raise ValueError("assignment schema differs from synthesis contract")

    _, train_utterances = _load_split(protocol.real_train, "train")
    profiles, references = _load_frozen_references(protocol, train_utterances)
    generation_choices, generation_cards = {}, {}
    if protocol.preference_loop is not None and protocol.preference_loop.generation_design_path is not None:
        from .generation_controller import load_design, load_catalog
        from .generation_design import load_generation_choices
        if generation_plan_path is None:
            raise ValueError("controlled generation requires its bound decision plan")
        generation_cards = load_catalog(load_design(protocol), list(train_utterances.values()))
        generation_choices = load_generation_choices(generation_plan_path, {x['sample_id']:x for x in assignments})
    speaker_allocation = _allocate_speakers(proposal, protocol, profiles)
    speaker_groups = {
        speaker_id: group
        for groups in profiles.values()
        for group, members in groups.items()
        for speaker_id in members
    }

    cluster_ids = {item.evidence_id for item in evidence.error_clusters} | {
        item.evidence_id for item in evidence.capability_anchors
    }
    rows: list[MaterializedSampleIndex] = []
    role_stimuli: dict[str, set[str]] = {
        "target": set(),
        "contrast": set(),
        "protection": set(),
        "maintenance": set(),
    }
    role_copies: Counter[str] = Counter()
    role_carriers: Counter[str] = Counter()
    cluster_stimuli: dict[str, set[str]] = defaultdict(set)
    cluster_copies: Counter[str] = Counter()
    cluster_carriers: Counter[str] = Counter()

    assignment_index = 0
    for stimulus in proposal.stimuli:
        grounding = stimulus_cluster_grounding(stimulus, evidence)
        cited_cluster_ids = [
            evidence_id
            for evidence_id in stimulus.evidence_ids
            if evidence_id in cluster_ids
        ]
        carrier_counts = [
            CarrierCount(evidence_id=evidence_id, count=grounding[evidence_id][0])
            for evidence_id in cited_cluster_ids
        ]
        for copy_index in range(1, stimulus.copies + 1):
            assignment = assignments[assignment_index]
            assignment_index += 1
            expected_sample_id = (
                f"{protocol.run_id}.r{proposal.round_index:03d}."
                f"{stimulus.stimulus_id}.c{copy_index:04d}"
            )
            if assignment["sample_id"] != expected_sample_id:
                raise ValueError("assignment order or sample binding differs")
            if assignment["target_text"] != stimulus.text:
                raise ValueError("assignment target text differs from its stimulus")
            speaker_id = speaker_allocation[stimulus.stimulus_id][copy_index - 1]
            reference_text, reference_wav, reference_hash = references[speaker_id]
            if generation_choices:
                selected = generation_cards[generation_choices[expected_sample_id]['reference_id']]
                if selected['speaker_id'] != speaker_id:
                    raise ValueError("generation decision changed the allocated patient")
                reference_text, reference_wav, reference_hash = selected['reference_text'], selected['reference_wav'], selected['reference_wav_sha256']
            if (
                assignment["reference_text"],
                assignment["reference_wav"],
                assignment["reference_wav_sha256"],
            ) != (reference_text, str(reference_wav), reference_hash):
                raise ValueError("assignment reference differs from frozen allocation")
            rows.append(
                MaterializedSampleIndex(
                    sample_id=expected_sample_id,
                    stimulus_id=stimulus.stimulus_id,
                    copy_index=copy_index,
                    role=stimulus.role,
                    evidence_ids=cited_cluster_ids,
                    speaker_profile_id=stimulus.speaker_profile_id,
                    reference_difficulty_group=speaker_groups[speaker_id],
                    carrier_counts=carrier_counts,
                )
            )
            role_copies[stimulus.role] += 1
            role_carriers[stimulus.role] += sum(item.count for item in carrier_counts)
            for item in carrier_counts:
                cluster_copies[item.evidence_id] += 1
                cluster_carriers[item.evidence_id] += item.count
        role_stimuli[stimulus.role].add(stimulus.stimulus_id)
        for evidence_id in cited_cluster_ids:
            cluster_stimuli[evidence_id].add(stimulus.stimulus_id)

    sidecar = MaterializationSidecar(
        schema_version="1",
        run_id=protocol.run_id,
        round_index=proposal.round_index,
        protocol_canonical_sha256=canonical_sha256(protocol),
        evidence_canonical_sha256=canonical_sha256(evidence),
        proposal_canonical_sha256=canonical_sha256(proposal),
        assignments_canonical_sha256=_canonical_assignment_sha256(assignments),
        whole_batch_text_carrier_exposures=[
            CarrierCount(evidence_id=evidence_id, count=count)
            for evidence_id, count in whole_proposal_carrier_exposures(
                proposal, evidence
            ).items()
        ],
        samples=rows,
        by_role=[
            PlannedRoleDose(
                role=role,
                unique_stimuli=len(role_stimuli[role]),
                copies=role_copies[role],
                carrier_exposures=role_carriers[role],
            )
            for role in ("target", "contrast", "protection", "maintenance")
        ],
        by_cluster=[
            PlannedClusterDose(
                evidence_id=evidence_id,
                unique_stimuli=len(cluster_stimuli[evidence_id]),
                linked_copies=cluster_copies[evidence_id],
                carrier_exposures=cluster_carriers[evidence_id],
            )
            for evidence_id in sorted(cluster_stimuli)
        ],
    )
    validate_materialization_sidecar(sidecar, proposal, evidence, protocol)
    return sidecar


def validate_materialization_sidecar(
    sidecar: MaterializationSidecar,
    proposal: Proposal,
    evidence: EvidencePack,
    protocol: RunProtocol,
) -> None:
    """Rebuild every semantic sidecar row from its bound Proposal."""

    validate_proposal(proposal, evidence, protocol)
    validate_stimulus_grounding(proposal, evidence)
    expected_bindings = (
        protocol.run_id,
        proposal.round_index,
        canonical_sha256(protocol),
        canonical_sha256(evidence),
        canonical_sha256(proposal),
    )
    actual_bindings = (
        sidecar.run_id,
        sidecar.round_index,
        sidecar.protocol_canonical_sha256,
        sidecar.evidence_canonical_sha256,
        sidecar.proposal_canonical_sha256,
    )
    if actual_bindings != expected_bindings:
        raise ValueError("materialization sidecar bindings differ from proposal")
    if len(sidecar.samples) != protocol.synthetic_count_per_round:
        raise ValueError("sidecar sample count differs from frozen budget")

    _, train_utterances = _load_split(protocol.real_train, "train")
    profiles, _ = _load_frozen_references(protocol, train_utterances)
    speaker_allocation = _allocate_speakers(proposal, protocol, profiles)
    speaker_groups = {
        speaker_id: group
        for groups in profiles.values()
        for group, members in groups.items()
        for speaker_id in members
    }

    cluster_ids = {item.evidence_id for item in evidence.error_clusters} | {
        item.evidence_id for item in evidence.capability_anchors
    }
    expected_samples: list[MaterializedSampleIndex] = []
    for stimulus in proposal.stimuli:
        grounding = stimulus_cluster_grounding(stimulus, evidence)
        cited = [
            evidence_id
            for evidence_id in stimulus.evidence_ids
            if evidence_id in cluster_ids
        ]
        carriers = [
            CarrierCount(evidence_id=evidence_id, count=grounding[evidence_id][0])
            for evidence_id in cited
        ]
        for copy_index in range(1, stimulus.copies + 1):
            speaker_id = speaker_allocation[stimulus.stimulus_id][copy_index - 1]
            expected_samples.append(
                MaterializedSampleIndex(
                    sample_id=(
                        f"{protocol.run_id}.r{proposal.round_index:03d}."
                        f"{stimulus.stimulus_id}.c{copy_index:04d}"
                    ),
                    stimulus_id=stimulus.stimulus_id,
                    copy_index=copy_index,
                    role=stimulus.role,
                    evidence_ids=cited,
                    speaker_profile_id=stimulus.speaker_profile_id,
                    reference_difficulty_group=speaker_groups[speaker_id],
                    carrier_counts=carriers,
                )
            )
    if sidecar.samples != expected_samples:
        raise ValueError("materialization sidecar rows differ from Proposal grounding")
    expected_exposures = [
        CarrierCount(evidence_id=evidence_id, count=count)
        for evidence_id, count in whole_proposal_carrier_exposures(
            proposal, evidence
        ).items()
    ]
    if sidecar.whole_batch_text_carrier_exposures != expected_exposures:
        raise ValueError("whole-batch text exposure differs from Proposal grounding")


def _canonical_assignment_sha256(
    assignments: Sequence[dict[str, str | int]],
) -> str:
    payload = json.dumps(
        list(assignments),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "REQUIRED_ASSIGNMENT_FIELDS",
    "build_materialization_sidecar",
    "materialize_proposal",
    "validate_materialization_sidecar",
]
