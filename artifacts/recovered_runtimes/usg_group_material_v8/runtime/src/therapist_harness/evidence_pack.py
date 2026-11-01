"""Build the sanitized EvidencePack consumed by the Therapist Agent."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pycantonese

from .capabilities import extract_capability_anchors
from .evidence import (
    MAX_TARGET_ALIGNMENT_UNCERTAINTY,
    MIN_TARGET_CONTENT_COUNT,
    MIN_TARGET_ERROR_COUNT,
    MIN_TARGET_PATIENT_COUNT,
    extract_evidence,
)
from .schema import (
    ActionContract,
    ClusterSelectionPolicy,
    EvidenceDataScope,
    EvidencePack,
    InterventionSummary,
    RunProtocol,
    SnapshotEvidence,
    SnapshotPhonologyCoverageSet,
    SnapshotSet,
)
from .store import sha256_directory, sha256_file, verify_artifact


PAIRING_FIELDS = (
    "utt_id",
    "speaker_id",
    "prompt_id",
    "zero_shot_bucket",
    "clean_gt",
)


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def load_prediction_rows(path: Path) -> list[dict[str, object]]:
    """Read a complete CSV or JSONL prediction snapshot without guessing columns."""

    if not path.is_absolute():
        raise ValueError("prediction path must be absolute")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or len(set(reader.fieldnames)) != len(
                reader.fieldnames
            ):
                raise ValueError("prediction CSV has missing or duplicate columns")
            rows: list[dict[str, object]] = [dict(row) for row in reader]
    elif path.suffix.lower() in {".jsonl", ".json"}:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"prediction JSONL line {line_number} is blank")
                try:
                    payload = json.loads(line, object_pairs_hook=_json_object)
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(
                        f"invalid prediction JSONL line {line_number}: {error}"
                    ) from error
                if not isinstance(payload, dict):
                    raise TypeError(
                        f"prediction JSONL line {line_number} is not an object"
                    )
                rows.append(payload)
    else:
        raise ValueError("prediction file must end in .csv, .jsonl, or .json")
    if not rows:
        raise ValueError("prediction snapshot is empty")
    return rows


def _load_feedback_metadata(protocol: RunProtocol) -> list[dict[str, object]]:
    verify_artifact(protocol.feedback_split)
    path = Path(protocol.feedback_split.path)
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"feedback split line {line_number} is blank")
            payload = json.loads(line, object_pairs_hook=_json_object)
            if not isinstance(payload, dict):
                raise TypeError(f"feedback split line {line_number} is not an object")
            rows.append(payload)
    return rows


def _validate_against_feedback_split(
    rows: Sequence[Mapping[str, object]], protocol: RunProtocol
) -> None:
    feedback_rows = _load_feedback_metadata(protocol)
    if len(rows) != len(feedback_rows):
        raise ValueError(
            "prediction snapshot row count differs from the frozen feedback split"
        )
    by_id: dict[str, Mapping[str, object]] = {}
    for row_index, row in enumerate(rows, start=1):
        utt_id = row.get("utt_id")
        if not isinstance(utt_id, str) or not utt_id:
            raise ValueError(f"prediction row {row_index} has an invalid utt_id")
        if utt_id in by_id:
            raise ValueError(f"prediction snapshot contains duplicate utt_id {utt_id}")
        by_id[utt_id] = row
    feedback_by_id = {str(row.get("utt_id")): row for row in feedback_rows}
    if set(by_id) != set(feedback_by_id):
        raise ValueError("prediction utterance IDs differ from the feedback split")
    for utt_id, prediction_row in by_id.items():
        feedback_row = feedback_by_id[utt_id]
        for field_name in PAIRING_FIELDS:
            if prediction_row.get(field_name) != feedback_row.get(field_name):
                raise ValueError(
                    f"prediction metadata differs from feedback split for "
                    f"{utt_id}, field={field_name}"
                )


def build_evidence_pack(
    protocol: RunProtocol,
    *,
    round_index: int,
    current_rows: Sequence[Mapping[str, object]],
    current_prediction_column: str,
    current_checkpoint_sha256: str,
    current_predictions_sha256: str,
    m0_rows: Sequence[Mapping[str, object]],
    m0_prediction_column: str,
    m0_predictions_sha256: str,
    zero_shot_rows: Sequence[Mapping[str, object]],
    zero_shot_prediction_column: str,
    zero_shot_predictions_sha256: str,
    previous_rows: Sequence[Mapping[str, object]] | None = None,
    previous_prediction_column: str | None = None,
    previous_checkpoint_sha256: str | None = None,
    previous_predictions_sha256: str | None = None,
    intervention_history: Sequence[InterventionSummary] = (),
    max_clusters_per_level: int = 10,
    artifact_mode: str = "recorded_execution",
) -> EvidencePack:
    """Construct one strict, identifier-free Agent input from paired snapshots."""

    if round_index < 1 or round_index > protocol.round_count:
        raise ValueError("round_index is outside the protocol")
    previous_values = (
        previous_rows,
        previous_prediction_column,
        previous_checkpoint_sha256,
        previous_predictions_sha256,
    )
    if any(value is None for value in previous_values) != all(
        value is None for value in previous_values
    ):
        raise ValueError("all previous snapshot inputs must be supplied together")
    if round_index == 1 and previous_rows is not None:
        raise ValueError("round 1 cannot include a previous snapshot")
    if round_index > 1 and previous_rows is None:
        raise ValueError("round 2+ requires a previous snapshot")
    if (
        m0_predictions_sha256 != protocol.initial_adapter_dev_predictions.sha256
        or m0_prediction_column != protocol.initial_adapter_prediction_column
    ):
        raise ValueError("M0 predictions do not match the frozen protocol anchor")
    if (
        zero_shot_predictions_sha256 != protocol.zero_shot_dev_predictions.sha256
        or zero_shot_prediction_column != protocol.zero_shot_prediction_column
    ):
        raise ValueError(
            "zero-shot predictions do not match the frozen protocol anchor"
        )

    _validate_against_feedback_split(current_rows, protocol)
    prior_target_ids = [
        evidence_id
        for item in intervention_history
        for evidence_id in item.target_evidence_ids
    ]
    prior_targets = [
        descriptor
        for item in intervention_history
        for descriptor in item.target_descriptors
    ]
    extraction = extract_evidence(
        current_rows,
        m0_rows,
        zero_shot_rows,
        current_prediction_column=current_prediction_column,
        m0_prediction_column=m0_prediction_column,
        zero_shot_prediction_column=zero_shot_prediction_column,
        previous_rows=previous_rows,
        previous_prediction_column=previous_prediction_column,
        critical_cer_threshold=protocol.evaluation.critical_cer_threshold,
        max_clusters_per_level=max_clusters_per_level,
        prior_target_ids=prior_target_ids,
        prior_targets=prior_targets,
    )
    if round_index == 1 and (
        current_checkpoint_sha256 != protocol.initial_adapter.sha256
        or current_predictions_sha256 != m0_predictions_sha256
        or current_prediction_column != protocol.initial_adapter_prediction_column
    ):
        raise ValueError("round 1 current snapshot must be the frozen M0 snapshot")

    previous_snapshot = (
        SnapshotEvidence(
            model_role="previous_pre_intervention",
            closed_loop_model_index=round_index - 2,
            checkpoint_sha256=previous_checkpoint_sha256,
            predictions_sha256=previous_predictions_sha256,
            metrics=extraction.previous_metrics,
        )
        if previous_checkpoint_sha256 is not None
        and previous_predictions_sha256 is not None
        and extraction.previous_metrics is not None
        else None
    )
    snapshots = SnapshotSet(
        current=SnapshotEvidence(
            model_role="current_pre_intervention",
            closed_loop_model_index=round_index - 1,
            checkpoint_sha256=current_checkpoint_sha256,
            predictions_sha256=current_predictions_sha256,
            metrics=extraction.current_metrics,
        ),
        m0=SnapshotEvidence(
            model_role="real_only_closed_loop_start",
            closed_loop_model_index=0,
            checkpoint_sha256=protocol.initial_adapter.sha256,
            predictions_sha256=m0_predictions_sha256,
            metrics=extraction.m0_metrics,
        ),
        zero_shot=SnapshotEvidence(
            model_role="zero_shot_diagnostic_only",
            closed_loop_model_index=None,
            checkpoint_sha256=protocol.base_model_fingerprint.sha256,
            predictions_sha256=zero_shot_predictions_sha256,
            metrics=extraction.zero_shot_metrics,
        ),
        previous=previous_snapshot,
    )
    return EvidencePack(
        schema_version="3",
        artifact_mode=artifact_mode,
        run_id=protocol.run_id,
        round_index=round_index,
        data_scope=EvidenceDataScope(
            setting_id=protocol.setting_id,
            feedback_split_sha256=protocol.feedback_split.sha256,
            sample_count=extraction.sample_count,
            speaker_count=extraction.speaker_count,
            content_count=extraction.content_count,
            sealed_test_used=False,
            critical_cer_threshold=protocol.evaluation.critical_cer_threshold,
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
                current=extraction.current_phonology_coverage,
                m0=extraction.m0_phonology_coverage,
                zero_shot=extraction.zero_shot_phonology_coverage,
                previous=extraction.previous_phonology_coverage,
            ),
            cluster_selection=ClusterSelectionPolicy(
                max_clusters_per_level=max_clusters_per_level,
                historical_targets_exempt_from_limit=True,
                candidate_count=extraction.candidate_cluster_count,
                included_count=len(extraction.error_clusters),
                minimum_current_error_count=MIN_TARGET_ERROR_COUNT,
                minimum_patient_count=MIN_TARGET_PATIENT_COUNT,
                minimum_content_count=MIN_TARGET_CONTENT_COUNT,
                maximum_alignment_uncertainty=MAX_TARGET_ALIGNMENT_UNCERTAINTY,
                character_insertions_targetable=False,
                ordering=(
                    "all_historical_targets_then_targetable_new_clusters_then_"
                    "current_error_count_then_patient_content_support_then_anchor_burden"
                ),
            ),
        ),
        snapshots=snapshots,
        comparisons=extraction.comparisons,
        error_clusters=list(extraction.error_clusters),
        capability_anchors=extract_capability_anchors(
            current_rows, m0_rows,
            current_prediction_column=current_prediction_column,
            m0_prediction_column=m0_prediction_column,
            previous_rows=previous_rows,
            previous_prediction_column=previous_prediction_column,
        ),
        intervention_history=list(intervention_history),
        action_contract=ActionContract(
            synthetic_count=protocol.synthetic_count_per_round,
            unique_stimulus_count=protocol.unique_stimulus_count_per_round,
            real_replay_count=protocol.real_replay_count_per_round,
            permitted_speaker_profiles=protocol.speaker_profiles,
            proposal_constraints=protocol.proposal_constraints,
        ),
    )


def build_evidence_pack_from_files(
    protocol: RunProtocol,
    *,
    round_index: int,
    current_predictions: Path,
    current_prediction_column: str,
    current_checkpoint: Path,
    m0_predictions: Path,
    m0_prediction_column: str,
    zero_shot_predictions: Path,
    zero_shot_prediction_column: str,
    previous_predictions: Path | None = None,
    previous_prediction_column: str | None = None,
    previous_checkpoint: Path | None = None,
    intervention_history: Sequence[InterventionSummary] = (),
    max_clusters_per_level: int = 10,
    artifact_mode: str = "recorded_execution",
) -> EvidencePack:
    """File-backed EvidencePack construction used by the Harness CLI."""

    if getattr(pycantonese, "__version__", None) != "5.0.0":
        raise RuntimeError("EvidencePack v3 requires exactly PyCantonese 5.0.0")
    if not current_checkpoint.is_absolute() or not current_checkpoint.is_dir():
        raise FileNotFoundError("current checkpoint must be an absolute directory")
    previous_values = (
        previous_predictions,
        previous_prediction_column,
        previous_checkpoint,
    )
    if any(value is None for value in previous_values) != all(
        value is None for value in previous_values
    ):
        raise ValueError("previous file inputs must be supplied together")
    if previous_checkpoint is not None and (
        not previous_checkpoint.is_absolute() or not previous_checkpoint.is_dir()
    ):
        raise FileNotFoundError("previous checkpoint must be an absolute directory")
    for reference in (
        protocol.initial_adapter,
        protocol.initial_adapter_dev_predictions,
        protocol.zero_shot_dev_predictions,
    ):
        verify_artifact(reference)

    current_rows = load_prediction_rows(current_predictions)
    m0_rows = load_prediction_rows(m0_predictions)
    zero_shot_rows = load_prediction_rows(zero_shot_predictions)
    previous_rows = (
        load_prediction_rows(previous_predictions)
        if previous_predictions is not None
        else None
    )
    return build_evidence_pack(
        protocol,
        round_index=round_index,
        current_rows=current_rows,
        current_prediction_column=current_prediction_column,
        current_checkpoint_sha256=sha256_directory(current_checkpoint),
        current_predictions_sha256=sha256_file(current_predictions),
        m0_rows=m0_rows,
        m0_prediction_column=m0_prediction_column,
        m0_predictions_sha256=sha256_file(m0_predictions),
        zero_shot_rows=zero_shot_rows,
        zero_shot_prediction_column=zero_shot_prediction_column,
        zero_shot_predictions_sha256=sha256_file(zero_shot_predictions),
        previous_rows=previous_rows,
        previous_prediction_column=previous_prediction_column,
        previous_checkpoint_sha256=(
            sha256_directory(previous_checkpoint)
            if previous_checkpoint is not None
            else None
        ),
        previous_predictions_sha256=(
            sha256_file(previous_predictions)
            if previous_predictions is not None
            else None
        ),
        intervention_history=intervention_history,
        max_clusters_per_level=max_clusters_per_level,
        artifact_mode=artifact_mode,
    )


__all__ = [
    "PAIRING_FIELDS",
    "build_evidence_pack",
    "build_evidence_pack_from_files",
    "load_prediction_rows",
]
