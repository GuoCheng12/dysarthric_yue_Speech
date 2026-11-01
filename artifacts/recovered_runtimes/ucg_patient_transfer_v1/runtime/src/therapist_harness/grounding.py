"""Pure pronunciation grounding shared by validation and materialization."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .capabilities import count_capability_carriers
from .phonology import map_text_to_syllables
from .schema import EvidencePack, Proposal, Stimulus
from .text import tts_phonology_text


def stimulus_cluster_grounding(
    stimulus: Stimulus,
    evidence: EvidencePack,
    *,
    evidence_ids: Iterable[str] | None = None,
) -> dict[str, tuple[int, bool]]:
    """Count text carriers for cited or explicitly requested clusters.

    These are pronunciation-facing text opportunities, not measurements of the
    eventual audio. Explicit IDs also expose incidental carriers in stimuli that
    do not cite that cluster.
    """

    clusters = {item.evidence_id: item for item in evidence.error_clusters}
    phonology_text = tts_phonology_text(stimulus.text)
    cited_clusters = [
        clusters[evidence_id]
        for evidence_id in (
            stimulus.evidence_ids if evidence_ids is None else evidence_ids
        )
        if evidence_id in clusters
    ]
    phone_levels = {"onset", "nucleus", "coda", "tone"}
    mapping = (
        map_text_to_syllables(phonology_text)
        if any(item.level in phone_levels for item in cited_clusters)
        else None
    )
    result: dict[str, tuple[int, bool]] = {}
    for cluster in cited_clusters:
        if cluster.level == "character":
            if (
                cluster.reference_surface_units is None
                or cluster.hypothesis_surface_units is None
            ):
                raise ValueError(
                    f"{cluster.evidence_id}: character grounding requires attested "
                    "surface units; regenerate legacy evidence before synthesis"
                )
            reference_carriers = sum(
                phonology_text.count(unit) for unit in cluster.reference_surface_units
            )
            hypothesis_present = (
                not cluster.hypothesis_unit
                or any(
                    unit in phonology_text for unit in cluster.hypothesis_surface_units
                )
            )
        else:
            if mapping is None:
                raise RuntimeError("phone cluster was not mapped")
            position_units = [
                getattr(position, cluster.level) or "∅"
                for position in mapping.positions
                if position.resolved
            ]
            reference_carriers = position_units.count(cluster.reference_unit)
            hypothesis_present = cluster.hypothesis_unit in position_units
        result[cluster.evidence_id] = (reference_carriers, hypothesis_present)
    requested = set(stimulus.evidence_ids if evidence_ids is None else evidence_ids)
    for anchor in evidence.capability_anchors:
        if anchor.evidence_id in requested:
            result[anchor.evidence_id] = (
                count_capability_carriers(stimulus.text, anchor), True
            )
    return result


def _monitored_cluster_ids(proposal: Proposal, evidence: EvidencePack) -> list[str]:
    cluster_ids = {item.evidence_id for item in evidence.error_clusters} | {
        item.evidence_id for item in evidence.capability_anchors
    }
    return sorted(
        (set(proposal.target_evidence_ids)
        | {item for stimulus in proposal.stimuli for item in stimulus.evidence_ids}
        | {
            item.subject_evidence_id
            for item in proposal.expected_outcomes
            if item.subject_evidence_id in cluster_ids
        }) & cluster_ids
    )


def whole_proposal_carrier_exposures(
    proposal: Proposal, evidence: EvidencePack
) -> dict[str, int]:
    """Count each monitored cluster's text carriers across the entire batch.

    Shared carriers remain separately reported for each cluster signature; their
    counts must not be summed as independent acoustic exercises.
    """

    monitored_ids = _monitored_cluster_ids(proposal, evidence)
    exposures = Counter({evidence_id: 0 for evidence_id in monitored_ids})
    for stimulus in proposal.stimuli:
        grounding = stimulus_cluster_grounding(
            stimulus, evidence, evidence_ids=monitored_ids
        )
        for evidence_id, (carrier_count, _) in grounding.items():
            exposures[evidence_id] += carrier_count * stimulus.copies
    return dict(exposures)


def validate_stimulus_grounding(proposal: Proposal, evidence: EvidencePack) -> None:
    """Ground cluster citations and enforce pronunciation-facing dose limits."""

    clusters = {item.evidence_id: item for item in evidence.error_clusters}
    constraints = evidence.action_contract.proposal_constraints
    primary_ids = set(proposal.target_evidence_ids)
    monitored_ids = _monitored_cluster_ids(proposal, evidence)
    carrier_exposures: Counter[str] = Counter()
    failures: list[str] = []
    for stimulus in proposal.stimuli:
        grounding = stimulus_cluster_grounding(
            stimulus, evidence, evidence_ids=monitored_ids
        )
        for evidence_id, (reference_carriers, hypothesis_present) in grounding.items():
            if evidence_id not in clusters:
                if evidence_id in stimulus.evidence_ids and reference_carriers <= 0:
                    failures.append(
                        f"{stimulus.stimulus_id}:{evidence_id}:"
                        "capability_carrier_missing_from_tts_text"
                    )
                continue
            cluster = clusters[evidence_id]
            carrier_exposures[cluster.evidence_id] += (
                reference_carriers * stimulus.copies
            )
            if evidence_id in stimulus.evidence_ids and reference_carriers <= 0:
                failures.append(
                    f"{stimulus.stimulus_id}:{cluster.evidence_id}:"
                    "reference_unit_missing_from_tts_text"
                )
            if (
                stimulus.role == "contrast"
                and evidence_id in stimulus.evidence_ids
                and cluster.evidence_id not in primary_ids
                and not hypothesis_present
            ):
                failures.append(
                    f"{stimulus.stimulus_id}:{cluster.evidence_id}:"
                    "comparator_unit_missing_from_tts_text"
                )

    selection = evidence.data_scope.cluster_selection
    for evidence_id, exposure in sorted(carrier_exposures.items()):
        cluster = clusters[evidence_id]
        current = cluster.observations.current
        is_threshold_edge = cluster.targetable and (
            current.error_count == selection.minimum_current_error_count
            or current.patient_count == selection.minimum_patient_count
            or current.content_count == selection.minimum_content_count
        )
        if (
            is_threshold_edge
            and exposure > constraints.threshold_edge_max_carrier_exposures
        ):
            failures.append(
                f"{evidence_id}:threshold_edge_carrier_exposures={exposure}/"
                f"{constraints.threshold_edge_max_carrier_exposures}"
            )
    if failures:
        raise ValueError("stimulus evidence grounding failed: " + "; ".join(failures))


__all__ = [
    "stimulus_cluster_grounding",
    "validate_stimulus_grounding",
    "whole_proposal_carrier_exposures",
]
