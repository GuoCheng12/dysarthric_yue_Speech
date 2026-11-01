from __future__ import annotations

from pathlib import Path

import pytest

from test_therapist_harness import add_cluster, evidence, proposal, protocol
from therapist_harness.grounding import (
    stimulus_cluster_grounding,
    validate_stimulus_grounding,
    whole_proposal_carrier_exposures,
)
from therapist_harness.schema import EvidencePack, Proposal, validate_proposal


def _case_one(tmp_path: Path) -> tuple[EvidencePack, Proposal]:
    pack = add_cluster(evidence(protocol(tmp_path)), evidence_id="e2")
    pack = add_cluster(pack, evidence_id="e3", hypothesis_unit="ng")
    payload = proposal(pack).model_dump(mode="json")
    template = payload["stimuli"][0]
    stimuli = [
        ("朝早食碗粥先出門口。", "target", ["e1"]),
        ("阿明執好間屋就出街。", "target", ["e1"]),
        ("呢件黑色外套好襯你。", "target", ["e1"]),
        ("我搭港鐵去食晏先。", "contrast", ["e1", "e2"]),
        ("今晚八點食完飯去散步。", "contrast", ["e1", "e2"]),
        ("阿明執好行李先返屋企。", "protection", ["e3"]),
    ]
    payload["stimuli"] = [
        dict(
            template,
            stimulus_id=f"s{index}",
            text=text,
            role=role,
            evidence_ids=evidence_ids,
            copies=4,
        )
        for index, (text, role, evidence_ids) in enumerate(stimuli, start=1)
    ]
    return pack, Proposal.model_validate(payload)


def test_uncited_carriers_are_included_without_summing_shared_families(
    tmp_path: Path,
) -> None:
    pack, prescription = _case_one(tmp_path)
    linked_primary = sum(
        stimulus_cluster_grounding(item, pack).get("e1", (0, False))[0]
        * item.copies
        for item in prescription.stimuli
    )
    assert linked_primary == 28
    assert whole_proposal_carrier_exposures(prescription, pack) == {
        "e1": 32,
        "e2": 32,
        "e3": 32,
    }


def test_uncited_protection_carriers_cannot_hide_threshold_edge_overdose(
    tmp_path: Path,
) -> None:
    pack, prescription = _case_one(tmp_path)
    payload = pack.model_dump(mode="json")
    payload["data_scope"]["cluster_selection"].update(
        minimum_current_error_count=2,
        minimum_patient_count=2,
        minimum_content_count=2,
    )
    payload["action_contract"]["proposal_constraints"][
        "threshold_edge_max_carrier_exposures"
    ] = 31
    pack = EvidencePack.model_validate(payload)
    with pytest.raises(ValueError, match="e1:threshold_edge_carrier_exposures=32/31"):
        validate_stimulus_grounding(prescription, pack)


def test_uncited_protection_carriers_are_counted_without_per_sentence_rejection(
    tmp_path: Path,
) -> None:
    pack, prescription = _case_one(tmp_path)
    payload = prescription.model_dump(mode="json")
    payload["stimuli"][-1]["text"] = "阿叔今日食白粥。"
    prescription = Proposal.model_validate(payload)
    validate_stimulus_grounding(prescription, pack)
    assert whole_proposal_carrier_exposures(prescription, pack) == {
        "e1": 44,
        "e2": 44,
        "e3": 44,
    }


def test_concentrated_target_passes_proposal_and_keeps_exposure_counts(tmp_path):
    spec = protocol(tmp_path)
    pack = evidence(spec)
    prescription = proposal(pack)
    validate_proposal(prescription, pack, spec)
    validate_stimulus_grounding(prescription, pack)
    assert [
        stimulus_cluster_grounding(item, pack)["e1"][0]
        for item in prescription.stimuli
    ] == [4, 4]
    assert whole_proposal_carrier_exposures(prescription, pack) == {"e1": 8}


@pytest.mark.parametrize("text", ["姐姐朝早去街市買菜。", "阿詩今晚去睇戲。"])
def test_smoke_tone_sentences_no_longer_fail_on_three_carriers(tmp_path, text):
    spec = protocol(tmp_path)
    payload = evidence(spec).model_dump(mode="json")
    payload["error_clusters"][0].update(
        operation="substitution", level="tone", reference_unit="1", hypothesis_unit="6"
    )
    pack = EvidencePack.model_validate(payload)
    prescription_payload = proposal(pack).model_dump(mode="json")
    prescription_payload["stimuli"][0]["text"] = text
    prescription_payload["stimuli"][1]["text"] = "我哋朝早去街市。"
    prescription = Proposal.model_validate(prescription_payload)
    validate_proposal(prescription, pack, spec)
    validate_stimulus_grounding(prescription, pack)
    assert stimulus_cluster_grounding(prescription.stimuli[0], pack)["e1"][0] == 3
    assert whole_proposal_carrier_exposures(prescription, pack) == {"e1": 5}
