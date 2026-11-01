import pytest

from test_therapist_harness import (
    add_cluster,
    counterbalance_outcome,
    evidence,
    proposal,
    protocol,
)
from therapist_harness.schema import Proposal, validate_proposal


def test_target_cannot_smuggle_an_unassigned_cluster(tmp_path):
    spec = protocol(tmp_path)
    pack = add_cluster(evidence(spec), evidence_id="e2")
    payload = proposal(pack).model_dump(mode="json")
    payload["stimuli"][0]["evidence_ids"].append("e2")
    with pytest.raises(ValueError, match="only primary target clusters"):
        validate_proposal(Proposal.model_validate(payload), pack, spec)


def test_every_comparator_must_match_reference_family(tmp_path):
    spec = protocol(tmp_path)
    pack = add_cluster(evidence(spec), evidence_id="e2")
    pack = add_cluster(pack, evidence_id="e3", reference_unit="t", hypothesis_unit="p")
    payload = proposal(pack).model_dump(mode="json")
    payload["stimuli"][1].update(role="contrast", evidence_ids=["e1", "e2", "e3"])
    payload["expected_outcomes"].extend(
        counterbalance_outcome(key) for key in ("e2", "e3")
    )
    with pytest.raises(ValueError, match="comparator_not_same_reference_family"):
        validate_proposal(Proposal.model_validate(payload), pack, spec)
