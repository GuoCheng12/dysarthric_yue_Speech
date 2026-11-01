"""Prepare real synthesis from an accepted prescription-only smoke.

This preserves the source prescription, reference allocation and seeds. It does
not promote counterfactual evidence into a recorded ASR intervention or train ASR.
Run the maintained synthesize_manifest.py on the resulting manifest separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from therapist_harness.materialize import (
    build_materialization_sidecar,
    materialize_proposal,
)
from therapist_harness.round_schema import MaterializationSidecar
from therapist_harness.schema import EvidencePack, Proposal, RunProtocol
from therapist_harness.store import sha256_file, write_immutable_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    protocol_path = source / "smoke_protocol.json"
    evidence_path = source / "round_001_evidence.json"
    proposal_path = source / "01_prescribe/proposal.json"
    sidecar_path = source / "materialization_sidecar.json"
    protocol = RunProtocol.model_validate_json(protocol_path.read_text("utf-8"))
    evidence = EvidencePack.model_validate_json(evidence_path.read_text("utf-8"))
    proposal = Proposal.model_validate_json(proposal_path.read_text("utf-8"))
    source_sidecar = MaterializationSidecar.model_validate_json(
        sidecar_path.read_text("utf-8")
    )
    report = json.loads((source / "smoke_report.json").read_text("utf-8"))
    if report["mode"] != "prescription_only_smoke" or not report["automatic_prescription_passed"]:
        raise ValueError("source must be a passed prescription-only smoke")
    output.mkdir(parents=True, exist_ok=False)
    audio = output / "audio"
    audio.mkdir()
    assignments = materialize_proposal(proposal, evidence, protocol, output_root=audio)
    sidecar = build_materialization_sidecar(proposal, evidence, protocol, assignments)
    # Output paths change, but the accepted prescription and its allocation do not.
    comparison = sidecar.model_dump(mode="json")
    comparison["assignments_canonical_sha256"] = source_sidecar.assignments_canonical_sha256
    if comparison != source_sidecar.model_dump(mode="json"):
        raise ValueError("synthesis allocation differs from accepted smoke")
    manifest = output / "synthesis_manifest.jsonl"
    with manifest.open("x", encoding="utf-8") as handle:
        for row in assignments:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_immutable_json(output / "materialization_sidecar.json", sidecar.model_dump(mode="json"))
    plan = {
        "mode": "real_synthesis_from_accepted_prescription_smoke",
        "source_artifact_mode": evidence.artifact_mode,
        "source_directory": str(source),
        "source_file_sha256": {
            str(path): sha256_file(path)
            for path in (protocol_path, evidence_path, proposal_path, sidecar_path)
        },
        "manifest_path": str(manifest),
        "manifest_file_sha256": sha256_file(manifest),
        "unique_texts": len(proposal.stimuli),
        "requested_audio": len(assignments),
        "reference_policy_changed": False,
        "seed_policy_changed": False,
        "synthetic_text_changed": False,
        "asr_training_authorized_by_this_plan": False,
        "execution_note": "Real TTS preview only; not a recorded ASR adaptation round.",
    }
    write_immutable_json(output / "synthesis_plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
