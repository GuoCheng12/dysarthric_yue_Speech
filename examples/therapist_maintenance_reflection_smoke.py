"""Full smoke: three Agent calls; prescription-only or reflection-only: one call.

No TTS, ASR training, or sealed-test evaluation.

Real M0/zero-shot dev predictions exercise a clearly labelled counterfactual
post-round interface. The receipts contain placeholders, not generated audio or a
trained adapter. A failure stops immediately; requests/responses remain inspectable.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import time
from pathlib import Path

from therapist_harness.agent import (
    build_reflection_request,
    build_therapist_request,
    execute_therapist_request,
    parse_reflection_response,
    parse_therapist_response,
)
from therapist_harness.cli import load_protocol
from therapist_harness.evidence_pack import build_evidence_pack, load_prediction_rows
from therapist_harness.materialize import (
    build_materialization_sidecar,
    materialize_proposal,
)
from therapist_harness.memory import SkillMemory, compile_intervention_summary
from therapist_harness.outcome import (
    compile_evaluation_plan,
    compile_reflection_pack,
    compile_round_result,
)
from therapist_harness.round_schema import (
    SynthesisReceipt,
    SynthesizedSample,
    TrainingReceipt,
)
from therapist_harness.schema import (
    Proposal,
    RunProtocol,
    canonical_sha256,
    validate_proposal,
)
from therapist_harness.grounding import validate_stimulus_grounding
from therapist_harness.store import sha256_file, verify_artifact, write_immutable_json


MODE = "counterfactual_interface_smoke"
PACKAGE = Path(__file__).resolve().parents[1] / "src" / "therapist_harness"


def save(directory, name, value):
    payload = (
        value.model_dump(mode="json", by_alias=True)
        if hasattr(value, "model_dump")
        else value
    )
    write_immutable_json(directory / f"{name}.json", payload)


def request_once(protocol, request, directory):
    save(directory, "request", request)
    started = time.monotonic()
    response = execute_therapist_request(protocol, request)
    save(directory, "response", response)
    timing = {
        "elapsed_seconds": time.monotonic() - started,
        "usage": response.usage.model_dump(mode="json") if response.usage else None,
    }
    save(directory, "timing", timing)
    print(
        json.dumps(
            {
                "stage": directory.name,
                "elapsed_seconds": timing["elapsed_seconds"],
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        ),
        flush=True,
    )
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--read-api-key", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prescription-only",
        action="store_true",
        help="Call prescribe once and validate/materialize its plan; no reflection fixture or training.",
    )
    mode.add_argument(
        "--reflection-only-fixture",
        type=Path,
        help="Explicit hand-authored test input: call only reflect, then compile next request; never an automatic fallback.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original = load_protocol(args.protocol)
    if args.read_api_key:
        os.environ[original.agent.api_key_env] = getpass.getpass("Gateway API key: ")
    for reference in (
        original.initial_adapter_dev_predictions,
        original.zero_shot_dev_predictions,
        original.feedback_split,
    ):
        verify_artifact(reference)
    payload = original.model_dump(mode="json")
    payload["protocol_status"] = "frozen"
    payload["synthetic_count_per_round"] = 24
    payload["unique_stimulus_count_per_round"] = 6
    payload["lora"]["optimizer_steps_per_round"] = math.ceil(
        (original.real_replay_count_per_round + 24)
        / (
            original.lora.per_device_batch_size
            * original.lora.gradient_accumulation_steps
        )
    )
    protocol = RunProtocol.model_validate(payload)
    save(args.output, "smoke_protocol", protocol)
    for round_index in (1, 2):
        (args.output / f"not_generated_round_{round_index:03d}").mkdir()
    save(
        args.output,
        "fixture_manifest",
        {
            "artifact_mode": MODE,
            "tts_run": False,
            "asr_training_run": False,
            "sealed_test_evaluated": False,
            "description": (
                "Real M0 dev evidence for a prescription-only smoke; no AFTER result."
                if args.prescription_only else
                "Real M0 dev BEFORE, real zero-shot dev AFTER; NOT a training transition."
            ),
            "receipt_placeholders": (
                "None: prescription-only mode creates no execution receipts."
                if args.prescription_only else
                "No synthesized audio, no trained adapter; duration 0 and hashes of fixture labels."
            ),
            "original_protocol_sha256": canonical_sha256(original),
            "smoke_budget_only": {"unique_texts": 6, "copies": 24},
        },
    )
    m0_rows = load_prediction_rows(Path(protocol.initial_adapter_dev_predictions.path))
    zero_rows = load_prediction_rows(Path(protocol.zero_shot_dev_predictions.path))
    common = {
        "m0_rows": m0_rows,
        "m0_prediction_column": protocol.initial_adapter_prediction_column,
        "m0_predictions_sha256": protocol.initial_adapter_dev_predictions.sha256,
        "zero_shot_rows": zero_rows,
        "zero_shot_prediction_column": protocol.zero_shot_prediction_column,
        "zero_shot_predictions_sha256": protocol.zero_shot_dev_predictions.sha256,
        "artifact_mode": MODE,
    }
    evidence = build_evidence_pack(
        protocol,
        round_index=1,
        current_rows=m0_rows,
        current_prediction_column=protocol.initial_adapter_prediction_column,
        current_checkpoint_sha256=protocol.initial_adapter.sha256,
        current_predictions_sha256=protocol.initial_adapter_dev_predictions.sha256,
        **common,
    )
    save(args.output, "round_001_evidence", evidence)
    print(
        json.dumps(
            {
                "stage": "evidence_ready",
                "clusters": len(evidence.error_clusters),
                "capability_anchors": len(evidence.capability_anchors),
            }
        ),
        flush=True,
    )
    skills = SkillMemory.initialize(
        args.output / "skills", PACKAGE / "skills/v000/SKILL.md"
    )
    skill = skills.bind_round(1)
    prescribe_prompt = PACKAGE / "prompts/therapist.md"
    reflect_prompt = PACKAGE / "prompts/reflection.md"
    if args.reflection_only_fixture:
        proposal = Proposal.model_validate_json(
            args.reflection_only_fixture.read_text("utf-8")
        )
        validate_proposal(proposal, evidence, protocol)
        validate_stimulus_grounding(proposal, evidence)
        save(
            args.output,
            "prescription_source",
            {
                "mode": "explicit_hand_authored_reflection_test_fixture",
                "fixture_file_sha256": sha256_file(args.reflection_only_fixture),
                "automatic_prescription_passed": False,
            },
        )
    else:
        request = build_therapist_request(
            protocol,
            evidence,
            system_prompt=prescribe_prompt.read_text("utf-8"),
            skill_text=skill.read_text(),
        )
        response = request_once(protocol, request, args.output / "01_prescribe")
        proposal = parse_therapist_response(response, protocol, evidence)
    save(args.output / "01_prescribe", "proposal", proposal)
    assignments = materialize_proposal(
        proposal,
        evidence,
        protocol,
        output_root=args.output / "not_generated_round_001",
    )
    sidecar = build_materialization_sidecar(proposal, evidence, protocol, assignments)
    save(args.output, "materialization_sidecar", sidecar)
    if args.prescription_only:
        report = {
            "mode": "prescription_only_smoke",
            "artifact_mode": MODE,
            "automatic_prescription_passed": True,
            "unique_texts": len(proposal.stimuli),
            "copies": sum(item.copies for item in proposal.stimuli),
            "materialized_assignments": len(assignments),
            "tts_run": False,
            "asr_training_run": False,
            "reflection_called": False,
        }
        save(args.output, "smoke_report", report)
        print(json.dumps(report), flush=True)
        return
    plan = compile_evaluation_plan(
        protocol,
        evidence,
        proposal,
        before_prediction_column=protocol.initial_adapter_prediction_column,
        after_prediction_column=protocol.zero_shot_prediction_column,
    )
    save(args.output, "evaluation_plan", plan)
    placeholder = hashlib.sha256(
        b"counterfactual interface fixture; not an artifact"
    ).hexdigest()
    synthesis = SynthesisReceipt(
        schema_version="1",
        run_id=protocol.run_id,
        round_index=1,
        artifact_mode=MODE,
        status="complete",
        assignments_canonical_sha256=sidecar.assignments_canonical_sha256,
        requested=24,
        generated=24,
        failed=0,
        samples=[
            SynthesizedSample(
                sample_id=item.sample_id,
                audio_file_sha256=placeholder,
                duration_seconds=0.0,
            )
            for item in sidecar.samples
        ],
    )
    training = TrainingReceipt(
        schema_version="1",
        run_id=protocol.run_id,
        round_index=1,
        artifact_mode=MODE,
        status="complete",
        parent_checkpoint_directory_sha256=protocol.initial_adapter.sha256,
        new_checkpoint_directory_sha256=placeholder,
        train_manifest_file_sha256=placeholder,
        training_index_file_sha256=placeholder,
        trainer_state_file_sha256=placeholder,
        real_rows=protocol.real_replay_count_per_round,
        synthetic_rows=24,
        total_rows=protocol.real_replay_count_per_round + 24,
        expected_optimizer_steps=protocol.lora.optimizer_steps_per_round,
        actual_optimizer_steps=protocol.lora.optimizer_steps_per_round,
    )
    result = compile_round_result(
        protocol,
        evidence,
        proposal,
        plan,
        sidecar,
        synthesis,
        training,
        before_rows=m0_rows,
        after_rows=zero_rows,
        before_predictions_file_sha256=protocol.initial_adapter_dev_predictions.sha256,
        after_predictions_file_sha256=protocol.zero_shot_dev_predictions.sha256,
    )
    pack = compile_reflection_pack(
        protocol,
        evidence,
        proposal,
        plan,
        sidecar,
        result,
        prescription_prompt_file_sha256=sha256_file(prescribe_prompt),
        reflection_prompt_file_sha256=sha256_file(reflect_prompt),
        skill_version=skill.version,
        skill_file_sha256=skill.file_sha256,
    )
    for name, value in (
        ("synthesis_fixture_receipt", synthesis),
        ("training_fixture_receipt", training),
        ("round_result", result),
        ("reflection_pack", pack),
    ):
        save(args.output, name, value)
    request = build_reflection_request(
        protocol,
        evidence,
        proposal,
        pack,
        system_prompt=reflect_prompt.read_text("utf-8"),
        skill_text=skill.read_text(),
    )
    response = request_once(protocol, request, args.output / "02_reflect")
    reflection = parse_reflection_response(response, pack)
    save(args.output / "02_reflect", "reflection", reflection)
    episode = compile_intervention_summary(
        protocol, evidence, proposal, pack, reflection
    )
    save(args.output, "intervention_history", [episode.model_dump(mode="json")])
    next_evidence = build_evidence_pack(
        protocol,
        round_index=2,
        current_rows=zero_rows,
        current_prediction_column=protocol.zero_shot_prediction_column,
        current_checkpoint_sha256=placeholder,
        current_predictions_sha256=protocol.zero_shot_dev_predictions.sha256,
        previous_rows=m0_rows,
        previous_prediction_column=protocol.initial_adapter_prediction_column,
        previous_checkpoint_sha256=protocol.initial_adapter.sha256,
        previous_predictions_sha256=protocol.initial_adapter_dev_predictions.sha256,
        intervention_history=[episode],
        **common,
    )
    save(args.output, "round_002_evidence", next_evidence)
    next_skill = skills.bind_round(2)
    request = build_therapist_request(
        protocol,
        next_evidence,
        system_prompt=prescribe_prompt.read_text("utf-8"),
        skill_text=next_skill.read_text(),
    )
    if args.reflection_only_fixture:
        save(args.output / "03_next_prescribe", "request", request)
        save(
            args.output,
            "smoke_report",
            {
                "status": "reflection_and_feedback_interface_passed",
                "artifact_mode": MODE,
                "real_agent_calls": 1,
                "automatic_prescription_passed": False,
                "next_agent_called": False,
                "immediate_history_present": next_evidence.intervention_history[
                    0
                ].episode_memory.next_round_guidance
                == reflection.next_round_guidance,
                "skill_same_until_scheduled_update": next_skill == skill,
                "compiled_expectation_verdict": reflection.compiled_expectation_verdict,
                "hypothesis_verdict": reflection.hypothesis_verdict,
                "next_guidance": reflection.next_round_guidance.model_dump(mode="json"),
                "tts_run": False,
                "asr_training_run": False,
            },
        )
        print(
            "Reflection and next-request memory interface passed; prescription fixture was hand-authored.",
            flush=True,
        )
        return
    response = request_once(protocol, request, args.output / "03_next_prescribe")
    next_proposal = parse_therapist_response(response, protocol, next_evidence)
    save(args.output / "03_next_prescribe", "proposal", next_proposal)
    next_assignments = materialize_proposal(
        next_proposal,
        next_evidence,
        protocol,
        output_root=args.output / "not_generated_round_002",
    )
    next_sidecar = build_materialization_sidecar(
        next_proposal, next_evidence, protocol, next_assignments
    )
    save(args.output, "round_002_materialization_sidecar", next_sidecar)
    history_ref = f"intervention.{episode.ledger_id}"
    history_cited = any(
        history_ref in claim.evidence_ids for claim in next_proposal.assessment_claims
    )
    report = {
        "status": "passed",
        "artifact_mode": MODE,
        "real_agent_calls": 3,
        "model": protocol.agent.model,
        "reasoning_effort": protocol.agent.reasoning_effort,
        "round_001_roles": {item.role: item.copies for item in sidecar.by_role},
        "round_002_roles": {item.role: item.copies for item in next_sidecar.by_role},
        "capability_results": [
            item.model_dump(mode="json") for item in result.response.capability_changes
        ],
        "compiled_expectation_verdict": reflection.compiled_expectation_verdict,
        "hypothesis_verdict": reflection.hypothesis_verdict,
        "next_guidance": reflection.next_round_guidance.model_dump(mode="json"),
        "next_proposal_cites_history": history_cited,
        "immediate_history_present": next_evidence.intervention_history[
            0
        ].episode_memory.next_round_guidance
        == reflection.next_round_guidance,
        "skill_same_until_scheduled_update": next_skill == skill,
        "tts_run": False,
        "asr_training_run": False,
    }
    save(args.output, "smoke_report", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
