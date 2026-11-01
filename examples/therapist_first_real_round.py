"""Execute the first 6-text/24-audio real intervention on the new 8x4 M0.

Stages are explicit and stop on failure. No counterfactual outcomes, retries,
checkpoint selection, or test evaluation are used here.
"""

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from therapist_harness.agent import (
    build_therapist_request, execute_therapist_request,
    parse_therapist_response, serialize_response,
)
from therapist_harness.cli import load_protocol
from therapist_harness.evidence_pack import build_evidence_pack_from_files
from therapist_harness.materialize import materialize_proposal, build_materialization_sidecar
from therapist_harness.memory import SkillMemory
from therapist_harness.outcome import compile_evaluation_plan
from therapist_harness.schema import RunProtocol
from therapist_harness.store import (
    RunStore, sha256_directory, sha256_file, write_immutable_bytes, write_immutable_json,
)

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src/therapist_harness"
M0 = Path("/data/qwen3-asr/records/therapist_mas_setting_d_v1/real_only_m0_b8g4_utterance_mean_v1")


def save(path, value):
    write_immutable_json(path, value.model_dump(mode="json") if hasattr(value, "model_dump") else value)


def request_once(protocol, request, directory):
    save(directory / "request.json", request)
    start = time.monotonic()
    response = execute_therapist_request(protocol, request)
    save(directory / "response.json", serialize_response(response))
    timing = {"seconds": time.monotonic() - start, "usage": response.usage.model_dump(mode="json")}
    save(directory / "timing.json", timing)
    print(json.dumps(timing), flush=True)
    return response


def prepare(root):
    payload = load_protocol(REPO / "configs/therapist_setting_d_v1.yaml").model_dump(mode="json")
    adapter = M0 / "training/checkpoint-50"
    predictions = M0 / "evaluation/dev_step_50/predictions.csv"
    payload.update(protocol_status="frozen", run_id="therapist_m0_b8g4_pilot_v1", round_count=1,
                   synthetic_count_per_round=24, unique_stimulus_count_per_round=6,
                   skill_update_rounds=[], initial_adapter_prediction_column="prediction")
    payload["initial_adapter"] = {"path": str(adapter), "kind": "directory", "sha256": sha256_directory(adapter)}
    payload["initial_adapter_dev_predictions"] = {"path": str(predictions), "sha256": sha256_file(predictions), "expected_rows": 258}
    payload["lora"].update(per_device_batch_size=8, gradient_accumulation_steps=4, optimizer_steps_per_round=67)
    protocol = RunProtocol.model_validate(payload)
    store = RunStore.initialize(protocol, root)
    evidence = build_evidence_pack_from_files(
        protocol, round_index=1, current_predictions=predictions,
        current_prediction_column="prediction", current_checkpoint=adapter,
        m0_predictions=predictions, m0_prediction_column="prediction",
        zero_shot_predictions=Path(protocol.zero_shot_dev_predictions.path),
        zero_shot_prediction_column=protocol.zero_shot_prediction_column,
    )
    save(root / "evidence.json", evidence)
    store.advance("assessed", {"evidence": root / "evidence.json"})
    SkillMemory.initialize(root / "skills", PACKAGE / "skills/v000/SKILL.md").bind_round(1)
    for name in ("therapist.md", "reflection.md"):
        write_immutable_bytes(root / "prompts" / name, (PACKAGE / "prompts" / name).read_bytes())
    print(json.dumps({"phase": "assessed", "run": str(root), "clusters": len(evidence.error_clusters)}), flush=True)


def prescribe(root):
    from therapist_harness.schema import EvidencePack

    store = RunStore(root)
    protocol = store.protocol()
    evidence = EvidencePack.model_validate_json((root / "evidence.json").read_text())
    skill = SkillMemory(root / "skills").bind_round(1)
    os.environ[protocol.agent.api_key_env] = getpass.getpass("Gateway API key (not saved): ")
    request = build_therapist_request(protocol, evidence,
        system_prompt=(root / "prompts/therapist.md").read_text(), skill_text=skill.read_text())
    response = request_once(protocol, request, root / "prescribe")
    proposal = parse_therapist_response(response, protocol, evidence)
    save(root / "proposal.json", proposal)
    plan = compile_evaluation_plan(protocol, evidence, proposal,
                                  before_prediction_column="prediction", after_prediction_column="prediction")
    save(root / "evaluation_plan.json", plan)
    store.advance("proposed", {"agent_request": root / "prescribe/request.json",
        "agent_response": root / "prescribe/response.json", "proposal": root / "proposal.json",
        "evaluation_plan": root / "evaluation_plan.json"})
    (root / "audio").mkdir()
    assignments = materialize_proposal(proposal, evidence, protocol, output_root=root / "audio")
    sidecar = build_materialization_sidecar(proposal, evidence, protocol, assignments)
    save(root / "materialization_sidecar.json", sidecar)
    write_immutable_bytes(root / "synthesis_manifest.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in assignments).encode())
    store.advance("materialized", {"synthesis_manifest": root / "synthesis_manifest.jsonl",
                                    "materialization_sidecar": root / "materialization_sidecar.json"})
    print(json.dumps({"phase": "materialized", "proposal": proposal.model_dump(mode="json")}, ensure_ascii=False), flush=True)


def synthesize(root):
    from therapist_harness.round_schema import SynthesisReceipt, SynthesizedSample

    store = RunStore(root)
    protocol = store.protocol()
    assert store.current().phase == "materialized"
    with (root / "synthesis.log").open("x") as log:
        subprocess.run([sys.executable, str(REPO / "synthesis/cosyvoice3_patient_sft/synthesize_manifest.py"),
            "--manifest", str(root / "synthesis_manifest.jsonl"), "--summary", str(root / "tts_outputs.json")],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    output = json.loads((root / "tts_outputs.json").read_text())
    sidecar = json.loads((root / "materialization_sidecar.json").read_text())
    assert output["status"] == "complete" and output["output_count"] == 24
    receipt = SynthesisReceipt(schema_version="1", run_id=protocol.run_id, round_index=1,
        status="complete", assignments_canonical_sha256=sidecar["assignments_canonical_sha256"],
        requested=24, generated=24, failed=0, samples=[SynthesizedSample(
            sample_id=row["sample_id"], audio_file_sha256=row["output_wav_sha256"],
            duration_seconds=row["duration_seconds"]) for row in output["outputs"]])
    save(root / "synthesis_receipt.json", receipt)
    real = [json.loads(line) for line in Path(protocol.real_train.path).read_text().splitlines()]
    rows = [{"audio": r["audio"], "text": r["text"], "prompt": r["prompt"]} for r in real]
    index = [{"row": i, "source": "real", "utt_id": r["utt_id"], "audio": r["audio"]} for i, r in enumerate(real)]
    for r in output["outputs"]:
        index.append({"row": len(rows), "source": "synthetic", "sample_id": r["sample_id"],
                      "audio": r["output_wav"], "audio_sha256": r["output_wav_sha256"]})
        rows.append({"audio": r["output_wav"], "text": "language Cantonese<asr_text>" + r["target_text"], "prompt": ""})
    assert len(real) == 2117 and len(rows) == 2141
    write_immutable_bytes(root / "train.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())
    save(root / "training_index.json", index)
    store.advance("synthesized", {"synthesis_summary": root / "synthesis_receipt.json",
        "train_manifest": root / "train.jsonl", "training_index": root / "training_index.json"})
    print(json.dumps({"phase": "synthesized", "real": len(real), "synthetic": 24,
                      "synthetic_seconds": sum(r["duration_seconds"] for r in output["outputs"])}), flush=True)


def reassess(root):
    from therapist_harness.evidence_pack import load_prediction_rows
    from therapist_harness.schema import EvidencePack, Proposal
    from therapist_harness.round_schema import EvaluationPlan, MaterializationSidecar, SynthesisReceipt, TrainingReceipt
    from therapist_harness.outcome import compile_round_result, compile_reflection_pack

    store = RunStore(root)
    protocol = store.protocol()
    assert store.current().phase == "synthesized"
    checkpoint = root / "training/checkpoint-67"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    receipt = TrainingReceipt(schema_version="1", run_id=protocol.run_id, round_index=1, status="complete",
        parent_checkpoint_directory_sha256=protocol.initial_adapter.sha256,
        new_checkpoint_directory_sha256=sha256_directory(checkpoint),
        train_manifest_file_sha256=sha256_file(root / "train.jsonl"),
        training_index_file_sha256=sha256_file(root / "training_index.json"),
        trainer_state_file_sha256=sha256_file(checkpoint / "trainer_state.json"),
        real_rows=2117, synthetic_rows=24, total_rows=2141,
        expected_optimizer_steps=67, actual_optimizer_steps=state["global_step"])
    save(root / "training_receipt.json", receipt)
    store.advance("trained", {"checkpoint": checkpoint, "training_receipt": root / "training_receipt.json"})
    evidence = EvidencePack.model_validate_json((root / "evidence.json").read_text())
    proposal = Proposal.model_validate_json((root / "proposal.json").read_text())
    plan = EvaluationPlan.model_validate_json((root / "evaluation_plan.json").read_text())
    sidecar = MaterializationSidecar.model_validate_json((root / "materialization_sidecar.json").read_text())
    synthesis = SynthesisReceipt.model_validate_json((root / "synthesis_receipt.json").read_text())
    before = Path(protocol.initial_adapter_dev_predictions.path)
    after = root / "evaluation/dev_step_67/predictions.csv"
    result = compile_round_result(protocol, evidence, proposal, plan, sidecar, synthesis, receipt,
        before_rows=load_prediction_rows(before), after_rows=load_prediction_rows(after),
        before_predictions_file_sha256=sha256_file(before), after_predictions_file_sha256=sha256_file(after))
    save(root / "round_result.json", result)
    skill = SkillMemory(root / "skills").bind_round(1)
    pack = compile_reflection_pack(protocol, evidence, proposal, plan, sidecar, result,
        prescription_prompt_file_sha256=sha256_file(root / "prompts/therapist.md"),
        reflection_prompt_file_sha256=sha256_file(root / "prompts/reflection.md"),
        skill_version=skill.version, skill_file_sha256=skill.file_sha256)
    save(root / "reflection_pack.json", pack)
    store.advance("reassessed", {"after_predictions": after, "round_result": root / "round_result.json",
                                 "reflection_pack": root / "reflection_pack.json"})
    print(json.dumps({"phase": "reassessed", "delta": result.response.paired_delta.model_dump(mode="json")}), flush=True)


def reflect(root):
    from therapist_harness.agent import build_reflection_request, parse_reflection_response
    from therapist_harness.memory import compile_intervention_summary
    from therapist_harness.schema import EvidencePack, Proposal
    from therapist_harness.round_schema import ReflectionPack

    store = RunStore(root)
    protocol = store.protocol()
    assert store.current().phase == "reassessed"
    evidence = EvidencePack.model_validate_json((root / "evidence.json").read_text())
    proposal = Proposal.model_validate_json((root / "proposal.json").read_text())
    pack = ReflectionPack.model_validate_json((root / "reflection_pack.json").read_text())
    skill = SkillMemory(root / "skills").bind_round(1)
    os.environ[protocol.agent.api_key_env] = getpass.getpass("Gateway API key (not saved): ")
    request = build_reflection_request(protocol, evidence, proposal, pack,
        system_prompt=(root / "prompts/reflection.md").read_text(), skill_text=skill.read_text())
    response = request_once(protocol, request, root / "reflect")
    reflection = parse_reflection_response(response, pack)
    save(root / "reflection.json", reflection)
    store.advance("reflected", {"reflection_request": root / "reflect/request.json",
        "reflection_response": root / "reflect/response.json", "reflection": root / "reflection.json"})
    ledger = compile_intervention_summary(protocol, evidence, proposal, pack, reflection)
    save(root / "ledger_entry.json", ledger)
    store.advance("committed", {"ledger_entry": root / "ledger_entry.json"})
    store.verify_integrity()
    print(json.dumps({"phase": "committed", "ledger": ledger.model_dump(mode="json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = {"prepare": prepare, "prescribe": prescribe, "synthesize": synthesize,
              "reassess": reassess, "reflect": reflect}
    parser.add_argument("stage", choices=stages)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stages[args.stage](args.output.resolve())
