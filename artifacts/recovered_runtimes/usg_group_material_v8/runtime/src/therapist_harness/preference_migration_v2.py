"""Migrate the stopped SG boundary into a fresh execution ledger; never rewrite v1."""

from pathlib import Path

from . import preference_loop as loop
from .memory import SkillMemory
from .schema import RunProtocol
from .store import RunStore, sha256_file, write_immutable_bytes


def migrate(source, destination):
    """New ledger/version, same experiment/sample identities; preserve the stopped v1 ledger."""
    source, destination = Path(source), Path(destination)
    old_store = RunStore(source); old_store.verify_integrity()
    state = old_store.current()
    if state.round_index != 1 or state.phase != "materialized":
        raise ValueError("migration currently supports the stopped first-round generation boundary")
    stop = loop.read(source / "stop_receipt.json")
    if stop.get("status") != "stopped_by_user_for_execution_optimization":
        raise ValueError("source execution must have an explicit stop receipt")
    for path, reference in stop["preserved_artifacts"].items():
        if sha256_file(source / path) != reference["sha256"]:
            raise ValueError("stopped source artifact changed")
    snapshots = loop.read(Path(stop["source_snapshot"]))
    old_hashes = loop.read(source / "sources.json")
    for path, digest in old_hashes.items():
        if snapshots[path]["sha256"] != digest or sha256_file(Path(snapshots[path]["snapshot"])) != digest:
            raise ValueError("old execution source snapshot changed")
    data = old_store.protocol().model_dump(mode="json")
    if data["preference_loop"]["comparison"] != "dpo_only":
        raise ValueError("only the already-authorized DPO-only experiment can be migrated")
    data["preference_loop"]["execution_version"] = "persistent_dual_gpu_v2"
    p = RunProtocol.model_validate(data)
    store = RunStore.initialize(p, destination)
    copied = {}
    def copy(name):
        original, target = source / name, destination / name
        write_immutable_bytes(target, original.read_bytes())
        copied[name] = {"source": str(original), "sha256": sha256_file(original)}
    for name in ("initial_skill.md", "prompts/therapist.md", "prompts/reflection.md", "real_selection.jsonl", "split_audit.json"):
        copy(name)
    SkillMemory.initialize(destination / "skills", destination / "initial_skill.md")
    SkillMemory(destination / "skills").bind_round(1)
    loop.freeze_sources(destination)
    old_policy = loop.read(source / "execution_policy.json")
    loop.save(destination / "execution_policy.json", {**old_policy, "execution_version": "persistent_dual_gpu_v2",
        "migrated_from": str(source), "run_id_policy": "same experiment run_id and sample_ids; new execution root/ledger"})
    d = loop.round_dir(destination, 1)
    copy("round_001/evidence.json")
    store.advance("assessed", {"evidence": d / "evidence.json"})
    for name in ("prescribe/request.json", "prescribe/response.json", "proposal.json"):
        copy("round_001/" + name)
    old_plan = loop.read(source / "round_001/evaluation_plan.json")
    new_plan = loop.compile_evaluation_plan(p, loop.load(d / "evidence.json", loop.EvidencePack),
        loop.load(d / "proposal.json", loop.Proposal),
        before_prediction_column=old_plan["before_snapshot"]["prediction_column"],
        after_prediction_column=old_plan["semantics"]["after_prediction_column"],
        off_target_max_per_level_per_direction=old_plan["off_target_policy"]["max_per_level_per_direction"])
    comparison_plan = new_plan.model_dump(mode="json")
    comparison_plan["bindings"]["protocol_canonical_sha256"] = old_plan["bindings"]["protocol_canonical_sha256"]
    if comparison_plan != old_plan:
        raise ValueError("execution migration changed the evaluation semantics")
    loop.save(d / "evaluation_plan.json", new_plan)
    store.advance("proposed", {"agent_request": d / "prescribe/request.json", "agent_response": d / "prescribe/response.json",
        "proposal": d / "proposal.json", "evaluation_plan": d / "evaluation_plan.json"})
    loop.materialize_saved_proposal(destination, 1)
    before, after = loop.rows(source / "round_001/synthesis_manifest.jsonl"), loop.rows(d / "synthesis_manifest.jsonl")
    without_output = lambda item: {k: v for k, v in item.items() if k != "output_wav"}
    if [without_output(x) for x in before] != [without_output(x) for x in after]:
        raise ValueError("migration changed content/reference/seed/sample order")
    assignments = {x["sample_id"]: x for x in before}
    reused = []
    from .transcript_review import validate_transcript_review
    for path in sorted((source / "round_001/generation/attempts").glob("*/assignment_result.json")):
        result = loop.read(path); assignment = assignments[result["sample_id"]]
        for key in ("target_text", "reference_text", "reference_wav", "reference_wav_sha256", "output_wav"):
            if result[key] != assignment[key]:
                raise ValueError("completed assignment differs from original manifest")
        if result["assignment_seed"] != assignment["seed"] or sha256_file(Path(result["output_wav"])) != result["output_wav_sha256"]:
            raise ValueError("completed assignment seed/audio changed")
        review = result["review"]
        validate_transcript_review(review, p.preference_loop.review_asr_adapter_sha256, p.preference_loop.review_asr_base_model_sha256)
        if (review["candidate_audio_sha256"] != result["output_wav_sha256"] or review["target_text"] != assignment["target_text"]
                or review["reference_text"] != assignment["reference_text"] or review["reference_audio_sha256"] != assignment["reference_wav_sha256"]):
            raise ValueError("completed review conditioning differs from the frozen assignment")
        reused.append({"sample_id": result["sample_id"], "assignment_result": str(path),
                       "assignment_result_sha256": sha256_file(path), "audio_sha256": result["output_wav_sha256"]})
    migration = {"schema_version": "mas_dpo_execution_migration_v1", "source": str(source), "destination": str(destination),
        "source_protocol_sha256": sha256_file(source / "protocol.json"), "new_protocol_sha256": sha256_file(destination / "protocol.json"),
        "source_stop_receipt_sha256": sha256_file(source / "stop_receipt.json"),
        "same_experiment_run_id": p.run_id, "changed_protocol_fields": ["preference_loop.execution_version"],
        "original_ledger_modified": False, "copied": copied, "assignments": reused,
        "rebuilt_evaluation_plan": {"source": str(source / "round_001/evaluation_plan.json"),
            "source_sha256": sha256_file(source / "round_001/evaluation_plan.json"),
            "new_sha256": sha256_file(d / "evaluation_plan.json"),
            "only_change": "bindings.protocol_canonical_sha256", "evaluation_semantics_identical": True},
        "requested_audio": len(after), "reusable_completed_audio": len(reused), "pending_audio": len(after) - len(reused)}
    loop.save(d / "generation_reuse.json", migration)
    loop.save(destination / "migration.json", migration)
    store.verify_integrity()
    return migration
