"""Bounded three-arm real-data curriculum pilot; no automatic retry or TTS."""

# ruff: noqa: E402 -- standalone experiment reuses the maintained training monitor.
import argparse
from collections import Counter
import csv
import getpass
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "examples"), str(REPO / "finetune/scripts")]

from openai.lib._pydantic import to_strict_json_schema
from absorption_probe import rows, read, save_rows, retention
from train_m0_small_dose_pair import stratified_sample
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.real_curriculum import (CurriculumProposal, CurriculumReflection,
    catalog, compile_proposal, rule_selection, selection_stats, validate_selection)
from therapist_harness.schema import AgentConfig
from therapist_harness.store import sha256_file, sha256_directory, write_immutable_json as save

RECORDS = Path("/data/qwen3-asr/records/therapist_mas_setting_d_v1")
ROOT = RECORDS / "real_curriculum_pilot_v1"
M0 = RECORDS / "real_only_m0_b8g4_utterance_mean_v1/training/checkpoint-50"
TRAIN = Path("/data/qwen3-asr/finetune/data_setting_d_v1/setting_d_train.jsonl")
FEEDBACK = Path("/data/qwen3-asr/finetune/data_setting_d_merged_test_v1/setting_d_merged_test.jsonl")
M0_PRED = RECORDS / "therapist_m0_b8g4_pilot_v1/merged_test_reassessment_v1/m0_step50/predictions.jsonl"
ZERO_TRAIN = Path("/data/qwen3-asr/inference/outputs/setting_d_seen_patient_unseen_content_v1_qc_frozen/setting_d_train.csv")
BASE = "/data/qwen3-asr/models/Qwen3-ASR-1.7B"
PROMPTS = REPO / "src/therapist_harness/prompts"
ARMS = ("random", "difficulty", "agent")


def metrics(path):
    return evaluate_snapshot(rows(path), prediction_column="prediction", critical_cer_threshold=.5).model_dump(mode="json")


def prepare(root):
    train, feedback = rows(TRAIN), rows(FEEDBACK)
    assert len(train) == 2117 and len(feedback) == 511
    for field in ("utt_id", "prompt_id", "prompt_normalized_sha16", "prompt_jyutping_sha16"):
        assert not {r[field] for r in train} & {r[field] for r in feedback}, field
    current = rows(root / "m0_train/predictions.jsonl")
    assert {r["utt_id"] for r in current} == {r["utt_id"] for r in train}
    with ZERO_TRAIN.open() as f:
        zero = {r["utt_id"]: r for r in csv.DictReader(f)}
    cat, mapping = catalog(current, zero)
    save(root / "catalog_m0.json", cat)
    save(root / "local_recording_map.json", mapping)
    source_paths = [Path(__file__), REPO / "src/therapist_harness/real_curriculum.py",
        REPO / "src/therapist_harness/agent.py", REPO / "src/therapist_harness/evidence.py",
        REPO / "src/therapist_harness/phonology.py", REPO / "src/therapist_harness/text.py",
        REPO / "examples/absorption_probe.py", REPO / "examples/therapist_dose_monitor.py",
        *[REPO / "finetune/scripts" / name for name in ("qwen3_asr_lora_sft.py", "train_m0_small_dose_pair.py",
            "train_setting_d_m0.py", "evaluate_asr_checkpoint.py")],
        REPO / "finetune/official/qwen3_asr_sft.py", PROMPTS / "real_curriculum.md",
        PROMPTS / "real_curriculum_reflection.md", REPO / "docs/real_curriculum_pilot_v1.zh.md"]
    agent = read(RECORDS / "therapist_m0_b8g4_pilot_v1/protocol.json")["agent"]
    save(root / "protocol.json", {"m0_predictions": str(M0_PRED), "m0_metrics": metrics(M0_PRED),
        "M0": str(M0), "M0_sha256": sha256_directory(M0), "agent": agent,
        "source_sha256": {str(p): sha256_file(p) for p in source_paths},
        "data_sha256": {str(p): sha256_file(p) for p in (TRAIN, FEEDBACK, ZERO_TRAIN, M0_PRED)},
        "rounds_max": 3, "doublecheck_max": 1, "rows_per_round": 96, "synthetic": 0,
        "optimizer_steps": 3, "passes": 1, "shared_background": 32, "method_selected": 64,
        "feedback_scope": "existing merged 511, exploratory, not a sealed test",
        "disk_free_start_bytes": shutil.disk_usage(root).free})
    save(root / "data_audit.json", {"train_rows": len(train), "train_patients": len({r['speaker_id'] for r in train}),
        "train_contents": len({r['clean_gt'] for r in train}), "feedback_rows": len(feedback),
        "content_overlap": 0, "m0_train_metrics": metrics(root / "m0_train/predictions.jsonl")})


def background(root, number):
    mapping = read(root / "local_recording_map.json")
    reverse = {r["utt_id"]: k for k, r in mapping.items()}
    chosen = stratified_sample(list(mapping.values()), 32, random.Random(1000 + number))
    return [reverse[r["utt_id"]] for r in chosen]


def packet(root, number, cat, feedback, history):
    # Compact rows avoid repeating schema keys 2117 times. Transcripts stay complete.
    columns = list(cat["recordings"][0])
    return {"round": number, "task": "Select 64 real recording IDs; prescribe, do not reflect here.",
        "background_ids": background(root, number), "contents": cat["contents"],
        "recording_columns": columns, "recordings": [[r[k] for k in columns] for r in cat["recordings"]],
        "table_encoding": "Each recordings array follows recording_columns exactly.",
        "feedback_metrics": feedback, "history": history}


def call(root, directory, payload, model, prompt):
    cfg = AgentConfig.model_validate(read(root / "protocol.json")["agent"])
    request = {"model": cfg.model, "reasoning": {"effort": cfg.reasoning_effort}, "store": False,
        "stream": True, "max_output_tokens": cfg.max_output_tokens,
        "input": [{"role": "system", "content": prompt.read_text()},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
        "text": {"format": {"type": "json_schema", "name": model.__name__, "strict": True,
                              "schema": to_strict_json_schema(model)}}}
    raw = json.dumps(request, ensure_ascii=False)
    assert "/data/" not in raw and "vlink" not in raw and "speaker_id" not in raw
    save(directory / "request.json", request)
    started = time.monotonic()
    response = execute_therapist_request(SimpleNamespace(agent=cfg), request)
    save(directory / "response.json", serialize_response(response))
    save(directory / "timing.json", {"seconds": time.monotonic()-started,
         "usage": response.usage.model_dump(mode="json"), "request_characters": len(raw)})
    assert response.status == "completed", response.status
    value = model.model_validate_json(response.output_text)
    save(directory / "parsed.json", value.model_dump(mode="json"))
    print(f"[agent] {directory.name} completed in {time.monotonic()-started:.1f}s", flush=True)
    return value


def smoke(root):
    cat = read(root / "catalog_m0.json")
    evidence = packet(root, 1, cat, read(root / "protocol.json")["m0_metrics"], [])
    save(root / "smoke/evidence.json", evidence)
    proposals, ids = [], []
    for i in (1, 2):
        proposal = call(root, root / f"smoke/call_{i}", evidence, CurriculumProposal, PROMPTS / "real_curriculum.md")
        proposals.append(proposal)
        ids.append(compile_proposal(proposal, cat["recordings"], evidence["background_ids"]))
    save(root / "smoke/acceptance.json", {"valid_calls": 2, "training_uses": "call_1 (predeclared)",
        "intersection": len(set(ids[0]) & set(ids[1])), "union": len(set(ids[0]) | set(ids[1])),
        "selected_stats": [selection_stats(x + evidence["background_ids"], cat["recordings"]) for x in ids],
        "role_counts": [dict(Counter({g.role: len(g.sample_ids) for g in p.groups})) for p in proposals]})


def make_arm(root, name, ids, cat, parent, seed, shared):
    validate_selection(ids, cat["recordings"], shared)
    mapping = read(root / "local_recording_map.json")
    # Same background positions in each effective batch; same shuffle seed per arm.
    shared, ids = list(shared), list(ids)
    rng = random.Random(seed)
    rng.shuffle(shared)
    rng.shuffle(ids)
    s, b = iter(ids), iter(shared)
    ordered = [next(b) if (i+1)*32//96 > i*32//96 else next(s) for i in range(96)]
    d = root / name
    save_rows(d / "train.jsonl", [{k: mapping[i][k] for k in ("audio", "text", "prompt")} for i in ordered])
    save(d / "training_index.json", [{"row": n, "id": mapping[i]["utt_id"], "catalog_id": i,
        "source": "real", "audio": mapping[i]["audio"], "role": "background" if i in shared else "selected"}
        for n, i in enumerate(ordered)])
    command = [sys.executable, str(REPO / "finetune/scripts/qwen3_asr_lora_sft.py"),
        "--model_path", BASE, "--train_file", str(d / "train.jsonl"), "--eval_file", "",
        "--output_dir", str(d / "training"), "--lora_init_adapter_path", str(parent),
        "--lora_rank", "16", "--lora_alpha", "4", "--lora_dropout", ".05",
        "--lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--batch_size", "8", "--grad_acc", "4", "--lr", "2e-5", "--epochs", "1", "--max_steps", "3",
        "--lr_scheduler_type", "constant", "--warmup_ratio", "0", "--seed", str(seed),
        "--save_strategy", "steps", "--save_steps", "1000000", "--save_total_limit", "0",
        "--save_final_checkpoint", "0", "--report_to", "none", "--log_steps", "1"]
    save(d / "config.json", {"command": command, "parent": str(parent), "parent_sha256": sha256_directory(parent),
        "train_sha256": sha256_file(d / "train.jsonl"), "audio_sha256": {mapping[i]["audio"]: sha256_file(Path(mapping[i]["audio"])) for i in ordered},
        "epochs": [1], "observe_steps": [3], "real_count": 96, "synthetic_count": 0, "seed": seed,
        "selected_ids": ids, "background_ids": shared, "statistics": selection_stats(ordered, cat["recordings"])})


def train_arm(root, name):
    started = time.monotonic()
    print(f"[train] start {name}", flush=True)
    with (root / f"{name}.log").open("x") as log:
        subprocess.run([sys.executable, str(REPO / "examples/absorption_probe.py"), "--root", str(root), "--train", name],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    result = read(root / name / "result.json")
    state = read(root / name / "training/checkpoint-3/trainer_state.json")
    assert state["global_step"] == 3 and state["epoch"] == 1.
    assert all(abs(x["learning_rate"]-2e-5) < 1e-12 for x in state["log_history"] if "learning_rate" in x)
    save(root / name / "acceptance.json", {"steps": 3, "passes": 1, "training_seconds_including_evaluation": time.monotonic()-started,
        "disk_free_bytes": shutil.disk_usage(root).free})
    print(f"[train] {name}: CER={100*result['history'][-1]['metrics']['overall']['mean_utterance_cer']:.4f}%", flush=True)
    return result


def can_continue(results, m0):
    a = results["agent"]["history"][-1]["metrics"]
    best = min(results[k]["history"][-1]["metrics"]["overall"]["mean_utterance_cer"] for k in ("random", "difficulty"))
    return (a["overall"]["mean_utterance_cer"] < m0["overall"]["mean_utterance_cer"]
        and a["overall"]["mean_utterance_cer"] <= best + .001
        and all(a[k]["mean_utterance_cer"] <= m0[k]["mean_utterance_cer"]+.002 for k in ("easy", "medium", "hard")))


def assess_train(root, number, arm, parent):
    destination = root / f"r{number:02d}_{arm}_train_diagnostic"
    with destination.with_suffix(".log").open("x") as log:
        subprocess.run([sys.executable, str(REPO / "finetune/scripts/evaluate_asr_checkpoint.py"),
            "--base-model-path", BASE, "--adapter-path", str(parent), "--split-jsonl", str(TRAIN),
            "--out-dir", str(destination), "--batch-size", "8", "--critical-cer-threshold", ".5",
            "--max-new-tokens", "256"], stdout=log, stderr=subprocess.STDOUT, check=True)
    with ZERO_TRAIN.open() as f:
        zero = {r["utt_id"]: r for r in csv.DictReader(f)}
    cat, _ = catalog(rows(destination / "predictions.jsonl"), zero)
    save(destination / "catalog.json", cat)
    return cat


def run_round(root, number):
    m0 = read(root / "protocol.json")["m0_metrics"]
    shared = background(root, number)
    results = {}
    for arm in ARMS:
        name = f"r{number:02d}_{arm}"
        parent = M0 if number == 1 else root / f"r{number-1:02d}_{arm}/training/checkpoint-3"
        cat = read(root / "catalog_m0.json") if number == 1 else assess_train(root, number, arm, parent)
        if arm == "agent":
            if number == 1:
                proposal = CurriculumProposal.model_validate(read(root / "smoke/call_1/parsed.json"))
            else:
                history = [read(root / f"round_{n:02d}_episode.json") for n in range(1, number)]
                previous = history[-1]["after_metrics"]
                evidence = packet(root, number, cat, previous, history)
                proposal = call(root, root / f"r{number:02d}_prescribe", evidence, CurriculumProposal, PROMPTS / "real_curriculum.md")
            ids = compile_proposal(proposal, cat["recordings"], shared)
        else:
            ids = rule_selection(cat["recordings"], shared, arm, 2000+number)
        make_arm(root, name, ids, cat, parent, 42, shared)
        results[arm] = train_arm(root, name)
    save(root / f"round_{number:02d}_results.json", results)
    after_path = root / f"r{number:02d}_agent/evaluation/merged_step_3/predictions.jsonl"
    before_path = M0_PRED if number == 1 else root / f"r{number-1:02d}_agent/evaluation/merged_step_3/predictions.jsonl"
    episode = {"round": number, "proposal": proposal.model_dump(mode="json"),
        "selection": results["agent"]["config"]["statistics"], "before_metrics": metrics(before_path),
        "after_metrics": metrics(after_path), "paired_transitions": retention(rows(before_path), rows(after_path)),
        "fixed_train_fit": results["agent"]["fit_history"]}
    reflection = call(root, root / f"r{number:02d}_reflect", episode, CurriculumReflection, PROMPTS / "real_curriculum_reflection.md")
    episode["reflection"] = reflection.model_dump(mode="json")
    save(root / f"round_{number:02d}_episode.json", episode)
    proceed = can_continue(results, m0)
    save(root / f"round_{number:02d}_decision.json", {"continue": proceed, "evaluated_fixed_endpoint": 3})
    print(f"[decision] round {number} continue={proceed}", flush=True)


def doublecheck(root):
    # One fixed-course seed replication, not another proposal or hyperparameter search.
    cat = read(root / "catalog_m0.json")
    results = {}
    for arm in ARMS:
        original = read(root / f"r01_{arm}/config.json")
        name = f"check_{arm}"
        make_arm(root, name, original["selected_ids"], cat, M0, 43, original["background_ids"])
        results[arm] = train_arm(root, name)
    save(root / "doublecheck_results.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "round", "doublecheck"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--number", type=int, default=1)
    args = parser.parse_args()
    if args.stage in ("smoke", "round") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Gateway API key (memory only): ")
    if args.stage == "prepare":
        prepare(args.root)
    elif args.stage == "smoke":
        smoke(args.root)
    elif args.stage == "round":
        run_round(args.root, args.number)
    else:
        doublecheck(args.root)
