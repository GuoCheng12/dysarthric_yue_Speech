"""Two GPU execution of the unchanged MAS/DPO experiment, with explicit provenance."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import preference_loop as loop
from .store import RunStore, sha256_file
from .preference_sampling_merge import merge_shards


def event(root, number, name, **details):
    payload = {"time": datetime.now(timezone.utc).isoformat(), "monotonic": time.monotonic(),
               "pid": os.getpid(), "round": number, "event": name, **details}
    fd = os.open(Path(root) / "execution_timeline.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, (json.dumps(payload, ensure_ascii=False) + '\n').encode())
    finally:
        os.close(fd)


def sample_source(root, number, source):
    p = RunStore(root).protocol(); c = p.preference_loop; d = loop.round_dir(root, number)
    output = d / f"sampling_{source}"
    if (output / "preference_audit.json").exists():
        return
    inputs = Path(root) / "real_selection.jsonl" if source == "real" else d / "generation/accepted.jsonl"
    command = [sys.executable, str(loop.REPO / "examples/sample_asr_preferences.py"), "sample",
        "--inputs", str(inputs), "--real-train", p.real_train.path, "--feedback", p.feedback_split.path,
        "--out-dir", str(output), "--base-model", p.base_model_dir,
        "--current-adapter", str(loop.checkpoint(root, number - 1)),
        "--expected-parent-sha256", loop.sha256_directory(loop.checkpoint(root, number - 1)),
        "--samples", str(c.samples_per_audio), "--seed", str(p.lora.seed),
        "--temperature", str(c.temperature), "--top-p", str(c.top_p), "--max-new-tokens", str(c.max_new_tokens)]
    if source == "synthetic":
        command += ["--reviews", str(d / "generation/reviews.jsonl")]
    event(root, number, f"{source}_sampling_start", gpu=1 if source == "real" else 0,
          parent_sha256=loop.sha256_directory(loop.checkpoint(root, number - 1)))
    loop.run_command(command, d / f"sampling_{source}.log",
                     env_overrides={"CUDA_VISIBLE_DEVICES": "1" if source == "real" else "0"})
    event(root, number, f"{source}_sampling_end")


def real_branch(root, number):
    loop.verify_sources(root)
    p = RunStore(root).protocol()
    if p.preference_loop.execution_version != "persistent_dual_gpu_v2":
        raise ValueError("real worker requires the dual GPU protocol")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    d = loop.round_dir(root, number)
    sample_source(root, number, "real")
    pairs = loop.rows(d / "sampling_real/preferences.jsonl")
    loop.jsonl(d / "r_dpo/train.jsonl", pairs)
    loop.save(d / "r_dpo/training_index.json", [{"row": i, "source": x["source"], "audio_id": x["audio_id"]}
                                               for i, x in enumerate(pairs)])
    if not (d / "r_dpo/training/training_result.json").exists() and not (d / "r_dpo/no_update.json").exists():
        event(root, number, "real_training_start", gpu=1)
        loop.train_branch(root, number, "r_dpo")
        event(root, number, "real_training_end")
    if not (d / "r_dpo/evaluation/summary.json").exists():
        event(root, number, "real_evaluation_start", gpu=1)
        loop.evaluate_branch(root, number, "r_dpo")
        event(root, number, "real_evaluation_end")
    loop.save(d / "real_branch_complete.json", {"status": "complete", "parent_sha256": loop.sha256_directory(loop.checkpoint(root, number - 1)),
        "sampling_receipt_sha256": sha256_file(d / "sampling_real/sampling_receipt.json"),
        "evaluation_sha256": sha256_file(d / "r_dpo/evaluation/summary.json")})


def stop_child(process):
    if process.poll() is None:
        import psutil
        descendants = psutil.Process(process.pid).children(recursive=True)
        for child in reversed(descendants):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        process.terminate()
        process.wait(timeout=60)


def run_parallel_round(root, number):
    root = Path(root); store = RunStore(root); p = store.protocol(); d = loop.round_dir(root, number)
    phase = store.current().phase
    expected = 1 if phase == "initialized" else store.current().round_index + (phase == "committed")
    if number != expected:
        raise ValueError(f"journal expects round {expected}, received {number}")
    loop.verify_sources(root)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    d.mkdir(parents=True, exist_ok=True)
    event(root, number, "round_start", execution_version="persistent_dual_gpu_v2")
    child = None
    log = None
    try:
        if not (d / "real_branch_complete.json").exists():
            log = (d / "real_branch.log").open("x")
            env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "1"
            child = subprocess.Popen([sys.executable, str(loop.REPO / "examples/mas_asr_dpo_loop.py"),
                "real-branch", str(root), "--round", str(number)], cwd=loop.REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            event(root, number, "real_worker_started", child_pid=child.pid, gpu=1)
        if phase in {"initialized", "committed"}:
            loop.assess(root, number)
        if store.current().phase in {"assessed", "proposed"}:
            loop.prescribe(root, number)
        has_pairs = True
        if store.current().phase == "materialized":
            if not (d / "synthesis_receipt.json").exists():
                event(root, number, "sg_start", gpu=0)
                loop.generate(root, number)
                event(root, number, "sg_end", gpu=0)
            sample_source(root, number, "synthetic")
            begin = time.monotonic()
            while not (d / "sampling_real/preference_audit.json").exists():
                if child is None or child.poll() is not None:
                    raise RuntimeError("real sampling worker exited before complete preferences")
                time.sleep(1)
            event(root, number, "shards_ready", waiting_for_real_seconds=time.monotonic() - begin)
            if not (d / "sampling/preference_audit.json").exists():
                merge_shards(root, number)
            has_pairs = loop.sample_and_pair(root, number)
        if not (d / "mas_dpo/training/training_result.json").exists() and not (d / "mas_dpo/no_update.json").exists():
            event(root, number, "mixed_training_start", gpu=0)
            loop.train_branch(root, number, "mas_dpo")
            event(root, number, "mixed_training_end")
        if not (d / "mas_dpo/evaluation/summary.json").exists():
            event(root, number, "mixed_evaluation_start", gpu=0)
            loop.evaluate_branch(root, number, "mas_dpo")
            event(root, number, "mixed_evaluation_end")
        begin = time.monotonic()
        if child is not None and child.wait() != 0:
            raise RuntimeError("real DPO branch failed; see real_branch.log")
        event(root, number, "both_branches_complete", waiting_for_real_branch_seconds=time.monotonic() - begin)
        if not has_pairs:
            return False
        if store.current().phase in {"synthesized", "trained"}:
            loop.reassess(root, number)
        if store.current().phase in {"reassessed", "reflected"}:
            loop.reflect(root, number)
        if store.current().phase != "committed":
            raise RuntimeError("parallel round did not commit")
        event(root, number, "round_complete")
        return True
    finally:
        if child is not None:
            stop_child(child)
        if log is not None:
            log.close()


def migrate(source, destination):
    from .preference_migration_v2 import migrate as migrate_execution
    return migrate_execution(source, destination)
