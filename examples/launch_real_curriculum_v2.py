"""Run the frozen curriculum experiment outside the launching terminal session."""

import argparse
import getpass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import real_curriculum_v2 as runner


def prepare_dynamic_restart(root, previous_root):
    """Fresh Dynamic; preserve the completed controls and their first prescription."""
    root, previous_root = root.resolve(), previous_root.resolve()
    before = runner.read(previous_root / "protocol.json")
    for arm in ("static", "random"):
        result = runner.read(previous_root / arm / "result.json")
        assert result["status"] == "complete"
        assert [h["step"] for h in result["history"]] == list(runner.ENDPOINTS)
    assert runner.read(previous_root / "first_course.json") == runner.read(previous_root / "static/active_course_20.json")
    runner.prepare(root)
    after = runner.read(root / "protocol.json")
    for key in set(before) | set(after):
        if key not in ("source_sha256", "prepared_artifact_sha256", "free_bytes_preparation"):
            assert before[key] == after[key], key
    changed = {name for name in before["source_sha256"] if before["source_sha256"][name] != after["source_sha256"][name]}
    allowed = {str(runner.REPO / name) for name in (
        "src/therapist_harness/curriculum_v2.py",
        "src/therapist_harness/prompts/curriculum_v2_reflect.md", "docs/real_curriculum_v2.zh.md")}
    assert changed <= allowed, changed
    for name, digest in after["prepared_artifact_sha256"].items():
        assert digest == before["prepared_artifact_sha256"][str(previous_root / Path(name).name)]
    runner.write_immutable_bytes(root / "first_course.json", (previous_root / "first_course.json").read_bytes())
    inputs = [previous_root / "protocol.json", previous_root / "first_course.json", root / "first_course.json"]
    for arm in ("static", "random"):
        (root / arm).symlink_to(previous_root / arm, target_is_directory=True)
        inputs.extend(sorted((previous_root / arm).glob("*.json")))
    runner.save(root / "dynamic_restart.json", {
        "previous_root": str(previous_root), "run_arms": ["dynamic"],
        "reused_completed_arms": ["static", "random"], "changed_sources": sorted(changed),
        "initialization": "original Qwen3-ASR, fresh LoRA; not checkpoint-40 resume",
        "input_sha256": {str(p): runner.sha256_file(p) for p in inputs},
    })


def run_dynamic(root):
    runner.verify_frozen(root)
    for name, digest in runner.read(root / "dynamic_restart.json")["input_sha256"].items():
        assert runner.sha256_file(Path(name)) == digest, name
    assert os.environ.get("OPENAI_API_KEY")
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    runner.save(root / "started.json", {"status": "authorized_dynamic_restart", "time_utc_unix": time.time()})
    process = spawn_arm(root, "dynamic", gpu)
    try:
        code = process.wait()
        runner.save(root / "exit_dynamic.json", {
            "pid": process.pid, "exit_code": code, "finished_utc_unix": time.time()})
        if code != 0:
            raise subprocess.CalledProcessError(code, ["train-arm", "dynamic"])
        runner.report(root)
    finally:
        stop_arm(process)


def spawn_arm(root, arm, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    command = [sys.executable, str(runner.REPO / "examples/real_curriculum_v2.py"),
               "train-arm", "--root", str(root), "--arm", arm]
    with (root / f"{arm}.log").open("x") as log:
        process = subprocess.Popen(command, cwd=runner.REPO, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    runner.save(root / f"process_{arm}.json", {
        "pid": process.pid, "gpu": gpu, "command": command, "started_utc_unix": time.time()})
    print(f"Started {arm}: PID={process.pid}, CUDA_VISIBLE_DEVICES={gpu}", flush=True)
    return process


def stop_arm(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_parallel(root, gpus):
    assert len(gpus) == 2 and gpus[0] != gpus[1]
    runner.verify_frozen(root)
    assert os.environ.get("OPENAI_API_KEY")
    runner.save(root / "parallel_execution.json", {
        "overrides": "serial launch order only; scientific protocol unchanged",
        "gpu_assignment": {"static": gpus[0], "dynamic": gpus[0], "random": gpus[1]},
        "schedule": "static and random start together; dynamic starts after static completes",
        "evaluation": "all arms independently evaluate every endpoint; random never waits for static evaluation",
        "failure_policy": "stop remaining arm process groups on any failure; no retry",
    })
    runner.save(root / "started.json", {"status": "authorized_run_started", "time_utc_unix": time.time()})
    active = {}
    try:
        active["static"] = spawn_arm(root, "static", gpus[0])
        active["random"] = spawn_arm(root, "random", gpus[1])
        while active:
            for arm, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                runner.save(root / f"exit_{arm}.json", {
                    "pid": process.pid, "exit_code": code, "finished_utc_unix": time.time()})
                if code != 0:
                    raise subprocess.CalledProcessError(code, ["train-arm", arm])
                del active[arm]
                if arm == "static":
                    active["dynamic"] = spawn_arm(root, "dynamic", gpus[0])
            if active:
                time.sleep(1)
        runner.report(root)
    finally:
        for process in active.values():
            stop_arm(process)


def worker(root, gpus=None, dynamic_only=False):
    runner.save(root / "background_worker.json", {
        "pid": os.getpid(), "ppid_at_start": os.getppid(),
        "session_id": os.getsid(0), "stdin_is_tty": os.isatty(0),
        "started_utc_unix": time.time(),
    })
    try:
        if dynamic_only:
            assert gpus is None
            run_dynamic(root)
        elif gpus is None:
            runner.run(root)
        else:
            run_parallel(root, gpus)
    except Exception as error:
        traceback.print_exc()
        runner.save(root / "background_exit.json", {
            "status": "failed", "exit_code": 1,
            "exception_type": type(error).__name__, "finished_utc_unix": time.time(),
        })
        return 1
    runner.save(root / "background_exit.json", {
        "status": "complete", "exit_code": 0, "finished_utc_unix": time.time(),
    })
    return 0


def launch(root, gpus=None, dynamic_only=False):
    root = root.resolve()
    runner.verify_frozen(root)
    ready = runner.read(root / "readiness.json")
    assert ready["offline_checks_passed"] and ready["real_audio_backward_passed"]
    assert not (root / "started.json").exists(), "launch into a newly prepared directory"
    if dynamic_only:
        assert gpus is None and (root / "dynamic_restart.json").is_file()
    env = dict(os.environ)
    if not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = getpass.getpass("Gateway API key (memory only): ")
    assert env["OPENAI_API_KEY"], "API key is required before training begins"
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               PYTHONPATH=f"{runner.REPO}:{runner.REPO / 'src'}",
               CUDA_VISIBLE_DEVICES=gpus[0] if gpus else env.get("CUDA_VISIBLE_DEVICES", "0"))
    command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root)]
    if gpus is not None:
        assert len(gpus) == 2 and gpus[0] != gpus[1]
        command.extend(["--gpus", *gpus])
    if dynamic_only:
        command.append("--dynamic-only")
    command.append("--worker")
    with (root / "background.log").open("x") as log:
        process = subprocess.Popen(command, cwd=runner.REPO, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    receipt = {
        "supervisor_pid": process.pid, "command": command,
        "start_new_session": True, "stdin": "DEVNULL", "credentials_written": False,
        "launcher_sha256": runner.sha256_file(Path(__file__)),
        "protocol_sha256": runner.sha256_file(root / "protocol.json"),
        "gpus": gpus,
        "dynamic_only": dynamic_only,
        "created_utc_unix": time.time(),
    }
    runner.save(root / "background_launch.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--gpus", nargs=2, metavar=("COURSE_GPU", "RANDOM_GPU"))
    execution.add_argument("--dynamic-only", action="store_true", help="run only Dynamic with prepared completed controls")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    return worker(args.root, args.gpus, args.dynamic_only) if args.worker else launch(args.root, args.gpus, args.dynamic_only)


if __name__ == "__main__":
    raise SystemExit(main())
