"""Prepare or run the two real-data branches on two dedicated L20s each."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from therapist_harness.store import write_immutable_json, sha256_file

RUNNER = REPO / "examples/run_setting_e_sampling_curriculum_20260911.py"
DEFAULT = Path("/data/qwen3-asr/records/therapist_mas_setting_e_v1/sampling_curriculum_dev_every29_total590_seed42_20260911_v4/protocol.json")


def command(mode, protocol, branch):
    return [sys.executable, "-u", str(RUNNER), mode, "--protocol", str(protocol), "--branch", branch]


def prepare(protocol_path):
    protocol = json.loads(protocol_path.read_text())
    root = Path(protocol["root"])
    for branch in ("uniform", "mas"):
        subprocess.run(command("prepare", protocol_path, branch), check=True, cwd=REPO)
    subprocess.run(command("cpu-smoke", protocol_path, "uniform"), check=True, cwd=REPO)
    print("Prepared both branches:", root)


def start(protocol_path):
    from therapist_harness.laicai_credentials import load_laicai_key
    protocol = json.loads(protocol_path.read_text())
    root = Path(protocol["root"])
    smoke_root = Path(protocol.get('validation_smoke_root', root / 'smoke_v1'))
    for name in ("uniform", "mas"):
        if not json.loads((smoke_root / name / "complete.json").read_text())["passed"]:
            raise RuntimeError("complete the isolated GPU smoke before the comparison")
        receipt = json.loads((root / name / 'preparation.json').read_text())
        if any(sha256_file(Path(path)) != digest for key in ('source_sha256', 'input_sha256')
               for path, digest in receipt[key].items()):
            raise RuntimeError('prepared source or input changed before launch')
    api_result = Path(protocol.get('validation_api_result', root / 'api_smoke/result.json'))
    if not json.loads(api_result.read_text())["passed"]:
        raise RuntimeError("complete the actual laicai API smoke first")
    if any(sha256_file(Path(path)) != digest for path, digest in protocol.get('validation_sha256', {}).items()):
        raise RuntimeError('the recorded smoke validation changed')
    gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.splitlines()
    gpu_ids = {line.split(",")[2].strip() for line in gpu_info if line.split(",")[0].strip() in ("0", "1", "2", "3")}
    if len(gpu_ids) != 4 or any("L20" not in line for line in gpu_info[:4]):
        raise RuntimeError("this frozen comparison expects four L20 GPUs")
    active = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True).stdout.splitlines()
    if any(line.split(",")[0].strip() in gpu_ids for line in active):
        raise RuntimeError("the comparison needs dedicated GPUs 0,1 and 2,3; existing GPU jobs are still running")
    credential = load_laicai_key()
    write_immutable_json(root / "comparison_launch.json", {
        "started_unix": time.time(), "hardware": gpu_info,
        "branches": {"uniform": [0, 1], "mas": [2, 3]},
        "credential": credential, "wall_budget_per_branch": protocol["wall_time_budget_seconds"]})
    processes, handles = [], []
    try:
        for branch, gpus in (("uniform", "0,1"), ("mas", "2,3")):
            handle = (root / f"{branch}_supervisor.log").open("x")
            handles.append(handle)
            processes.append(subprocess.Popen(command("supervise", protocol_path, branch) + ["--gpus", gpus],
                stdout=handle, stderr=subprocess.STDOUT, cwd=REPO))
        reported = set()
        while any(p.poll() is None for p in processes):
            for branch, process in zip(('uniform', 'mas'), processes):
                code = process.poll()
                if code not in (None, 0) and branch not in reported:
                    write_immutable_json(root / f'{branch}_failed.json',
                                         dict(branch=branch, exit_code=code, other_branch_continues=True))
                    reported.add(branch)
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("comparison branch did not finish successfully")
        subprocess.run(command("report", protocol_path, "uniform"), check=True, cwd=REPO)
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=75)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "start"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "start": start}[args.mode](args.protocol)
