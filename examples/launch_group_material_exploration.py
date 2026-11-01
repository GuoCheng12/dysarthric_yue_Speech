"""Wait for a running experiment, then audit grouped-material GPU smoke and train."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from therapist_harness.store import sha256_file, write_immutable_json


def read(path):
    return json.loads(Path(path).read_text())


def process_alive(pid, protocol):
    try:
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return b'supervise' in argv and str(protocol).encode() in argv and b'mas' in argv
    except FileNotFoundError:
        return False


def validate(root, deployment):
    formal = read(root / 'formal/protocol.json')
    smoke = read(root / 'smoke/protocol.json')
    assert formal['material_curriculum']['sampling_mode'] == 'group_then_material'
    differences = {k for k in formal.keys() | smoke.keys() if formal.get(k) != smoke.get(k)}
    assert differences <= {'root', 'validation_api_result', 'validation_smoke_root'}, differences
    assert formal['total_optimizer_steps_limit'] == 590 and formal['steps_per_stage'] == 29
    assert formal['setting'] == 'E' and formal['seed'] == 42
    result = read(root / 'cpu_api_smoke/result.json')
    assert result['passed'] and result['actual_model'] == 'gpt-5.6-sol'
    assert result['actual_api_calls'] == 3 and result['state_restore_exact'] and result['train_only']
    assert not result['probe_decisions_used_for_training']
    checks = {}
    for phase in ('smoke', 'formal'):
        preparation = read(root / phase / 'mas/preparation.json')
        for key in ('source_sha256', 'input_sha256'):
            checks.update(preparation[key])
    runner = deployment / 'examples/run_setting_e_sampling_curriculum_20260911.py'
    assert str(runner) in checks
    reuse = read(root / 'formal/uniform_reuse.json')
    assert reuse['source_path'] == formal['uniform_comparison_path']
    checks.update(reuse['source_sha256'])
    helper = deployment / 'examples/run_setting_e_material_exploration_20260911.py'
    for p in (helper, Path(__file__).resolve(), root / 'cpu_api_smoke/result.json',
              root / 'formal/uniform_reuse.json'):
        checks[str(p)] = sha256_file(p)
    assert all(sha256_file(Path(p)) == h for p, h in checks.items())
    return checks


def free_hardware(gpus):
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid',
                                     '--format=csv,noheader'], text=True).splitlines()
    selected = [line for line in lines if line.split(',')[0].strip() in gpus]
    assert len(selected) == 2 and all('L20' in line for line in selected)
    uuids = {line.split(',')[2].strip() for line in selected}
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                      '--format=csv,noheader'], text=True).splitlines()
    assert not any(line.split(',')[0].strip() in uuids for line in active), 'Selected GPUs are occupied.'
    return selected


def main(args):
    root, deployment = args.root.resolve(), args.deployment.resolve()
    checks = validate(root, deployment)
    if args.check_only:
        print(json.dumps(dict(passed=True, checked_files=len(checks), gpu_training_started=False)))
        return
    wait_protocol = args.wait_protocol.resolve()
    predecessor = Path(read(wait_protocol)['root']) / 'mas'
    supervisor = read(predecessor / 'supervisor.json')
    gpus = args.gpus.split(',')
    assert len(gpus) == len(set(gpus)) == 2
    assert supervisor['gpus'] == gpus
    pid = supervisor['pid']
    assert process_alive(pid, wait_protocol), 'Expected predecessor is not live; inspect before queuing.'
    write_immutable_json(root / 'queue.json', dict(pid=os.getpid(), started_unix=time.time(),
        predecessor_pid=pid, predecessor_protocol=str(wait_protocol), gpus=gpus,
        source_and_input_sha256=checks,
        sequence=['wait_for_successful_590_completion', 'two_gpu_smoke', 'audit_smoke', 'fresh_590_training', 'report']))
    while process_alive(pid, wait_protocol):
        print(f'Waiting for live MAS PID {pid}; requested GPUs {args.gpus}.', flush=True)
        time.sleep(30)
    complete = read(predecessor / 'complete.json')
    assert complete['passed'] and complete['steps'] == 590 and complete['actual_audio_exposures'] == 18880
    assert not (predecessor / 'failure.json').exists()
    hardware = free_hardware(gpus)
    assert all(sha256_file(Path(p)) == h for p, h in checks.items()), 'Prepared files changed while queued.'
    write_immutable_json(root / 'queue_released.json', dict(unix=time.time(), hardware=hardware,
        predecessor_complete_sha256=sha256_file(predecessor / 'complete.json')))
    runner = deployment / 'examples/run_setting_e_sampling_curriculum_20260911.py'
    helper = deployment / 'examples/run_setting_e_material_exploration_20260911.py'
    for phase, mode in (('smoke', 'smoke'), ('formal', 'supervise')):
        if phase == 'formal':
            subprocess.run([sys.executable, str(helper), 'audit-smoke', '--root', str(root)],
                           cwd=deployment, check=True)
            audit = read(root / 'smoke_audit.json')
            assert audit['passed'] and audit['group_and_material_exposure_verified']
            assert audit['optimizer_steps'] == 4 and audit['successful_real_api_calls'] == 10
            assert all(sha256_file(Path(p)) == h for p, h in checks.items())
            free_hardware(gpus)
            write_immutable_json(root / 'formal/launch_validation.json', dict(unix=time.time(),
                smoke_audit_sha256=sha256_file(root / 'smoke_audit.json'), fresh_initialization=True,
                smoke_or_probe_decisions_replayed=False, total_optimizer_steps=590))
        with (root / phase / 'mas_supervisor.log').open('x') as log:
            subprocess.run([sys.executable, '-u', str(runner), mode, '--protocol',
                            str(root / phase / 'protocol.json'), '--branch', 'mas', '--gpus', args.gpus],
                           cwd=deployment, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(helper), 'report', '--root', str(root)], cwd=deployment, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--deployment', required=True, type=Path)
    parser.add_argument('--wait-protocol', required=True, type=Path)
    parser.add_argument('--gpus', default='2,3')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if not args.check_only:
            write_immutable_json(args.root / 'queue_failed.json',
                                 dict(error_type=type(error).__name__, detail=str(error), unix=time.time()))
        raise
