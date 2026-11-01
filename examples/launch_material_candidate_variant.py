"""Launch one prepared MAS variant after its existing uniform control releases two GPUs."""
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


def main(protocol_path):
    protocol = read(protocol_path)
    root = Path(protocol['root'])
    control = Path(protocol['uniform_comparison_path'])
    control_protocol = read(control.parent / 'protocol.json')
    # A new MAS catalog budget does not require rerunning an identical ordinary control.
    differences = {k for k in protocol.keys() | control_protocol.keys()
                   if protocol.get(k) != control_protocol.get(k)}
    allowed = {'root', 'version', 'material_curriculum', 'requested_change',
               'uniform_comparison_path', 'candidate_probe', 'validation_sha256'}
    assert differences <= allowed, differences - allowed
    assert protocol['material_curriculum']['index_path'] == control_protocol['material_curriculum']['index_path']
    preparation = read(root / 'mas/preparation.json')
    control_preparation = read(control / 'preparation.json')
    assert preparation['source_sha256'] == control_preparation['source_sha256']
    assert read(protocol['candidate_probe'])['passed']
    assert read(protocol['validation_api_result'])['passed']
    assert all(read(Path(protocol['validation_smoke_root']) / b / 'complete.json')['passed']
               for b in ('mas', 'uniform'))
    assert read(root / 'candidate_sampler_validation.json')['passed']
    checks = {**preparation['source_sha256'], **preparation['input_sha256'],
              **protocol['validation_sha256']}
    assert all(sha256_file(Path(p)) == h for p, h in checks.items())
    supervisor = read(control / 'supervisor.json')
    pid = supervisor['pid']
    def alive():
        try:
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            return 'supervise' in cmd and str(control.parent / 'protocol.json') in cmd and '--branch uniform' in cmd
        except FileNotFoundError:
            return False
    write_immutable_json(root / 'queue.json', dict(pid=os.getpid(), started_unix=time.time(),
        control_pid=pid, control_path=str(control), gpus=[0, 1],
        source_sha256={str(Path(__file__).resolve()): sha256_file(Path(__file__))}))
    while True:
        if (control / 'failure.json').exists():
            raise RuntimeError('The control stopped without successful completion; inspect its original logs.')
        if not alive():
            if (control / 'complete.json').exists():
                break
            raise RuntimeError('The control process disappeared without completion; inspect its original logs.')
        print(f'Waiting for live uniform PID {pid} to finish and release GPUs 0,1.', flush=True)
        time.sleep(30)
    completed = read(control / 'complete.json')
    assert completed['passed'] and completed['steps'] == 590 and completed['actual_audio_exposures'] == 18880
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid', '--format=csv,noheader'], text=True)
    selected = [line for line in hardware.splitlines() if line.split(',')[0].strip() in ('0', '1')]
    assert len(selected) == 2 and all('L20' in line for line in selected)
    gpu_ids = {line.split(',')[2].strip() for line in selected}
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert not any(line.split(',')[0].strip() in gpu_ids for line in active.splitlines()), 'GPUs 0,1 are occupied.'
    assert all(sha256_file(Path(p)) == h for p, h in checks.items()), 'Prepared inputs or source changed while waiting.'
    sources = [control.parent / 'protocol.json', control / 'preparation.json', control / 'complete.json',
               control / 'initialization.json', control / 'selected_checkpoint.json']
    sources.extend(control.glob('evaluations/step-*/test/summary.json'))
    sources.extend(control.glob('evaluations/step-*/feedback/summary.json'))
    write_immutable_json(root / 'uniform_reuse.json', dict(source_path=str(control),
        source_sha256={str(p): sha256_file(p) for p in sources}, protocol_differences=sorted(differences),
        source_code_identical=True, training_settings_identical=True, data_inputs_identical=True,
        ordinary_control_retrained=False, reused_at_unix=time.time()))
    write_immutable_json(root / 'variant_launch.json', dict(started_unix=time.time(),
        gpus=[0, 1], hardware=selected, fresh_initialization=True,
        validation_scope='Reuse unchanged runtime two-GPU smoke; actual candidate-budget API probe and sampler check. No new GPU smoke claimed.'))
    runner = REPO / 'examples/run_setting_e_sampling_curriculum_20260911.py'
    with (root / 'mas_supervisor.log').open('x') as log:
        subprocess.run([sys.executable, '-u', str(runner), 'supervise', '--protocol', str(protocol_path),
                        '--branch', 'mas', '--gpus', '0,1'], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(REPO / 'examples/run_setting_e_material_exploration_20260911.py'),
                    'report', '--root', str(root.parent)], cwd=REPO, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    args = parser.parse_args()
    try:
        main(args.protocol)
    except Exception as error:
        root = Path(read(args.protocol)['root'])
        write_immutable_json(root / 'variant_launcher_failed.json',
                             dict(error_type=type(error).__name__, detail=str(error)))
        raise
