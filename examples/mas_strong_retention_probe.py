#!/usr/bin/env python3
"""Retain real-speech behavior while strengthening observed MAS DPO corrections."""

import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys

import mas_residual_retention_probe as retention
from mas_selection_dpo_probe import REPO, execute, read, rows, save
from therapist_harness.store import sha256_file

TRAINER = REPO / 'finetune/scripts/qwen3_asr_shared_teacher_kl_dpo.py'


def prepare(base, root):
    root.mkdir(exist_ok=False)
    old = base / 'residual_retention_probe_v1'
    selected = base / 'candidate_coverage_recovery_v2'
    p = read(selected / 'protocol.json')['source_protocol']
    cache = old / 'weight_10/training/real_teacher_log_probs.pt'
    metadata = old / 'weight_10/training/real_teacher_cache.json'
    assert sha256_file(cache) == read(metadata)['cache_file_sha256']
    assert read(metadata)['frozen_reference_sha256'] == p['initial_adapter']['sha256']
    os.link(old / 'real.jsonl', root / 'real.jsonl')
    (root / 'smoke').mkdir()
    assert read(old / 'smoke/verified.json')['status'] == 'passed'
    save(root / 'smoke/verified.json', dict(status='passed', actual_new_smoke=False,
        reused_smoke=str(old / 'smoke/verified.json'), reused_smoke_sha256=sha256_file(old / 'smoke/verified.json'),
        scope='Existing real two-step smoke of this unchanged residual KL/DPO engine. New persistence wrapper validates all96 freshly computed teacher tensors before policy updates; current runs also check zero initial KL,log2 DPO,finite gradients and frozen base/reference. No current-data or new-lr smoke is claimed.'))
    files = [Path(__file__), TRAINER, REPO / 'examples/mas_residual_retention_probe.py',
        REPO / 'finetune/scripts/qwen3_asr_residual_kl_dpo.py', REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py',
        REPO / 'finetune/scripts/qwen3_asr_kl_dpo_probe.py', REPO / 'src/therapist_harness/dpo_training.py',
        REPO / 'examples/evaluate_residual_asr.py', old / 'real.jsonl', old / 'real_selection_receipt.json',
        old / 'mas_allocation.json', old / 'smoke/verified.json', cache, metadata,
        selected / 'protocol.json', selected / 'cross_arm_verification.json',
        selected / 'all16/preferences.jsonl', selected / 'allocation.json',
        selected / 'all16/selection_receipt.json', selected / 'sampling/sampling_receipt.json',
        base / 'residual_lr_probe_v1/results.json', base / 'residual_lr_probe_v1/target_response.json',
        base / 'residual_lr_probe_v1/preference_response.json', root / 'real.jsonl', root / 'smoke/verified.json']
    save(root / 'protocol.json', dict(source_protocol=p, source_sha256={str(f): sha256_file(f) for f in files},
        preferences=str(selected / 'all16/preferences.jsonl'), allocation=str(selected / 'allocation.json'),
        sampling_receipt=str(selected / 'sampling/sampling_receipt.json'),
        verified_teacher_cache=str(cache), verified_teacher_metadata=str(metadata),
        objective='Independent branch: standard synthetic audio DPO plus real-training symmetric KL to fixed initial R-only; no real CE, no inference postprocessing.',
        lr=1e-4, beta=.1, seed=42, steps=12, synthetic_per_step=8, real_per_step=16, kl_weights=[1,10],
        hypothesis='Stronger DPO produced26 primary target-position recoveries but increased insertions in their real sentence cohort from107 to158. Anchor existing real-training behavior while retaining correction on new MAS content.',
        controls='Completed same-data pure residual DPO at lr1e-4; both new KL strengths share exact real pool,synthetic pairs,exposure schedule,parent and reference.',
        real_allocation='Reuse actual MAS-selected96 real-training recordings,64hard/16medium/16easy,54patients;192 exposures. The511 development audio never enters training.',
        storage='Teacher still recomputed for every run. Before any policy update, compare every actual cached tensor to the previously verified immutable teacher; hardlink its equivalent archive to avoid664MB duplicate per run.',
        actual_new_generation=False, actual_new_sampling=False, actual_new_dpo_training=True))


def run(base, root, weight):
    cfg = retention.verify(root)
    retention.TRAINER = TRAINER

    def execute_bound(directory, command, name):
        if name == 'training':
            command = list(command)
            assert command[1] == str(TRAINER)
            command[command.index('--lr')+1] = str(cfg['lr'])
            command.extend(['--verified_teacher_cache', cfg['verified_teacher_cache'],
                '--verified_teacher_metadata', cfg['verified_teacher_metadata']])
        execute(directory, command, name)

    retention.execute = execute_bound
    retention.run(base, root, weight)
    reused = read(root / f'weight_{weight:g}/training/teacher_storage_reuse.json')
    assert reused['actual_fresh_teacher_computation']
    assert reused['identical_actual_tensor_values_verified'] == 96


def supervise(base, root):
    jobs = []
    for gpu, weight in enumerate([1.,10.]):
        command = [sys.executable, str(Path(__file__).resolve()), 'run', str(base), '--weight', str(weight)]
        with (root / f'weight_{weight:g}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        save(root / f'weight_{weight:g}.launch.json', dict(pid=child.pid, gpu=gpu, command=command))
        jobs.append(child)
    codes = [p.wait() for p in jobs]
    assert codes == [0,0], codes
    data = {w: rows(root / f'weight_{w}/training/training_exposures.jsonl') for w in [1,10]}
    normalized = lambda w: [dict(x, loss_weight=x['loss_weight']/(w if x['source']=='real' else 1)) for x in data[w]]
    assert normalized(1) == normalized(10)
    for w in [1,10]:
        assert Counter(x['source'] for x in data[w]) == {'real':192,'synthetic':96}
    assert read(root / 'weight_1/training/reference_log_probs.json')['values'] == read(root / 'weight_10/training/reference_log_probs.json')['values']
    save(root / 'cross_arm_verification.json', dict(status='passed', same_real_synthetic_exposures=True,
        same_DPO_reference_scores=True, identical_actual_real_teacher_verified_before_updates=True))
    save(root / 'results.json', dict(status='complete', arms={str(w):read(root / f'weight_{w}/results.json') for w in [1,10]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','run','supervise'])
    parser.add_argument('base', type=Path)
    parser.add_argument('--weight', type=float, choices=[1.,10.], default=10.)
    args = parser.parse_args()
    base = args.base.resolve()
    root = base / 'strong_retention_probe_v1'
    if args.stage == 'prepare':
        prepare(base, root)
    elif args.stage == 'run':
        run(base, root, args.weight)
    else:
        supervise(base, root)
