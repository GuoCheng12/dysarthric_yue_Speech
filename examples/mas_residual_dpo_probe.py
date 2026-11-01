#!/usr/bin/env python3
"""Verify merged-parent identity, then train matched MAS/random residual DPO."""

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file

TRAINER = REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py'
EVALUATOR = REPO / 'examples/evaluate_residual_asr.py'


def prepare(base, root):
    root.mkdir(exist_ok=False)
    p = read(base / 'lexical96_mas_v1/protocol.json')
    sources = dict(mas=base / 'lexical96_mas_v1/round_001/mas_dpo',
                   random=base / 'lexical96_random_control_v1')
    arguments = {arm: read(d / 'training/run_protocol.json')['arguments'] for arm, d in sources.items()}
    receipts = dict(mas=base / 'lexical96_mas_v1/round_001/sampling/sampling_receipt.json',
                    random=base / 'lexical96_random_control_v1/sampling/sampling_receipt.json')
    files = [Path(__file__), TRAINER, EVALUATOR, REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py',
        REPO / 'finetune/scripts/evaluate_asr_checkpoint.py', REPO / 'src/therapist_harness/dpo_training.py',
        base / 'lexical96_mas_v1/protocol.json', Path(p['feedback_split']['path']),
        Path(p['initial_adapter_dev_predictions']['path']), base / 'peft_runtime_probe_v1/parent/results.json',
        base / 'peft_runtime_probe_v1/dpo/results.json']
    for arm, a in arguments.items():
        assert a['real_per_step'] == 0 and a['max_steps'] == 12 and a['synthetic_per_step'] == 8
        assert a['policy_adapter_path'] == a['reference_adapter_path'] == p['initial_adapter']['path']
        assert read(receipts[arm])['current_asr_sha256'] == p['initial_adapter']['sha256']
        files.extend([sources[arm] / 'training/run_protocol.json', sources[arm] / 'training/training_exposures.jsonl',
                      Path(a['train_file']), Path(a['synthetic_allocation_file']), receipts[arm]])
    keys = read(arguments['mas']['synthetic_allocation_file'])['steps'][0][:2]
    jsonl(root / 'smoke_preferences.jsonl', [x for x in rows(arguments['mas']['train_file']) if x['audio_id'] in keys])
    save(root / 'smoke_allocation.json', dict(steps=[keys, list(reversed(keys))]))
    files.extend([root / 'smoke_preferences.jsonl', root / 'smoke_allocation.json'])
    save(root / 'protocol.json', dict(source_protocol=p, source_arguments=arguments,
        source_sha256={str(f): sha256_file(f) for f in files},
        source_exposures={a: str(d / 'training/training_exposures.jsonl') for a, d in sources.items()},
        reused_independent_sampling_receipts={a: str(f) for a, f in receipts.items()},
        actual_new_generation=False, actual_new_sampling=False, actual_new_dpo_training=True,
        hypothesis='Align the DPO starting function with the merged R-only used for actual candidate sampling and evaluation. Freeze that merged model and optimize a new zero-output residual LoRA.',
        intervention_scope='Changes optimization coordinates as well as execution. Matched existing MAS and random-reference/recipe data remain unchanged, with their original pair availability and exposure schedules.',
        preflight='All 511 zero-residual raw predictions must match the previously accepted R-only. Two actual update steps must start at log(2), change only the new residual, and preserve frozen merged base/reference.',
        acceptance='Original raw-ASR gate against the matched R-only; no LLM postprocessing.'))


def verify(root):
    config = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == digest for f, digest in config['source_sha256'].items())
    return config


def evaluation_command(p, directory, residual=None):
    command = [sys.executable, str(EVALUATOR), '--base-model', p['base_model_dir'],
        '--initial-adapter', p['initial_adapter']['path'], '--split', p['feedback_split']['path'],
        '--out', str(directory)]
    if residual:
        command.extend(['--residual-adapter', str(residual)])
    return command


def initial(root):
    p = verify(root)['source_protocol']
    execute(root, evaluation_command(p, root / 'initial'), 'initial')
    before = rows(p['initial_adapter_dev_predictions']['path'])
    after = rows(root / 'initial/evaluation/predictions.jsonl')
    assert [(x['utt_id'], x['prediction']) for x in before] == [(x['utt_id'], x['prediction']) for x in after]
    save(root / 'initial/identity_verified.json', dict(status='passed', exact_predictions=len(after),
        zero_residual_matches_original_merged_parent=True))


def train(base, root, arm, smoke=False):
    cfg = verify(root)
    if not smoke:
        assert read(root / 'initial/identity_verified.json')['status'] == 'passed'
        assert read(root / 'smoke/verified.json')['status'] == 'passed'
    d = root / ('smoke' if smoke else arm)
    d.mkdir(exist_ok=False)
    a = dict(cfg['source_arguments'][arm], output_dir=str(d / 'training'))
    if smoke:
        a.update(train_file=str(root / 'smoke_preferences.jsonl'),
            synthetic_allocation_file=str(root / 'smoke_allocation.json'), max_steps=2, synthetic_per_step=2)
    command = [sys.executable, str(TRAINER)]
    for key, value in a.items():
        if value is not None:
            command.extend(['--' + key, str(value)])
    execute(d, command, 'training')
    result = read(d / 'training/training_result.json')
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert math.isclose(result['history'][0]['loss'], math.log(2), abs_tol=1e-4)
    assert read(d / 'training/residual_verified.json')['status'] == 'passed'
    if not smoke:
        assert rows(d / 'training/training_exposures.jsonl') == rows(cfg['source_exposures'][arm])
    save(d / 'verified.json', dict(status='passed', initial_dpo_log2=True, reference_unchanged=True,
        merged_parent_base_unchanged=True, original_exposure_records_match=None if smoke else True))
    verify(root)
    if smoke:
        return
    p = cfg['source_protocol']
    execute(d, evaluation_command(p, d / 'evaluation_run', d / 'training/checkpoint-12'), 'evaluation')
    metrics = read(d / 'evaluation_run/evaluation/summary.json')['metrics']
    parent = read(root / 'initial/evaluation/summary.json')['metrics']
    checks = dict(pooled_improves=metrics['overall']['pooled_cer'] < parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer']-parent['overall']['patient_macro_cer'] <= .001,
        hard_regression_within_limit=metrics['hard']['pooled_cer']-parent['hard']['pooled_cer'] <= .005)
    save(d / 'paired_parent.json', compare(root / 'initial/evaluation/predictions.jsonl',
                                          d / 'evaluation_run/evaluation/predictions.jsonl'))
    save(d / 'results.json', dict(status='complete', metrics=metrics, gate_checks=checks,
        eligible_for_next_round_parent=all(checks.values()), actual_new_dpo_training=True, postprocessing_used=False))
    verify(root)
    print(dict(arm=arm, metrics=metrics['overall'], gate=checks), flush=True)


def supervise(base, root):
    def parallel(jobs):
        running = []
        for gpu, stage, arm in jobs:
            label = stage + '_' + arm
            command = [sys.executable, str(Path(__file__).resolve()), stage, str(base), '--arm', arm]
            with (root / (label + '.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
            save(root / (label + '.launch.json'), dict(pid=process.pid, command=command, gpu=gpu))
            running.append(process)
        codes = [process.wait() for process in running]
        assert codes == [0] * len(codes), codes
    parallel([(0, 'initial', 'mas'), (1, 'smoke', 'mas')])
    parallel([(0, 'train', 'mas'), (1, 'train', 'random')])
    save(root / 'results.json', dict(status='complete', arms={a: read(root / a / 'results.json') for a in ['mas', 'random']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'initial', 'smoke', 'train', 'supervise'])
    parser.add_argument('base', type=Path)
    parser.add_argument('--arm', choices=['mas', 'random'], default='mas')
    args = parser.parse_args()
    base = args.base.resolve()
    root = base / 'residual_dpo_probe_v1'
    if args.stage == 'prepare':
        prepare(base, root)
    elif args.stage == 'initial':
        initial(root)
    elif args.stage in ['smoke', 'train']:
        train(base, root, args.arm, args.stage == 'smoke')
    else:
        supervise(base, root)
