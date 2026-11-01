#!/usr/bin/env python3
"""Prepare and validate frozen-real-teacher retention alongside synthetic DPO."""

import argparse
from collections import Counter
import math
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file

TRAINER = REPO / 'finetune/scripts/qwen3_asr_kl_dpo_probe.py'


def prepare(base, root):
    root.mkdir(exist_ok=False)
    p = read(base / 'lexical96_mas_v1/protocol.json')
    old = base / 'real_anchor_probe'
    old_protocol = read(old / 'protocol.json')
    assert sha256_file(old / 'real.jsonl') == old_protocol['source_sha256'][str(old / 'real.jsonl')]
    real = rows(old / 'real.jsonl')
    train = {x['utt_id']: x for x in rows(p['real_train']['path'])}
    feedback_ids = {x['utt_id'] for x in rows(p['feedback_split']['path'])}
    assert len(real) == 96
    assert all(x['audio_id'] in train and x['audio_id'] not in feedback_ids for x in real)
    assert all(Path(x['audio']).resolve() == Path(train[x['audio_id']]['audio']).resolve() for x in real)
    assert old_protocol['isolation']['real_audio_hash_overlap'] == 0
    jsonl(root / 'real.jsonl', real)
    source = base / 'lexical96_mas_v1/round_001'
    pairs = source / 'mas_dpo/train.jsonl'
    allocation = source / 'training_allocation.json'
    receipt = source / 'sampling/sampling_receipt.json'
    assert read(receipt)['current_asr_sha256'] == p['initial_adapter']['sha256']
    files = [Path(__file__), TRAINER, REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py',
        REPO / 'src/therapist_harness/dpo_training.py', REPO / 'finetune/scripts/evaluate_asr_checkpoint.py',
        old / 'real.jsonl', old / 'protocol.json', root / 'real.jsonl', pairs, allocation, receipt,
        Path(p['real_train']['path']), Path(p['feedback_split']['path']), base / 'lexical96_mas_v1/protocol.json']
    save(root / 'protocol.json', dict(source_protocol=p, source_sha256={str(f): sha256_file(f) for f in files},
        preferences=str(pairs), allocation=str(allocation), sampling_receipt=str(receipt),
        steps=12, real_per_step=16, synthetic_per_step=8, real_audio_pool=96,
        kl_weights=[1., 10.], synthetic_weight=1., lr=1e-5, beta=.1, seed=42,
        objective='Mean synthetic audio DPO + lambda * mean real-audio symmetric KL to frozen initial R-only',
        scope='Independent objective containing real teacher-forced retention, separate from synthetic-only results; no real-label CE or development training.',
        hypothesis='Constrain changes on real training speech while preserving the corrective synthetic update. Teacher KL should be zero with zero gradient at the common R-only start.',
        selection='Reuse the previously isolated fixed-seed 96 real-training audio. Each arm has 192 real exposures and the same 96 synthetic exposures; no new SG or candidate sampling.'))
    keys = read(allocation)['steps'][0][:2]
    assert len(set(keys)) == 2
    jsonl(root / 'smoke_real.jsonl', real[:2])
    jsonl(root / 'smoke_preferences.jsonl', [x for x in rows(pairs) if x['audio_id'] in keys])
    save(root / 'smoke_allocation.json', dict(steps=[keys, list(reversed(keys))]))


def verify(root):
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(p)) == digest for p, digest in cfg['source_sha256'].items())
    return cfg


def run_training(root, directory, cfg, weight, smoke=False):
    p = cfg['source_protocol']
    execute(directory, [sys.executable, str(TRAINER), '--model_path', p['base_model_dir'],
        '--policy_adapter_path', p['initial_adapter']['path'], '--reference_adapter_path', p['initial_adapter']['path'],
        '--real_train_file', str(root / ('smoke_real.jsonl' if smoke else 'real.jsonl')),
        '--synthetic_train_file', str(root / 'smoke_preferences.jsonl') if smoke else cfg['preferences'],
        '--synthetic_allocation_file', str(root / 'smoke_allocation.json') if smoke else cfg['allocation'],
        '--output_dir', str(directory / 'training'), '--max_steps', '2' if smoke else '12',
        '--real_per_step', '2' if smoke else '16', '--synthetic_per_step', '2' if smoke else '8',
        '--synthetic_weight', '1', '--kl_weight', str(weight), '--lr', '1e-5', '--beta', '.1', '--seed', '42'], 'training')
    result = read(directory / 'training/training_result.json')
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert abs(result['probe_before'][0]['real_symmetric_kl']) < 1e-8
    assert abs(result['history'][0]['weighted_losses']['synthetic'] - math.log(2)) < 1e-4
    assert abs(result['history'][0]['weighted_losses']['real']) < 1e-7
    exposures = rows(directory / 'training/training_exposures.jsonl')
    weights = Counter()
    counts = Counter()
    seen = set()
    for x in exposures:
        weights[(x['step'], x['source'])] += x['loss_weight']
        key = (x['step'], x['source'], x['audio_id'], x['occurrence'])
        if key not in seen:
            counts[x['source']] += 1
            seen.add(key)
    assert counts == ({'real':4, 'synthetic':4} if smoke else {'real':192, 'synthetic':96})
    assert all(math.isclose(value, weight if source == 'real' else 1., abs_tol=1e-7)
               for (step, source), value in weights.items())
    verify(root)
    save(directory / 'verified.json', dict(status='passed', initial_teacher_kl_zero=True,
        initial_dpo_log2=True, reference_files_unchanged=True, actual_audio_exposures=dict(counts),
        real_teacher_weight=weight, synthetic_weight=1., per_step_source_weights_verified=True))


def main(args):
    base = args.base.resolve()
    root = base / 'kl_retention_probe_v1'
    if args.stage == 'prepare':
        prepare(base, root)
        print('prepared', flush=True)
        return
    cfg = verify(root)
    smoke = args.stage == 'smoke'
    if not smoke:
        assert read(root / 'smoke/verified.json')['status'] == 'passed'
    weight = 10. if smoke else args.weight
    directory = root / ('smoke' if smoke else f'weight_{weight:g}')
    directory.mkdir(exist_ok=False)
    run_training(root, directory, cfg, weight, smoke)
    if smoke:
        print('actual two-step KL/DPO smoke passed', flush=True)
        return
    p = cfg['source_protocol']
    execute(directory, [sys.executable, str(REPO / 'finetune/scripts/evaluate_asr_checkpoint.py'),
        '--base-model-path', p['base_model_dir'], '--adapter-path', str(directory / 'training/checkpoint-12'),
        '--split-jsonl', p['feedback_split']['path'], '--out-dir', str(directory / 'evaluation'),
        '--batch-size', '8', '--critical-cer-threshold', '.5', '--max-new-tokens', '256'], 'evaluation')
    save(directory / 'paired_parent.json', compare(Path(p['initial_adapter_dev_predictions']['path']),
                                                  directory / 'evaluation/predictions.jsonl'))
    verify(root)
    save(directory / 'results.json', dict(status='complete', metrics=read(directory / 'evaluation/summary.json')['metrics'],
        objective=cfg['objective'], kl_weight=weight, automatic_promotion=False))
    print(read(directory / 'results.json')['metrics']['overall'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'run'])
    parser.add_argument('base', type=Path)
    parser.add_argument('--weight', type=float, choices=[1., 10.], default=10.)
    main(parser.parse_args())
