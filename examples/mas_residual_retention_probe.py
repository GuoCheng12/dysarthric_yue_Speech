#!/usr/bin/env python3
"""MAS allocates real-training retention examples for merged-parent residual DPO."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_evaluation_report import compare
from mas_residual_dpo_probe import evaluation_command
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file

TRAINER = REPO / 'finetune/scripts/qwen3_asr_residual_kl_dpo.py'


class Allocation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    easy: int = Field(ge=8, le=80)
    medium: int = Field(ge=8, le=80)
    hard: int = Field(ge=8, le=80)
    reason: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)


def prepare(base, root):
    from openai.lib._pydantic import to_strict_json_schema

    root.mkdir(exist_ok=False)
    p = RunProtocol.model_validate(read(base / 'lexical96_mas_v1/protocol.json'))
    real = rows(p.real_train.path)
    dev = rows(p.feedback_split.path)
    prior = base / 'kl_retention_probe_v1'
    by_id = {x['utt_id']: x for x in real}
    old_counts = Counter(by_id[x['audio_id']]['zero_shot_bucket'] for x in rows(prior / 'real.jsonl'))
    observations = read(base / 'residual_hard_retention_next_hypothesis_v1.json')
    request = dict(model=p.agent.model, reasoning={'effort': p.agent.reasoning_effort},
        store=False, stream=True, max_output_tokens=8192,
        input=[dict(role='system', content='''Allocate a fixed pool of 96 REAL TRAINING recordings among easy, medium and hard for an ASR capability-retention probe.
Use the supplied observed effects of actual MAS TTS DPO. The retention term compares
the current model's token distributions against the FROZEN accepted R-only model on
these real training recordings. It is symmetric KL, not real-label cross entropy.
The unchanged synthetic same-audio DPO term supplies correction. The initial merged
R-only plus zero residual has already reproduced all 511 baseline predictions.
Choose exactly 96 recordings total, each bucket at least 8, in multiples of 8.
The executor samples only from the 2117 real-training pool with seed42, round-robin
over patients within each bucket. No development recordings enter training. The
existing bucket labels are descriptive selection metadata, not new TTS admission
thresholds. Keep both KL-strength arms on exactly the same pool and sequence.
Explain the tradeoff between protecting the degraded hard group and retaining easy/
medium corrections. Do not claim the proposed allocation has already improved ASR.
Return only the allocation, a brief reason, and a prospective expected effect.'''),
               dict(role='user', content=json.dumps(dict(current_real_development_effect=observations['observed'],
                    real_training_available=dict(Counter(x['zero_shot_bucket'] for x in real)),
                    previous_random_96_pool=dict(old_counts),
                    planned_training=dict(synthetic_audio_exposures=96, real_retention_exposures=192,
                        steps=12, candidate_kl_weights=[1, 10], teacher='frozen merged initial R-only',
                        previous_unmerged_teacher_KL='No raw CER gain; new probe changes both teacher runtime/residual parameterization and pool allocation')),
                    ensure_ascii=False))],
        text={'format': {'type': 'json_schema', 'name': 'real_retention_allocation', 'strict': True,
                         'schema': to_strict_json_schema(Allocation)}})
    save(root / 'allocation_request.json', request)
    response = serialize_response(execute_therapist_request(p, request))
    save(root / 'allocation_response.json', response)
    assert response['status'] == 'completed'
    final = [x for x in response['output'] if x.get('type') == 'message' and x.get('phase') == 'final_answer']
    assert len(final) == 1
    decision = Allocation.model_validate_json(''.join(x['text'] for x in final[0]['content'] if x['type'] == 'output_text'))
    quotas = {k: getattr(decision, k) for k in ['easy', 'medium', 'hard']}
    assert sum(quotas.values()) == 96 and all(x % 8 == 0 for x in quotas.values())
    save(root / 'mas_allocation.json', decision.model_dump(mode='json'))
    rng = random.Random(42)
    selected = []
    for bucket, count in quotas.items():
        patients = defaultdict(list)
        for row in sorted(real, key=lambda x: x['utt_id']):
            if row['zero_shot_bucket'] == bucket:
                patients[row['speaker_id']].append(row)
        speakers = sorted(patients)
        rng.shuffle(speakers)
        for speaker in speakers:
            rng.shuffle(patients[speaker])
        chosen = []
        while len(chosen) < count:
            for speaker in speakers:
                if patients[speaker] and len(chosen) < count:
                    chosen.append(patients[speaker].pop())
        selected.extend(chosen)
    assert len({x['utt_id'] for x in selected}) == 96
    dev_ids = {x['utt_id'] for x in dev}
    dev_paths = {str(Path(x['audio']).resolve()) for x in dev}
    dev_hashes = {sha256_file(Path(x['audio'])) for x in dev}
    prepared = []
    for row in selected:
        digest = sha256_file(Path(row['audio']))
        assert row['utt_id'] not in dev_ids and str(Path(row['audio']).resolve()) not in dev_paths
        assert digest not in dev_hashes
        prepared.append(dict(source='real', audio_id=row['utt_id'], audio=row['audio'], audio_sha256=digest,
                             chosen=row['clean_gt'], prompt='', language='Cantonese'))
    jsonl(root / 'real.jsonl', prepared)
    save(root / 'real_selection_receipt.json', dict(actual_mas_decision=True, quotas=quotas,
        patients=len({x['speaker_id'] for x in selected}), ids=[x['utt_id'] for x in selected],
        source_train_sha256=p.real_train.sha256, development_overlap_ids_paths_hashes=0,
        selection='Seed42 patient round-robin within each existing difficulty bucket; all buckets retained.'))
    old = read(prior / 'protocol.json')
    cfg = {k: old[k] for k in ['preferences', 'allocation', 'sampling_receipt', 'steps',
        'real_per_step', 'synthetic_per_step', 'kl_weights', 'synthetic_weight', 'lr', 'beta', 'seed']}
    keys = read(cfg['allocation'])['steps'][0][:2]
    # Include a hard example in the actual smoke, without using any development audio.
    smoke_ids = [next(x['utt_id'] for x in selected if x['zero_shot_bucket'] == bucket) for bucket in ['hard', 'easy']]
    jsonl(root / 'smoke_real.jsonl', [next(x for x in prepared if x['audio_id'] == key) for key in smoke_ids])
    jsonl(root / 'smoke_preferences.jsonl', [x for x in rows(cfg['preferences']) if x['audio_id'] in keys])
    save(root / 'smoke_allocation.json', dict(steps=[keys, list(reversed(keys))]))
    files = [Path(__file__), TRAINER, REPO / 'finetune/scripts/qwen3_asr_kl_dpo_probe.py',
        REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py', REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py',
        REPO / 'src/therapist_harness/dpo_training.py', REPO / 'examples/evaluate_residual_asr.py',
        REPO / 'examples/mas_residual_dpo_probe.py', REPO / 'finetune/scripts/evaluate_asr_checkpoint.py',
        base / 'residual_dpo_probe_v1/initial/identity_verified.json',
        base / 'residual_dpo_probe_v1/initial/evaluation/predictions.jsonl',
        base / 'residual_dpo_probe_v1/mas/results.json', base / 'residual_hard_retention_next_hypothesis_v1.json',
        base / 'lexical96_mas_v1/protocol.json', prior / 'protocol.json', prior / 'real.jsonl',
        Path(p.real_train.path), Path(p.feedback_split.path), Path(cfg['preferences']), Path(cfg['allocation']),
        Path(cfg['sampling_receipt'])] + list(root.glob('*.json')) + list(root.glob('*.jsonl'))
    save(root / 'protocol.json', dict(cfg, source_protocol=p.model_dump(mode='json'),
        source_sha256={str(f): sha256_file(f) for f in files}, real_audio_pool=96,
        objective='Synthetic audio DPO plus lambda times real-training symmetric KL to fixed merged R-only',
        real_history='Original real-training GT prefixes, identical to earlier KL implementation; no real CE',
        comparison='KL1 versus KL10 with same MAS-allocated real pool, synthetic data and exposures; compare both to matched residual DPO without KL',
        scope='Independent regularized DPO experiment, not a synthetic-only result. Different pool and runtime from the prior KL probe.',
        next_control='If a raw gain appears, compare matched random real allocation and random TTS generation before claiming MAS allocation value.',
        initial_identity_reused=str(base / 'residual_dpo_probe_v1/initial/identity_verified.json'),
        actual_new_generation=False, actual_new_sampling=False, actual_new_dpo_training=True))
    print(dict(quotas=quotas, real_audio=96, actual_mas_decision=True), flush=True)


def verify(root):
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == digest for f, digest in cfg['source_sha256'].items())
    return cfg


def run(base, root, weight, smoke=False):
    cfg = verify(root)
    if not smoke:
        assert read(root / 'smoke/verified.json')['status'] == 'passed'
    d = root / ('smoke' if smoke else f'weight_{weight:g}')
    d.mkdir(exist_ok=False)
    p = cfg['source_protocol']
    execute(d, [sys.executable, str(TRAINER), '--model_path', p['base_model_dir'],
        '--policy_adapter_path', p['initial_adapter']['path'], '--reference_adapter_path', p['initial_adapter']['path'],
        '--real_train_file', str(root / ('smoke_real.jsonl' if smoke else 'real.jsonl')),
        '--synthetic_train_file', str(root / 'smoke_preferences.jsonl') if smoke else cfg['preferences'],
        '--synthetic_allocation_file', str(root / 'smoke_allocation.json') if smoke else cfg['allocation'],
        '--output_dir', str(d / 'training'), '--max_steps', '2' if smoke else '12',
        '--real_per_step', '2' if smoke else '16', '--synthetic_per_step', '2' if smoke else '8',
        '--synthetic_weight', '1', '--kl_weight', str(weight), '--lr', '1e-5', '--beta', '.1', '--seed', '42'], 'training')
    result = read(d / 'training/training_result.json')
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert abs(result['probe_before'][0]['real_symmetric_kl']) < 1e-8
    assert abs(result['history'][0]['weighted_losses']['synthetic'] - math.log(2)) < 1e-4
    assert abs(result['history'][0]['weighted_losses']['real']) < 1e-7
    assert read(d / 'training/residual_verified.json')['status'] == 'passed'
    exposures = rows(d / 'training/training_exposures.jsonl')
    seen = {(x['step'], x['source'], x['audio_id'], x['occurrence']) for x in exposures}
    counts = Counter(x[1] for x in seen)
    assert counts == ({'real': 4, 'synthetic': 4} if smoke else {'real': 192, 'synthetic': 96})
    save(d / 'verified.json', dict(status='passed', initial_kl_zero=True, initial_dpo_log2=True,
        frozen_merged_base_unchanged=True, actual_audio_exposures=dict(counts), weight=weight))
    verify(root)
    if smoke:
        return
    execute(d, evaluation_command(p, d / 'evaluation_run', d / 'training/checkpoint-12'), 'evaluation')
    metrics = read(d / 'evaluation_run/evaluation/summary.json')['metrics']
    parent = read(base / 'parent_eval_check/summary.json')['metrics']
    gate = dict(pooled_improves=metrics['overall']['pooled_cer'] < parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer'] - parent['overall']['patient_macro_cer'] <= .001,
        hard_regression_within_limit=metrics['hard']['pooled_cer'] - parent['hard']['pooled_cer'] <= .005)
    save(d / 'paired_parent.json', compare(Path(p['initial_adapter_dev_predictions']['path']),
                                         d / 'evaluation_run/evaluation/predictions.jsonl'))
    save(d / 'results.json', dict(status='complete', metrics=metrics, gate_checks=gate,
        eligible_for_next_round_parent=all(gate.values()), objective=cfg['objective'], kl_weight=weight,
        actual_new_dpo_training=True, postprocessing_used=False))
    verify(root)
    print(dict(weight=weight, overall=metrics['overall'], gate=gate), flush=True)


def supervise(base, root):
    run(base, root, 10., smoke=True)
    jobs = []
    for gpu, weight in enumerate([1., 10.]):
        command = [sys.executable, str(Path(__file__).resolve()), 'run', str(base), '--weight', str(weight)]
        with (root / f'weight_{weight:g}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        save(root / f'weight_{weight:g}.launch.json', dict(pid=child.pid, command=command, gpu=gpu))
        jobs.append(child)
    codes = [p.wait() for p in jobs]
    assert codes == [0, 0], codes
    save(root / 'results.json', dict(status='complete', arms={str(w): read(root / f'weight_{w}/results.json') for w in [1, 10]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'run', 'supervise'])
    parser.add_argument('base', type=Path)
    parser.add_argument('--weight', type=float, choices=[1., 10.], default=10.)
    args = parser.parse_args()
    base = args.base.resolve()
    root = base / 'residual_retention_probe_v1'
    if args.stage == 'prepare':
        prepare(base, root)
    elif args.stage == 'supervise':
        supervise(base, root)
    else:
        run(base, root, args.weight, args.stage == 'smoke')
