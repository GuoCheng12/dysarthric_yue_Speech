#!/usr/bin/env python3
"""Paired reference policies with observed 4/8/12 checkpoints in each run."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from mas_selection_dpo_probe import REPO, read, rows, save, jsonl, execute
from mas_exploratory_round2_dpo_v1 import verify, exposure_plan, match_exposures, gate
from mas_ASR_target_format_probe import copy_receipts
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from therapist_harness.evidence import compare_prediction_snapshots
from therapist_harness.round_schema import ClusterDescriptor
from therapist_harness.store import sha256_file, sha256_directory

TRAINER = REPO / 'finetune/scripts/qwen3_asr_warm_dpo_trajectory_v1.py'
ARMS = ['original_reference', 'MAS_reference']
SCRATCH = Path('/tmp/qwen3_asr_round2_reference_trajectory_v1')


def command(a):
    return [sys.executable, str(TRAINER)] + [z for k, v in a.items() if v is not None for z in ['--' + k, str(v)]]


def check_training(cfg, a, out, control, expected):
    training = Path(a['output_dir']); result = read(training / 'training_result.json')
    assert result['policy_trainable_sha256_before'] == cfg['previous_policy_trainable_sha256']
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert read(training / 'residual_verified.json')['starting_parameters_equal_previous_endpoint']
    assert rows(training / 'training_exposures.jsonl') == expected
    for name, field in [('reference_log_probs.json', 'values'), ('real_teacher_cache.json', 'audio')]:
        actual, previous = read(training / name)[field], read(control / 'training_receipts' / name)[field]
        assert all(value == previous[key] for key, value in actual.items())
    masks = read(training / 'preference_token_masks.json'); old_masks = read(control / 'training_receipts/preference_token_masks.json')
    assert masks['mode'] == old_masks['mode'] == 'edit_eos'
    assert all(value == old_masks['pairs'][key] for key, value in masks['pairs'].items())
    assert read(training / 'run_protocol.json')['reference_asr_sha256'] == cfg['source_protocol']['initial_adapter']['sha256']
    trajectory = read(training / 'trajectory_verified.json'); assert trajectory['status'] == 'passed'
    assert trajectory['last_snapshot_tensor_identical_to_final_checkpoint']
    for item in trajectory['snapshots']:
        checkpoint = Path(item['checkpoint'])
        assert sha256_directory(checkpoint) == item['checkpoint_sha256']
        prefix = read(checkpoint.parent / 'training_result.json')
        assert prefix['history'] == result['history'][:item['step']]
        assert prefix['policy_trainable_sha256_after'] == item['trainable_sha256']
    copy_receipts(training, out)
    return trajectory


def smoke(base):
    source = base / 'round2_reference_selection_DPO_v1'; cfg = verify(source)
    root = base / 'warm_DPO_trajectory_smoke_v1'; root.mkdir(exist_ok=False)
    a = dict(cfg['arguments']['MAS_reference']); real = rows(a['real_train_file'])[:4]
    all_pairs = rows(a['synthetic_train_file']); keys = list(dict.fromkeys(x['audio_id'] for x in all_pairs))[:2]
    selected = [x for x in all_pairs if x['audio_id'] in keys]
    jsonl(root / 'real.jsonl', real); jsonl(root / 'preferences.jsonl', selected)
    factors = {x['audio_id']: 1. for x in real}; save(root / 'factors.json', dict(factors=factors))
    scheduled = dict(steps=[[keys[0]], [keys[1]]]); save(root / 'allocation.json', scheduled)
    a.update(real_train_file=str(root / 'real.jsonl'), synthetic_train_file=str(root / 'preferences.jsonl'),
        real_loss_factors=str(root / 'factors.json'), synthetic_allocation_file=str(root / 'allocation.json'),
        max_steps=2, real_per_step=4, synthetic_per_step=1, trajectory_steps='1,2',
        output_dir='/tmp/qwen3_asr_warm_DPO_trajectory_smoke_v1/training')
    expected = exposure_plan(a, real, selected, scheduled, factors)
    save(root / 'training_arguments.json', a); save(root / 'expected_exposures.json', expected)
    files = [Path(__file__), TRAINER, source / 'protocol.json'] + list(root.glob('*.json')) + list(root.glob('*.jsonl'))
    hashes = {**cfg['source_sha256'], **{str(f): sha256_file(f) for f in files}}
    save(root / 'protocol.json', dict(source_sha256=hashes, source_protocol=str(source / 'protocol.json'),
        actual_optimizer_steps=2, observed_checkpoints=[1, 2], scope='Technical smoke; no changed training objective or data admission.'))
    execute(root, command(a), 'training')
    trajectory = check_training(cfg, a, root, source / 'MAS_reference', expected)
    assert trajectory['snapshots'][0]['trainable_sha256'] != trajectory['snapshots'][1]['trainable_sha256']
    assert all(sha256_file(Path(f)) == h for f, h in hashes.items())
    save(root / 'verified.json', dict(status='passed', actual_steps=2, actual_audio=dict(real=4, synthetic=2),
        same_initial_R_only_DPO_scores_and_real_KL_tensors=True, exact_actual_exposures=True,
        same_warm_parent=True, snapshot_does_not_modify_parameters=True,
        last_snapshot_equals_actual_final_weights=True, actual_training_runs=1))
    print(read(root / 'verified.json'), flush=True)


def prepare(base, root):
    source = base / 'round2_reference_selection_DPO_v1'; old = verify(source)
    smoke_path = base / 'warm_DPO_trajectory_smoke_v1/verified.json'
    assert read(smoke_path)['status'] == 'passed'
    assert shutil.disk_usage('/tmp').free > 700 * 1024 ** 2
    root.mkdir(exist_ok=False); SCRATCH.mkdir(exist_ok=False)
    args = {}; expected = {}; files = [Path(__file__), TRAINER, source / 'protocol.json', smoke_path,
        base / 'round2_DPO_prefix_dose_v1/results.json']
    for arm in ARMS:
        out = root / arm; out.mkdir()
        a = dict(old['arguments'][arm], trajectory_steps='4,8,12', output_dir=str(SCRATCH / arm / 'training'))
        args[arm] = a
        expected[arm] = exposure_plan(a, rows(a['real_train_file']), rows(a['synthetic_train_file']),
            read(a['synthetic_allocation_file']), read(a['real_loss_factors'])['factors'])
        control = Path(old['result_roots'][arm])
        assert expected[arm] == rows(control / 'training_receipts/training_exposures.jsonl')
        save(out / 'expected_exposures.json', expected[arm]); files.append(out / 'expected_exposures.json')
    match_exposures(*[expected[x] for x in ARMS])
    save(root / 'evaluation_target_descriptor.json', read(source / 'evaluation_target_descriptor.json'))
    files.append(root / 'evaluation_target_descriptor.json')
    save(root / 'protocol.json', dict(plan=old['plan'], source_protocol=old['source_protocol'], arguments=args,
        previous_policy_trainable_sha256=old['previous_policy_trainable_sha256'], result_roots=old['result_roots'],
        source_sha256={**old['source_sha256'], **{str(f): sha256_file(f) for f in files}},
        new_execution_version='observed_warm_DPO_trajectory_v1', actual_new_training_runs=2,
        actual_new_generation=False, actual_new_sampling=False, same_engine_GPU_smoke=str(smoke_path),
        prospective_checkpoint_steps=[4, 8, 12], original_ledgers_unchanged=True, parent_promoted=False,
        selection_rule='Report every checkpoint; lowest pooled CER among candidates passing original pooled/macro/hard gates against both initial R-only and current parent is eligible. Single-seed development candidate, not proof of generalization.',
        hypothesis='Check whether earlier actual optimizer checkpoints retain global performance and target recovery; compare original versus the existing MAS-selected reference policy at every dose.',
        scope='Both arms already use the new MAS content. This contrast measures additional reference selection, not MAS content versus generic content. Same13 logical audios/39pairs, initial R-only KL/DPO teachers, shared current1373parent, seed42, actual exposure order/masses.'))
    verify(root); print(dict(status='prepared', arms=ARMS, checkpoints=[4, 8, 12]), flush=True)


def train(base, root, arm):
    cfg = verify(root); out = root / arm; a = cfg['arguments'][arm]
    started = time.monotonic(); execute(out, command(a), 'training')
    training_seconds = time.monotonic() - started
    trajectory = check_training(cfg, a, out, Path(cfg['result_roots'][arm]), read(out / 'expected_exposures.json'))
    save(out / 'training_verified.json', dict(status='passed', wall_seconds=training_seconds,
        actual_optimizer_steps=12, actual_training_runs=1, observed_checkpoints=[4, 8, 12],
        exact_previous_exposures_and_initial_teacher_tensors=True))
    initial = base / 'parent_eval_check'; current = base / 'ASR_target_format_dpo_probe_v1/edit_eos/evaluation_run/evaluation'
    descriptor = ClusterDescriptor.model_validate(read(root / 'evaluation_target_descriptor.json'))
    results = {}
    for item in trajectory['snapshots']:
        step = item['step']; destination = out / f'step_{step}'; destination.mkdir()
        checkpoint = Path(item['checkpoint']); started = time.monotonic()
        execute(destination, evaluation_command(cfg['source_protocol'], destination / 'evaluation_run', checkpoint), 'evaluation')
        elapsed = time.monotonic() - started
        prediction = destination / 'evaluation_run/evaluation/predictions.jsonl'
        metrics = read(prediction.with_name('summary.json'))['metrics']
        gates = {name: gate(metrics, read(path / 'summary.json')['metrics']) for name, path in [('initial_R_only', initial), ('current_parent', current)]}
        for name, path in [('initial_R_only', initial), ('current_parent', current)]:
            save(destination / ('paired_' + name + '.json'), compare(path / 'predictions.jsonl', prediction))
        response = compare_prediction_snapshots(rows(current / 'predictions.jsonl'), rows(prediction),
            before_prediction_column='prediction', after_prediction_column='prediction', critical_cer_threshold=.5,
            monitored_clusters=[descriptor], off_target_max_per_level_per_direction=3)
        response = json.loads(json.dumps(asdict(response), default=lambda x: x.model_dump(mode='json')))
        save(destination / 'current_target_response.json', response)
        results[str(step)] = dict(status='complete', metrics=metrics, gates=gates,
            eligible=all(all(x.values()) for x in gates.values()), checkpoint=str(checkpoint),
            checkpoint_sha256=item['checkpoint_sha256'], evaluation_wall_seconds=elapsed,
            target=response['monitored_cluster_changes'][0],
            actual_training_run=str(Path(a['output_dir']) / 'training_result.json'),
            scope='Actual step checkpoint of the shared run, not an independent restart.', parent_promoted=False)
        save(destination / 'results.json', results[str(step)])
        assert sha256_directory(checkpoint) == item['checkpoint_sha256']
        print(dict(arm=arm, step=step, errors=metrics['overall']['total_edits'], eligible=results[str(step)]['eligible']), flush=True)
    save(out / 'results.json', dict(status='complete', steps=results, training_wall_seconds=training_seconds,
        actual_training_runs=1, parent_promoted=False))
    verify(root)


def report(base, root):
    verify(root); results = {arm: read(root / arm / 'results.json') for arm in ARMS}
    assert all(x['status'] == 'complete' for x in results.values())
    candidates = []
    for step in [4, 8, 12]:
        paths = [root / arm / f'step_{step}/evaluation_run/evaluation/predictions.jsonl' for arm in ARMS]
        save(root / f'paired_reference_step_{step}.json', compare(*paths))
        for arm in ARMS:
            item = results[arm]['steps'][str(step)]
            if item['eligible']:
                candidates.append(dict(arm=arm, step=step, errors=item['metrics']['overall']['total_edits'], checkpoint=item['checkpoint']))
    candidates.sort(key=lambda x: (x['errors'], x['step'], x['arm']))
    save(root / 'results.json', dict(status='complete', arms=results, eligible_candidates=candidates,
        actual_training_runs=2, actual_evaluated_checkpoints=6, parent_promoted=False,
        interpretation='Within-arm dose comparison follows one actual trajectory. Across-arm contrast estimates the additional existing MAS reference policy, conditional on common MAS content. Repeated511 development, seed42 only.'))
    print(dict(status='complete', eligible_candidates=candidates), flush=True)


def supervise(base, root):
    children = []
    for gpu, arm in enumerate(ARMS):
        cmd = [sys.executable, str(Path(__file__).resolve()), 'train', str(base), '--arm', arm]
        with (root / (arm + '.log')).open('x') as log:
            process = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        save(root / (arm + '.launch.json'), dict(pid=process.pid, gpu=gpu, command=cmd)); children.append(process)
    codes = [p.wait() for p in children]; assert codes == [0, 0], codes
    report(base, root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['smoke', 'prepare', 'train', 'report', 'all'])
    parser.add_argument('base', type=Path); parser.add_argument('--arm', choices=ARMS, default=ARMS[0])
    args = parser.parse_args(); base = args.base.resolve(); root = base / 'round2_reference_trajectory_v1'
    if args.stage == 'smoke': smoke(base)
    if args.stage in ['prepare', 'all']: prepare(base, root)
    if args.stage == 'train': train(base, root, args.arm)
    if args.stage == 'report': report(base, root)
    if args.stage == 'all': supervise(base, root)
