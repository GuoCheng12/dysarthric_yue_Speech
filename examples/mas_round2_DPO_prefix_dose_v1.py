#!/usr/bin/env python3
"""Four/eight-step prefixes of the completed twelve-step warm MAS DPO update."""

import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys

from mas_selection_dpo_probe import REPO, read, rows, save
from mas_exploratory_round2_dpo_v1 import verify, exposure_plan, train
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file

ARMS = ['step_4', 'step_8']
SCRATCH = Path('/tmp/qwen3_asr_round2_DPO_prefix_dose_v1')


def prepare(base, root):
    source = base / 'round2_reference_selection_DPO_v1'; old = verify(source); control = source / 'MAS_reference'
    plan_path = base / 'round2_DPO_prefix_dose_plan_v1.json'; plan = read(plan_path)
    assert all(sha256_file(Path(f)) == h for f, h in plan['source_sha256'].items())
    original = old['arguments']['MAS_reference']; full_allocation = read(original['synthetic_allocation_file'])
    full_exposures = rows(control / 'training_receipts/training_exposures.jsonl')
    assert original['max_steps'] == 12 and len(full_allocation['steps']) == 12
    assert read(old['same_engine_GPU_smoke'])['status'] == 'passed'
    root.mkdir(exist_ok=False); SCRATCH.mkdir(exist_ok=False)
    arguments, delivery = {}, {}
    files = [Path(__file__), plan_path, source / 'protocol.json', control / 'results.json',
        control / 'training_receipts/training_exposures.jsonl', control / 'training_receipts/training_result.json',
        control / 'training_receipts/reference_log_probs.json', control / 'training_receipts/preference_token_masks.json']
    roles = {x['sample_id']: x['role'] for x in read(base / 'exploratory_round2_v1/adaptive_data/slots.json')}
    for arm, steps in zip(ARMS, [4, 8]):
        out = root / arm; out.mkdir()
        scheduled = dict(schema_version='literal_update_prefix_v1', steps=full_allocation['steps'][:steps],
            source_allocation=original['synthetic_allocation_file'], full_steps=12,
            scope='Literal prefix of the same12step schedule; no reallocation or equal-total-exposure claim.')
        save(out / 'allocation.json', scheduled)
        a = dict(original, max_steps=steps, synthetic_allocation_file=str(out / 'allocation.json'),
            output_dir=str(SCRATCH / arm / 'training')); arguments[arm] = a
        expected = exposure_plan(a, rows(a['real_train_file']), rows(a['synthetic_train_file']), scheduled, read(a['real_loss_factors'])['factors'])
        assert expected == [x for x in full_exposures if x['step'] <= steps]
        save(out / 'expected_exposures.json', expected)
        occurrences = list(dict.fromkeys((x['step'], x['source'], x['occurrence'], x['audio_id']) for x in expected))
        delivery[arm] = dict(steps=steps, source_exposures=dict(Counter(x[1] for x in occurrences)),
            unique_audio={source: len({x[3] for x in occurrences if x[1] == source}) for source in ['real', 'synthetic']},
            role_exposures=dict(Counter(roles[x[3]] for x in occurrences if x[1] == 'synthetic')))
        files += [out / 'allocation.json', out / 'expected_exposures.json']
    save(root / 'prefix_delivery.json', delivery)
    save(root / 'evaluation_target_descriptor.json', read(source / 'evaluation_target_descriptor.json'))
    save(root / 'cpu_smoke.json', dict(status='passed', exact_actual12step_exposure_prefixes=True,
        same_actual39pairs_and96real_pool=True, initial_R_only_DPO_and_KL_reference=True,
        reused_warm_GPU_smoke=old['same_engine_GPU_smoke'], new_GPU_smoke=False))
    files += [root / 'prefix_delivery.json', root / 'evaluation_target_descriptor.json', root / 'cpu_smoke.json']
    save(root / 'protocol.json', dict(plan=old['plan'], source_protocol=old['source_protocol'], arguments=arguments,
        previous_policy_trainable_sha256=old['previous_policy_trainable_sha256'], same_engine_GPU_smoke=old['same_engine_GPU_smoke'],
        control_result_root=str(control), source_sha256={**old['source_sha256'], **{str(f): sha256_file(f) for f in files}},
        actual_new_training_runs=2, actual_new_generation=False, actual_new_sampling=False,
        original_ledgers_unchanged=True, parent_promoted=False,
        scope='Only update prefix length differs. Fewer total exposures and prefix-specific coverage are explicit. Retain initialR-only KL and DPO teacher; do not inherit the failed current-teacher endpoint.'))
    verify(root); print(dict(status='prepared', delivery=delivery), flush=True)


def report(base, root):
    cfg = verify(root); control = Path(cfg['control_result_root'])
    paths = {**{a: root / a for a in ARMS}, 'step_12': control}
    results = {a: read(p / 'results.json') for a, p in paths.items()}
    assert all(x['status'] == 'complete' for x in results.values())
    trajectory = {}; old_history = read(control / 'training_receipts/training_result.json')['history']
    for arm in ARMS:
        path = paths[arm]
        assert rows(path / 'training_receipts/training_exposures.jsonl') == read(path / 'expected_exposures.json')
        assert read(path / 'training_receipts/reference_log_probs.json')['values'] == read(control / 'training_receipts/reference_log_probs.json')['values']
        assert read(path / 'training_receipts/preference_token_masks.json') == read(control / 'training_receipts/preference_token_masks.json')
        history = read(path / 'training_receipts/training_result.json')['history']
        trajectory[arm] = dict(exact_logged_trajectory_prefix=history == old_history[:len(history)],
            max_absolute_logged_loss_difference=max(abs(x['loss'] - y['loss']) for x, y in zip(history, old_history)),
            scope='Same logical update prefix; numeric trajectory identity is reported rather than assumed.')
        save(root / (arm + '_vs_step12.json'), compare(control / 'evaluation_run/evaluation/predictions.jsonl', path / 'evaluation_run/evaluation/predictions.jsonl'))
    targets = {a: read(p / 'current_target_response.json')['monitored_cluster_changes'][0] for a, p in paths.items()}
    save(root / 'results.json', dict(status='complete', arms=results, target_changes=targets, trajectory_verification=trajectory,
        prefix_delivery=read(root / 'prefix_delivery.json'), actual_new_training_runs=2, step12_control_reused=True,
        same39_initial_R_only_DPO_reference_scores_and_masks=True, parent_promoted=False,
        evaluation_scope='Repeated511 real development feedback and selected seed42 parent. No inference postprocessing.'))
    lines = ['从同一1373错的parent，使用同一批13音频39对和初始R-only DPO/KL teacher，比较原12step更新的4/8step前缀。', '',
        '| 更新步数 | CER % | 字符错误 | macro CER % | hard 错误 | 織删除 | 明确目标恢复 |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm, value in results.items():
        m = value['metrics']; target = targets[arm]
        lines.append(f'| {arm} | {100*m["overall"]["pooled_cer"]:.6f} | {m["overall"]["total_edits"]} | {100*m["overall"]["patient_macro_cer"]:.6f} | {m["hard"]["total_edits"]} | {target["after"]["error_count"]} | {target["target_positions"]["recovered_count"]} |')
    lines += ['', '初始R-only为1384错，本轮parent为1373错；原hard门槛最多694错。实际曝光与覆盖见prefix_delivery.json。',
        '两个较短前缀为真实新训练；12step对照复用已完成结果。旧账本和checkpoint门槛没有改变。']
    with (root / 'results.zh.md').open('x') as handle: handle.write('\n'.join(lines) + '\n')
    print({a: dict(errors=x['metrics']['overall']['total_edits'], CER_percent=100*x['metrics']['overall']['pooled_cer']) for a, x in results.items()}, flush=True)


def supervise(base, root):
    processes = []
    for gpu, arm in enumerate(ARMS):
        command = [sys.executable, str(Path(__file__).resolve()), 'train', str(base), '--arm', arm]
        with (root / (arm + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        save(root / (arm + '.launch.json'), dict(pid=child.pid, gpu=gpu, command=command)); processes.append(child)
    codes = [p.wait() for p in processes]; assert codes == [0, 0], codes
    report(base, root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['prepare', 'train', 'report', 'supervise', 'all']); parser.add_argument('base', type=Path)
    parser.add_argument('--arm', choices=ARMS, default=ARMS[0]); args = parser.parse_args()
    base = args.base.resolve(); root = base / 'round2_DPO_prefix_dose_v1'
    if args.stage in ['prepare', 'all']: prepare(base, root)
    if args.stage == 'train': train(base, root, args.arm)
    elif args.stage == 'report': report(base, root)
    elif args.stage in ['supervise', 'all']: supervise(base, root)
