#!/usr/bin/env python3
"""Replicate the observed four-step reference-policy contrast at seeds43/44."""

import argparse
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from mas_selection_dpo_probe import REPO, read, rows, save
from mas_exploratory_round2_dpo_v1 import verify, exposure_plan, match_exposures, train
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file

ARMS = ['original_reference', 'MAS_reference']
ROOT = Path('/tmp/qwen3_asr_round2_reference_dose4_seeds_v1')


def prepare(base):
    source = base / 'round2_reference_selection_DPO_v1'; old = verify(source)
    trajectory = base / 'round2_reference_trajectory_v1'; verify(trajectory)
    completed = read(trajectory / 'results.json'); assert completed['status'] == 'complete'
    assert completed['eligible_candidates'] == [dict(arm='MAS_reference', step=4, errors=1367,
        checkpoint=completed['arms']['MAS_reference']['steps']['4']['checkpoint'])]
    assert shutil.disk_usage('/tmp').free > 360 * 1024 ** 2
    ROOT.mkdir(exist_ok=False)
    allocation = dict(steps=read(old['arguments']['MAS_reference']['synthetic_allocation_file'])['steps'][:4],
        scope='Fixed literal four-step synthetic exposure schedule; no allocation search on replication seeds.')
    save(ROOT / 'allocation.json', allocation)
    save(ROOT / 'evaluation_target_descriptor.json', read(trajectory / 'evaluation_target_descriptor.json'))
    args = {}; plans = {}; files = [Path(__file__), source / 'protocol.json', trajectory / 'protocol.json',
        trajectory / 'results.json', ROOT / 'allocation.json', ROOT / 'evaluation_target_descriptor.json']
    for seed in [43, 44]:
        for arm in ARMS:
            name = f'{arm}_seed{seed}'; out = ROOT / name; out.mkdir()
            a = dict(old['arguments'][arm], seed=seed, max_steps=4,
                synthetic_allocation_file=str(ROOT / 'allocation.json'),
                output_dir=str(ROOT / 'weights' / name / 'training'))
            args[name] = a
            plans[name] = exposure_plan(a, rows(a['real_train_file']), rows(a['synthetic_train_file']),
                allocation, read(a['real_loss_factors'])['factors'])
            save(out / 'expected_exposures.json', plans[name]); files.append(out / 'expected_exposures.json')
        match_exposures(*[plans[f'{arm}_seed{seed}'] for arm in ARMS])
    save(ROOT / 'protocol.json', dict(plan=old['plan'], source_protocol=old['source_protocol'], arguments=args,
        previous_policy_trainable_sha256=old['previous_policy_trainable_sha256'], same_engine_GPU_smoke=old['same_engine_GPU_smoke'],
        source_sha256={**old['source_sha256'], **{str(p): sha256_file(p) for p in files}},
        seed42_result_roots={arm: str(trajectory / arm / 'step_4') for arm in ARMS},
        actual_new_training_runs=4, actual_new_generation=False, actual_new_sampling=False,
        parent_promoted=False, original_ledgers_unchanged=True,
        intervention='Only training seed changes to43/44. Same learned1373parent and initialR-only DPO/KL teacher, same13audio39pairs per policy, same4step synthetic sequence,64real/32synthetic exposures. Real order may differ across seeds, but is exactly matched between reference policies within each seed.',
        runtime='Existing validated warm trainer. The seed42 controls are observed step4 weights from completed uninterrupted12step runs; they are not claimed to be new independent4step executions.',
        decision='Report all six outcomes and per-policy three-training-seed means. Do not choose the best replication seed or alter step count after seeing43/44.',
        scope='Replicates training conditional on selected seed42 parent and fixed generated dataset; not independent end-to-end MAS seeds or a sealed test.',
        storage='New weights and receipts use /tmp because /data is nearly full. Old artifacts are preserved.'))
    save(base / 'round2_reference_dose4_seeds_location_v1.json', dict(root=str(ROOT),
        protocol_sha256=sha256_file(ROOT / 'protocol.json'), storage='Explicit new experiment location; original ledgers unchanged.'))
    verify(ROOT); print(dict(status='prepared', root=str(ROOT), actual_new_training_runs=4), flush=True)


def report(base):
    cfg = verify(ROOT); results = {}; files = [Path(__file__), ROOT / 'protocol.json']
    for arm in ARMS:
        for seed in [42, 43, 44]:
            out = Path(cfg['seed42_result_roots'][arm]) if seed == 42 else ROOT / f'{arm}_seed{seed}'
            result = read(out / 'results.json'); assert result['status'] == 'complete'
            results[f'{arm}_seed{seed}'] = dict(result_root=str(out), metrics=result['metrics'],
                eligible=result['eligible'] if seed == 42 else result['eligible_against_both_parents'],
                checkpoint=result['checkpoint'], checkpoint_sha256=result['checkpoint_sha256'])
            files.append(out / 'results.json')
            if seed != 42:
                assert rows(out / 'training_receipts/training_exposures.jsonl') == read(out / 'expected_exposures.json')
    for seed in [42, 43, 44]:
        paths = [Path(results[f'{arm}_seed{seed}']['result_root']) / 'evaluation_run/evaluation/predictions.jsonl' for arm in ARMS]
        save(ROOT / f'paired_seed{seed}.json', compare(*paths))
        if seed != 42:
            left, right = [ROOT / f'{arm}_seed{seed}' for arm in ARMS]
            match_exposures(rows(left / 'training_receipts/training_exposures.jsonl'), rows(right / 'training_receipts/training_exposures.jsonl'))
    means = {}
    for arm in ARMS:
        values = [results[f'{arm}_seed{seed}'] for seed in [42, 43, 44]]
        errors = [x['metrics']['overall']['total_edits'] for x in values]
        means[arm] = dict(mean_errors=statistics.mean(errors), mean_CER_percent=statistics.mean(errors) / 6656 * 100,
            sample_std_errors=statistics.stdev(errors), gate_passes=sum(x['eligible'] for x in values), training_seeds=3)
    save(ROOT / 'results.json', dict(status='complete', results=results, training_seed_means=means,
        actual_new_training_runs=4, reused_seed42_checkpoints=2, parent_promoted=False,
        source_sha256={str(p): sha256_file(p) for p in files}))
    lines = ['固定4步更新：原reference与MAS选择reference的训练seed复核', '',
        '| 条件 | seed | CER % | 字符错误 | hard 错误 | 原验收 |', '|---|---:|---:|---:|---:|---|']
    for arm in ARMS:
        for seed in [42, 43, 44]:
            x = results[f'{arm}_seed{seed}']; m = x['metrics']
            lines.append(f'| {arm} | {seed} | {100*m["overall"]["pooled_cer"]:.6f} | {m["overall"]["total_edits"]} | {m["hard"]["total_edits"]} | {"通过" if x["eligible"] else "未通过"} |')
        lines.append(f'| {arm}均值 | 3 seeds | {means[arm]["mean_CER_percent"]:.6f} | {means[arm]["mean_errors"]:.3f} | — | {means[arm]["gate_passes"]}/3 |')
    lines += ['', '初始R-only1384错/20.793269%；共同parent1373错/20.628005%。各分支仍从该共同parent开始；初始R-only DPO/KL teacher始终固定。',
        'seed42复用同次轨迹的实际第4步checkpoint；seed43/44为4次真实新训练。生成数据和ASR采样不变；训练seed改变真实池顺序，同一seed两种reference政策的真实顺序和曝光权重严格相同。',
        '这是固定已选parent及生成数据下的训练随机性复核；511条是反复使用的开发反馈，不是完整MAS独立重复或封存测试。未修改旧账本、没有自动参数晋级。']
    with (ROOT / 'results.zh.md').open('x') as handle: handle.write('\n'.join(lines) + '\n')
    print(dict(status='complete', means=means), flush=True)


def supervise(base):
    processes = []
    for gpu, arm in enumerate(ARMS):
        cmd = [sys.executable, str(Path(__file__).resolve()), 'arm', str(base), '--arm', arm]
        with (ROOT / (arm + '.log')).open('x') as log:
            child = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        save(ROOT / (arm + '.launch.json'), dict(pid=child.pid, gpu=gpu, command=cmd)); processes.append(child)
    codes = [p.wait() for p in processes]; assert codes == [0, 0], codes
    report(base)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'arm', 'report', 'all']); parser.add_argument('base', type=Path)
    parser.add_argument('--arm', choices=ARMS, default=ARMS[0]); args = parser.parse_args(); base = args.base.resolve()
    if args.stage in ['prepare', 'all']: prepare(base)
    if args.stage == 'arm':
        for seed in [43, 44]: train(base, ROOT, f'{args.arm}_seed{seed}')
    if args.stage == 'report': report(base)
    if args.stage == 'all': supervise(base)
