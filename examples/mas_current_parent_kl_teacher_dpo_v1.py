#!/usr/bin/env python3
"""Matched real-KL teacher ablation on unchanged current-parent MAS preferences."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

from mas_selection_dpo_probe import REPO, read, rows, save, execute
from mas_exploratory_round2_dpo_v1 import verify, exposure_plan, gate
from mas_ASR_target_format_probe import copy_receipts
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from therapist_harness.evidence import compare_prediction_snapshots
from therapist_harness.round_schema import ClusterDescriptor
from therapist_harness.store import sha256_file, sha256_directory

TRAINER = REPO / 'finetune/scripts/qwen3_asr_current_parent_kl_teacher_v1.py'


def prepare(base, root):
    source = base / 'round2_reference_selection_DPO_v1'; old = verify(source)
    plan_path = base / 'current_parent_KL_teacher_plan_v1.json'; plan = read(plan_path)
    assert all(sha256_file(Path(f)) == h for f, h in plan['source_sha256'].items())
    smoke = base / 'current_parent_KL_teacher_smoke_v1'
    assert read(smoke / 'verified.json')['status'] == 'passed'
    control = source / 'MAS_reference'; a = dict(old['arguments']['MAS_reference'],
        output_dir='/tmp/qwen3_asr_current_parent_KL_teacher_DPO_v1/training')
    assert len(rows(a['real_train_file'])) == 96 and len(rows(a['synthetic_train_file'])) == 39
    expected = exposure_plan(a, rows(a['real_train_file']), rows(a['synthetic_train_file']),
        read(a['synthetic_allocation_file']), read(a['real_loss_factors'])['factors'])
    assert expected == rows(control / 'training_receipts/training_exposures.jsonl')
    root.mkdir(exist_ok=False); out = root / 'current_parent_teacher'; out.mkdir()
    save(out / 'expected_exposures.json', expected)
    save(root / 'evaluation_target_descriptor.json', read(source / 'evaluation_target_descriptor.json'))
    save(root / 'control_reuse.json', dict(result_root=str(control), exact39_actual_pairs_reused=True,
        exact192real96synthetic_exposure_order_and_mass=True, unchanged_all_training_arguments_except_output_path=True,
        changed_computation='Only the real KL frozen teacher is current1373parent instead of initialR-only. DPO reference remains initialR-only.',
        actual_control_training_reexecuted=False))
    files = [Path(__file__), TRAINER, plan_path, smoke / 'protocol.json', smoke / 'verified.json',
        smoke / 'training_receipts/real_teacher_cache.json', source / 'protocol.json', source / 'results.json',
        control / 'results.json', control / 'verified.json', control / 'training_receipts/reference_log_probs.json',
        control / 'training_receipts/preference_token_masks.json', control / 'training_receipts/real_teacher_cache.json',
        control / 'training_receipts/training_exposures.jsonl', out / 'expected_exposures.json',
        root / 'evaluation_target_descriptor.json', root / 'control_reuse.json']
    save(root / 'protocol.json', dict(plan=old['plan'], source_protocol=old['source_protocol'], arguments=a,
        previous_policy_trainable_sha256=old['previous_policy_trainable_sha256'],
        control_result_root=str(control), smoke=str(smoke), intervention_plan=str(plan_path),
        source_sha256={**old['source_sha256'], **read(smoke / 'protocol.json')['source_sha256'], **{str(f): sha256_file(f) for f in files}},
        original_ledgers_unchanged=True, DPO_reference_initial_R_only=True, KL_teacher_frozen_current_parent=True,
        actual_new_generation=False, actual_new_sampling=False, actual_new_DPO_training=True, parent_promoted=False))
    verify(root); print(dict(status='prepared', paired_audio=13, pairs=39, control_errors=1380), flush=True)


def train(base, root):
    cfg = verify(root); out = root / 'current_parent_teacher'; a = cfg['arguments']; control = Path(cfg['control_result_root'])
    started = time.monotonic()
    execute(out, [sys.executable, str(TRAINER)] + [z for k, v in a.items() if v is not None for z in ['--' + k, str(v)]], 'training')
    train_seconds = time.monotonic() - started
    training = Path(a['output_dir']); result = read(training / 'training_result.json')
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert result['policy_trainable_sha256_before'] == cfg['previous_policy_trainable_sha256']
    assert read(training / 'residual_verified.json')['fixed_current_parent_KL_reference']
    assert read(training / 'current_KL_teacher_verified.json')['status'] == 'passed'
    assert read(training / 'current_KL_teacher_verified.json')['actual_teacher_audio'] == 96
    assert rows(training / 'training_exposures.jsonl') == read(out / 'expected_exposures.json')
    assert rows(training / 'training_exposures.jsonl') == rows(control / 'training_receipts/training_exposures.jsonl')
    teacher = read(training / 'real_teacher_cache.json'); old_teacher = read(control / 'training_receipts/real_teacher_cache.json')
    assert teacher['frozen_reference_sha256'] == cfg['plan']['current_parent_sha256']
    assert teacher['teacher_parameters_unchanged'] and teacher['teacher_frozen_before_first_policy_update']
    assert teacher['audio'].keys() == old_teacher['audio'].keys() and len(teacher['audio']) == 96
    smoke_teacher = read(Path(cfg['smoke']) / 'training_receipts/real_teacher_cache.json')['audio']
    assert all(x == teacher['audio'][k] for k, x in smoke_teacher.items())
    changed = sum(x['tensor_sha256'] != old_teacher['audio'][k]['tensor_sha256'] for k, x in teacher['audio'].items())
    assert changed > 0 and not teacher['tensor_data_persisted']
    reference = read(training / 'reference_log_probs.json')
    assert reference['values'] == read(control / 'training_receipts/reference_log_probs.json')['values']
    assert read(training / 'preference_token_masks.json') == read(control / 'training_receipts/preference_token_masks.json')
    assert reference['mask_manifest_sha256'] == sha256_file(training / 'preference_token_masks.json')
    assert read(training / 'run_protocol.json')['reference_asr_sha256'] == cfg['source_protocol']['initial_adapter']['sha256']
    assert abs(result['history'][0]['weighted_losses']['real']) < 1e-7
    assert result['history'][1]['weighted_losses']['real'] > 0
    checkpoint = Path(result['checkpoint']); digest = sha256_directory(checkpoint)
    assert not (checkpoint / 'optimizer.pt').exists() and not (training / 'real_teacher_log_probs.pt').exists()
    copy_receipts(training, out)
    save(out / 'verified.json', dict(status='passed', exact_common_starting_parameters=True,
        all39_initial_R_only_DPO_reference_scores_identical=True, exact_score_masks_and_exposures=True,
        current_KL_teacher_frozen_before_update=True, all4_smoke_teacher_distributions_identical=True,
        changed_real_teacher_audio=changed, current_KL_initial_loss=result['history'][0]['weighted_losses']['real'],
        checkpoint=str(checkpoint), checkpoint_sha256=digest))
    started = time.monotonic()
    execute(out, evaluation_command(cfg['source_protocol'], out / 'evaluation_run', checkpoint), 'evaluation')
    evaluation_seconds = time.monotonic() - started
    predictions = out / 'evaluation_run/evaluation/predictions.jsonl'
    metrics = read(predictions.with_name('summary.json'))['metrics']
    initial = base / 'parent_eval_check'; current = base / 'ASR_target_format_dpo_probe_v1/edit_eos/evaluation_run/evaluation'
    initial_gate = gate(metrics, read(initial / 'summary.json')['metrics']); current_gate = gate(metrics, read(current / 'summary.json')['metrics'])
    for name, before in [('initial_R_only', initial / 'predictions.jsonl'), ('current_parent', current / 'predictions.jsonl'),
        ('initial_teacher_control', control / 'evaluation_run/evaluation/predictions.jsonl')]:
        save(out / ('paired_' + name + '.json'), compare(before, predictions))
    descriptor = ClusterDescriptor.model_validate(read(root / 'evaluation_target_descriptor.json'))
    response = compare_prediction_snapshots(rows(current / 'predictions.jsonl'), rows(predictions),
        before_prediction_column='prediction', after_prediction_column='prediction', critical_cer_threshold=.5,
        monitored_clusters=[descriptor], off_target_max_per_level_per_direction=3)
    save(out / 'current_target_response.json', json.loads(json.dumps(asdict(response), default=lambda x: x.model_dump(mode='json'))))
    save(out / 'results.json', dict(status='complete', metrics=metrics, initial_R_only_gate=initial_gate,
        current_parent_gate=current_gate, eligible_against_both_parents=all(initial_gate.values()) and all(current_gate.values()),
        checkpoint=str(checkpoint), checkpoint_sha256=digest, training_wall_seconds=train_seconds,
        evaluation_wall_seconds=evaluation_seconds, actual_new_DPO_training=True, parent_promoted=False, postprocessing_used=False))
    verify(root); report(base, root)


def report(base, root):
    cfg = verify(root); current = root / 'current_parent_teacher'; control = Path(cfg['control_result_root'])
    paths = dict(initial_R_only_teacher=control, current_parent_teacher=current)
    results = {k: read(p / 'results.json') for k, p in paths.items()}
    paired = read(current / 'paired_initial_teacher_control.json')
    targets = {k: read(p / 'current_target_response.json')['monitored_cluster_changes'][0] for k, p in paths.items()}
    save(root / 'results.json', dict(status='complete', arms=results, target_changes=targets,
        current_teacher_minus_initial_teacher_CER_pp=paired['groups']['overall']['delta_pp'],
        bootstrap95_pp=paired['groups']['overall']['patient_cluster_bootstrap_delta_pp_95'],
        actual_new_training_runs=1, control_reused=True, DPO_reference_initial_R_only_unchanged=True,
        same_actual_pairs_masks_and_exposures=True, parent_promoted=False,
        scope='Explicit real-KL teacher ablation; repeated511 development feedback and selected seed42 starting parent. No generation, resampling, relabeling or inference postprocessing.'))
    lines = ['仅将真实 KL teacher 从初始 R-only 改为冻结的本轮1373错 parent；DPO reference 仍为初始 R-only。', '',
        '| 条件 | CER % | 字符错误 | macro CER % | hard 错误 | 織删除 |', '|---|---:|---:|---:|---:|---:|']
    for name, value in results.items():
        m = value['metrics']; lines.append(f'| {name} | {100*m["overall"]["pooled_cer"]:.6f} | {m["overall"]["total_edits"]} | {100*m["overall"]["patient_macro_cer"]:.6f} | {m["hard"]["total_edits"]} | {targets[name]["after"]["error_count"]} |')
    lines += ['', '共享13音频39对，原始两份候选采样来源、39对冻结 DPO 分数、全部 score masks 与真实/合成曝光保持一致。',
        '这是新的保持约束消融；没有改动原实验账本或放宽 checkpoint 门槛。']
    with (root / 'results.zh.md').open('x') as handle: handle.write('\n'.join(lines) + '\n')
    print({k: dict(CER_percent=100*x['metrics']['overall']['pooled_cer'], errors=x['metrics']['overall']['total_edits']) for k, x in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['prepare', 'train', 'report', 'all']); parser.add_argument('base', type=Path)
    args = parser.parse_args(); base = args.base.resolve(); root = base / 'current_parent_KL_teacher_DPO_v1'
    if args.stage == 'all':
        smoke = base / 'current_parent_KL_teacher_smoke_v1/verified.json'
        launch = read(base / 'current_parent_KL_teacher_smoke_execution_v1/supervisor_launch.json')
        while not smoke.exists():
            assert Path(f'/proc/{launch["pid"]}').exists(), 'Current-parent KL teacher smoke ended before verification.'
            time.sleep(10)
        prepare(base, root); train(base, root)
    elif args.stage == 'prepare': prepare(base, root)
    elif args.stage == 'train': train(base, root)
    else: report(base, root)
