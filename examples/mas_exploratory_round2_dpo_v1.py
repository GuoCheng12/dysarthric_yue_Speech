#!/usr/bin/env python3
"""Matched fixed/adaptive curriculum DPO continuing the same learned residual."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_exploratory_round2_data_v1 import verify as verify_data
from mas_edit_mask_dpo_probe import exposure_plan
from mas_ASR_target_format_probe import copy_receipts
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from therapist_harness.evidence import compare_prediction_snapshots
from therapist_harness.preference_data import build_preference_pairs
from therapist_harness.round_schema import ClusterDescriptor
from therapist_harness.store import sha256_file, sha256_directory
from therapist_harness.text import text_normalize

ARMS = ['fixed_curriculum', 'adaptive_MAS']
SCRATCH = Path('/tmp/qwen3_asr_exploratory_round2_DPO_v1')
TRAINER = REPO / 'finetune/scripts/qwen3_asr_edit_kl_dpo_continuation_v1.py'


def allocation(slots, common, policy):
    """Use explicit optimizer exposure shares, independent of generated copies."""
    pools = defaultdict(list)
    roles = {x['sample_id']: x['role'] for x in slots}
    for key in sorted(common): pools[roles[key]].append(key)
    assert pools
    requested = policy['requested_training_exposures']; total = policy['total_audio_exposures']
    exact = {r: total * requested[r] / sum(requested[k] for k in pools) for r in pools}
    quotas = {r: int(v) for r, v in exact.items()}
    for r in sorted(pools, key=lambda r: (-(exact[r] - quotas[r]), r))[:total - sum(quotas.values())]:
        quotas[r] += 1
    labels = [r for r in sorted(quotas) for _ in range(quotas[r])]
    rng = random.Random(42 + 2017); rng.shuffle(labels)
    decks = {}; sequence = []
    for r in labels:
        if not decks.get(r):
            decks[r] = list(pools[r]); rng.shuffle(decks[r])
        sequence.append(decks[r].pop())
    assert len(sequence) == total == 96 and set(sequence) == common
    assert Counter(roles[k] for k in sequence) == quotas
    return dict(schema_version='explicit_role_exposure_allocation_v1',
        requested_training_exposures=requested, paired_audio={r: len(v) for r, v in pools.items()},
        unavailable_roles=sorted(set(requested) - set(pools)), role_exposures=quotas,
        total_audio_exposures=total, steps=[sequence[i:i + 8] for i in range(0, total, 8)],
        rule='Explicit exposure shares, largest remainder over available roles; seeded shuffled cycling within role. Same actual logical slots and masses on both sides.')


def collapse(records):
    result = defaultdict(float)
    for x in records:
        result[(x['step'], x['source'], x['occurrence'], x['audio_id'])] += x['loss_weight']
    return result


def match_exposures(left, right):
    assert [x for x in left if x['source'] == 'real'] == [x for x in right if x['source'] == 'real']
    a, b = collapse(left), collapse(right)
    assert a.keys() == b.keys() and all(math.isclose(a[k], b[k], abs_tol=1e-12) for k in a)


def verify(root):
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == h for f, h in cfg['source_sha256'].items())
    assert sha256_directory(Path(cfg['plan']['current_parent'])) == cfg['plan']['current_parent_sha256']
    return cfg


def prepare(base, root):
    experiment = root.parent; data_root = experiment / 'adaptive_data'
    data_cfg = verify_data(data_root); plan = data_cfg['plan']; p = plan['source_protocol']
    assert read(data_root / 'data_complete.json')['status'] == 'complete'
    decision = read(data_root / 'proposal_receipt.json')
    assert all(sha256_file(Path(f)) == h for f, h in decision['bound_sha256'].items())
    old_path = base / 'ASR_target_format_dpo_probe_v1/protocol.json'; old = read(old_path)
    parent_root = base / 'ASR_target_format_dpo_probe_v1/edit_eos'
    parent_result = read(parent_root / 'training_receipts/training_result.json')
    smoke = base / 'edit_KL_continuation_smoke_v1/verified.json'
    assert read(smoke)['status'] == 'passed'
    roots = dict(fixed_curriculum=experiment / 'fixed_curriculum_sampling', adaptive_MAS=data_root / 'sampling')
    files = [Path(__file__), TRAINER, old_path, smoke, data_root / 'execution.json', data_root / 'data_complete.json',
        data_root / 'proposal_receipt.json', parent_root / 'training_receipts/training_result.json',
        parent_root / 'training_receipts/policy_initial_runtime.json', parent_root / 'training_receipts/real_teacher_cache.json',
        parent_root / 'evaluation_run/evaluation/summary.json', parent_root / 'evaluation_run/evaluation/predictions.jsonl']
    data, receipts, inputs, draws = {}, {}, {}, {}
    for arm, sample in roots.items():
        receipt = read(sample / 'sampling_receipt.json'); receipts[arm] = receipt
        assert receipt['status'] == 'complete' and receipt['parent_files_unchanged']
        assert receipt['current_asr_sha256'] == plan['current_parent_sha256']
        assert receipt['initial_R_only_sha256'] == p['initial_adapter']['sha256']
        assert receipt['learned_residual_loaded'] and not receipt['current_residual_merged']
        for name, field in [('candidates.jsonl', 'candidates_sha256'), ('greedy.jsonl', 'greedy_sha256'), ('validated_inputs.jsonl', 'validated_inputs_sha256')]:
            assert sha256_file(sample / name) == receipt[field]
        manifest = rows(sample / 'validated_inputs.jsonl'); inputs[arm] = {x['audio_id']: x for x in manifest}
        for x in manifest:
            assert sha256_file(Path(x['audio'])) == x['audio_sha256']
            review = x['review']; assert review['candidate_audio_sha256'] == x['audio_sha256']
            trans = Path(review['transcription_receipt_path'])
            assert sha256_file(trans) == review['transcription_receipt_sha256']; files.append(trans)
        candidates = rows(sample / 'candidates.jsonl'); draws[arm] = {x['sample_id']: x for x in candidates}
        rebuilt, audit = build_preference_pairs(manifest, candidates, plan['current_parent_sha256'],
            expected_samples=4, greedy_candidates=rows(sample / 'greedy.jsonl'))
        assert rebuilt == rows(sample / 'preferences.jsonl') and audit == read(sample / 'preference_audit.json')
        data[arm] = rebuilt
        files += [sample / n for n in ['sampling_receipt.json', 'candidates.jsonl', 'greedy.jsonl', 'validated_inputs.jsonl', 'preferences.jsonl', 'preference_audit.json']]
    fields = ['base_model_sha256', 'current_asr_sha256', 'initial_R_only_sha256', 'samples_per_audio', 'seed',
        'temperature', 'top_p', 'top_k', 'max_new_tokens', 'language', 'prompt', 'do_sample',
        'real_train_sha256', 'feedback_sha256', 'greedy_config']
    assert all(receipts[ARMS[0]][k] == receipts[ARMS[1]][k] for k in fields)
    common = set.intersection(*[{x['audio_id'] for x in data[a]} for a in ARMS])
    assert common, 'No common actual errors; do not manufacture preferences.'
    for key in common:
        a, b = [inputs[arm][key] for arm in ARMS]
        assert a['clean_gt'] != b['clean_gt'] and a['audio_sha256'] != b['audio_sha256']
        for index in range(4):
            x, y = [draws[arm][f'{key}::{index}'] for arm in ARMS]
            assert x['sample_index'] == y['sample_index'] and x['seed'] == y['seed']
    scheduled = allocation(read(data_root / 'slots.json'), common, read(data_root / 'role_policy.json'))
    assert shutil.disk_usage('/tmp').free > 250 * 1024 ** 2
    root.mkdir(exist_ok=False); SCRATCH.mkdir(exist_ok=False)
    save(root / 'allocation.json', scheduled)
    # The data-stage descriptor is a compact selection record. Bind the full
    # existing evaluator schema from the same frozen evidence, including rates.
    target = read(data_root / 'target_descriptor.json')
    cluster = next(x for x in read(data_root / 'evidence_components.json')['extraction']['error_clusters']
        if x['evidence_id'] == target['evidence_id'])
    assert all(cluster[k] == v for k, v in target.items())
    descriptor = ClusterDescriptor.model_validate({k: cluster[k] for k in ClusterDescriptor.model_fields})
    save(root / 'evaluation_target_descriptor.json', descriptor.model_dump(mode='json'))
    arguments, expected = {}, {}
    for arm in ARMS:
        out = root / arm; out.mkdir()
        selected = []
        for row in data[arm]:
            if row['audio_id'] not in common: continue
            chosen = row['chosen'].rstrip('。！？!?')
            assert text_normalize(chosen) == text_normalize(row['chosen']) != text_normalize(row['rejected'])
            selected.append(dict(row, chosen=chosen, metadata=dict(row['metadata'], original_target_text=row['chosen'],
                label_transform='strip_terminal_sentence_punctuation_v2')))
        jsonl(out / 'preferences.jsonl', selected)
        a = dict(old['arguments']['edit_eos'], policy_adapter_path=plan['current_parent'],
            synthetic_train_file=str(out / 'preferences.jsonl'), synthetic_allocation_file=str(root / 'allocation.json'),
            output_dir=str(SCRATCH / arm / 'training'),
            expected_frozen_base_sha256=read(parent_root / 'training_receipts/policy_initial_runtime.json')['merged_base_tensor_sha256'])
        assert a['reference_adapter_path'] == p['initial_adapter']['path']
        assert (a['max_steps'], a['real_per_step'], a['synthetic_per_step'], a['lr'], a['beta'], a['kl_weight'], a['seed']) == (12, 16, 8, .0001, .1, 10., 42)
        arguments[arm] = a
        expected[arm] = exposure_plan(a, rows(a['real_train_file']), selected, scheduled, read(a['real_loss_factors'])['factors'])
        save(out / 'expected_exposures.json', expected[arm])
        files += [out / 'preferences.jsonl', out / 'expected_exposures.json']
    match_exposures(*[expected[a] for a in ARMS])
    save(root / 'data_receipt.json', dict(common_logical_slots=sorted(common), common_audio=len(common),
        full_paired_audio={a: len({x['audio_id'] for x in data[a]}) for a in ARMS},
        training_pair_rows={a: len(rows(root / a / 'preferences.jsonl')) for a in ARMS},
        independent_sampling_receipts={a: str(r / 'sampling_receipt.json') for a, r in roots.items()},
        fixed_audio_reuse='Old accepted waveforms and original coarse reviews verified; all ASR candidates newly drawn from current learned parent. No fictitious joint sampling receipt.',
        matching='Same current parent, decoders, logical-slot sampling seeds, actual role/audio exposure schedule, real order and masses. New content, patient, reference and recipe jointly form the adaptive policy.',
        all_correct='Remains in the data report; omitted from matched DPO without invented negatives.',
        label_semantics='Intended text after unchanged coarse review; not word-verified audio. Distinct actual errors averaged within audio.'))
    save(root / 'cpu_smoke.json', dict(status='passed', actual_pairs_rebuilt=True,
        common_current_parent_decode_and_sampling_seeds=True, exact_matched_exposure_schedule_and_mass=True,
        warm_GPU_smoke=str(smoke), new_GPU_smoke=False))
    files += [root / 'allocation.json', root / 'data_receipt.json', root / 'cpu_smoke.json', root / 'evaluation_target_descriptor.json']
    save(root / 'protocol.json', dict(plan=plan, source_protocol=p, arguments=arguments,
        previous_policy_trainable_sha256=parent_result['policy_trainable_sha256_after'],
        source_sha256={**old['source_sha256'], **data_cfg['source_sha256'], **decision['bound_sha256'], **{str(f): sha256_file(f) for f in files}},
        initial_R_only_teacher_fixed=True, current_policy_continued=True, same_engine_GPU_smoke=str(smoke),
        parent_promoted=False, original_ledgers_unchanged=True,
        scope='Exploratory second round from a selected seed42 checkpoint; prior three-seed mean tied R-only. Repeated511 development feedback, no inference postprocessing.'))
    verify(root)
    print(dict(stage='prepared', common_audio=len(common), role_exposures=scheduled['role_exposures'],
        pair_rows=read(root / 'data_receipt.json')['training_pair_rows']), flush=True)


def gate(metrics, parent):
    return dict(pooled_improves=metrics['overall']['pooled_cer'] < parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer'] - parent['overall']['patient_macro_cer'] <= .001,
        hard_regression_within_limit=metrics['hard']['pooled_cer'] - parent['hard']['pooled_cer'] <= .005)


def train(base, root, arm):
    cfg = verify(root); out = root / arm; a = cfg['arguments'][arm]
    started = time.monotonic()
    execute(out, [sys.executable, str(TRAINER)] + [z for k, v in a.items() if v is not None for z in ['--' + k, str(v)]], 'training')
    training_seconds = time.monotonic() - started
    save(out / 'training_timing.json', dict(wall_seconds=training_seconds, includes_model_load_and_teacher_cache=True))
    training = Path(a['output_dir']); result = read(training / 'training_result.json')
    assert result['policy_trainable_sha256_before'] == cfg['previous_policy_trainable_sha256']
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert read(training / 'residual_verified.json')['starting_parameters_equal_previous_endpoint']
    assert rows(training / 'training_exposures.jsonl') == read(out / 'expected_exposures.json')
    teacher = read(training / 'real_teacher_cache.json')
    assert teacher['audio'] == read(base / 'ASR_target_format_dpo_probe_v1/edit_eos/training_receipts/real_teacher_cache.json')['audio']
    assert len(teacher['audio']) == 96 and not teacher['tensor_data_persisted']
    assert not (training / 'real_teacher_log_probs.pt').exists()
    assert read(training / 'run_protocol.json')['reference_asr_sha256'] == cfg['source_protocol']['initial_adapter']['sha256']
    assert not read(training / 'policy_initial_runtime.json')['zero_residual']
    assert read(training / 'reference_runtime.json')['zero_residual']
    masks = read(training / 'preference_token_masks.json'); scores = read(training / 'reference_log_probs.json')
    assert masks['mode'] == scores['preference_token_mask'] == 'edit_eos'
    assert scores['mask_manifest_sha256'] == sha256_file(training / 'preference_token_masks.json')
    counts = Counter()
    for pair in masks['pairs'].values():
        assert pair['token_edit_distance'] > 0
        for side in ['chosen', 'rejected']:
            ids = pair[side + '_token_ids']; kept = pair[side + '_scored_indices']
            assert kept == pair[side + '_edit_eos_indices'] and kept[-1] == len(ids) - 1
            counts['full'] += len(ids); counts['scored'] += len(kept)
    checkpoint = Path(result['checkpoint']); digest = sha256_directory(checkpoint)
    assert not (checkpoint / 'optimizer.pt').exists()
    assert not read(checkpoint / 'optimizer_omission.json')['optimizer_state_persisted']
    copy_receipts(training, out)
    metadata = out / 'checkpoint_metadata_receipts'; metadata.mkdir()
    for name in ['residual_composition.json', 'trainer_state.json', 'optimizer_omission.json', 'adapter_config.json']:
        shutil.copyfile(checkpoint / name, metadata / name)
        assert sha256_file(checkpoint / name) == sha256_file(metadata / name)
    save(out / 'verified.json', dict(status='passed', actual_exposures_match=True, all96_real_teachers_unchanged=True,
        starting_parameters_exact_previous_endpoint=True, initial_R_only_DPO_KL_reference_fixed=True,
        initial_weighted_losses=result['history'][0]['weighted_losses'], initial_losses_not_forced_to_zero_or_log2=True,
        actual_scored_token_counts=dict(counts), checkpoint=str(checkpoint), checkpoint_sha256=digest))
    started = time.monotonic()
    execute(out, evaluation_command(cfg['source_protocol'], out / 'evaluation_run', checkpoint), 'evaluation')
    evaluation_seconds = time.monotonic() - started
    prediction = out / 'evaluation_run/evaluation/predictions.jsonl'
    metrics = read(prediction.with_name('summary.json'))['metrics']
    initial = base / 'parent_eval_check'; current = base / 'ASR_target_format_dpo_probe_v1/edit_eos/evaluation_run/evaluation'
    initial_gate = gate(metrics, read(initial / 'summary.json')['metrics'])
    current_gate = gate(metrics, read(current / 'summary.json')['metrics'])
    save(out / 'paired_initial_R_only.json', compare(initial / 'predictions.jsonl', prediction))
    save(out / 'paired_current_parent.json', compare(current / 'predictions.jsonl', prediction))
    descriptor = ClusterDescriptor.model_validate(read(root / 'evaluation_target_descriptor.json'))
    response = compare_prediction_snapshots(rows(current / 'predictions.jsonl'), rows(prediction),
        before_prediction_column='prediction', after_prediction_column='prediction', critical_cer_threshold=.5,
        monitored_clusters=[descriptor], off_target_max_per_level_per_direction=3)
    response = json.loads(json.dumps(asdict(response), default=lambda x: x.model_dump(mode='json')))
    save(out / 'current_target_response.json', response)
    save(out / 'results.json', dict(status='complete', metrics=metrics, initial_R_only_gate=initial_gate,
        current_parent_gate=current_gate, eligible_against_both_parents=all(initial_gate.values()) and all(current_gate.values()),
        checkpoint=str(checkpoint), checkpoint_sha256=digest, actual_new_DPO_training=True,
        training_wall_seconds=training_seconds, evaluation_wall_seconds=evaluation_seconds,
        postprocessing_used=False, parent_promoted=False))
    assert sha256_directory(checkpoint) == digest
    verify(root)
    print(dict(arm=arm, CER_percent=100 * metrics['overall']['pooled_cer'], errors=metrics['overall']['total_edits'],
        initial_gate=initial_gate, current_gate=current_gate), flush=True)


def report(base, root):
    cfg = verify(root); results = {a: read(root / a / 'results.json') for a in ARMS}
    assert all(x['status'] == 'complete' for x in results.values())
    match_exposures(*[rows(root / a / 'training_receipts/training_exposures.jsonl') for a in ARMS])
    assert all(read(root / a / 'training_receipts/training_result.json')['policy_trainable_sha256_before'] == cfg['previous_policy_trainable_sha256'] for a in ARMS)
    paired = compare(root / ARMS[0] / 'evaluation_run/evaluation/predictions.jsonl', root / ARMS[1] / 'evaluation_run/evaluation/predictions.jsonl')
    save(root / 'paired_adaptive_policy.json', paired)
    proposal = read(root.parent / 'adaptive_data/MAS_proposal/proposal.json')
    key_map = read(root.parent / 'adaptive_data/feedback_key_map.json')
    cited = sorted({k for x in proposal['assignments'] for k in x['evidence_ids'] if k in key_map})
    snapshots = dict(initial_R_only=base / 'parent_eval_check/predictions.jsonl',
        current_parent=base / 'ASR_target_format_dpo_probe_v1/edit_eos/evaluation_run/evaluation/predictions.jsonl',
        **{a: root / a / 'evaluation_run/evaluation/predictions.jsonl' for a in ARMS})
    concrete = {}
    for name, path in snapshots.items():
        records = {x['utt_id']: x for x in rows(path)}
        concrete[name] = {k: {f: records[key_map[k]][f] for f in ['clean_gt', 'prediction', 'edit_count']} for k in cited}
    target_changes = {a: read(root / a / 'current_target_response.json')['monitored_cluster_changes'][0] for a in ARMS}
    save(root / 'declared_target_observations.json', dict(primary_descriptor=read(root / 'evaluation_target_descriptor.json'),
        target_cluster_changes=target_changes, actual_predictions_on_cited_feedback=concrete,
        expected_observation=proposal['expected_observation'], observation_against_hypothesis=proposal['observation_against_hypothesis'],
        semantics='Reports the primary cluster and actual feedback cited before training. Deletion disappearance alone does not establish correct character recovery; paired target-position transitions are retained.'))
    save(root / 'results.json', dict(status='complete', arms=results,
        MAS_minus_fixed_CER_pp=paired['groups']['overall']['delta_pp'],
        bootstrap95_pp=paired['groups']['overall']['patient_cluster_bootstrap_delta_pp_95'],
        exposure_identity_order_weights_match=True, exact_common_learned_parent=True,
        frozen_initial_R_only_reference=True, parent_promoted=False, original_ledgers_unchanged=True,
        evaluation_scope='Exploratory selected-parent second round on repeated511 real development recordings. No independent test or stable multi-seed gain claimed.'))
    baseline_paths = [('初始 R-only', base / 'parent_eval_check/summary.json'),
        ('本轮共同 parent', base / 'ASR_target_format_dpo_probe_v1/edit_eos/evaluation_run/evaluation/summary.json')]
    metrics = [(label, read(path)['metrics']) for label, path in baseline_paths] + [(a, results[a]['metrics']) for a in ARMS]
    lines = ['两侧均继续同一个已学习的 residual；DPO 与真实 KL teacher 固定在初始 R-only。', '',
        '| 条件 | CER % | 字符错误 | macro CER % | hard 错误 |', '|---|---:|---:|---:|---:|']
    for label, m in metrics:
        lines.append(f'| {label} | {100*m["overall"]["pooled_cer"]:.6f} | {m["overall"]["total_edits"]} | {100*m["overall"]["patient_macro_cer"]:.6f} | {m["hard"]["total_edits"]} |')
    lines += ['', f'MAS 相对固定旧练习 CER 变化：{paired["groups"]["overall"]["delta_pp"]:+.6f} 个百分点。',
        f'本轮主目标「織」删除：共同 parent 8 次；固定练习 {target_changes[ARMS[0]]["after"]["error_count"]} 次；MAS 新练习 {target_changes[ARMS[1]]["after"]["error_count"]} 次。',
        '各分支均保留相对当前 parent、相对初始 R-only 的患者分组比较，以及本轮实际选中目标的字符/音节位置响应。',
        '共同 parent 的前期三 seed 平均等于 R-only；本轮是探索性的跨轮实验。511 条为反复使用的开发反馈。',
        '粗审准入不等于逐字音频保真；没有推理后处理。权重位于 /tmp，预测与训练回执保留在记录目录。']
    (root / 'results.zh.md').write_text('\n'.join(lines) + '\n')
    print({a: dict(CER_percent=100*x['metrics']['overall']['pooled_cer'], errors=x['metrics']['overall']['total_edits']) for a, x in results.items()}, flush=True)


def supervise(base, root):
    processes = []
    for gpu, arm in enumerate(ARMS):
        command = [sys.executable, str(Path(__file__).resolve()), 'train', str(base), '--arm', arm]
        with (root / (arm + '.log')).open('x') as log:
            p = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        save(root / (arm + '.launch.json'), dict(pid=p.pid, gpu=gpu, command=command)); processes.append(p)
    codes = [p.wait() for p in processes]; assert codes == [0, 0], codes
    report(base, root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'report', 'supervise'])
    parser.add_argument('base', type=Path); parser.add_argument('--arm', choices=ARMS, default=ARMS[0])
    args = parser.parse_args(); base = args.base.resolve(); root = base / 'exploratory_round2_v1/DPO'
    if args.stage == 'prepare': prepare(base, root)
    elif args.stage == 'train': train(base, root, args.arm)
    elif args.stage == 'report': report(base, root)
    else: supervise(base, root)
