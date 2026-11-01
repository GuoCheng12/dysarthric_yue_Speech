#!/usr/bin/env python3
"""Two actual warm updates with current-parent KL and initial-R-only DPO."""

import argparse
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO, read, rows, save, jsonl, execute
from mas_exploratory_round2_dpo_v1 import verify, exposure_plan
from mas_ASR_target_format_probe import copy_receipts
from therapist_harness.store import sha256_file, sha256_directory


def main(base):
    source = base / 'round2_reference_selection_DPO_v1'; cfg = verify(source)
    trial_plan = base / 'current_parent_KL_teacher_plan_v1.json'; plan = read(trial_plan)
    assert all(sha256_file(Path(f)) == h for f, h in plan['source_sha256'].items())
    root = base / 'current_parent_KL_teacher_smoke_v1'; root.mkdir(exist_ok=False)
    a = dict(cfg['arguments']['MAS_reference']); real = rows(a['real_train_file'])[:4]
    pairs = rows(a['synthetic_train_file']); ids = list(dict.fromkeys(x['audio_id'] for x in pairs)); keys = [ids[0], ids[-1]]
    selected = [x for x in pairs if x['audio_id'] in keys]
    jsonl(root / 'real.jsonl', real); jsonl(root / 'preferences.jsonl', selected)
    save(root / 'factors.json', dict(factors={x['audio_id']: 1. for x in real}))
    scheduled = dict(steps=[[keys[0]], [keys[1]]]); save(root / 'allocation.json', scheduled)
    a.update(real_train_file=str(root / 'real.jsonl'), synthetic_train_file=str(root / 'preferences.jsonl'),
        real_loss_factors=str(root / 'factors.json'), synthetic_allocation_file=str(root / 'allocation.json'),
        max_steps=2, real_per_step=4, synthetic_per_step=1, output_dir='/tmp/qwen3_asr_current_parent_KL_teacher_smoke_v1/training')
    expected = exposure_plan(a, real, selected, scheduled, read(root / 'factors.json')['factors'])
    save(root / 'expected_exposures.json', expected); save(root / 'training_arguments.json', a)
    trainer = REPO / 'finetune/scripts/qwen3_asr_current_parent_kl_teacher_v1.py'
    files = [Path(__file__), trainer, trial_plan, source / 'protocol.json'] + list(root.glob('*.json')) + list(root.glob('*.jsonl'))
    hashes = {**cfg['source_sha256'], **{str(f): sha256_file(f) for f in files}}
    save(root / 'protocol.json', dict(source_sha256=hashes, current_parent=a['policy_adapter_path'],
        current_parent_sha256=cfg['plan']['current_parent_sha256'], initial_R_only=a['reference_adapter_path'],
        initial_R_only_sha256=cfg['source_protocol']['initial_adapter']['sha256'],
        objective='Technical smoke: change only frozen real-KL teacher to current round parent. InitialR-only DPO reference and actual current-parent negative sources remain intact.',
        real_audio=4, synthetic_audio=2, pair_rows=len(selected), actual_steps=2,
        new_ASR_sampling=False, original_sampling_provenance=str(source / 'source_composition.json')))
    execute(root, [sys.executable, str(trainer)] + [z for k, v in a.items() if v is not None for z in ['--' + k, str(v)]], 'training')
    training = Path(a['output_dir']); result = read(training / 'training_result.json')
    assert result['policy_trainable_sha256_before'] == cfg['previous_policy_trainable_sha256']
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert rows(training / 'training_exposures.jsonl') == expected
    assert read(training / 'residual_verified.json')['starting_parameters_equal_previous_endpoint']
    assert read(training / 'residual_verified.json')['fixed_current_parent_KL_reference']
    assert read(training / 'current_KL_teacher_verified.json')['status'] == 'passed'
    runtime = read(training / 'current_KL_teacher_runtime.json')
    assert runtime['trainable_parameters'] == 0 and not runtime['current_residual_merged']
    assert runtime['checkpoint_sha256'] == cfg['plan']['current_parent_sha256']
    assert runtime['merged_base_tensor_sha256'] == a['expected_frozen_base_sha256']
    teacher = read(training / 'real_teacher_cache.json')
    assert teacher['frozen_reference_sha256'] == cfg['plan']['current_parent_sha256']
    assert teacher['teacher_parameters_unchanged'] and teacher['teacher_frozen_before_first_policy_update']
    old_teacher = read(source / 'MAS_reference/training_receipts/real_teacher_cache.json')['audio']
    changed_teacher_audio = sum(x['tensor_sha256'] != old_teacher[k]['tensor_sha256'] for k, x in teacher['audio'].items())
    assert changed_teacher_audio > 0 and len(teacher['audio']) == 4
    assert not teacher['tensor_data_persisted'] and not (training / 'real_teacher_log_probs.pt').exists()
    reference = read(training / 'reference_log_probs.json')
    old_reference = read(source / 'MAS_reference/training_receipts/reference_log_probs.json')
    assert all(v == old_reference['values'][k] for k, v in reference['values'].items())
    protocol = read(training / 'run_protocol.json')
    assert protocol['reference_asr_sha256'] == cfg['source_protocol']['initial_adapter']['sha256']
    assert protocol['real_KL_teacher_sha256'] == cfg['plan']['current_parent_sha256']
    assert abs(result['history'][0]['weighted_losses']['real']) < 1e-7
    assert result['history'][1]['weighted_losses']['real'] > 0
    assert reference['mask_manifest_sha256'] == sha256_file(training / 'preference_token_masks.json')
    assert read(training / 'preference_token_masks.json')['mode'] == 'edit_eos'
    assert sha256_directory(Path(a['policy_adapter_path'])) == cfg['plan']['current_parent_sha256']
    assert all(sha256_file(Path(f)) == h for f, h in hashes.items())
    copy_receipts(training, root)
    save(root / 'verified.json', dict(status='passed', actual_steps=2, actual_audio=dict(real=4, synthetic=2),
        initial_weighted_losses=result['history'][0]['weighted_losses'], initial_current_parent_KL_zero=True,
        all_synthetic_initial_R_only_DPO_reference_scores_identical=True,
        changed_KL_teacher_audio=changed_teacher_audio, frozen_current_KL_teacher_verified=True,
        residual_warm_start_and_updates_verified=True, actual_exposures_match=True,
        current_parent_and_initial_R_only_unchanged=True, diagnostic_only=True,
        checkpoint=result['checkpoint'], checkpoint_sha256=sha256_directory(Path(result['checkpoint']))))
    print(read(root / 'verified.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
