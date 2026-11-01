#!/usr/bin/env python3
"""Four actual current-policy audio samples followed by two warm KL/DPO steps."""

import argparse
from pathlib import Path
import sys
import time

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_edit_mask_dpo_probe import exposure_plan
from mas_ASR_target_format_probe import copy_receipts
from therapist_harness.store import sha256_file, sha256_directory
from therapist_harness.text import text_normalize


def main(base):
    root = base / 'edit_KL_continuation_smoke_v1'; root.mkdir(exist_ok=False)
    source = base / 'ASR_target_format_dpo_probe_v1/edit_eos'
    previous = read(source / 'training_receipts/training_result.json')
    parent = Path(previous['checkpoint']); parent_hash = sha256_directory(parent)
    assert parent_hash == read(source / 'verified.json')['checkpoint_sha256']
    cfg = read(base / 'patient_content_DPO_v1/protocol.json'); p = cfg['source_protocol']
    assert all(sha256_file(Path(f)) == h for f,h in cfg['source_sha256'].items())
    initial = p['initial_adapter']['path']; a = dict(cfg['arguments']['patient_error_content'])
    inputs = rows(base / 'patient_content_probe_v1/sampling/validated_inputs.jsonl')[:4]
    real = rows(a['real_train_file'])[:4]
    jsonl(root / 'inputs.jsonl', inputs); jsonl(root / 'real.jsonl', real)
    save(root / 'factors.json', dict(factors={x['audio_id']:1. for x in real}))
    sampler = REPO / 'examples/sample_residual_asr_preferences_v1.py'
    trainer = REPO / 'finetune/scripts/qwen3_asr_edit_kl_dpo_continuation_v1.py'
    files = [Path(__file__), sampler, trainer, root / 'inputs.jsonl', root / 'real.jsonl', root / 'factors.json',
        source / 'training_receipts/training_result.json', source / 'verified.json',
        base / 'patient_content_DPO_v1/protocol.json']
    hashes = {**cfg['source_sha256'], **{str(f):sha256_file(f) for f in files}}
    save(root / 'protocol.json', dict(source_sha256=hashes, current_residual=str(parent), current_residual_sha256=parent_hash,
        frozen_initial_R_only=initial, initial_R_only_sha256=p['initial_adapter']['sha256'],
        purpose='Technical continuation smoke only. Uses a previously eligible seed42 endpoint, not a claim of stable multi-seed gain or scientific promotion.',
        actual_sampling_audio=4, actual_draws_per_audio=4, training_steps=2,
        objective='Same edit+EOS synthetic DPO and realGT-prefix symmetricKL10, with the policy residual warm-started and initialR-only teacher fixed.',
        wait_for='GPU0 previous_content full511 evaluation completes before this smoke starts.'))
    ready = base / 'patient_content_DPO_v1/previous_content/results.json'
    while not ready.exists(): time.sleep(10)
    assert read(ready)['status'] == 'complete'
    execute(root, [sys.executable, str(sampler), 'sample', '--inputs', str(root / 'inputs.jsonl'),
        '--real-train', p['real_train']['path'], '--feedback', p['feedback_split']['path'],
        '--out-dir', str(root / 'sampling'), '--base-model', p['base_model_dir'],
        '--current-adapter', str(parent), '--initial-r-only-adapter', initial,
        '--expected-parent-sha256', parent_hash, '--samples', '4', '--include-greedy',
        '--seed', '42', '--temperature', '.8', '--top-p', '.95', '--max-new-tokens', '128'], 'sampling')
    receipt = read(root / 'sampling/sampling_receipt.json')
    assert receipt['current_asr_sha256'] == parent_hash != receipt['initial_R_only_sha256']
    assert receipt['initial_R_only_sha256'] == p['initial_adapter']['sha256']
    assert receipt['learned_residual_loaded'] and not receipt['current_residual_merged']
    pairs = rows(root / 'sampling/preferences.jsonl')
    if not pairs:
        save(root / 'no_update.json', dict(reason='All actual draws correct; no invented negatives.')); return
    normalized = []
    for x in pairs:
        chosen = x['chosen'].rstrip('。！？!?'); assert text_normalize(chosen) == text_normalize(x['chosen'])
        normalized.append(dict(x, chosen=chosen, metadata=dict(x['metadata'], original_target_text=x['chosen'],
            label_transform='strip_terminal_sentence_punctuation_v2')))
    jsonl(root / 'preferences.jsonl', normalized)
    keys = list(dict.fromkeys(x['audio_id'] for x in normalized))
    allocation = dict(steps=[[keys[0]], [keys[-1]]]); save(root / 'allocation.json', allocation)
    a.update(policy_adapter_path=str(parent), real_train_file=str(root / 'real.jsonl'),
        synthetic_train_file=str(root / 'preferences.jsonl'), synthetic_allocation_file=str(root / 'allocation.json'),
        real_loss_factors=str(root / 'factors.json'), max_steps=2, real_per_step=4, synthetic_per_step=1,
        output_dir='/tmp/qwen3_asr_edit_KL_continuation_smoke_v1/training',
        expected_frozen_base_sha256=read(source / 'training_receipts/policy_initial_runtime.json')['merged_base_tensor_sha256'])
    expected = exposure_plan(a, real, normalized, allocation, read(root / 'factors.json')['factors'])
    save(root / 'expected_exposures.json', expected); save(root / 'training_arguments.json', a)
    execute(root, [sys.executable, str(trainer)] + [z for k,v in a.items() for z in ['--' + k, str(v)]], 'training')
    directory = Path(a['output_dir']); result = read(directory / 'training_result.json')
    assert result['policy_trainable_sha256_before'] == previous['policy_trainable_sha256_after']
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert read(directory / 'residual_verified.json')['starting_parameters_equal_previous_endpoint']
    assert rows(directory / 'training_exposures.jsonl') == expected
    teacher = read(directory / 'real_teacher_cache.json')['audio']
    prior_teacher = read(source / 'training_receipts/real_teacher_cache.json')['audio']
    assert all(v == prior_teacher[k] for k,v in teacher.items())
    assert read(directory / 'run_protocol.json')['reference_asr_sha256'] == p['initial_adapter']['sha256']
    assert result['history'][0]['weighted_losses']['real'] > 0
    assert not read(directory / 'policy_initial_runtime.json')['zero_residual']
    assert read(directory / 'reference_runtime.json')['zero_residual']
    assert sha256_directory(parent) == parent_hash
    assert all(sha256_file(Path(f)) == h for f,h in hashes.items())
    copy_receipts(directory, root)
    save(root / 'verified.json', dict(status='passed', actual_draws=receipt['candidate_count'],
        paired_audio=len(keys), pair_rows=len(pairs), actual_training_steps=2,
        initial_losses=result['history'][0]['weighted_losses'], initial_losses_not_forced_to_zero_or_log2=True,
        starting_parameters_exact_previous_endpoint=True, sampled_current_parent_equals_training_parent=True,
        initial_R_only_DPO_KL_reference_fixed=True, all_four_real_teacher_distributions_unchanged=True,
        exposure_plan_matches=True, frozen_base_and_parent_unchanged=True,
        checkpoint=result['checkpoint'], checkpoint_sha256=sha256_directory(Path(result['checkpoint'])),
        diagnostic_only=True, scientific_parent_promoted=False))
    print(read(root / 'verified.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
