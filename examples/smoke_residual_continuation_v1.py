#!/usr/bin/env python3
"""Actual current-residual sampling and two-step warm-start training smoke."""

import argparse
from pathlib import Path
import sys
import time

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from therapist_harness.store import sha256_file, sha256_directory
from therapist_harness.text import text_normalize


def main(base):
    root = base / 'residual_continuation_smoke_v1'
    root.mkdir(exist_ok=False)
    source = base / 'adaptive96_matched_DPO_recovery_v2/MAS_reference'
    old = read(source / 'training_receipts/training_result.json')
    parent = Path(old['checkpoint']); parent_hash = sha256_directory(parent)
    cfg = read(base / 'real2117_residual_SFT_DPO_v1/protocol.json')
    p = cfg['source_protocol']; initial = Path(p['initial_adapter']['path'])
    assert parent_hash == read(source / 'verified.json')['checkpoint_sha256']
    ids = list(dict.fromkeys(x['audio_id'] for x in rows(source / 'preferences.jsonl')))[:4]
    inputs = rows(base / 'adaptive_mas96_data_v1/round_001/sampling/validated_inputs.jsonl')
    jsonl(root / 'inputs.jsonl', [x for x in inputs if x['audio_id'] in ids])
    jsonl(root / 'real.jsonl', rows(base / 'real2117_residual_SFT_DPO_v1/real.jsonl')[:2])
    sampler = REPO / 'examples/sample_residual_asr_preferences_v1.py'
    trainer = REPO / 'finetune/scripts/qwen3_asr_residual_sft_dpo_continuation_v1.py'
    files = [Path(__file__), sampler, trainer, root / 'inputs.jsonl', root / 'real.jsonl',
        source / 'training_receipts/training_result.json', source / 'verified.json',
        REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py',
        REPO / 'finetune/scripts/qwen3_asr_sft_dpo_probe.py',
        REPO / 'src/therapist_harness/preference_data.py',
        REPO / 'src/therapist_harness/dpo_training.py',
        REPO / 'src/therapist_harness/preference_token_mask.py']
    hashes = {str(f): sha256_file(f) for f in files}
    save(root / 'protocol.json', dict(source_sha256=hashes, initial_R_only=str(initial),
        initial_R_only_sha256=p['initial_adapter']['sha256'], current_residual=str(parent),
        current_residual_sha256=parent_hash, diagnostic_only=True,
        interpretation='Use an existing rejected endpoint only to test continuation mechanics. This does not promote it or advance the scientific round parent.',
        audio=4, stochastic_draws_per_audio=4, separate_greedy=True,
        training='Two actual steps from the sampled learned residual; frozen original R-only DPO reference. Real CE2/step, synthetic1/step, weight.05, edit+EOS.',
        wait_for_gpu0='Wait for the running full-real CE arm to finish training and evaluation.'))
    print(dict(stage='waiting_for_real_CE_arm', root=str(root)), flush=True)
    ready = base / 'real2117_residual_SFT_DPO_v1/R_CE/results.json'
    while not ready.exists(): time.sleep(10)
    assert read(ready)['status'] == 'complete'
    execute(root, [sys.executable, str(sampler), 'sample', '--inputs', str(root / 'inputs.jsonl'),
        '--real-train', p['real_train']['path'], '--feedback', p['feedback_split']['path'],
        '--out-dir', str(root / 'sampling'), '--base-model', p['base_model_dir'],
        '--current-adapter', str(parent), '--initial-r-only-adapter', str(initial),
        '--expected-parent-sha256', parent_hash, '--samples', '4', '--include-greedy',
        '--seed', '42', '--temperature', '.8', '--top-p', '.95', '--max-new-tokens', '256'], 'sampling')
    receipt = read(root / 'sampling/sampling_receipt.json')
    assert receipt['current_asr_sha256'] == parent_hash != receipt['initial_R_only_sha256']
    assert receipt['initial_R_only_sha256'] == p['initial_adapter']['sha256']
    assert receipt['learned_residual_loaded'] and not receipt['current_residual_merged']
    data = rows(root / 'sampling/preferences.jsonl')
    if not data:
        save(root / 'no_update.json', dict(reason='All actual sampled transcripts correct; no fabricated negatives.',
            sampling_smoke_completed=True, warm_training_smoke_completed=False))
        return
    normalized = []
    for x in data:
        target = x['chosen'].rstrip('。！？!?')
        assert text_normalize(target) == text_normalize(x['chosen'])
        normalized.append(dict(x, chosen=target, metadata=dict(x['metadata'],
            original_target_text=x['chosen'], label_transform='strip_terminal_sentence_punctuation_v2')))
    jsonl(root / 'preferences.jsonl', normalized)
    keys = list(dict.fromkeys(x['audio_id'] for x in normalized))
    save(root / 'allocation.json', dict(steps=[[keys[0]], [keys[-1]]]))
    args = dict(cfg['smoke_arguments']['R_CE_MAS_DPO'], policy_adapter_path=str(parent),
        real_train_file=str(root / 'real.jsonl'), synthetic_train_file=str(root / 'preferences.jsonl'),
        synthetic_allocation_file=str(root / 'allocation.json'),
        output_dir='/tmp/qwen3_asr_residual_continuation_smoke_v1/training')
    command = [sys.executable, str(trainer)]
    for k, v in args.items(): command += ['--' + k, str(v)]
    save(root / 'training_arguments.json', args)
    execute(root, command, 'training')
    directory = Path(args['output_dir']); result = read(directory / 'training_result.json')
    assert result['policy_trainable_sha256_before'] == old['policy_trainable_sha256_after']
    assert result['policy_changed'] and result['frozen_base_unchanged'] and result['reference_files_unchanged']
    assert read(directory / 'residual_verified.json')['learned_residual_continued']
    assert not read(directory / 'policy_initial_runtime.json')['zero_residual']
    assert read(directory / 'reference_runtime.json')['zero_residual']
    assert read(directory / 'run_protocol.json')['reference_asr_sha256'] == p['initial_adapter']['sha256']
    assert sha256_directory(parent) == parent_hash
    assert all(sha256_file(Path(f)) == h for f, h in hashes.items())
    from mas_ASR_target_format_probe import copy_receipts
    copy_receipts(directory, root)
    save(root / 'verified.json', dict(status='passed', actual_sampling_audio=4,
        actual_stochastic_draws=receipt['candidate_count'], actual_new_training_steps=2,
        starting_trainable_parameters_exactly_match_previous_endpoint=True,
        current_sampled_parent_matches_training_parent=True, initial_R_only_reference_fixed=True,
        initial_DPO_loss=result['history'][0]['weighted_losses']['synthetic'] / .05,
        initial_loss_need_not_be_log2=True, frozen_base_and_parent_files_unchanged=True,
        checkpoint=result['checkpoint'], checkpoint_sha256=sha256_directory(Path(result['checkpoint'])),
        diagnostic_only=True, scientific_parent_promoted=False))
    print(read(root / 'verified.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
