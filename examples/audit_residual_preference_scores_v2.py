#!/usr/bin/env python3
"""Measure absolute chosen/rejected scores alongside DPO margins on actual pairs."""

import argparse
from collections import defaultdict
import gc
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO, read, rows, save
from therapist_harness.store import sha256_file, sha256_directory

sys.path.insert(0, str(REPO / 'finetune/scripts'))
sys.path.insert(0, str(REPO / 'finetune/official'))


def main(base):
    import torch
    from transformers import GenerationConfig, set_seed
    from qwen3_asr_sft import patch_outer_forward
    from qwen3_asr_residual_dpo import load_merged_parent, add_residual, frozen_base_digest
    from therapist_harness.dpo_training import AudioPreferenceCollator, move_batch, transcript_log_probs, preference_key
    from therapist_harness.preference_token_mask import edit_eos_labels

    source = base / 'adaptive96_matched_DPO_recovery_v2/MAS_reference'
    root = base / 'adaptive96_preference_score_audit_v2'; root.mkdir(exist_ok=False)
    cfg = read(source.parent / 'protocol.json'); p = cfg['source_protocol']
    assert read(source / 'verified.json')['status'] == 'passed'
    data = rows(source / 'preferences.jsonl')
    checkpoint = Path(read(source / 'results.json')['checkpoint'])
    checkpoint_hash = sha256_directory(checkpoint)
    assert checkpoint_hash == read(source / 'verified.json')['checkpoint_sha256']
    cached = read(source / 'training_receipts/reference_log_probs.json')['values']
    files = [Path(__file__), REPO / 'finetune/official/qwen3_asr_sft.py', source / 'preferences.jsonl', source / 'results.json', source / 'verified.json',
        source / 'training_receipts/reference_log_probs.json', source / 'training_receipts/training_exposures.jsonl',
        base / 'adaptive_mas96_data_v1/round_001/sampling/preference_audit.json',
        REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py',
        REPO / 'src/therapist_harness/dpo_training.py', REPO / 'src/therapist_harness/preference_token_mask.py']
    hashes = {str(f): sha256_file(f) for f in files}
    save(root / 'protocol.json', dict(source_sha256=hashes, checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_hash, scope='Existing107 preference pairs; actual teacher-forced scoring only. No new training, decoding, admission or acoustic-fidelity judgment.'))
    scores = {}; base_hash = None
    for label, adapter in [('before', None), ('after', checkpoint)]:
        set_seed(42)
        wrapper = load_merged_parent(p['base_model_dir'], p['initial_adapter']['path'], 'cuda:0', torch.bfloat16)
        patch_outer_forward(wrapper.model)
        wrapper.model.generation_config = GenerationConfig.from_model_config(wrapper.model.config)
        wrapper.model.config.use_cache = False
        model = add_residual(wrapper.model, p['initial_adapter']['path'], adapter)
        model.requires_grad_(False); model.eval()
        digest = frozen_base_digest(model)
        if base_hash is None: base_hash = digest
        else: assert digest == base_hash
        collator = AudioPreferenceCollator(wrapper.processor); values = {}
        with torch.inference_mode():
            for i, row in enumerate(data):
                batch = move_batch(collator([row]), 'cuda:0', torch.bfloat16)
                result = model(**{k: v for k, v in batch.items() if k != 'labels'})
                logits = result['logits'] if isinstance(result, dict) else result.logits
                masked, _ = edit_eos_labels(batch['labels'], 1, wrapper.processor.tokenizer.eos_token_id)
                full, counts = transcript_log_probs(logits, batch['labels'])
                edit, edit_counts = transcript_log_probs(logits, masked)
                values[preference_key(row)] = dict(full_logp=full.float().cpu().tolist(),
                    full_counts=counts.cpu().tolist(), edit_eos_logp=edit.float().cpu().tolist(),
                    edit_eos_counts=edit_counts.cpu().tolist())
                del batch, result, logits, masked, full, counts, edit, edit_counts
                if (i + 1) % 25 == 0: print(dict(stage=label, scored=i + 1, total=len(data)), flush=True)
        assert frozen_base_digest(model) == base_hash
        save(root / (label + '.json'), values); scores[label] = values
        del model, wrapper, collator; gc.collect(); torch.cuda.empty_cache()
    audit = {x['audio_id']: x for x in read(base / 'adaptive_mas96_data_v1/round_001/sampling/preference_audit.json')['audio_results']}
    exposure = rows(source / 'training_receipts/training_exposures.jsonl')
    weights = defaultdict(float)
    for x in exposure:
        if x['source'] == 'synthetic': weights[x['preference_key']] += x['loss_weight']
    records = []; max_reference_difference = 0.0
    for row in data:
        key = preference_key(row); before = scores['before'][key]; after = scores['after'][key]
        assert before['full_counts'] == after['full_counts'] and before['edit_eos_counts'] == after['edit_eos_counts']
        max_reference_difference = max(max_reference_difference, *(abs(x - y) for x, y in zip(before['edit_eos_logp'], cached[key])))
        delta = {mode: [a - b for a, b in zip(after[mode], before[mode])] for mode in ['full_logp', 'edit_eos_logp']}
        records.append(dict(audio_id=row['audio_id'], preference_key=key, correct_samples=audit[row['audio_id']]['correct_samples'],
            actual_aggregate_loss_weight=weights[key], delta=delta,
            edit_margin_delta=delta['edit_eos_logp'][0] - delta['edit_eos_logp'][1]))
    groups = {}
    for group in ['all', 'some_correct_draw', 'zero_correct_draw']:
        selected = [x for x in records if group == 'all' or (x['correct_samples'] > 0) == (group == 'some_correct_draw')]
        total = sum(x['actual_aggregate_loss_weight'] for x in selected)
        groups[group] = dict(pairs=len(selected), audio=len({x['audio_id'] for x in selected}),
            edit_margin_improved_pairs=sum(x['edit_margin_delta'] > 0 for x in selected),
            edit_margin_improved_but_full_chosen_logp_fell=sum(x['edit_margin_delta'] > 0 and x['delta']['full_logp'][0] < 0 for x in selected),
            weighted_full_chosen_logp_delta=sum(x['actual_aggregate_loss_weight'] * x['delta']['full_logp'][0] for x in selected) / total,
            weighted_edit_chosen_logp_delta=sum(x['actual_aggregate_loss_weight'] * x['delta']['edit_eos_logp'][0] for x in selected) / total,
            weighted_edit_rejected_logp_delta=sum(x['actual_aggregate_loss_weight'] * x['delta']['edit_eos_logp'][1] for x in selected) / total)
    assert sha256_directory(checkpoint) == checkpoint_hash
    assert all(sha256_file(Path(f)) == h for f, h in hashes.items())
    save(root / 'results.json', dict(status='complete', groups=groups, pair_changes=records,
        max_difference_from_original_frozen_reference_edit_scores=max_reference_difference,
        frozen_base_unchanged=True, no_new_training=True, no_new_ASR_decoding=True,
        interpretation='Teacher-forced score changes on the synthetic training set; not real-speech transfer, generated-label correctness, or an independent performance gain.'))
    print(dict(groups=groups, max_reference_difference=max_reference_difference), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
