#!/usr/bin/env python3
"""Measure realized DPO preference changes on all actually trained audio."""

import argparse
from collections import Counter
import math
from pathlib import Path
import sys

import torch

from mas_selection_dpo_probe import REPO, read, rows, save
from therapist_harness.dpo_training import AudioPreferenceCollator, batch_log_probs, move_batch, preference_key
from therapist_harness.store import sha256_file

sys.path.insert(0, str(REPO / 'finetune/scripts'))
from qwen3_asr_residual_dpo import load_merged_parent, add_residual, frozen_base_digest
from qwen3_asr_sft import patch_outer_forward


def measure(training, output):
    protocol = read(training / 'run_protocol.json')
    a = protocol['arguments']
    cache = read(training / 'reference_log_probs.json')
    assert cache['protocol_sha256'] == sha256_file(training / 'run_protocol.json')
    assert cache['reference_unchanged'] and cache['trainable_parameters'] == 0
    assert sha256_file(Path(a['train_file'])) == protocol['training_file_sha256']
    checkpoint = training / f"checkpoint-{a['max_steps']}"
    assert read(checkpoint / 'residual_composition.json')['parent_adapter'] == a['reference_adapter_path']
    data = rows(a['train_file'])
    exposures = Counter(x['audio_id'] for x in rows(training / 'training_exposures.jsonl'))
    assert all(sum(x['pair_weight'] for x in data if x['audio_id'] == k) == 1 for k in exposures)
    files = [Path(__file__), Path(a['train_file']), training / 'run_protocol.json',
        training / 'reference_log_probs.json', training / 'training_result.json',
        checkpoint / 'adapter_model.safetensors', checkpoint / 'residual_composition.json',
        REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py', REPO / 'src/therapist_harness/dpo_training.py']
    hashes = {str(f): sha256_file(f) for f in files}
    dtype, device = torch.bfloat16, torch.device('cuda:0')
    wrapper = load_merged_parent(a['model_path'], a['policy_adapter_path'], device, dtype)
    patch_outer_forward(wrapper.model)
    model = add_residual(wrapper.model, a['policy_adapter_path'], checkpoint)
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    assert frozen_base_digest(model) == read(training / 'reference_runtime.json')['merged_base_tensor_sha256']
    collator = AudioPreferenceCollator(wrapper.processor, a['sr'])
    measured = []
    with torch.inference_mode():
        for row in data:
            key = preference_key(row)
            logs, _ = batch_log_probs(model, move_batch(collator([row]), device, dtype))
            before = cache['values'][key]
            after = logs.float().cpu().tolist()
            gain = after[0] - after[1] - (before[0] - before[1])
            measured.append(dict(audio_id=row['audio_id'], key=key, chosen=row['chosen'], rejected=row['rejected'],
                weight=exposures[row['audio_id']] * row['pair_weight'],
                before_chosen_logp=before[0], before_rejected_logp=before[1],
                after_chosen_logp=after[0], after_rejected_logp=after[1],
                before_margin=before[0]-before[1], after_margin=after[0]-after[1], margin_gain=gain,
                dpo_loss=math.log1p(math.exp(-a['beta'] * gain))))
    total = sum(x['weight'] for x in measured)
    summary = dict(pairs=len(measured), audio=len(exposures),
        pairs_with_margin_gain=sum(x['margin_gain'] > 0 for x in measured),
        pairs_favoring_chosen_before=sum(x['before_margin'] > 0 for x in measured),
        pairs_favoring_chosen_after=sum(x['after_margin'] > 0 for x in measured),
        weighted_margin_gain=sum(x['weight'] * x['margin_gain'] for x in measured) / total,
        initial_dpo_loss=math.log(2),
        realized_dpo_loss=sum(x['weight'] * x['dpo_loss'] for x in measured) / total)
    assert all(sha256_file(Path(f)) == h for f, h in hashes.items())
    save(output, dict(source_sha256=hashes, summary=summary, pairs=measured,
        scope='Actual frozen-checkpoint teacher-forced pair scores, not new training or greedy decoding. Uses saved unchanged initial R-only reference scores. Does not establish word fidelity or modify admission.'))
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('training', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    measure(args.training.resolve(), args.output.resolve())
