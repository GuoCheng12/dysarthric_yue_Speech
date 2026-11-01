#!/usr/bin/env python3
"""Save actual optimizer-step checkpoints of one unchanged warm KL/DPO run."""

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

import qwen3_asr_edit_kl_dpo_continuation_v1 as warm
import qwen3_asr_kl_dpo_storage_v2 as engine


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--trajectory_steps', required=True)
    parser.add_argument('--expected_frozen_base_sha256', required=True)
    extra, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index('--output_dir') + 1]).resolve()
    maximum = int(remaining[remaining.index('--max_steps') + 1])
    steps = [int(x) for x in extra.trajectory_steps.split(',')]
    assert steps == sorted(set(steps)) and steps[0] > 0 and steps[-1] == maximum
    source = Path(__file__).resolve(); source_hash = engine.sha256(source)
    original_digest, original_write = engine.parameter_digest, engine.write_json
    state = dict(step=0); snapshots = []

    def digest(model, *, trainable_only=False):
        if trainable_only:
            state['policy'] = model
        return original_digest(model, trainable_only=trainable_only)

    def write_record(path, value):
        if Path(path).name == 'run_protocol.json':
            value = dict(value, source_sha256={**value['source_sha256'], str(source): source_hash},
                trajectory=dict(steps=steps, hook='after actual AdamW optimizer.step',
                    objective_unchanged=True, independent_restart=False,
                    rng='Python/NumPy/CPU/CUDA states restored after each snapshot',
                    optimizer='One uninterrupted fresh AdamW; snapshot files omit optimizer state.'))
        original_write(path, value)

    def snapshot(optimizer, args, kwargs):
        state['step'] += 1
        if state['step'] not in steps:
            return
        started = time.monotonic(); policy = state['policy']; step = state['step']
        assert isinstance(optimizer, torch.optim.AdamW) and policy.training
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        checkpoint = output / 'trajectory' / f'step_{step}' / f'checkpoint-{step}'
        before = original_digest(policy, trainable_only=True)
        optimizer_steps = {int(x['step']) for x in optimizer.state.values() if 'step' in x}
        assert optimizer_steps == {step}
        try:
            policy.save_pretrained(checkpoint, safe_serialization=True)
        finally:
            random.setstate(rng[0]); np.random.set_state(rng[1])
            torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
        assert original_digest(policy, trainable_only=True) == before and policy.training
        snapshots.append(dict(step=step, checkpoint=str(checkpoint), trainable_sha256=before,
            adapter_tensor_file_sha256=engine.sha256(checkpoint / 'adapter_model.safetensors'),
            actual_optimizer_steps=sorted(optimizer_steps), model_parameters_unchanged_by_save=True,
            rng_restored=True, snapshot_wall_seconds=time.monotonic() - started))

    engine.parameter_digest = digest; engine.write_json = write_record
    hook = register_optimizer_step_post_hook(snapshot)
    sys.argv = [sys.argv[0]] + remaining
    warm.residual.main = lambda: warm.continue_training(extra.expected_frozen_base_sha256)
    try:
        warm.masked.main()
    finally:
        hook.remove()
    result = json.loads((output / 'training_result.json').read_text())
    final_checkpoint = Path(result['checkpoint'])
    assert state['step'] == maximum and [x['step'] for x in snapshots] == steps
    assert snapshots[-1]['trainable_sha256'] == result['policy_trainable_sha256_after']
    assert snapshots[-1]['adapter_tensor_file_sha256'] == engine.sha256(final_checkpoint / 'adapter_model.safetensors')
    composition = json.loads((final_checkpoint / 'residual_composition.json').read_text())
    for item in snapshots:
        checkpoint = Path(item['checkpoint']); step = item['step']
        original_write(checkpoint / 'residual_composition.json', dict(composition,
            trajectory_source=str(output / 'training_result.json'), observed_optimizer_step=step))
        original_write(checkpoint / 'trainer_state.json', dict(global_step=step,
            log_history=result['history'][:step], source_training_result=str(output / 'training_result.json')))
        original_write(checkpoint / 'optimizer_omission.json', dict(optimizer_state_persisted=False,
            continuation='Fresh optimizer only; this snapshot records weights from the uninterrupted run.'))
        original_write(checkpoint.parent / 'training_result.json', dict(status='complete',
            checkpoint=str(checkpoint), history=result['history'][:step], policy_changed=True,
            policy_trainable_sha256_before=result['policy_trainable_sha256_before'],
            policy_trainable_sha256_after=item['trainable_sha256'], reference_files_unchanged=True,
            shared_run_result=str(output / 'training_result.json'),
            provenance='Observed weights after this optimizer step in the shared run, not a separate training execution.'))
        item['checkpoint_sha256'] = engine.sha256_directory(checkpoint)
    assert engine.sha256(source) == source_hash
    original_write(output / 'trajectory_verified.json', dict(status='passed', snapshots=snapshots,
        actual_training_runs=1, actual_optimizer_steps=maximum,
        last_snapshot_tensor_identical_to_final_checkpoint=True,
        source_training_result=str(output / 'training_result.json'), source_sha256={str(source): source_hash}))


if __name__ == '__main__':
    main()
