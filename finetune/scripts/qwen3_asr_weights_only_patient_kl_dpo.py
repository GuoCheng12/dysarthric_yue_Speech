#!/usr/bin/env python3
"""Keep model checkpoints and optimizer hashes, without duplicate resume state."""

import hashlib
from pathlib import Path
import sys

import torch

import qwen3_asr_patient_weighted_kl_dpo as weighted


def optimizer_manifest(state):
    tensors = {}
    for parameter, values in state['state'].items():
        tensors[str(parameter)] = {}
        for name, value in values.items():
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                tensors[str(parameter)][name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                    sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
            else:
                tensors[str(parameter)][name] = value
    return dict(optimizer_state_persisted=False, exact_optimizer_resume_available=False,
        reason='This execution saves learned adapter weights and full training records; subsequent rounds use fresh AdamW. Actual final optimizer tensor hashes are retained.',
        parameter_groups=state['param_groups'], parameter_states=tensors)


def main():
    output = Path(sys.argv[sys.argv.index('--output_dir') + 1]).resolve()
    steps = int(sys.argv[sys.argv.index('--max_steps') + 1])
    optimizer_path = output / f'checkpoint-{steps}/optimizer.pt'
    engine = weighted.shared.residual.trainer
    original_save, original_write = torch.save, engine.write_json
    source = Path(__file__).resolve()
    source_hash = engine.sha256(source)
    skipped = []

    def save_tensor_data(value, path, *args, **kwargs):
        if Path(path).resolve() != optimizer_path:
            return original_save(value, path, *args, **kwargs)
        assert not optimizer_path.exists()
        original_write(optimizer_path.with_name('optimizer_omission.json'), optimizer_manifest(value))
        skipped.append(str(optimizer_path))

    def write_record(path, value):
        if Path(path).name == 'run_protocol.json':
            value = dict(value, checkpoint_storage=dict(adapter_weights='persisted',
                optimizer_state='not persisted; final tensor hashes recorded',
                exact_optimizer_resume_available=False, training_computation_unchanged=True),
                source_sha256={**value['source_sha256'], str(source): source_hash})
        if Path(path).name == 'teacher_storage_reuse.json':
            value = dict(value, unchanged_optimizer_checkpoint_serialization=False,
                optimizer_checkpoint_policy='Weights-only execution: optimizer tensor hashes are recorded in checkpoint/optimizer_omission.json; no optimizer.pt is stored.')
        original_write(path, value)

    torch.save, engine.write_json = save_tensor_data, write_record
    try:
        weighted.main()
    finally:
        torch.save, engine.write_json = original_save, original_write
    assert skipped == [str(optimizer_path)] and not optimizer_path.exists()
    assert engine.sha256(source) == source_hash


if __name__ == '__main__':
    main()
