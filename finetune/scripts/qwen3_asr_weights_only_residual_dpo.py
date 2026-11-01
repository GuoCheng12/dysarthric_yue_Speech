#!/usr/bin/env python3
"""Unchanged residual audio DPO, retaining weights and final optimizer hashes."""

from pathlib import Path
import sys

import torch
import qwen3_asr_residual_dpo as residual
from qwen3_asr_weights_only_patient_kl_dpo import optimizer_manifest


def main():
    output = Path(sys.argv[sys.argv.index('--output_dir') + 1]).resolve()
    steps = int(sys.argv[sys.argv.index('--max_steps') + 1])
    optimizer_path = output / f'checkpoint-{steps}/optimizer.pt'
    engine = residual.trainer
    original_save, original_write = torch.save, engine.write_json
    files = [Path(__file__).resolve(), Path(__file__).with_name('qwen3_asr_weights_only_patient_kl_dpo.py').resolve()]
    hashes = {str(f): engine.sha256(f) for f in files}
    skipped = []

    def save_tensor(value, path, *args, **kwargs):
        if Path(path).resolve() != optimizer_path:
            return original_save(value, path, *args, **kwargs)
        assert not optimizer_path.exists()
        original_write(optimizer_path.with_name('optimizer_omission.json'), optimizer_manifest(value))
        skipped.append(str(optimizer_path))

    def write_record(path, value):
        if Path(path).name == 'run_protocol.json':
            value = dict(value, checkpoint_storage=dict(adapter_weights='saved in declared output directory',
                optimizer_state='final tensor hashes only; no exact optimizer resume', training_computation_unchanged=True),
                source_sha256={**value.get('source_sha256', {}), **hashes})
        original_write(path, value)

    torch.save, engine.write_json = save_tensor, write_record
    try:
        residual.main()
    finally:
        torch.save, engine.write_json = original_save, original_write
    assert skipped == [str(optimizer_path)] and not optimizer_path.exists()
    assert all(engine.sha256(Path(f)) == h for f, h in hashes.items())


if __name__ == '__main__':
    main()
