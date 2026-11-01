#!/usr/bin/env python3
"""Evaluate half-sized adapter-parameter updates from existing lexical DPO runs."""

import argparse
from pathlib import Path
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

from mas_selection_dpo_probe import REPO, execute, read, save
from therapist_harness.store import sha256_file


def main(base):
    root = base / 'lexical_adapter_damping_probe_v2'
    root.mkdir(exist_ok=False)
    p = read(base / 'lexical24_loop_v1/protocol.json')
    parent = Path(p['initial_adapter']['path'])
    arms = dict(step4=base / 'lexical24_loop_v1/round_001/mas_dpo/training/checkpoint-4',
                step12=base / 'lexical24_dose12_probe/training/checkpoint-12')
    files = [Path(__file__), parent / 'adapter_model.safetensors', parent / 'adapter_config.json',
             REPO / 'finetune/scripts/evaluate_asr_checkpoint.py']
    files += [path / name for path in arms.values() for name in ['adapter_model.safetensors', 'adapter_config.json']]
    hashes = {str(f): sha256_file(f) for f in files}
    save(root / 'protocol.json', dict(source_sha256=hashes, coefficient=0.5, source_protocol=p,
        intervention='Interpolate saved rank-16 LoRA parameter tensors halfway from R-only to each candidate. This is not exact dense-weight/model-output interpolation.',
        purpose='Test whether damping an existing lexical update retains correction with less overall drift. No new sampling, training or optimizer continuation.',
        resource_context='GPU0 also generates the matched lexical96 control; timing is shared-resource diagnostic time.'))
    initial = load_file(str(parent / 'adapter_model.safetensors'))
    parent_config = read(parent / 'adapter_config.json')
    results = {}
    for name, source in arms.items():
        directory = root / name
        adapter = directory / 'adapter'
        adapter.mkdir(parents=True)
        config = read(source / 'adapter_config.json')
        assert all(config[k] == parent_config[k] for k in ['r', 'lora_alpha', 'bias'])
        assert set(config['target_modules']) == set(parent_config['target_modules'])
        candidate = load_file(str(source / 'adapter_model.safetensors'))
        assert candidate.keys() == initial.keys()
        updated = {}
        for key, value in candidate.items():
            assert value.shape == initial[key].shape and torch.isfinite(value).all()
            updated[key] = (initial[key].float() + 0.5 * (value.float() - initial[key].float())).to(value.dtype)
            assert torch.isfinite(updated[key]).all()
        save_file(updated, str(adapter / 'adapter_model.safetensors'), metadata={'format': 'pt'})
        shutil.copy2(source / 'adapter_config.json', adapter / 'adapter_config.json')
        save(directory / 'derivation.json', dict(parent=str(parent), candidate=str(source), coefficient=0.5,
             adapter_tensor_sha256=sha256_file(adapter / 'adapter_model.safetensors'), new_training=False))
        execute(directory, [sys.executable, str(REPO / 'finetune/scripts/evaluate_asr_checkpoint.py'),
            '--base-model-path', p['base_model_dir'], '--adapter-path', str(adapter),
            '--split-jsonl', p['feedback_split']['path'], '--out-dir', str(directory / 'evaluation'),
            '--batch-size', '8', '--critical-cer-threshold', '.5', '--max-new-tokens', '256'], 'evaluation')
        results[name] = read(directory / 'evaluation/summary.json')['metrics']
        print(name, results[name]['overall']['pooled_cer'], flush=True)
    assert all(sha256_file(Path(f)) == digest for f, digest in hashes.items())
    save(root / 'results.json', dict(status='complete', metrics=results, automatic_promotion=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
