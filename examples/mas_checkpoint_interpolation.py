#!/usr/bin/env python3
"""Derive and evaluate a smaller LoRA parameter update without further training."""

import argparse
from pathlib import Path
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

from mas_selection_dpo_probe import REPO, execute, read, save
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file


def main(args):
    root = args.output_dir.resolve()
    root.mkdir(exist_ok=False)
    assert 0 < args.coefficient < 1
    files = [Path(__file__), REPO / 'finetune/scripts/evaluate_asr_checkpoint.py',
             args.feedback, args.parent_predictions]
    files += [d / n for d in [args.parent, args.source] for n in ['adapter_model.safetensors', 'adapter_config.json']]
    hashes = {str(p.resolve()): sha256_file(p) for p in files}
    save(root / 'protocol.json', dict(source_sha256=hashes, parent=str(args.parent),
        candidate=str(args.source), coefficient=args.coefficient,
        intervention='Rank-preserving interpolation of LoRA A/B parameter tensors; not exact interpolation of dense updates or output distributions.',
        actual_new_training=False, original_optimizer_reused=False,
        resource_context='Independent matched checkpoint probe; may share GPU with generation or allocation training.'))
    previous = load_file(str(args.parent / 'adapter_model.safetensors'))
    candidate = load_file(str(args.source / 'adapter_model.safetensors'))
    a = read(args.parent / 'adapter_config.json')
    z = read(args.source / 'adapter_config.json')
    assert all(a[k] == z[k] for k in ['r', 'lora_alpha', 'bias'])
    assert set(a['target_modules']) == set(z['target_modules']) and candidate.keys() == previous.keys()
    updated = {}
    for key, value in candidate.items():
        assert value.shape == previous[key].shape
        updated[key] = (previous[key].float() + args.coefficient * (value.float() - previous[key].float())).to(value.dtype)
        assert torch.isfinite(updated[key]).all()
    adapter = root / 'adapter'
    adapter.mkdir()
    save_file(updated, str(adapter / 'adapter_model.safetensors'), metadata={'format': 'pt'})
    shutil.copy2(args.source / 'adapter_config.json', adapter / 'adapter_config.json')
    save(root / 'derivation.json', dict(adapter_tensor_sha256=sha256_file(adapter / 'adapter_model.safetensors'),
        source_sha256=hashes, coefficient=args.coefficient, training_executed=False))
    execute(root, [sys.executable, str(REPO / 'finetune/scripts/evaluate_asr_checkpoint.py'),
        '--base-model-path', str(args.base_model), '--adapter-path', str(adapter),
        '--split-jsonl', str(args.feedback), '--out-dir', str(root / 'evaluation'),
        '--batch-size', '8', '--critical-cer-threshold', '.5', '--max-new-tokens', '256'], 'evaluation')
    assert all(sha256_file(Path(p)) == digest for p, digest in hashes.items())
    save(root / 'paired_parent.json', compare(args.parent_predictions, root / 'evaluation/predictions.jsonl'))
    save(root / 'results.json', dict(status='complete', metrics=read(root / 'evaluation/summary.json')['metrics'],
        automatic_promotion=False, source_protocol_sha256=sha256_file(root / 'protocol.json')))
    print(read(root / 'results.json')['metrics']['overall'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent', 'source', 'base-model', 'feedback', 'parent-predictions', 'output-dir']:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--coefficient', required=True, type=float)
    main(parser.parse_args())
