#!/usr/bin/env python3
"""Separate the realized audio-encoder and language-decoder LoRA updates."""

import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

from mas_selection_dpo_probe import REPO, execute, read, save
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file


def component(name):
    if '.thinker.audio_tower.' in name:
        return 'audio'
    if '.thinker.model.' in name:
        return 'decoder'
    raise ValueError(f'Unclassified adapter parameter: {name}')


def main(args):
    base = args.base.resolve()
    root = base / 'lexical96_component_probe_v1' / args.component
    root.mkdir(parents=True, exist_ok=False)
    p = read(base / 'lexical96_mas_v1/protocol.json')
    parent = Path(p['initial_adapter']['path'])
    source = base / 'lexical96_mas_v1/round_001/mas_dpo/training/checkpoint-12'
    files = [Path(__file__), base / 'lexical96_mas_v1/protocol.json',
        REPO / 'finetune/scripts/evaluate_asr_checkpoint.py',
        Path(p['feedback_split']['path']), Path(p['initial_adapter_dev_predictions']['path'])]
    files += [d / n for d in [parent, source] for n in ['adapter_model.safetensors', 'adapter_config.json']]
    hashes = {str(f): sha256_file(f) for f in files}
    save(root / 'protocol.json', dict(source_sha256=hashes, parent=str(parent), source=str(source),
        retained_component=args.component, actual_training=False,
        intervention='Copy complete A/B tensors for one component from the completed DPO candidate and all other tensors from its R-only parent. No fractional interpolation.',
        interpretation='Locates useful/harmful realized parameter changes. Does not establish the result of training with a component frozen, because the original components were jointly optimized.',
        automatic_promotion=False, evaluation_scope='Repeated 511-row development feedback.'))
    previous = load_file(str(parent / 'adapter_model.safetensors'))
    candidate = load_file(str(source / 'adapter_model.safetensors'))
    assert previous.keys() == candidate.keys()
    a, z = read(parent / 'adapter_config.json'), read(source / 'adapter_config.json')
    assert all(a[k] == z[k] for k in ['r', 'lora_alpha', 'bias'])
    assert set(a['target_modules']) == set(z['target_modules'])
    counts, changed, squared_delta = Counter(), Counter(), Counter()
    updated = {}
    for name, value in candidate.items():
        group = component(name)
        counts[group] += 1
        changed[group] += int(not torch.equal(value, previous[name]))
        squared_delta[group] += (value.float() - previous[name].float()).square().sum().item()
        updated[name] = value if group == args.component else previous[name]
        assert torch.isfinite(updated[name]).all()
    assert set(counts) == {'audio', 'decoder'} and all(changed.values())
    adapter = root / 'adapter'
    adapter.mkdir()
    save_file(updated, str(adapter / 'adapter_model.safetensors'), metadata={'format': 'pt'})
    shutil.copy2(parent / 'adapter_config.json', adapter / 'adapter_config.json')
    saved = load_file(str(adapter / 'adapter_model.safetensors'))
    assert all(torch.equal(saved[name], candidate[name] if component(name) == args.component else previous[name])
               for name in saved)
    save(root / 'verified.json', dict(status='passed', tensor_counts=dict(counts),
        changed_tensors_in_source=dict(changed), source_delta_squared_norm=dict(squared_delta),
        exact_component_tensor_provenance=True, adapter_sha256=sha256_file(adapter / 'adapter_model.safetensors')))
    execute(root, [sys.executable, str(REPO / 'finetune/scripts/evaluate_asr_checkpoint.py'),
        '--base-model-path', p['base_model_dir'], '--adapter-path', str(adapter),
        '--split-jsonl', p['feedback_split']['path'], '--out-dir', str(root / 'evaluation'),
        '--batch-size', '8', '--critical-cer-threshold', '.5', '--max-new-tokens', '256'], 'evaluation')
    assert all(sha256_file(Path(f)) == digest for f, digest in hashes.items())
    save(root / 'paired_parent.json', compare(Path(p['initial_adapter_dev_predictions']['path']),
                                              root / 'evaluation/predictions.jsonl'))
    save(root / 'results.json', dict(status='complete', metrics=read(root / 'evaluation/summary.json')['metrics'],
        automatic_promotion=False, actual_training=False))
    print(read(root / 'results.json')['metrics']['overall'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('--component', choices=['audio', 'decoder'], required=True)
    main(parser.parse_args())
