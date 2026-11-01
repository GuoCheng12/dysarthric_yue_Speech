#!/usr/bin/env python3
"""Matched learning-rate/terminal-punctuation diagnostic on existing actual pairs."""

import argparse
import json
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO, execute, jsonl, read, rows, save
from therapist_harness.store import sha256_file


def main(base):
    root = base / 'format_dose_probe'
    root.mkdir(exist_ok=False)
    source = base / 'selection_dpo_probe/original'
    p = read(base / 'selection_dpo_probe/protocol.json')['source_protocol']
    pairs = rows(source / 'preferences.jsonl')
    allocation = read(source / 'allocation.json')
    # This is a declared label-format intervention, not a transcript replacement.
    trimmed = []
    for row in pairs:
        clean = row['chosen'].rstrip('。！？!?')
        assert clean and clean != row['rejected']
        trimmed.append(dict(row, chosen=clean, metadata=dict(row['metadata'],
            original_target_text=row['chosen'], label_transform='strip_terminal_sentence_punctuation_v1')))
    save(root / 'protocol.json', dict(
        source_protocol=p, source_preferences=str(source / 'preferences.jsonl'),
        source_preferences_sha256=sha256_file(source / 'preferences.jsonl'),
        source_allocation_sha256=sha256_file(source / 'allocation.json'),
        source_sampling_receipt=read(source / 'selection_receipt.json'),
        source_sha256={str(x):sha256_file(x) for x in [Path(__file__),
            REPO/'finetune/scripts/qwen3_asr_audio_dpo.py', REPO/'src/therapist_harness/dpo_training.py']},
        label_intervention='Only terminal sentence punctuation is removed in named trimmed arms; all spoken characters, audio, negatives and coarse reviews are unchanged.',
        arms=['raw_lr1e-5_existing', 'raw_lr1e-6', 'trimmed_lr1e-5', 'trimmed_lr1e-6'],
        steps=4, synthetic_per_step=8, reference='frozen initial R-only',
        scope='exploratory matched format and step-size diagnosis; not MAS efficacy'))
    results = {'raw_lr1e-5_existing':dict(metrics=read(source/'evaluation/summary.json')['metrics'],
        training_source=str(source), reused=True)}
    for name, data, lr in [('raw_lr1e-6',pairs,'1e-6'),
                            ('trimmed_lr1e-5',trimmed,'1e-5'), ('trimmed_lr1e-6',trimmed,'1e-6')]:
        d = root / name
        jsonl(d/'preferences.jsonl', data)
        save(d/'allocation.json', allocation)
        execute(d,[sys.executable,str(REPO/'finetune/scripts/qwen3_asr_audio_dpo.py'),
            '--model_path',p['base_model_dir'],'--policy_adapter_path',p['initial_adapter']['path'],
            '--reference_adapter_path',p['initial_adapter']['path'],'--train_file',str(d/'preferences.jsonl'),
            '--output_dir',str(d/'training'),'--max_steps','4','--real_per_step','0','--synthetic_per_step','8',
            '--synthetic_weight','1','--lr',lr,'--seed','42',
            '--synthetic_allocation_file',str(d/'allocation.json')],'training')
        execute(d,[sys.executable,str(REPO/'finetune/scripts/evaluate_asr_checkpoint.py'),
            '--base-model-path',p['base_model_dir'],'--adapter-path',str(d/'training/checkpoint-4'),
            '--split-jsonl',p['feedback_split']['path'],'--out-dir',str(d/'evaluation'),
            '--batch-size','8','--critical-cer-threshold','.5','--max-new-tokens','256'],'evaluation')
        results[name] = dict(metrics=read(d/'evaluation/summary.json')['metrics'],
                            training_source=str(d), reused=False)
        print(json.dumps(dict(arm=name,overall=results[name]['metrics']['overall'])),flush=True)
    results['parent'] = read(Path(p['initial_adapter_dev_predictions']['path']).parent/'summary.json')['metrics']
    save(root/'results.json', results)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path)
    main(parser.parse_args().base.resolve())
