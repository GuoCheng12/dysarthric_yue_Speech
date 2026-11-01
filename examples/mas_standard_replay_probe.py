#!/usr/bin/env python3
"""Replay an existing DPO update with its exact arguments and no save hooks."""

import argparse
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save
from therapist_harness.store import sha256_file


def main(base):
    root = base / 'standard_replay_probe'
    root.mkdir(exist_ok=False)
    source = base / 'formal_mas96_v3/round_001/mas_dpo'
    previous_protocol = read(source / 'training/run_protocol.json')
    arguments = dict(previous_protocol['arguments'], output_dir=str(root / 'training'))
    files = [Path(__file__), REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py',
             REPO / 'src/therapist_harness/dpo_training.py',
             source / 'training/run_protocol.json', source / 'training/training_result.json',
             source / 'training/training_exposures.jsonl', source / 'evaluation/summary.json',
             Path(arguments['train_file']), Path(arguments['synthetic_allocation_file'])]
    hashes = {str(p): sha256_file(p) for p in files}
    save(root / 'protocol.json', dict(source_sha256=hashes, arguments=arguments,
         purpose='Check ordinary DPO reproducibility without intermediate-save hooks; no new sampling or model selection.'))
    command = [sys.executable, str(REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py')]
    for key, value in arguments.items():
        if value is not None:
            command += ['--' + key, str(value)]
    execute(root, command, 'training')
    actual = read(root / 'training/training_result.json')
    previous = read(source / 'training/training_result.json')
    assert actual['reference_files_unchanged'] and actual['policy_changed']
    assert rows(root / 'training/training_exposures.jsonl') == rows(source / 'training/training_exposures.jsonl')
    identical = actual['policy_trainable_sha256_after'] == previous['policy_trainable_sha256_after']
    if identical:
        metrics = read(source / 'evaluation/summary.json')['metrics']
        save(root / 'evaluation_reuse.json', dict(source=str(source / 'evaluation/summary.json'),
             sha256=sha256_file(source / 'evaluation/summary.json'),
             reason='Final adapter tensors exactly match the previously evaluated model; no fresh evaluation claimed.'))
    else:
        protocol = read(base / 'formal_mas96_v3/protocol.json')
        execute(root, [sys.executable, str(REPO / 'finetune/scripts/evaluate_asr_checkpoint.py'),
            '--base-model-path', arguments['model_path'], '--adapter-path', str(root / 'training/checkpoint-12'),
            '--split-jsonl', protocol['feedback_split']['path'], '--out-dir', str(root / 'evaluation'),
            '--batch-size', '8', '--critical-cer-threshold', '.5', '--max-new-tokens', '256'], 'evaluation')
        metrics = read(root / 'evaluation/summary.json')['metrics']
    assert all(sha256_file(Path(p)) == digest for p, digest in hashes.items())
    save(root / 'results.json', dict(status='complete', final_adapter_tensors_identical=identical,
         exposure_records_identical=True, metrics=metrics, fresh_evaluation=not identical))
    print(dict(identical=identical, pooled_cer=metrics['overall']['pooled_cer']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
