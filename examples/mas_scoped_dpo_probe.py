#!/usr/bin/env python3
"""Matched MAS-TTS DPO training: joint LoRA versus frozen audio encoder."""

import argparse
import math
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file

TRAINER = REPO / 'finetune/scripts/qwen3_asr_scoped_dpo.py'


def prepare(base, root):
    root.mkdir(exist_ok=False)
    source = base / 'lexical96_mas_v1/round_001'
    p = read(base / 'lexical96_mas_v1/protocol.json')
    arguments = read(source / 'mas_dpo/training/run_protocol.json')['arguments']
    receipt = read(source / 'sampling/sampling_receipt.json')
    assert receipt['current_asr_sha256'] == p['initial_adapter']['sha256']
    pairs, schedule = Path(arguments['train_file']), Path(arguments['synthetic_allocation_file'])
    assert arguments['max_steps'] == 12 and arguments['real_per_step'] == 0
    assert arguments['synthetic_per_step'] == 8 and arguments['synthetic_weight'] == 1
    files = [Path(__file__), TRAINER, REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py',
        REPO / 'src/therapist_harness/dpo_training.py', REPO / 'finetune/scripts/evaluate_asr_checkpoint.py',
        base / 'lexical96_mas_v1/protocol.json', source / 'mas_dpo/training/run_protocol.json',
        source / 'mas_dpo/training/training_exposures.jsonl', source / 'sampling/sampling_receipt.json',
        source / 'generation/generation_summary.json', source / 'generation_plan.json', pairs, schedule]
    save(root / 'protocol.json', dict(source_sha256={str(f): sha256_file(f) for f in files},
        source_protocol=p, source_arguments=arguments, original_exposures=str(source / 'mas_dpo/training/training_exposures.jsonl'),
        hypothesis='Synthetic-only DPO may disturb real-speech acoustic processing when audio encoder LoRA is updated. Test freezing it from the beginning, unlike post-hoc component recombination.',
        arms=['all', 'decoder'], actual_new_generation=False, actual_new_sampling=False,
        reused_sampling_receipt=str(source / 'sampling/sampling_receipt.json'),
        preserved='Same 96 MAS-generated audio, 236 actual preference pairs, target labels, coarse review, parent, frozen R-only reference, DPO loss, learning rate, 12 steps and exact 96-audio exposure sequence.',
        acceptance='Original gate: pooled CER strictly improves, patient macro regression<=0.1pp, hard regression<=0.5pp. Raw ASR only; no postprocessing benefit counts.'))
    keys = read(schedule)['steps'][0][:2]
    assert len(set(keys)) == 2
    jsonl(root / 'smoke_preferences.jsonl', [x for x in rows(pairs) if x['audio_id'] in keys])
    save(root / 'smoke_allocation.json', dict(steps=[keys, list(reversed(keys))]))


def verify_sources(root):
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == digest for f, digest in cfg['source_sha256'].items())
    return cfg


def run(base, root, scope, smoke):
    cfg = verify_sources(root)
    if not smoke:
        assert read(root / 'smoke/verified.json')['status'] == 'passed'
    directory = root / ('smoke' if smoke else scope)
    directory.mkdir(exist_ok=False)
    arguments = dict(cfg['source_arguments'], output_dir=str(directory / 'training'))
    if smoke:
        arguments.update(train_file=str(root / 'smoke_preferences.jsonl'),
            synthetic_allocation_file=str(root / 'smoke_allocation.json'), max_steps=2, synthetic_per_step=2)
    command = [sys.executable, str(TRAINER), '--train_component', scope]
    for key, value in arguments.items():
        if value is not None:
            command.extend(['--' + key, str(value)])
    execute(directory, command, 'training')
    result = read(directory / 'training/training_result.json')
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert math.isclose(result['history'][0]['loss'], math.log(2), abs_tol=1e-4)
    assert read(directory / 'training/scope_verified.json')['status'] == 'passed'
    exposures = rows(directory / 'training/training_exposures.jsonl')
    if not smoke:
        assert exposures == rows(cfg['original_exposures'])
    save(directory / 'verified.json', dict(status='passed', initial_dpo_log2=True,
        reference_unchanged=True, actual_parameter_scope_verified=True,
        exact_original_exposure_records_match=None if smoke else True))
    verify_sources(root)
    if smoke:
        print('Actual two-step decoder-only DPO smoke passed', flush=True)
        return
    p = cfg['source_protocol']
    execute(directory, [sys.executable, str(REPO / 'finetune/scripts/evaluate_asr_checkpoint.py'),
        '--base-model-path', p['base_model_dir'], '--adapter-path', str(directory / 'training/checkpoint-12'),
        '--split-jsonl', p['feedback_split']['path'], '--out-dir', str(directory / 'evaluation'),
        '--batch-size', '8', '--critical-cer-threshold', '.5', '--max-new-tokens', '256'], 'evaluation')
    metrics = read(directory / 'evaluation/summary.json')['metrics']
    parent = read(base / 'parent_eval_check/summary.json')['metrics']
    checks = dict(pooled_improves=metrics['overall']['pooled_cer'] < parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer'] - parent['overall']['patient_macro_cer'] <= .001,
        hard_regression_within_limit=metrics['hard']['pooled_cer'] - parent['hard']['pooled_cer'] <= .005)
    save(directory / 'paired_parent.json', compare(Path(p['initial_adapter_dev_predictions']['path']), directory / 'evaluation/predictions.jsonl'))
    verify_sources(root)
    save(directory / 'results.json', dict(status='complete', train_component=scope, metrics=metrics,
        gate_checks=checks, eligible_for_next_round_parent=all(checks.values()),
        actual_new_dpo_training=True, postprocessing_used=False))
    print(dict(metrics=metrics['overall'], gate=checks), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'run'])
    parser.add_argument('base', type=Path)
    parser.add_argument('--scope', choices=['all', 'decoder'], default='decoder')
    args = parser.parse_args()
    base = args.base.resolve()
    root = base / 'scoped_dpo_probe_v1'
    if args.stage == 'prepare':
        prepare(base, root)
    else:
        run(base, root, args.scope, args.stage == 'smoke')
