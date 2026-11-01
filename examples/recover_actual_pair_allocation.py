#!/usr/bin/env python3
"""Recover the observed quota failure without overwriting the original execution."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save
from mas_evaluation_report import compare
from mas_residual_dpo_probe import evaluation_command
from therapist_harness.store import sha256_file, sha256_directory


def prepare(base, root):
    old = base / 'actual_pair_allocation_probe_v1'
    cfg = read(old / 'protocol.json')
    assert all(sha256_file(Path(f)) == digest for f, digest in cfg['source_sha256'].items())
    assert read(old / 'pure/training/training_result.json')['status'] == 'complete'
    assert read(old / 'pure/training/residual_verified.json')['status'] == 'passed'
    assert not (old / 'weight_10/training/training_result.json').exists()
    root.mkdir(exist_ok=False)
    command = read(old / 'weight_10/training.command.json')
    assert isinstance(command, list)
    files = [Path(__file__), old / 'protocol.json', old / 'pure/training/training_result.json',
        old / 'pure/training/residual_verified.json', old / 'weight_10/training.command.json',
        old / 'weight_10/training/training_exposures.jsonl', old / 'weight_10/training/reference_log_probs.json',
        old / 'weight_10/training/real_teacher_cache.json', old / 'weight_10/training/checkpoint-12/adapter_model.safetensors',
        old / 'pure/evaluation.log', old / 'weight_10/training.log',
        REPO / 'examples/evaluate_residual_asr.py', REPO / 'finetune/scripts/evaluate_asr_checkpoint.py',
        REPO / 'finetune/scripts/qwen3_asr_residual_kl_dpo.py', REPO / 'finetune/scripts/qwen3_asr_kl_dpo_probe.py',
        REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py']
    save(root / 'migration.json', dict(source_sha256={str(f): sha256_file(f) for f in files},
        original_root=str(old), source_protocol=cfg['source_protocol'], original_kl_command=command,
        pure_checkpoint=str(old / 'pure/training/checkpoint-12'),
        pure_checkpoint_sha256=sha256_directory(old / 'pure/training/checkpoint-12'),
        failure='Filesystem quota exhausted: pure evaluation failed writing predictions; KL optimizer save failed after step12.',
        pure_action='Reuse the complete verified checkpoint and actually decode again into a new directory.',
        retention_action='Repeat the identical training command from the original parent with a fresh output directory; no partial optimizer resume or fabricated completion.',
        cache_storage='After actual teacher computation/save, hardlink to the identical existing immutable cache if its whole-file SHA matches. No tensor or training change.',
        old_execution_unchanged=True))


def verify(root):
    cfg = read(root / 'migration.json')
    assert all(sha256_file(Path(f)) == digest for f, digest in cfg['source_sha256'].items())
    return cfg


def train(base, root):
    sys.path.insert(0, str(REPO / 'finetune/scripts'))
    import qwen3_asr_kl_dpo_probe as engine
    import qwen3_asr_residual_kl_dpo as wrapper

    cfg = verify(root)
    old = Path(cfg['original_root'])
    output = root / 'weight_10/training'
    output.parent.mkdir(exist_ok=False)
    command = list(cfg['original_kl_command'])
    command[command.index('--output_dir') + 1] = str(output)
    save(root / 'weight_10/training.command.json', command)
    cache_source = old / 'weight_10/training/real_teacher_log_probs.pt'
    expected_cache_hash = read(old / 'weight_10/training/real_teacher_cache.json')['cache_file_sha256']
    assert sha256_file(cache_source) == expected_cache_hash
    original_write = engine.write_json

    def write_record(path, value):
        original_write(path, value)
        if Path(path).name == 'real_teacher_cache.json':
            identical = value['cache_file_sha256'] == expected_cache_hash
            if identical:
                destination = Path(path).parent / 'real_teacher_log_probs.pt'
                temporary = destination.with_name(destination.name + '.dedup_tmp')
                os.link(cache_source, temporary)
                os.replace(temporary, destination)
            original_write(output / 'cache_storage_receipt.json', dict(
                actual_teacher_computation=True, identical_bytes_hardlinked=identical,
                source=str(cache_source), source_sha256=expected_cache_hash,
                destination_sha256=value['cache_file_sha256']))

    engine.write_json = write_record
    sys.argv = command[1:]
    wrapper.main()
    result = read(output / 'training_result.json')
    assert result['status'] == 'complete' and result['policy_changed'] and result['reference_files_unchanged']
    assert read(output / 'residual_verified.json')['status'] == 'passed'
    assert rows(output / 'training_exposures.jsonl') == rows(old / 'weight_10/training/training_exposures.jsonl')
    assert read(output / 'reference_log_probs.json')['values'] == read(old / 'weight_10/training/reference_log_probs.json')['values']
    save(root / 'weight_10/replay_verified.json', dict(status='passed', actual_new_training=True,
        original_exposure_records_identical=True, original_reference_scores_identical=True,
        model_file_sha_matches_failed_execution=sha256_file(output / 'checkpoint-12/adapter_model.safetensors') ==
            sha256_file(old / 'weight_10/training/checkpoint-12/adapter_model.safetensors'),
        incomplete_optimizer_not_reused=True))
    verify(root)


def evaluate(base, root, arm):
    cfg = verify(root)
    p = cfg['source_protocol']
    directory = root / arm
    directory.mkdir(exist_ok=True)
    checkpoint = Path(cfg['pure_checkpoint']) if arm == 'pure' else directory / 'training/checkpoint-12'
    if arm == 'pure':
        assert sha256_directory(checkpoint) == cfg['pure_checkpoint_sha256']
    else:
        assert read(directory / 'replay_verified.json')['status'] == 'passed'
    execute(directory, evaluation_command(p, directory / 'evaluation_run', checkpoint), 'evaluation')
    metrics = read(directory / 'evaluation_run/evaluation/summary.json')['metrics']
    parent = read(base / 'parent_eval_check/summary.json')['metrics']
    gate = dict(pooled_improves=metrics['overall']['pooled_cer'] < parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer']-parent['overall']['patient_macro_cer'] <= .001,
        hard_regression_within_limit=metrics['hard']['pooled_cer']-parent['hard']['pooled_cer'] <= .005)
    save(directory / 'paired_parent.json', compare(Path(p['initial_adapter_dev_predictions']['path']),
                                                  directory / 'evaluation_run/evaluation/predictions.jsonl'))
    save(directory / 'results.json', dict(status='complete', metrics=metrics, gate_checks=gate,
        eligible_for_next_round_parent=all(gate.values()), actual_new_training_this_recovery=arm != 'pure',
        actual_new_decoding=True, postprocessing_used=False, checkpoint=str(checkpoint)))
    verify(root)
    print(dict(arm=arm, overall=metrics['overall'], gate=gate), flush=True)


def supervise(base, root):
    jobs = []
    for gpu, stage in enumerate(['pure', 'retention']):
        command = [sys.executable, str(Path(__file__).resolve()), stage, str(base)]
        with (root / (stage + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        save(root / (stage + '.launch.json'), dict(pid=child.pid, gpu=gpu, command=command))
        jobs.append(child)
    codes = [x.wait() for x in jobs]
    assert codes == [0, 0], codes
    save(root / 'results.json', dict(status='complete', arms={a: read(root / a / 'results.json') for a in ['pure', 'weight_10']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'pure', 'retention', 'supervise'])
    parser.add_argument('base', type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    root = base / 'actual_pair_allocation_recovery_v1'
    if args.stage == 'prepare':
        prepare(base, root)
    elif args.stage == 'supervise':
        supervise(base, root)
    elif args.stage == 'pure':
        evaluate(base, root, 'pure')
    else:
        train(base, root)
        evaluate(base, root, 'weight_10')
