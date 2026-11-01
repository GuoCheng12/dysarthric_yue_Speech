#!/usr/bin/env python3
"""Test stronger DPO updating on the unchanged completed MAS candidate selection."""

import argparse
import math
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file


def run(base):
    source = base / 'candidate_coverage_recovery_v2'
    root = base / 'residual_lr_probe_v1'
    root.mkdir(exist_ok=False)
    p = read(source / 'protocol.json')['source_protocol']
    old = source / 'all16/training'
    a = read(old / 'run_protocol.json')['arguments']
    assert a['lr'] == 1e-5 and a['max_steps'] == 12 and a['real_per_step'] == 0
    assert read(source / 'smoke/verified.json')['status'] == 'passed'
    assert read(source / 'all16/results.json')['status'] == 'complete'
    a = dict(a, lr=1e-4, output_dir=str(root / 'training'))
    files = [Path(__file__), REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py',
        REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py', REPO / 'src/therapist_harness/dpo_training.py',
        REPO / 'examples/evaluate_residual_asr.py', REPO / 'examples/measure_residual_preference_response.py',
        source / 'protocol.json', source / 'migration.json', source / 'all16/selection_receipt.json',
        source / 'all16/results.json', source / 'smoke/verified.json',
        old / 'run_protocol.json', old / 'reference_log_probs.json', old / 'training_exposures.jsonl',
        Path(a['train_file']), Path(a['synthetic_allocation_file']),
        base / 'actual_pair_allocation_preference_response_v1.json']
    hashes = {str(f): sha256_file(f) for f in files}
    save(root / 'protocol.json', dict(source_sha256=hashes, source_protocol=p, arguments=a,
        control=str(source / 'all16'), changed_training_arguments={'lr': [1e-5, 1e-4]},
        unchanged='Same15 selected actual pairs,96 audio exposures/order,12 steps,seed42,beta0.1,zero residual on merged R-only,fixed initial R-only reference,standard synthetic-only DPO.',
        hypothesis='Observed preference movement in the preceding actual15-pair DPO was small (weighted margin+0.201, loss0.693->0.683). Test whether tenfold learning rate translates this unchanged candidate selection into stronger real correction.',
        interpretation='A stronger fit to training preferences without raw real-ASR improvement would argue against simply increasing update strength; next change should address MAS target/content transfer.',
        smoke='Reuse the already passed two-step smoke of the unchanged trainer and this data; current run still checks initial log2, finite gradients, policy change and unchanged reference/base. No new-lr smoke is claimed.',
        acceptance='Existing raw-ASR gate against initial R-only, pooled improves,macro regression<=0.1pp,hard<=0.5pp. No postprocessing.'))
    command = [sys.executable, str(REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py')]
    for key, value in a.items():
        if value is not None:
            command.extend(['--'+key, str(value)])
    execute(root, command, 'training')
    result = read(root / 'training/training_result.json')
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert math.isclose(result['history'][0]['loss'], math.log(2), abs_tol=1e-4)
    assert read(root / 'training/residual_verified.json')['status'] == 'passed'
    assert rows(root / 'training/training_exposures.jsonl') == rows(old / 'training_exposures.jsonl')
    assert read(root / 'training/reference_log_probs.json')['values'] == read(old / 'reference_log_probs.json')['values']
    save(root / 'verified.json', dict(status='passed', same_actual_exposure_records=True,
        same_frozen_reference_scores=True, frozen_merged_base_unchanged=True, initial_log2=True))
    execute(root, evaluation_command(p, root / 'evaluation_run', root / 'training/checkpoint-12'), 'evaluation')
    metrics = read(root / 'evaluation_run/evaluation/summary.json')['metrics']
    parent = read(base / 'parent_eval_check/summary.json')['metrics']
    gate = dict(pooled_improves=metrics['overall']['pooled_cer'] < parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer']-parent['overall']['patient_macro_cer'] <= .001,
        hard_regression_within_limit=metrics['hard']['pooled_cer']-parent['hard']['pooled_cer'] <= .005)
    save(root / 'paired_parent.json', compare(Path(p['initial_adapter_dev_predictions']['path']), root / 'evaluation_run/evaluation/predictions.jsonl'))
    save(root / 'results.json', dict(status='complete', metrics=metrics, gate_checks=gate,
        eligible_for_next_round_parent=all(gate.values()), actual_new_dpo_training=True, postprocessing_used=False))
    execute(root, [sys.executable, str(REPO / 'examples/measure_residual_preference_response.py'),
        str(root / 'training'), str(root / 'preference_response.json')], 'preference_response')
    assert all(sha256_file(Path(f)) == h for f, h in hashes.items())
    print(dict(status='complete', metrics=metrics['overall'], gate=gate), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    args = parser.parse_args()
    run(args.base.resolve())
