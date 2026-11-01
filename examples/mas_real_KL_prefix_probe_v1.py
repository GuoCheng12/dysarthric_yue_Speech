#!/usr/bin/env python3
"""Match GT-prefix and frozen-reference-greedy-prefix real KL in MAS DPO."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_edit_mask_dpo_probe import exposure_plan, check_training
from mas_ASR_target_format_probe import copy_receipts
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from report_ASR_target_format import summarize
from therapist_harness.store import sha256_file, sha256_directory

ARMS = ['ground_truth', 'reference_greedy']
TRAINER = REPO / 'finetune/scripts/qwen3_asr_real_prefix_kl_dpo_v1.py'
SCRATCH = Path('/tmp/qwen3_asr_real_KL_prefix_probe_v1')


def prepare(base, root):
    previous = base / 'adaptive96_matched_DPO_recovery_v2'
    old = read(previous / 'protocol.json'); inputs = base / 'real_KL_greedy_prefix_inputs_v1'
    assert all(sha256_file(Path(f)) == h for f, h in old['source_sha256'].items())
    audit = read(inputs / 'tokenization_audit.json')
    assert audit['nonempty_complete_EOS'] == 96 and audit['noncanonical_greedy_tokenizations'] == 0
    transcriptions = read(inputs / 'transcriptions.json')
    assert transcriptions['adapter_sha256'] == old['source_protocol']['initial_adapter']['sha256']
    assert transcriptions['status'] == 'complete' and transcriptions['adapter_files_unchanged']
    original = old['arguments']['MAS_reference']
    real = rows(original['real_train_file']); greedy = rows(inputs / 'greedy_text_real.jsonl')
    assert len(real) == len(greedy) == 96
    for a, b in zip(real, greedy):
        assert {k:v for k,v in a.items() if k not in ['chosen', 'metadata']} == {k:v for k,v in b.items() if k not in ['chosen', 'metadata']}
        assert b['metadata']['original_ground_truth'] == a['chosen']
    root.mkdir(exist_ok=False); SCRATCH.mkdir(exist_ok=False)
    arguments = {}; expected = {}
    for arm in ARMS:
        (root / arm).mkdir()
        a = dict(original, output_dir=str(SCRATCH / arm / 'training'), real_prefix_kind=arm)
        if arm == 'reference_greedy': a['real_train_file'] = str(inputs / 'greedy_text_real.jsonl')
        arguments[arm] = a
        expected[arm] = exposure_plan(a, rows(a['real_train_file']), rows(a['synthetic_train_file']),
            read(a['synthetic_allocation_file']), read(a['real_loss_factors'])['factors'])
        save(root / arm / 'expected_exposures.json', expected[arm])
    identity = lambda xs: [{k:v for k,v in x.items() if k != 'preference_key'} for x in xs]
    assert identity(expected[ARMS[0]]) == identity(expected[ARMS[1]])
    assert [x for x in expected[ARMS[0]] if x['source'] == 'synthetic'] == [x for x in expected[ARMS[1]] if x['source'] == 'synthetic']
    out = root / 'smoke'; out.mkdir()
    synthetic = rows(original['synthetic_train_file']); key = synthetic[0]['audio_id']
    jsonl(out / 'real.jsonl', greedy[:4]); jsonl(out / 'preferences.jsonl', [x for x in synthetic if x['audio_id'] == key])
    save(out / 'allocation.json', dict(steps=[[key], [key]]))
    save(out / 'factors.json', dict(factors={x['audio_id']:1. for x in greedy[:4]}))
    s = dict(arguments['reference_greedy'], real_train_file=str(out / 'real.jsonl'),
        synthetic_train_file=str(out / 'preferences.jsonl'), real_loss_factors=str(out / 'factors.json'),
        synthetic_allocation_file=str(out / 'allocation.json'), max_steps=2, real_per_step=4,
        synthetic_per_step=1, output_dir=str(SCRATCH / 'smoke/training'))
    save(out / 'expected_exposures.json', exposure_plan(s, greedy[:4], rows(out / 'preferences.jsonl'),
        read(out / 'allocation.json'), read(out / 'factors.json')['factors']))
    files = [Path(__file__), TRAINER, previous / 'protocol.json', inputs / 'plan.json',
        inputs / 'audio_only.jsonl', inputs / 'transcriptions.json', inputs / 'tokenization_audit.json',
        inputs / 'greedy_text_real.jsonl']
    files += list(root.glob('*/expected_exposures.json')) + list(out.glob('*.json')) + list(out.glob('*.jsonl'))
    save(root / 'protocol.json', dict(source_protocol=old['source_protocol'], arguments=arguments, smoke_arguments=s,
        source_sha256={**old['source_sha256'], **{str(f):sha256_file(f) for f in files}},
        previous=str(previous), actual_new_DPO_training=True, actual_new_greedy_decoding_audio=96,
        actual_new_generation=False, actual_new_preference_sampling=False, actual_new_MAS_decision=False,
        intervention='Only the real KL teacher-forcing history changes: originalGT versus actual initialR-only greedy transcript. Same96real audio/order/weights,44MAS audio107 actual pairs and96 synthetic occurrences,12steps lr1e-4 beta.1 KL10 seed42.',
        caveat='Greedy-prefix retention also preserves existing errors; benefit is a hypothesis. Synthetic review/pairs and the frozen initialR-only reference are unchanged. No real CE or new real preference data.',
        parent_promoted=False))
    print(dict(stage='prepared', different_GT_transcripts=audit['different_normalized_GT']), flush=True)


def verify(root):
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == h for f, h in cfg['source_sha256'].items())
    return cfg


def run(base, root, arm, smoke=False):
    cfg = verify(root); out = root / ('smoke' if smoke else arm)
    if not smoke: assert read(root / 'smoke/verified.json')['status'] == 'passed'
    a = cfg['smoke_arguments'] if smoke else cfg['arguments'][arm]
    command = [sys.executable, str(TRAINER)] + [z for k,v in a.items() if v is not None for z in ['--' + k, str(v)]]
    execute(out, command, 'training')
    directory = Path(a['output_dir'])
    result, teacher, counts = check_training(directory.parent, read(out / 'expected_exposures.json'), 'edit_eos')
    assert read(directory / 'run_protocol.json')['real_prefix_kind'] == a['real_prefix_kind']
    assert ('Actual frozen initial R-only greedy' in teacher['conditional_history']) == (a['real_prefix_kind'] == 'reference_greedy')
    if not smoke and arm == 'ground_truth':
        assert teacher['audio'] == read(Path(cfg['previous']) / 'MAS_reference/training_receipts/real_teacher_cache.json')['audio']
    copy_receipts(directory, out)
    checkpoint = Path(result['checkpoint']); digest = sha256_directory(checkpoint)
    save(out / 'verified.json', dict(status='passed', real_prefix_kind=a['real_prefix_kind'],
        actual_exposures_match=True, frozen_base_and_reference_unchanged=True,
        checkpoint=str(checkpoint), checkpoint_sha256=digest, actual_scored_token_counts=counts))
    if smoke: return
    execute(out, evaluation_command(cfg['source_protocol'], out / 'evaluation_run', checkpoint), 'evaluation')
    metrics = read(out / 'evaluation_run/evaluation/summary.json')['metrics']
    parent = read(base / 'parent_eval_check/summary.json')['metrics']
    gates = dict(pooled_improves=metrics['overall']['pooled_cer'] < parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer'] - parent['overall']['patient_macro_cer'] <= .001,
        hard_regression_within_limit=metrics['hard']['pooled_cer'] - parent['hard']['pooled_cer'] <= .005)
    save(out / 'paired_parent.json', compare(base / 'parent_eval_check/predictions.jsonl', out / 'evaluation_run/evaluation/predictions.jsonl'))
    save(out / 'results.json', dict(status='complete', metrics=metrics, gate_checks=gates,
        eligible_for_next_round_parent=all(gates.values()), checkpoint=str(checkpoint), checkpoint_sha256=digest,
        actual_new_DPO_training=True, postprocessing_used=False))
    assert sha256_directory(checkpoint) == digest
    verify(root)


def report(base, root):
    cfg = verify(root)
    summaries = {a:summarize(base, root / a) for a in ARMS}
    for filename, field in [('training_result.json', 'policy_trainable_sha256_before'), ('reference_log_probs.json', 'values')]:
        assert read(root / ARMS[0] / 'training_receipts' / filename)[field] == read(root / ARMS[1] / 'training_receipts' / filename)[field]
    actual = [rows(root / a / 'training_receipts/training_exposures.jsonl') for a in ARMS]
    assert [{k:v for k,v in x.items() if k != 'preference_key'} for x in actual[0]] == [{k:v for k,v in x.items() if k != 'preference_key'} for x in actual[1]]
    teacher = [read(root / a / 'training_receipts/real_teacher_cache.json')['audio'] for a in ARMS]
    assert teacher[0].keys() == teacher[1].keys()
    paired = compare(root / 'ground_truth/evaluation_run/evaluation/predictions.jsonl', root / 'reference_greedy/evaluation_run/evaluation/predictions.jsonl')
    save(root / 'paired_prefix_policy.json', paired)
    save(root / 'results.json', dict(status='complete', arms=summaries, parent_promoted=False,
        greedy_minus_GT_CER_pp=paired['groups']['overall']['delta_pp'],
        bootstrap95_pp=paired['groups']['overall']['patient_cluster_bootstrap_delta_pp_95'],
        changed_real_teacher_distributions=sum(teacher[0][k] != teacher[1][k] for k in teacher[0]),
        actual_exposure_identity_order_weights_equal=True, synthetic_frozen_reference_scores_equal=True,
        initial_residual_equal=True, postprocessing_used=False))
    print(summaries, flush=True)


def supervise(base, root):
    run(base, root, 'reference_greedy', smoke=True)
    jobs = []
    for gpu, arm in enumerate(ARMS):
        command = [sys.executable, str(Path(__file__).resolve()), 'train', str(base), '--arm', arm]
        with (root / (arm + '.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        save(root / (arm + '.launch.json'), dict(pid=process.pid, gpu=gpu, command=command)); jobs.append(process)
    codes = [p.wait() for p in jobs]
    assert codes == [0, 0], codes
    report(base, root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['prepare', 'smoke', 'train', 'report', 'supervise'])
    parser.add_argument('base', type=Path); parser.add_argument('--arm', choices=ARMS, default=ARMS[0])
    args = parser.parse_args(); base = args.base.resolve(); root = base / 'real_KL_prefix_probe_v1'
    if args.stage == 'prepare': prepare(base, root)
    elif args.stage in ['smoke', 'train']: run(base, root, args.arm, args.stage == 'smoke')
    elif args.stage == 'report': report(base, root)
    else: supervise(base, root)
