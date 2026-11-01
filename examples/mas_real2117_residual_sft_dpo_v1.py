#!/usr/bin/env python3
"""Matched full-real CE versus full-real CE plus existing MAS audio DPO."""

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_ASR_target_format_probe import copy_receipts
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from report_ASR_target_format import summarize
from therapist_harness.dpo_training import SourceBalancedWindows, group_training_rows, preference_key
from therapist_harness.store import sha256_file, sha256_directory
from therapist_harness.text import text_normalize
from therapist_harness.training_allocation import compile_role_allocation

ARMS = ['R_CE', 'R_CE_MAS_DPO']
NAME = 'real2117_residual_SFT_DPO_v1'
SCRATCH = Path('/tmp/qwen3_asr_real2117_residual_SFT_DPO_v1')
TRAINER = REPO / 'finetune/scripts/qwen3_asr_residual_sft_dpo_v1.py'
BASE_SHA = '8bb636596c705bf5f0d9524656664ded7e3fd3e0cda7fa30dbc03b5022356d34'


def expected(a):
    real = rows(a['real_train_file'])
    synthetic = rows(a['synthetic_train_file']) if a.get('synthetic_train_file') else []
    groups = dict(real=group_training_rows(real, 'sft')['real'],
                  synthetic=group_training_rows(synthetic, 'dpo')['synthetic'] if synthetic else {})
    schedule = read(a['synthetic_allocation_file'])['steps'] if synthetic else None
    windows = SourceBalancedWindows(groups, a['real_per_step'], a['synthetic_per_step'],
        a['synthetic_weight'], a['seed'], synthetic_schedule=schedule)
    records = []
    for step in range(1, a['max_steps'] + 1):
        for row in windows.next():
            records.append(dict(step=step, source=row['source'], audio_id=row['_audio_group'],
                occurrence=row['_occurrence'], loss_weight=row['_loss_weight'],
                objective='dpo' if row['source'] == 'synthetic' else 'sft', preference_key=preference_key(row)))
    return records


def prepare(base, root):
    old_path = base / 'adaptive96_matched_DPO_recovery_v2/protocol.json'
    old = read(old_path); p = old['source_protocol']
    assert all(sha256_file(Path(f)) == h for f, h in old['source_sha256'].items())
    source = base / 'adaptive96_matched_DPO_recovery_v2/MAS_reference/preferences.jsonl'
    synthetic = rows(source)
    assert len(synthetic) == 107 and len({x['audio_id'] for x in synthetic}) == 44
    assert all(x['current_asr_sha256'] == p['initial_adapter']['sha256'] for x in synthetic)
    assert all(sha256_file(Path(x['audio'])) == x['audio_sha256'] for x in synthetic)
    train = rows(p['real_train']['path']); dev = rows(p['feedback_split']['path'])
    assert len(train) == 2117 and len(dev) == 511
    prefix = 'language Cantonese<asr_text>'
    real = []
    for x in train:
        assert x['text'].startswith(prefix) and x['prompt'] == ''
        chosen = x['text'][len(prefix):]
        assert chosen == x['clean_gt']
        real.append(dict(source='real', audio_id=x['utt_id'], audio=str(Path(x['audio']).resolve()),
            audio_sha256=sha256_file(Path(x['audio'])), chosen=chosen, prompt='', language='Cantonese'))
    assert len({x['audio_id'] for x in real}) == 2117
    assert not {x['audio_id'] for x in real}.intersection(x['utt_id'] for x in dev)
    assert not {x['audio'] for x in real}.intersection(str(Path(x['audio']).resolve()) for x in dev)
    assert not {x['audio_sha256'] for x in real}.intersection(sha256_file(Path(x['audio'])) for x in dev)
    assert not {text_normalize(x['chosen']) for x in real}.intersection(text_normalize(x['text'].removeprefix(prefix)) for x in dev)
    root.mkdir(exist_ok=False); SCRATCH.mkdir(exist_ok=False)
    jsonl(root / 'real.jsonl', real)
    jsonl(root / 'preferences.jsonl', synthetic)
    sidecar = base / 'adaptive_mas96_data_v1/round_001/materialization_sidecar.json'
    allocation = compile_role_allocation(read(sidecar), synthetic, 67, 8, 42)
    save(root / 'allocation.json', allocation)
    first_id = allocation['steps'][0][0]
    jsonl(root / 'smoke_real.jsonl', real[:2])
    jsonl(root / 'smoke_preferences.jsonl', [x for x in synthetic if x['audio_id'] == first_id])
    save(root / 'smoke_allocation.json', dict(steps=[[first_id], [first_id]]))
    arguments = {}; smoke_arguments = {}
    for arm in ARMS:
        (root / arm).mkdir(); (root / ('smoke_' + arm)).mkdir()
        a = dict(model_path=p['base_model_dir'], policy_adapter_path=p['initial_adapter']['path'],
            reference_adapter_path=p['initial_adapter']['path'], real_train_file=str(root / 'real.jsonl'),
            output_dir=str(SCRATCH / arm / 'training'), max_steps=67, real_per_step=32,
            synthetic_per_step=8, synthetic_weight=.05, lr=1e-5, beta=.1, seed=42,
            preference_token_mask='edit_eos', expected_frozen_base_sha256=BASE_SHA)
        if arm == 'R_CE_MAS_DPO':
            a.update(synthetic_train_file=str(root / 'preferences.jsonl'),
                     synthetic_allocation_file=str(root / 'allocation.json'))
        arguments[arm] = a
        s = dict(a, max_steps=2, real_per_step=2, synthetic_per_step=1,
                 real_train_file=str(root / 'smoke_real.jsonl'),
                 output_dir=str(SCRATCH / ('smoke_' + arm) / 'training'))
        if arm == 'R_CE_MAS_DPO':
            s.update(synthetic_train_file=str(root / 'smoke_preferences.jsonl'),
                     synthetic_allocation_file=str(root / 'smoke_allocation.json'))
        smoke_arguments[arm] = s
        save(root / arm / 'expected_exposures.json', expected(a))
        save(root / ('smoke_' + arm) / 'expected_exposures.json', expected(s))
    planned = [read(root / a / 'expected_exposures.json') for a in ARMS]
    assert planned[0] == [x for x in planned[1] if x['source'] == 'real']
    assert len(planned[0]) == 2144 and len({x['audio_id'] for x in planned[0]}) == 2117
    save(root / 'data_receipt.json', dict(real_pool=2117, dev_feedback=511,
        real_audio_exposures=2144, real_unique_audio=2117, repeated_real_audio_occurrences=27,
        synthetic_audio=44, synthetic_pairs=107, synthetic_audio_exposures=536,
        source_preference_manifest=str(source), source_preference_sha256=sha256_file(source),
        source_sampling_receipt=str(base / 'adaptive_mas96_data_v1/round_001/sampling/sampling_receipt.json'),
        no_new_generation=True, no_new_sampling=True, no_new_MAS_decision=True,
        real_dev_ID_path_hash_and_wholetext_overlap=0,
        coarse_review='Original three-text review unchanged; does not prove word-for-word audio fidelity.'))
    sources = [Path(__file__), TRAINER, old_path, source, sidecar,
        base / 'real2117_residual_SFT_DPO_plan_v1.json',
        REPO / 'finetune/scripts/qwen3_asr_sft_dpo_probe.py',
        REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py',
        REPO / 'src/therapist_harness/preference_token_mask.py',
        REPO / 'examples/evaluate_residual_asr.py']
    sources += list(root.glob('*.json')) + list(root.glob('*.jsonl')) + list(root.glob('*/expected_exposures.json'))
    save(root / 'protocol.json', dict(source_protocol=p, arguments=arguments, smoke_arguments=smoke_arguments,
        source_sha256={**old['source_sha256'], **{str(f): sha256_file(f) for f in sources}},
        scope='Matched realCE control versus realCE plus synthetic edit+EOS DPO; full2117 pool and corrected merged-parent residual runtime. No realDPO or realKL in this experiment.',
        formal_steps=67, real_order_identical=True, parent_promoted=False))
    print(read(root / 'data_receipt.json'), flush=True)


def verify(root):
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == h for f, h in cfg['source_sha256'].items())
    return cfg


def train(base, root, arm, smoke):
    cfg = verify(root)
    if not smoke: assert read(root / 'smoke_verified.json')['status'] == 'passed'
    out = root / (('smoke_' if smoke else '') + arm)
    a = cfg['smoke_arguments' if smoke else 'arguments'][arm]
    command = [sys.executable, str(TRAINER)]
    for k, v in a.items(): command += ['--' + k, str(v)]
    started = time.time(); execute(out, command, 'training')
    directory = Path(a['output_dir']); result = read(directory / 'training_result.json')
    actual = rows(directory / 'training_exposures.jsonl')
    assert actual == read(out / 'expected_exposures.json')
    assert result['policy_changed'] and result['reference_files_unchanged'] and result['frozen_base_unchanged']
    assert read(directory / 'residual_verified.json')['status'] == 'passed'
    assert not (Path(result['checkpoint']) / 'optimizer.pt').exists()
    if arm == 'R_CE_MAS_DPO':
        assert math.isclose(result['history'][0]['weighted_losses']['synthetic'], .05 * math.log(2), abs_tol=1e-4)
        masks = read(directory / 'preference_token_masks.json')
        assert masks['mode'] == 'edit_eos' and masks['pairs']
        assert read(directory / 'reference_runtime.json')['merged_base_tensor_sha256'] == BASE_SHA
    copy_receipts(directory, out)
    checkpoint = Path(result['checkpoint']); digest = sha256_directory(checkpoint)
    save(out / 'verified.json', dict(status='passed', actual_exposures_match=True,
        frozen_base_and_reference_unchanged=True, checkpoint=str(checkpoint), checkpoint_sha256=digest))
    if smoke: return
    execute(out, evaluation_command(cfg['source_protocol'], out / 'evaluation_run', checkpoint), 'evaluation')
    metrics = read(out / 'evaluation_run/evaluation/summary.json')['metrics']
    parent = read(base / 'parent_eval_check/summary.json')['metrics']
    gates = dict(pooled_improves=metrics['overall']['pooled_cer'] < parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer'] - parent['overall']['patient_macro_cer'] <= .001,
        hard_regression_within_limit=metrics['hard']['pooled_cer'] - parent['hard']['pooled_cer'] <= .005)
    save(out / 'paired_parent.json', compare(base / 'parent_eval_check/predictions.jsonl',
        out / 'evaluation_run/evaluation/predictions.jsonl'))
    save(out / 'results.json', dict(status='complete', metrics=metrics, gate_checks=gates,
        eligible_for_next_round_parent=all(gates.values()), checkpoint=str(checkpoint), checkpoint_sha256=digest,
        actual_new_DPO_training=arm == 'R_CE_MAS_DPO', real_CE=True, postprocessing_used=False,
        elapsed_seconds=time.time() - started))
    assert sha256_directory(checkpoint) == digest
    verify(root)
    print(dict(arm=arm, CER=metrics['overall']['pooled_cer'], gate=gates), flush=True)


def crosscheck(root, smoke):
    dirs = [root / (('smoke_' if smoke else '') + a) for a in ARMS]
    actual = [rows(d / 'training_receipts/training_exposures.jsonl') for d in dirs]
    assert actual[0] == [x for x in actual[1] if x['source'] == 'real']
    result = [read(d / 'training_receipts/training_result.json') for d in dirs]
    assert result[0]['policy_trainable_sha256_before'] == result[1]['policy_trainable_sha256_before']
    assert result[0]['probe_before'][0] == result[1]['probe_before'][0]
    assert result[0]['history'][0]['weighted_losses']['real'] == result[1]['history'][0]['weighted_losses']['real']
    for d in dirs: assert read(d / 'verified.json')['status'] == 'passed'
    save(root / ('smoke_verified.json' if smoke else 'cross_arm_verified.json'), dict(status='passed',
        same_initial_residual_and_real_CE_probe=True, same_first_step_real_CE=True,
        exact_real_exposures_order_and_weights=True, actual_two_step_GPU_smoke=smoke))


def report(base, root):
    verify(root); crosscheck(root, False)
    summaries = {a: summarize(base, root / a) for a in ARMS}
    paired = compare(root / 'R_CE/evaluation_run/evaluation/predictions.jsonl',
                     root / 'R_CE_MAS_DPO/evaluation_run/evaluation/predictions.jsonl')
    save(root / 'paired_MAS_increment.json', paired)
    save(root / 'results.json', dict(status='complete', arms=summaries, parent_promoted=False,
        MAS_increment_pp=paired['groups']['overall']['delta_pp'],
        bootstrap95_pp=paired['groups']['overall']['patient_cluster_bootstrap_delta_pp_95'],
        matched_real_training=True, postprocessing_used=False))
    print(summaries, flush=True)


def supervise(base, root):
    for stage in ['smoke', 'train']:
        jobs = []
        for gpu, arm in enumerate(ARMS):
            command = [sys.executable, str(Path(__file__).resolve()), stage, str(base), '--arm', arm]
            with (root / (stage + '_' + arm + '.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
            save(root / (stage + '_' + arm + '.launch.json'), dict(pid=process.pid, gpu=gpu, command=command))
            jobs.append(process)
        codes = [p.wait() for p in jobs]
        assert codes == [0, 0], dict(stage=stage, returncodes=codes)
        if stage == 'smoke': crosscheck(root, True)
    report(base, root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'train', 'report', 'supervise'])
    parser.add_argument('base', type=Path); parser.add_argument('--arm', choices=ARMS, default=ARMS[0])
    args = parser.parse_args(); base = args.base.resolve(); root = base / NAME
    if args.stage == 'prepare': prepare(base, root)
    elif args.stage in ['smoke', 'train']: train(base, root, args.arm, args.stage == 'smoke')
    elif args.stage == 'report': report(base, root)
    else: supervise(base, root)
