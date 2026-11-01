#!/usr/bin/env python3
"""Matched R/DPO versus R+MAS-TTS/DPO from the corrected common parent runtime."""

import argparse
from collections import Counter
import math
from pathlib import Path
import shutil
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_ASR_target_format_probe import copy_receipts
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from therapist_harness.dpo_training import SourceBalancedWindows, group_training_rows, preference_key
from therapist_harness.preference_data import validate_sampling_inputs, build_preference_pairs
from therapist_harness.store import sha256_file, sha256_directory
from therapist_harness.text import text_normalize

ARMS = ['R_DPO', 'R_MAS_DPO']
SCRATCH = Path('/tmp/qwen3_asr_real_preference_bridge_v1')
TRAINER = REPO / 'finetune/scripts/qwen3_asr_weights_only_residual_dpo.py'


def expected(a):
    groups = group_training_rows(rows(a['train_file']), 'dpo')
    allocation = read(a['synthetic_allocation_file'])['steps'] if a.get('synthetic_allocation_file') else None
    windows = SourceBalancedWindows(groups, a['real_per_step'], a['synthetic_per_step'],
                                    a['synthetic_weight'], a['seed'], synthetic_schedule=allocation)
    records = []
    for step in range(1, a['max_steps'] + 1):
        for row in windows.next():
            records.append(dict(step=step, source=row['source'], audio_id=row['_audio_group'],
                occurrence=row['_occurrence'], preference_key=preference_key(row), loss_weight=row['_loss_weight']))
    return records


def prepare(base, root):
    source = base / 'real96_dpo_probe/sampling'
    normalized = base / 'ASR_target_format_inputs_v1/preferences.jsonl'
    synthetic_source = base / 'candidate_coverage_recovery_v2/sampling'
    p = read(base / 'lexical96_mas_v1/protocol.json')
    parent = p['initial_adapter']['sha256']
    assert sha256_directory(Path(p['initial_adapter']['path'])) == parent
    inputs_by_source = {}
    receipts = {}
    files = [Path(__file__), TRAINER, REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py',
        REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py', REPO / 'finetune/scripts/qwen3_asr_weights_only_patient_kl_dpo.py',
        REPO / 'src/therapist_harness/dpo_training.py', REPO / 'src/therapist_harness/preference_data.py',
        REPO / 'examples/mas_ASR_target_format_probe.py', REPO / 'examples/mas_residual_dpo_probe.py',
        REPO / 'examples/evaluate_residual_asr.py', base / 'lexical96_mas_v1/protocol.json', normalized,
        base / 'ASR_target_format_inputs_v1/receipt.json', base / 'ASR_target_format_training_seeds_v1/results.json']
    real_train, feedback = rows(p['real_train']['path']), rows(p['feedback_split']['path'])
    assert len(real_train) == 2117 and len(feedback) == 511
    for name, directory in [('real', source), ('synthetic', synthetic_source)]:
        receipt = read(directory / 'sampling_receipt.json')
        assert receipt['status'] == 'complete' and receipt['parent_files_unchanged']
        assert receipt['current_asr_sha256'] == parent
        assert receipt['real_train_sha256'] == sha256_file(Path(p['real_train']['path']))
        assert receipt['feedback_sha256'] == sha256_file(Path(p['feedback_split']['path']))
        for filename, field in [('candidates.jsonl', 'candidates_sha256'), ('validated_inputs.jsonl', 'validated_inputs_sha256'), ('greedy.jsonl', 'greedy_sha256')]:
            assert sha256_file(directory / filename) == receipt[field]
        inputs = rows(directory / 'validated_inputs.jsonl')
        checked = validate_sampling_inputs(inputs, real_train, feedback)
        assert checked == inputs
        rebuilt, audit = build_preference_pairs(checked, rows(directory / 'candidates.jsonl'), parent,
            expected_samples=receipt['samples_per_audio'], greedy_candidates=rows(directory / 'greedy.jsonl'))
        assert rebuilt == rows(directory / 'preferences.jsonl')
        assert audit == read(directory / 'preference_audit.json')
        inputs_by_source[name] = rebuilt
        receipts[name] = dict(path=str(directory / 'sampling_receipt.json'), sha256=sha256_file(directory / 'sampling_receipt.json'),
            audio=len(inputs), pairs=len(rebuilt), actual_draws=receipt['candidate_count'],
            decoding={k: receipt[k] for k in ['samples_per_audio', 'max_new_tokens', 'temperature', 'top_p', 'top_k', 'prompt', 'language', 'seed']})
        files += [directory / n for n in ['sampling_receipt.json', 'validated_inputs.jsonl', 'candidates.jsonl',
                                         'greedy.jsonl', 'preferences.jsonl', 'preference_audit.json']]
    real = inputs_by_source['real']; synthetic = rows(normalized)
    available = {x['pair_id']: x for x in inputs_by_source['synthetic']}
    for item in synthetic:
        original = available[item['pair_id']]
        assert item['audio_sha256'] == original['audio_sha256'] and item['rejected'] == original['rejected']
        assert item['sample_indices'] == original['sample_indices'] and item['chosen'] == original['chosen'].rstrip('。！？!?')
        assert text_normalize(item['chosen']) != text_normalize(item['rejected'])
        assert item['pair_weight'] == 1. and item['current_asr_sha256'] == parent
    assert len(real) == 132 and len({x['audio_id'] for x in real}) == 50 and len(synthetic) == 15
    root.mkdir(exist_ok=False)
    assert shutil.disk_usage('/tmp').free > 400 * 1024 ** 2
    SCRATCH.mkdir(exist_ok=False)
    jsonl(root / 'real_preferences.jsonl', real)
    jsonl(root / 'mixed_preferences.jsonl', real + synthetic)
    allocation = base / 'candidate_coverage_recovery_v2/allocation.json'
    arguments = {}
    for arm in ARMS:
        (root / arm).mkdir(); (SCRATCH / arm).mkdir()
        a = dict(model_path=p['base_model_dir'], policy_adapter_path=p['initial_adapter']['path'],
            reference_adapter_path=p['initial_adapter']['path'], train_file=str(root / ('real_preferences.jsonl' if arm == 'R_DPO' else 'mixed_preferences.jsonl')),
            output_dir=str(SCRATCH / arm / 'training'), objective='dpo', max_steps=12, batch_size=1,
            real_per_step=16, synthetic_per_step=0 if arm == 'R_DPO' else 8, synthetic_weight=.25,
            lr=1e-5, beta=.1, seed=42)
        if arm == 'R_MAS_DPO': a['synthetic_allocation_file'] = str(allocation)
        arguments[arm] = a
        save(root / arm / 'expected_exposures.json', expected(a))
    a, b = [expected(arguments[arm]) for arm in ARMS]
    assert a == [x for x in b if x['source'] == 'real']
    audio_counts = lambda records: dict(Counter(x[1] for x in {(r['step'], r['source'], r['occurrence']) for r in records}))
    assert audio_counts(a) == {'real': 192} and audio_counts(b) == {'real': 192, 'synthetic': 96}
    for data in [a, b]:
        for step in range(1, 13):
            assert math.isclose(sum(x['loss_weight'] for x in data if x['source'] == 'real' and x['step'] == step), 1.)
    save(root / 'data_receipt.json', dict(status='passed', sources=receipts,
        common_initial_R_only_parent=parent, actual_new_sampling=False, actual_new_generation=False,
        real_contributing_audio=50, real_all_correct_audio_excluded=46, real_pairs=132, MAS_audio_and_pairs=15,
        decoding_differences='Preserved independent executions: real4 draws cap256; synthetic16 draws cap128. Same parent,seed,language,empty context,temperature,top_p,top_k. No single merged sampling execution is claimed.',
        review_unchanged=True, synthetic_chosen_transform='Only terminal punctuation stripped; original target/review retained.',
        real2117_feedback511_isolation='Revalidated audio hashes, identities and content through original input validator.'))
    save(root / 'cpu_smoke.json', dict(status='passed', identical_actual_real_pair_order_weights=True,
        planned_audio_exposures={ARMS[0]: audio_counts(a), ARMS[1]: audio_counts(b)},
        distinct_errors_average_per_audio=True, actual_GPU_smoke_pending=True))
    files += [Path(p['real_train']['path']), Path(p['feedback_split']['path']), allocation,
              root / 'real_preferences.jsonl', root / 'mixed_preferences.jsonl', root / 'data_receipt.json', root / 'cpu_smoke.json']
    files += [root / arm / 'expected_exposures.json' for arm in ARMS]
    save(root / 'protocol.json', dict(arguments=arguments, source_protocol=p,
        source_sha256={str(f): sha256_file(f) for f in files},
        actual_new_DPO_training=True, actual_new_MAS_decision=False,
        objective='Standard full transcript+EOS DPO on real preferences; mixed branch adds0.25 times MAS synthetic DPO. No real CE or KL retention.',
        motivation='Three normalized edit+EOS synthetic-DPO+real-KL seeds average exactly R-only CER. Test direct observed real error preferences as the real training signal and measure incremental MAS benefit in a matched comparison.',
        settings='Prespecified seed42,12steps,lr1e-5,beta0.1,192 common real audio exposures; mixed adds96 MAS exposures at weight0.25. This is a new objective/data experiment, not an isolated ablation of only label format.',
        acceptance='Original pooled/macro/hard development gate; also compare R+MAS/DPO against R/DPO. No automatic parent promotion.',
        storage='Temporary /tmp checkpoint weights; persistent workspace logs, receipts and evaluation. No legacy files removed.'))


def verify(root):
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == h for f, h in cfg['source_sha256'].items())
    return cfg


def run(base, root, arm, smoke=False):
    cfg = verify(root)
    a = dict(cfg['arguments'][arm]); out = root / ('smoke' if smoke else arm)
    if smoke:
        out.mkdir(); (SCRATCH / 'smoke').mkdir()
        data = rows(a['train_file']); selected = []
        for source in ['real', 'synthetic']:
            ids = list(dict.fromkeys(x['audio_id'] for x in data if x['source'] == source))[:2]
            selected += [x for x in data if x['audio_id'] in ids]
        jsonl(out / 'preferences.jsonl', selected)
        a.update(train_file=str(out / 'preferences.jsonl'), output_dir=str(SCRATCH / 'smoke/training'),
                 max_steps=2, real_per_step=2, synthetic_per_step=2)
        a.pop('synthetic_allocation_file', None)
        save(out / 'expected_exposures.json', expected(a))
    else:
        assert read(root / 'smoke/verified.json')['status'] == 'passed'
    execute(out, [sys.executable, str(TRAINER)] + [x for k, v in a.items() for x in ['--' + k, str(v)]], 'training')
    training = Path(a['output_dir']); result = read(training / 'training_result.json')
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert read(training / 'residual_verified.json')['status'] == 'passed'
    assert math.isclose(result['history'][0]['loss'], math.log(2) * (1 + (.25 if a['synthetic_per_step'] else 0)), abs_tol=1e-4)
    assert rows(training / 'training_exposures.jsonl') == read(out / 'expected_exposures.json')
    checkpoint = Path(result['checkpoint']); digest = sha256_directory(checkpoint)
    assert not (checkpoint / 'optimizer.pt').exists()
    assert read(checkpoint / 'optimizer_omission.json')['optimizer_state_persisted'] is False
    copy_receipts(training, out)
    save(out / 'verified.json', dict(status='passed', actual_new_GPU_training=True, actual_steps=a['max_steps'],
        initial_DPO_log2_per_source=True, exact_planned_real_synthetic_pair_exposures=True,
        residual_updated_frozen_parent_reference_preserved=True, checkpoint=str(checkpoint), checkpoint_sha256=digest))
    if smoke:
        verify(root)
        return
    execute(out, evaluation_command(cfg['source_protocol'], out / 'evaluation_run', checkpoint), 'evaluation')
    metrics = read(out / 'evaluation_run/evaluation/summary.json')['metrics']
    parent = read(base / 'parent_eval_check/summary.json')['metrics']
    gate = dict(pooled_improves=metrics['overall']['pooled_cer'] < parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer'] - parent['overall']['patient_macro_cer'] <= .001,
        hard_regression_within_limit=metrics['hard']['pooled_cer'] - parent['hard']['pooled_cer'] <= .005)
    save(out / 'paired_parent.json', compare(base / 'parent_eval_check/predictions.jsonl', out / 'evaluation_run/evaluation/predictions.jsonl'))
    save(out / 'results.json', dict(status='complete', metrics=metrics, gate_checks=gate,
        eligible_for_next_round_parent=all(gate.values()), actual_new_DPO_training=True, postprocessing_used=False,
        checkpoint=str(checkpoint), checkpoint_sha256=digest, checkpoint_storage='temporary scratch', persistent_checkpoint_exported=False))
    assert sha256_directory(checkpoint) == digest
    verify(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'run'])
    parser.add_argument('base', type=Path)
    parser.add_argument('--arm', choices=ARMS, default='R_MAS_DPO')
    args = parser.parse_args(); base = args.base.resolve(); root = base / 'real_preference_bridge_v1'
    if args.stage == 'prepare': prepare(base, root)
    else: run(base, root, args.arm, args.stage == 'smoke')
