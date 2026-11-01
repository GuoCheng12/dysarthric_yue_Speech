#!/usr/bin/env python3
"""Matched DPO on actual paired cases in the patient-assignment probe."""

import argparse
from collections import defaultdict
import math
import os
from pathlib import Path
import subprocess
import sys

from mas_selection_dpo_probe import REPO, read, rows, save, jsonl
from mas_patient_assignment_data_v1 import NAME, verify as verify_data
from mas_adaptive96_trial_recovery_v2 import train, verify
from mas_edit_mask_dpo_probe import exposure_plan
from mas_evaluation_report import compare
from report_ASR_target_format import summarize
from therapist_harness.preference_data import build_preference_pairs
from therapist_harness.store import sha256_file
from therapist_harness.text import text_normalize
from therapist_harness.training_allocation import compile_role_allocation

ARMS = ['original_patient', 'MAS_patient']
SCRATCH = Path('/tmp/qwen3_asr_patient_assignment_DPO_v1')


def collapse(records):
    result = defaultdict(float)
    for x in records:
        result[(x['step'], x['source'], x['occurrence'], x['audio_id'])] += x['loss_weight']
    return result


def prepare(base, root):
    data_root = base / NAME; plan = verify_data(data_root)
    assert read(data_root / 'data_complete.json')['status'] == 'complete'
    old = read(base / 'adaptive96_matched_DPO_recovery_v2/protocol.json')
    p = plan['source_protocol']; parent = p['initial_adapter']['sha256']
    selected_ids = set(plan['selected_cases'])
    roots = dict(original_patient=Path(plan['source_root']), MAS_patient=data_root)
    data, receipts, candidates, manifests, generation = {}, {}, {}, {}, {}
    files = [Path(__file__), Path(__file__).with_name('mas_adaptive96_trial_recovery_v2.py'),
        data_root / 'plan.json', data_root / 'data_complete.json', data_root / 'decision_receipt.json',
        data_root / 'patient_assignment_plan.json', data_root / 'role_sidecar.json']
    for arm, directory in roots.items():
        sample = directory / 'sampling'; receipt = read(sample / 'sampling_receipt.json')
        assert receipt['status'] == 'complete' and receipt['parent_files_unchanged']
        assert receipt['current_asr_sha256'] == parent and receipt['samples_per_audio'] == 4
        for filename, field in [('candidates.jsonl','candidates_sha256'), ('validated_inputs.jsonl','validated_inputs_sha256'), ('greedy.jsonl','greedy_sha256')]:
            assert sha256_file(sample / filename) == receipt[field]
        inputs = rows(sample / 'validated_inputs.jsonl')
        assert all(sha256_file(Path(x['audio'])) == x['audio_sha256'] for x in inputs)
        draws = rows(sample / 'candidates.jsonl')
        rebuilt, audit = build_preference_pairs(inputs, draws, parent, expected_samples=4,
            greedy_candidates=rows(sample / 'greedy.jsonl'))
        assert rebuilt == rows(sample / 'preferences.jsonl') and audit == read(sample / 'preference_audit.json')
        data[arm] = [x for x in rebuilt if x['audio_id'] in selected_ids]
        receipts[arm] = receipt
        candidates[arm] = {x['sample_id']:x for x in draws if x['audio_id'] in selected_ids}
        manifests[arm] = {x['audio_id']:x for x in inputs if x['audio_id'] in selected_ids}
        generation[arm] = {x['sample_id']:x for x in read(directory / 'generation/generation_summary.json')['outputs'] if x['sample_id'] in selected_ids}
        files += [sample / n for n in ['sampling_receipt.json','candidates.jsonl','validated_inputs.jsonl',
            'greedy.jsonl','preferences.jsonl','preference_audit.json']]
        files += [directory / 'generation_plan.json', directory / 'generation/generation_summary.json']
    fields = ['base_model_sha256','current_asr_sha256','samples_per_audio','seed','temperature','top_p','top_k',
        'max_new_tokens','language','prompt','do_sample','real_train_sha256','feedback_sha256','greedy_config']
    assert all(receipts[ARMS[0]][k] == receipts[ARMS[1]][k] for k in fields)
    common = set.intersection(*[{x['audio_id'] for x in data[a]} for a in ARMS])
    assert common, 'No shared actual paired case; do not invent negatives or train.'
    for key in common:
        assert manifests[ARMS[0]][key]['clean_gt'] == manifests[ARMS[1]][key]['clean_gt']
        for index in range(4):
            a = candidates[ARMS[0]][f'{key}::{index}']; b = candidates[ARMS[1]][f'{key}::{index}']
            assert a['seed'] == b['seed'] and a['sample_index'] == b['sample_index']
    root.mkdir(exist_ok=False); SCRATCH.mkdir(exist_ok=False)
    arguments, expected, allocations = {}, {}, {}
    for arm in ARMS:
        out = root / arm; out.mkdir()
        pairs = []
        for row in data[arm]:
            if row['audio_id'] not in common: continue
            chosen = row['chosen'].rstrip('。！？!?')
            assert text_normalize(chosen) == text_normalize(row['chosen']) != text_normalize(row['rejected'])
            pairs.append(dict(row, chosen=chosen, metadata=dict(row['metadata'], original_target_text=row['chosen'],
                label_transform='strip_terminal_sentence_punctuation_v2')))
        jsonl(out / 'preferences.jsonl', pairs)
        allocations[arm] = compile_role_allocation(read(data_root / 'role_sidecar.json'), pairs, 12, 8, 42)
        a = dict(old['arguments']['MAS_reference'], synthetic_train_file=str(out / 'preferences.jsonl'),
            synthetic_allocation_file=str(root / 'allocation.json'), output_dir=str(SCRATCH / arm / 'training'))
        arguments[arm] = a
        expected[arm] = exposure_plan(a, rows(a['real_train_file']), pairs, allocations[arm], read(a['real_loss_factors'])['factors'])
        save(out / 'expected_exposures.json', expected[arm])
        files += [out / 'preferences.jsonl', out / 'expected_exposures.json']
    assert allocations[ARMS[0]] == allocations[ARMS[1]]
    assert [x for x in expected[ARMS[0]] if x['source'] == 'real'] == [x for x in expected[ARMS[1]] if x['source'] == 'real']
    a, b = [collapse(expected[arm]) for arm in ARMS]
    assert a.keys() == b.keys() and all(math.isclose(a[k], b[k], abs_tol=1e-12) for k in a)
    save(root / 'allocation.json', allocations[ARMS[0]])
    save(root / 'data_receipt.json', dict(common_logical_case_ids=sorted(common), common_audio=len(common),
        prospective_selected_audio=24, selected_pair_audio={a:len({x['audio_id'] for x in data[a]}) for a in ARMS},
        training_pair_rows={a:len(rows(root / a / 'preferences.jsonl')) for a in ARMS},
        actual_delivery={a:[dict(sample_id=k, admitted=x['admitted'], attempt=x['attempt'],
            mode=x['generation_mode'], instruction_id=x['effective_instruction_id'],
            reference_wav_sha256=x['reference_wav_sha256'], audio_sha256=x['output_wav_sha256'],
            contributes_to_matched_DPO=k in common) for k,x in generation[a].items()] for a in ARMS},
        independent_receipts={a:str(roots[a] / 'sampling/sampling_receipt.json') for a in ARMS},
        control_reuse='Verified subset of the historical96audio/384draw receipt, not a new24audio sampling job.',
        matching='Same target/case/actual sampling seeds and decoders; waveform/patient/reference changed. DPO conditions on cases with actual pairs on both sides; excluded all-correct cases remain in generation/sampling reports.',
        objective='Existing edit+EOS scores, all actual distinct wrong negatives averaged per audio; terminal chosen punctuation stripped exactly as previous runs.'))
    smoke = base / 'edit_mask_dpo_probe_v1/smoke/verified.json'
    assert read(smoke)['status'] == 'passed'
    save(root / 'cpu_smoke.json', dict(status='passed', actual_preferences_rebuilt_from_receipts=True,
        matching_parent_decode_seed_and_real_exposure_order=True, matching_synthetic_occurrence_weights=True,
        unchanged_training_engine=True, reused_actual_two_step_GPU_smoke=str(smoke)))
    files += [root / 'allocation.json', root / 'data_receipt.json', root / 'cpu_smoke.json', smoke]
    save(root / 'protocol.json', dict(source_protocol=p, arguments=arguments,
        source_sha256={**plan['source_sha256'], **{str(f):sha256_file(f) for f in files},
            **read(data_root / 'decision_receipt.json')['bound_sha256']},
        source_data_roots={a:str(r) for a,r in roots.items()}, same_engine_GPU_smoke=str(smoke),
        intervention=plan['intervention'], acceptance=plan['acceptance'],
        patient_assignment_plan=str(data_root / 'patient_assignment_plan.json'), parent_promoted=False))
    verify(root)
    print(dict(stage='prepared', common_audio=len(common), pair_rows=read(root / 'data_receipt.json')['training_pair_rows']), flush=True)


def report(base, root):
    verify(root); summaries = {a:summarize(base, root / a) for a in ARMS}
    records = [rows(root / a / 'training_receipts/training_exposures.jsonl') for a in ARMS]
    assert [x for x in records[0] if x['source'] == 'real'] == [x for x in records[1] if x['source'] == 'real']
    x, y = [collapse(rs) for rs in records]
    assert x.keys() == y.keys() and all(math.isclose(x[k], y[k], abs_tol=1e-12) for k in x)
    assert read(root / ARMS[0] / 'training_receipts/training_result.json')['policy_trainable_sha256_before'] == read(root / ARMS[1] / 'training_receipts/training_result.json')['policy_trainable_sha256_before']
    paired = compare(root / ARMS[0] / 'evaluation_run/evaluation/predictions.jsonl', root / ARMS[1] / 'evaluation_run/evaluation/predictions.jsonl')
    save(root / 'paired_patient_assignment.json', paired)
    voice = read(base / NAME / 'patient_assignment_plan.json')['voice_group']
    patient = read(base / NAME / 'patient_key_map.json')[voice]
    focus = {}
    for arm, path in [('R_only', base / 'parent_eval_check/predictions.jsonl')] + [(a, root / a / 'evaluation_run/evaluation/predictions.jsonl') for a in ARMS]:
        xs = rows(path); focus[arm] = {}
        for label, data in [('selected_patient', [x for x in xs if x['speaker_id'] == patient]), ('other_patients', [x for x in xs if x['speaker_id'] != patient])]:
            edits = sum(x['edit_count'] for x in data); chars = sum(x['normalized_reference_characters'] for x in data)
            focus[arm][label] = dict(audio=len(data), edits=edits, chars=chars, CER_percent=100*edits/chars)
    save(root / 'results.json', dict(status='complete', arms=summaries, selected_voice_group=voice,
        selected_patient_and_others=focus, MAS_minus_original_CER_pp=paired['groups']['overall']['delta_pp'],
        bootstrap95_pp=paired['groups']['overall']['patient_cluster_bootstrap_delta_pp_95'],
        actual_exposure_identity_order_weights_match=True, common_initial_residual=True, parent_promoted=False,
        postprocessing_used=False))
    print(summaries, flush=True)


def supervise(base, root):
    processes = []
    for gpu, arm in enumerate(ARMS):
        command = [sys.executable, str(Path(__file__).resolve()), 'train', str(base), '--arm', arm]
        with (root / (arm + '.log')).open('x') as log:
            p = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        save(root / (arm + '.launch.json'), dict(pid=p.pid, gpu=gpu, command=command)); processes.append(p)
    codes = [p.wait() for p in processes]; assert codes == [0, 0], codes
    report(base, root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['prepare','train','report','supervise'])
    parser.add_argument('base', type=Path); parser.add_argument('--arm', choices=ARMS, default=ARMS[0])
    args = parser.parse_args(); base = args.base.resolve(); root = base / 'patient_assignment_DPO_v1'
    if args.stage == 'prepare': prepare(base, root)
    elif args.stage == 'train': train(base, root, args.arm)
    elif args.stage == 'report': report(base, root)
    else: supervise(base, root)
