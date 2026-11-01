#!/usr/bin/env python3
"""Match actual greedy and first-stochastic-error negatives on identical audio."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import shutil

from mas_selection_dpo_probe import read, rows, save, jsonl
from mas_edit_mask_dpo_probe import exposure_plan
from mas_adaptive96_trial_recovery_v2 import train
from mas_evaluation_report import compare
from report_ASR_target_format import summarize
from therapist_harness.store import sha256_file
from therapist_harness.text import text_normalize
from therapist_harness.training_allocation import compile_role_allocation

ARMS = ['greedy_rejected', 'stochastic_rejected']


def prepare(base, root):
    source = base / 'adaptive96_matched_DPO_recovery_v2'
    old = read(source / 'protocol.json')
    assert all(sha256_file(Path(p)) == h for p,h in old['source_sha256'].items())
    original = source / 'MAS_reference/preferences.jsonl'
    sample_root = base / 'adaptive_mas96_data_v1/round_001/sampling'
    receipt = read(sample_root / 'sampling_receipt.json')
    assert receipt['status'] == 'complete' and receipt['parent_files_unchanged']
    assert receipt['current_asr_sha256'] == old['source_protocol']['initial_adapter']['sha256']
    assert sha256_file(sample_root / 'greedy.jsonl') == receipt['greedy_sha256']
    assert sha256_file(sample_root / 'candidates.jsonl') == receipt['candidates_sha256']
    greedy = {x['audio_id']:x for x in rows(sample_root / 'greedy.jsonl')}
    candidates = defaultdict(list)
    for x in rows(sample_root / 'candidates.jsonl'): candidates[x['audio_id']].append(x)
    grouped = defaultdict(list)
    for x in rows(original): grouped[x['audio_id']].append(x)
    data = {arm:[] for arm in ARMS}; origins = []
    for audio_id, pairs in sorted(grouped.items()):
        template = pairs[0]; g = greedy[audio_id]
        if text_normalize(g['prediction']) == text_normalize(template['chosen']): continue
        assert g['ended_with_eos'] and not g['reached_max_new_tokens']
        assert g['do_sample'] is False and g['num_beams'] == 1 and g['prompt'] == '' and g['language'] == 'Cantonese'
        assert g['current_asr_sha256'] == template['current_asr_sha256'] and g['audio_sha256'] == template['audio_sha256']
        assert sha256_file(Path(template['audio'])) == g['audio_sha256']
        actual_wrong = sorted([x for x in candidates[audio_id]
                               if text_normalize(x['prediction']) != text_normalize(template['chosen'])],
                              key=lambda x:x['sample_index'])
        assert actual_wrong, 'Never manufacture a negative for an all-stochastic-correct case.'
        first = actual_wrong[0]
        selected = next(x for x in pairs if first['sample_index'] in x['sample_indices'])
        assert text_normalize(selected['rejected']) == text_normalize(first['prediction'])
        s = deepcopy(selected); s['pair_weight'] = 1.0
        s['metadata'].update(negative_selection='First actual wrong stochastic draw by original sample_index',
                             original_pair_weight=selected['pair_weight'], original_preference_manifest=str(original))
        data['stochastic_rejected'].append(s)
        row = deepcopy(template)
        row.update(pair_id=f'{audio_id}::greedy_rejected', rejected=g['prediction'],
                   normalized_rejected=text_normalize(g['prediction']), pair_weight=1.0,
                   sample_ids=[], sample_indices=[])
        row['metadata'].update(negative_origin='actual_greedy_decode_not_a_stochastic_draw',
            original_greedy_record=g, original_greedy_file=str(sample_root / 'greedy.jsonl'),
            original_greedy_file_sha256=receipt['greedy_sha256'],
            original_sampling_receipt=str(sample_root / 'sampling_receipt.json'))
        data['greedy_rejected'].append(row)
        origins.append(dict(audio_id=audio_id, audio_sha256=g['audio_sha256'], greedy=g,
                            first_wrong_stochastic=first, original_stochastic_pair_id=selected['pair_id']))
    assert data['greedy_rejected'] and len(data[ARMS[0]]) == len(data[ARMS[1]])
    assert shutil.disk_usage('/tmp').free > 200 * 1024**2
    root.mkdir(exist_ok=False)
    scratch = Path('/tmp/qwen3_asr_adaptive96_greedy_preference_v1'); scratch.mkdir(exist_ok=False)
    sidecar_path = base / 'adaptive_mas96_data_v1/round_001/materialization_sidecar.json'
    allocation = compile_role_allocation(read(sidecar_path), data[ARMS[0]],12,8,42)
    assert allocation == compile_role_allocation(read(sidecar_path), data[ARMS[1]],12,8,42)
    save(root / 'allocation.json', allocation); save(root / 'negative_origins.json', origins)
    arguments = {}; plans = {}
    for arm in ARMS:
        out = root / arm; out.mkdir(); (scratch / arm).mkdir()
        jsonl(out / 'preferences.jsonl', data[arm])
        a = dict(old['arguments']['MAS_reference'], synthetic_train_file=str(out / 'preferences.jsonl'),
            synthetic_allocation_file=str(root / 'allocation.json'), output_dir=str(scratch / arm / 'training'))
        arguments[arm] = a
        plans[arm] = exposure_plan(a, rows(a['real_train_file']), data[arm], allocation, read(a['real_loss_factors'])['factors'])
        save(out / 'expected_exposures.json', plans[arm])
    assert [x for x in plans[ARMS[0]] if x['source'] == 'real'] == [x for x in plans[ARMS[1]] if x['source'] == 'real']
    occurrences = lambda records: {(x['step'],x['source'],x['occurrence'],x['audio_id']) for x in records}
    assert occurrences(plans[ARMS[0]]) == occurrences(plans[ARMS[1]])
    save(root / 'data_receipt.json', dict(audio_per_arm=len(origins), pairs_per_audio=1,
        rejected_text_identical=sum(x['greedy']['prediction'] == x['first_wrong_stochastic']['prediction'] for x in origins),
        same_all96_synthetic_and192_real_audio_exposures=True,
        original_four_sample_and_greedy_receipt=str(sample_root / 'sampling_receipt.json'),
        actual_new_sampling=False, actual_new_generation=False,
        provenance='Derived preference manifests from independently recorded actual greedy and stochastic outputs within the original completed sampling job. No new or unified sampling receipt is invented. Greedy preference rows have empty stochastic sample_ids/indices; their complete original greedy record is retained.',
        selection='Common original44-case pool restricted to cases with both a normal greedy error and at least one actual stochastic error. All-correct cases are never supplied invented negatives.'))
    save(root / 'cpu_smoke.json', dict(status='passed', actual_new_GPU_smoke=False,
        actual_negative_records_audio_parent_and_decode_verified=True, both_arms_one_pair_per_same_audio=True,
        actual_exposure_compilation_matches=True, same_engine_GPU_smoke=old['same_engine_GPU_smoke']))
    sources = [Path(__file__), base / 'adaptive96_greedy_preference_plan_v1.json', original,
        source / 'protocol.json', sidecar_path, sample_root / 'sampling_receipt.json',
        sample_root / 'greedy.jsonl', sample_root / 'candidates.jsonl']
    sources += list(root.glob('*.json'))
    sources += [root / a / n for a in ARMS for n in ['preferences.jsonl','expected_exposures.json']]
    save(root / 'protocol.json', dict(source_protocol=old['source_protocol'], arguments=arguments,
        source_sha256={**old['source_sha256'], **{str(f):sha256_file(f) for f in sources}},
        same_engine_GPU_smoke=old['same_engine_GPU_smoke'], actual_new_DPO_training=True,
        actual_new_generation=False, actual_new_sampling=False, actual_new_MAS_decision=False,
        intervention='Actual greedy versus first actual stochastic wrong transcript, one rejected per same audio; identical exposure schedule, real KL retention, initialR-only/frozenreference and all training settings.'))
    print(read(root / 'data_receipt.json'),flush=True)


def report(base, root):
    summaries = {a:summarize(base,root/a) for a in ARMS}
    actual = [rows(root / a / 'training_receipts/training_exposures.jsonl') for a in ARMS]
    assert [x for x in actual[0] if x['source'] == 'real'] == [x for x in actual[1] if x['source'] == 'real']
    assert read(root / ARMS[0] / 'training_receipts/training_result.json')['policy_trainable_sha256_before'] == read(root / ARMS[1] / 'training_receipts/training_result.json')['policy_trainable_sha256_before']
    paired = compare(root / 'stochastic_rejected/evaluation_run/evaluation/predictions.jsonl',
                     root / 'greedy_rejected/evaluation_run/evaluation/predictions.jsonl')
    save(root / 'paired_negative_policy.json', paired)
    save(root / 'results.json', dict(status='complete', arms=summaries, parent_promoted=False,
        greedy_minus_stochastic_CER_pp=paired['groups']['overall']['delta_pp'],
        bootstrap95_pp=paired['groups']['overall']['patient_cluster_bootstrap_delta_pp_95'],
        actual_new_DPO_training=True, actual_new_generation=False, actual_new_sampling=False,
        postprocessing_used=False))
    print(summaries,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['prepare','train','report'])
    parser.add_argument('base',type=Path); parser.add_argument('--arm',choices=ARMS,default=ARMS[0])
    args = parser.parse_args(); base = args.base.resolve(); root = base / 'adaptive96_greedy_preference_v1'
    if args.stage == 'prepare': prepare(base,root)
    elif args.stage == 'train': train(base,root,args.arm)
    else: report(base,root)
