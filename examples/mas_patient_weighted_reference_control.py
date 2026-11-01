#!/usr/bin/env python3
"""Prepare a matched reference/recipe control for the patient-weighted DPO pilot."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_candidate_coverage_probe import Decision
from mas_patient_retention_probe import request_mas
from mas_preference_target_feedback import actual_events
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file


def verify(root):
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == h for f, h in cfg['source_sha256'].items())
    return cfg


def prepare(base, root):
    selected = base / 'candidate_coverage_recovery_v2'
    prior = base / 'lexical96_random_control_v1/sampling'
    p = read(base / 'patient_retention_dose_probe_v1/protocol.json')['source_protocol']
    actual = rows(selected / 'all16/preferences.jsonl')
    by_id = {x['audio_id']: x for x in actual}
    inputs = [x for x in rows(prior / 'validated_inputs.jsonl') if x['audio_id'] in by_id]
    assert len(inputs) == len(by_id) == 15
    for row in inputs:
        assert row['target_text'] == by_id[row['audio_id']]['chosen']
        assert row['review']['verdict'] == 'pass'
        assert sha256_file(Path(row['audio'])) == row['audio_sha256']
    receipt = read(prior / 'sampling_receipt.json')
    assert receipt['current_asr_sha256'] == p['initial_adapter']['sha256']
    root.mkdir(exist_ok=False)
    jsonl(root / 'inputs.jsonl', inputs)
    os.link(selected / 'allocation.json', root / 'allocation.json')
    os.link(selected / 'purposes.json', root / 'purposes.json')
    files = [Path(__file__), REPO / 'examples/sample_asr_preferences.py',
        REPO / 'examples/mas_patient_retention_probe.py', REPO / 'examples/mas_preference_target_feedback.py',
        REPO / 'examples/mas_candidate_coverage_probe.py', REPO / 'src/therapist_harness/preference_data.py',
        selected / 'all16/preferences.jsonl', selected / 'all16/request.json', selected / 'all16/response.json',
        selected / 'sampling/sampling_receipt.json', selected / 'cross_arm_verification.json',
        prior / 'validated_inputs.jsonl', prior / 'sampling_receipt.json',
        prior.parent / 'provenance.json', prior.parent / 'protocol.json', prior.parent / 'results.json',
        base / 'patient_retention_dose_probe_v1/results.json',
        root / 'inputs.jsonl', root / 'allocation.json', root / 'purposes.json']
    save(root / 'protocol.json', dict(source_protocol=p, source_sha256={str(f):sha256_file(f) for f in files},
        MAS_reference_control=str(selected), random_reference_audio_source=str(prior),
        actual_new_generation=False, actual_new_sampling=True,
        scope='Reuse the prior reviewed random-reference/recipe audio for the same15 MAS target texts and logical case IDs. Obtain16 actual initial-R-only samples per audio and use the same original MAS negative-selector instructions. Conditional reference/recipe control, not a non-MAS-content control.',
        training_if_available='Use the same real96 pool,50% focus weight,192 real and96 synthetic exposures,12steps,lr1e-4,KL10,beta.1,seed42. Both start from initial R-only. Reference remains initial R-only.',
        all_correct_rule='If any case has no actual wrong sample, report incomplete15-case pairing. Do not fabricate a negative or silently change the matched schedule.',
        source_receipts='Original SG/review and new ASR sampling retain separate provenance; no merged or invented generation/sampling receipt.'))


def sample(root):
    cfg = verify(root);p=cfg['source_protocol']
    execute(root, [sys.executable, str(REPO / 'examples/sample_asr_preferences.py'), 'sample',
        '--inputs', str(root/'inputs.jsonl'), '--real-train', p['real_train']['path'],
        '--feedback', p['feedback_split']['path'], '--out-dir', str(root/'sampling'),
        '--base-model', p['base_model_dir'], '--current-adapter', p['initial_adapter']['path'],
        '--expected-parent-sha256', p['initial_adapter']['sha256'], '--samples', '16', '--seed', '42',
        '--temperature', '.8', '--top-p', '.95', '--max-new-tokens', '128', '--include-greedy'], 'sampling')
    receipt=read(root/'sampling/sampling_receipt.json')
    assert receipt['candidate_count']==240 and receipt['parent_files_unchanged']
    assert receipt['candidates_sha256']==sha256_file(root/'sampling/candidates.jsonl')
    assert receipt['current_asr_sha256']==p['initial_adapter']['sha256']
    paired={x['audio_id'] for x in rows(root/'sampling/preferences.jsonl')}
    missing={x['audio_id'] for x in rows(root/'inputs.jsonl')}-paired
    save(root/'pair_availability.json',dict(actual_draws=240,paired_audio=len(paired),
        all_correct_case_ids=sorted(missing),complete_matching_available=not missing))
    verify(root)


def choose(root):
    cfg=verify(root)
    assert read(root/'pair_availability.json')['complete_matching_available'], 'No invented negative for all-correct audio'
    out=root/'selection';out.mkdir(exist_ok=False)
    groups=defaultdict(list)
    for row in rows(root/'sampling/preferences.jsonl'):groups[row['audio_id']].append(row)
    purposes={x['audio_id']:x for x in read(root/'purposes.json')}
    greedy={x['audio_id']:x['prediction'] for x in rows(root/'sampling/greedy.jsonl')}
    keys,audio_keys,cards={},{},[]
    for i,(audio_id,group) in enumerate(groups.items()):
        audio_key=f'a{i:02d}';audio_keys[audio_key]=audio_id;options=[]
        for j,row in enumerate(group):
            key=f'{audio_key}.n{j:02d}';keys[key]=row
            events=[dict(level=e.level,intended_unit=e.reference_unit,observed_unit=e.hypothesis_unit,
                intended_character=e.reference_character,observed_character=e.hypothesis_character)
                for e in actual_events(row['chosen'],row['rejected']) if e.context_sensitive is False]
            options.append(dict(negative_key=key,rejected=row['rejected'],sample_indices=row['sample_indices'],observed_stable_phone_events=events))
        cards.append(dict(audio_key=audio_key,target=group[0]['chosen'],reference_text=group[0]['metadata']['review']['reference_text'],
            greedy=greedy[audio_id],purpose=purposes[audio_id]['purpose'],exposures=purposes[audio_id]['exposures'],negatives=options))
    original=read(Path(cfg['MAS_reference_control'])/'all16/request.json')
    request=dict(original,input=[original['input'][0],dict(role='user',content=json.dumps(dict(audio_cards=cards),ensure_ascii=False))])
    save(out/'request.json',request)
    response=request_mas(RunProtocol.model_validate(cfg['source_protocol']),request,out)
    assert response['status']=='completed'
    final=[x for x in response['output'] if x.get('type')=='message' and x.get('phase')=='final_answer'];assert len(final)==1
    decision=Decision.model_validate_json(''.join(x['text'] for x in final[0]['content'] if x['type']=='output_text'))
    assert len({x.audio_key for x in decision.choices})==15 and {x.audio_key for x in decision.choices}==set(audio_keys)
    selected,mapping=[],[]
    for choice in decision.choices:
        row=keys[choice.negative_key];assert row['audio_id']==audio_keys[choice.audio_key]
        selected.append(dict(row,pair_weight=1.))
        mapping.append(dict(audio_id=row['audio_id'],source_pair_id=row['pair_id'],source_sample_indices=row['sample_indices'],reason=choice.reason))
    jsonl(out/'preferences.jsonl',selected);save(out/'mas_decision.json',decision.model_dump(mode='json'))
    save(out/'selection_receipt.json',dict(selections=mapping,changed_pair_fields=['pair_weight'],
        preferences_sha256=sha256_file(out/'preferences.jsonl'),actual_MAS_response_sha256=sha256_file(out/'response.json'),
        sampling_receipt_sha256=sha256_file(root/'sampling/sampling_receipt.json'),
        original_selector_request_sha256=sha256_file(Path(cfg['MAS_reference_control'])/'all16/request.json'),
        selector_system_schema_parameters_unchanged=True,selection_stochasticity_not_eliminated=True))
    verify(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','sample','choose']);parser.add_argument('base',type=Path)
    args=parser.parse_args();base=args.base.resolve();root=base/'patient_weighted_reference_control_v2'
    if args.stage=='prepare':prepare(base,root)
    elif args.stage=='sample':sample(root)
    else:choose(root)
