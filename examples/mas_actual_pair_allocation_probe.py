#!/usr/bin/env python3
"""MAS selects observed corrective pairs and their actual training exposure."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_evaluation_report import compare
from mas_explicit_feedback_correction import explicit_feedback
from mas_residual_dpo_probe import evaluation_command
from mas_residual_retention_probe import run as run_retention
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file


class Choice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    audio_key: str
    negative_keys: list[str] = Field(min_length=1, max_length=4)
    exposures: int = Field(ge=1, le=96)
    purpose: Literal['primary', 'counterbalance', 'protection', 'maintenance']
    reason: str = Field(min_length=1, max_length=240)


class Decision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    choices: list[Choice] = Field(min_length=6, max_length=24)
    hypothesis: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)


def prepare(base, root):
    from openai.lib._pydantic import to_strict_json_schema

    root.mkdir(exist_ok=False)
    p = RunProtocol.model_validate(read(base / 'lexical96_mas_v1/protocol.json'))
    source = base / 'lexical96_mas_v1/round_001'
    original_path = source / 'sampling/preferences.jsonl'
    original = rows(original_path)
    grouped = defaultdict(list)
    for row in original:
        assert row['source'] == 'synthetic' and row['current_asr_sha256'] == p.initial_adapter.sha256
        grouped[row['audio_id']].append(row)
    sidecar = {x['sample_id']: x for x in read(source / 'materialization_sidecar.json')['samples']}
    greedy = {x['audio_id']: x['prediction'] for x in rows(source / 'sampling/greedy.jsonl')}
    family_path = base / 'actual_preference_target_family_overlap_v2.json'
    family = read(family_path)
    assert family['source_sha256'][str(original_path)] == sha256_file(original_path)
    family_by_pair = {x['pair_id']: x['matches'] for x in family['matches']}
    lexical_path = base / 'actual_preference_feedback_overlap_v1.json'
    lexical = read(lexical_path)['arms']['mas96']
    assert lexical['source_sha256'] == sha256_file(original_path)
    lexical_by_pair = {x['pair_id']: x['shared_substitutions'] for x in lexical['matches']}
    cards, audio_keys, negative_keys = [], {}, {}
    for i, (audio_id, group) in enumerate(grouped.items()):
        key = f'a{i:03d}'
        audio_keys[key] = audio_id
        options = []
        for j, row in enumerate(group):
            negative = f'{key}.n{j}'
            negative_keys[negative] = row
            options.append(dict(negative_key=negative, rejected=row['rejected'],
                actual_sample_indices=row['sample_indices'], target_family_events=family_by_pair.get(row['pair_id'], []),
                shared_real_character_errors=lexical_by_pair.get(row['pair_id'], [])))
        cards.append(dict(audio_key=key, intended_target=group[0]['chosen'], original_role=sidecar[audio_id]['role'],
            reference_difficulty_group=sidecar[audio_id]['reference_difficulty_group'],
            reference_text=group[0]['metadata']['review']['reference_text'], greedy=greedy[audio_id], negatives=options))
    save(root / 'key_map.json', dict(audio_keys=audio_keys,
        negative_keys={key: value['pair_id'] for key, value in negative_keys.items()}))
    request = dict(model=p.agent.model, reasoning={'effort': p.agent.reasoning_effort}, store=False, stream=True,
        max_output_tokens=16384, input=[dict(role='system', content='''Allocate an actual training batch for the Cantonese patient-ASR MAS using OBSERVED preference opportunities.
Select 6–24 distinct supplied audio cards. For each, select one or more of its
actual negative keys and assign a positive integer audio exposure count. The
counts MUST sum to exactly 96, to be shuffled with seed42 into 12 steps of eight.
You may leave other admitted audio and irrelevant negatives unused for this
training probe; that does not revoke their coarse review. Previously the loop
assigned training by written sound carriers, but the sampled errors often did
not involve those targets. Now use actual chosen/rejected differences to decide
the training purpose and budget. A useful primary opportunity may occur in an
audio originally labelled protection or maintenance. Retain some capability
maintenance/counterbalance while concentrating on genuine correction opportunities.
Do not force every originally labelled primary exercise into training. Fewer or
more errors is not itself better. Automatic family matches are conservative
textual EvidencePack attributions, not proof of audio pronunciation or exhaustive
coverage; read the full pairs and distinguish relevant errors from unrelated ones.
The primary real feedback is i→yu, protected m→n, and counterbalance i→e, with
z/easy and j/medium maintenance. The same 鉗→鉛 events support both i→yu and m→n;
these are not independent clinical mechanisms. Orthographic matches alone need
not address that main issue. Keep all full intended targets and actual rejects
unchanged. No invented errors, target replacement, CER admission threshold,
listening claim, word audit, or new quality scoring. All-correct audio have no
pair and are unavailable here. Both compared training branches use your exact
same selected pairs and exposure sequence. One is pure residual DPO; the other
adds the already defined real-training KL10 retention. Both start from the same
evaluated merged R-only and keep the initial R-only reference frozen. Give a
brief rationale per audio and a prospective overall hypothesis, not a claim of
benefit before real evaluation.'''), dict(role='user', content=json.dumps(dict(
            actual_target_family_summary=family['summary'],
            real_feedback=explicit_feedback(read(base / 'lexical_feedback_v1.json')),
            prior_results=dict(R_only_CER=20.79326923, all_pairs_residual_DPO_CER=20.94350962,
                all_pairs_residual_DPO_KL10_CER=20.85336538,
                previous_one_negative_per_every_audio='Did not improve; this intervention may now omit audio, select subsets of its negatives and change exposure allocation.',
                accepted_model='Still initial R-only'), audio_cards=cards), ensure_ascii=False))],
        text={'format': {'type': 'json_schema', 'name': 'observed_pair_allocation', 'strict': True,
                         'schema': to_strict_json_schema(Decision)}})
    save(root / 'request.json', request)
    response = serialize_response(execute_therapist_request(p, request))
    save(root / 'response.json', response)
    assert response['status'] == 'completed'
    final = [x for x in response['output'] if x.get('type') == 'message' and x.get('phase') == 'final_answer']
    assert len(final) == 1
    decision = Decision.model_validate_json(''.join(x['text'] for x in final[0]['content'] if x['type'] == 'output_text'))
    assert len({x.audio_key for x in decision.choices}) == len(decision.choices)
    assert sum(x.exposures for x in decision.choices) == 96
    save(root / 'mas_decision.json', decision.model_dump(mode='json'))
    selected, sequence, selected_receipt = [], [], []
    for choice in decision.choices:
        audio_id = audio_keys[choice.audio_key]
        assert len(set(choice.negative_keys)) == len(choice.negative_keys)
        group = [negative_keys[k] for k in choice.negative_keys]
        assert all(x['audio_id'] == audio_id for x in group)
        denominator = sum(x['pair_weight'] for x in group)
        for row in group:
            updated = dict(row, pair_weight=row['pair_weight'] / denominator)
            assert {k: v for k, v in row.items() if k != 'pair_weight'} == {k: v for k, v in updated.items() if k != 'pair_weight'}
            selected.append(updated)
        sequence.extend([audio_id] * choice.exposures)
        selected_receipt.append(dict(audio_id=audio_id, original_pair_ids=[x['pair_id'] for x in group],
            original_pair_weights=[x['pair_weight'] for x in group], normalization_denominator=denominator,
            purpose=choice.purpose, exposures=choice.exposures, reason=choice.reason))
    random.Random(42).shuffle(sequence)
    jsonl(root / 'preferences.jsonl', selected)
    save(root / 'allocation.json', dict(steps=[sequence[i:i+8] for i in range(0,96,8)],
        source='Actual MAS per-audio exposure counts; deterministic seed42 shuffle', total_audio_exposures=96))
    save(root / 'selection_receipt.json', dict(selections=selected_receipt, actual_new_sampling=False,
        original_preferences=str(original_path), original_preferences_sha256=sha256_file(original_path),
        original_sampling_receipt=str(source / 'sampling/sampling_receipt.json'),
        original_sampling_receipt_sha256=sha256_file(source / 'sampling/sampling_receipt.json'),
        changed_pair_fields=['pair_weight'], review_and_original_admission_unchanged=True))
    old = base / 'residual_retention_probe_v1'
    jsonl(root / 'real.jsonl', rows(old / 'real.jsonl'))
    jsonl(root / 'smoke_real.jsonl', rows(old / 'smoke_real.jsonl'))
    keys = list(dict.fromkeys(sequence))[:2]
    jsonl(root / 'smoke_preferences.jsonl', [x for x in selected if x['audio_id'] in keys])
    save(root / 'smoke_allocation.json', dict(steps=[keys, list(reversed(keys))]))
    files = [Path(__file__), REPO / 'examples/mas_residual_retention_probe.py',
        REPO / 'examples/mas_residual_dpo_probe.py', REPO / 'examples/evaluate_residual_asr.py',
        REPO / 'examples/mas_explicit_feedback_correction.py', REPO / 'examples/mas_preference_target_feedback.py',
        REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py', REPO / 'finetune/scripts/qwen3_asr_kl_dpo_probe.py',
        REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py', REPO / 'finetune/scripts/qwen3_asr_residual_kl_dpo.py',
        REPO / 'finetune/scripts/evaluate_asr_checkpoint.py', REPO / 'src/therapist_harness/dpo_training.py',
        original_path, source / 'sampling/sampling_receipt.json', source / 'sampling/greedy.jsonl',
        source / 'materialization_sidecar.json', family_path, lexical_path, base / 'lexical_feedback_v1.json',
        base / 'lexical96_mas_v1/protocol.json', old / 'real.jsonl', old / 'smoke_real.jsonl',
        old / 'protocol.json', old / 'real_selection_receipt.json',
        base / 'residual_dpo_probe_v1/initial/identity_verified.json'] + list(root.glob('*.json')) + list(root.glob('*.jsonl'))
    save(root / 'protocol.json', dict(source_protocol=p.model_dump(mode='json'),
        source_sha256={str(f): sha256_file(f) for f in files}, preferences=str(root / 'preferences.jsonl'),
        allocation=str(root / 'allocation.json'), sampling_receipt=str(source / 'sampling/sampling_receipt.json'),
        objective='Actual MAS-selected same-audio residual DPO; compare pure with fixed real-training KL10',
        selected_audio=len(decision.choices), selected_pairs=len(selected), steps=12,
        actual_new_generation=False, actual_new_sampling=False, actual_new_dpo_training=True,
        controls='Previously completed all-pair residual DPO and all-pair KL10; if promising, obtain a matched selection control before claiming MAS selection benefit.'))
    print(dict(selected_audio=len(decision.choices), selected_pairs=len(selected),
        purpose_exposures=dict(Counter({purpose:sum(x.exposures for x in decision.choices if x.purpose==purpose)
                                      for purpose in ['primary','counterbalance','protection','maintenance']}))), flush=True)


def verify(root):
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == digest for f, digest in cfg['source_sha256'].items())
    return cfg


def pure(base, root):
    cfg = verify(root)
    assert read(root / 'smoke/verified.json')['status'] == 'passed'
    d = root / 'pure'
    d.mkdir(exist_ok=False)
    p = cfg['source_protocol']
    execute(d, [sys.executable, str(REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py'),
        '--model_path',p['base_model_dir'],'--policy_adapter_path',p['initial_adapter']['path'],
        '--reference_adapter_path',p['initial_adapter']['path'],'--train_file',cfg['preferences'],
        '--synthetic_allocation_file',cfg['allocation'],'--output_dir',str(d/'training'),
        '--max_steps','12','--real_per_step','0','--synthetic_per_step','8','--synthetic_weight','1',
        '--lr','1e-5','--beta','.1','--seed','42'], 'training')
    result=read(d/'training/training_result.json')
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert math.isclose(result['history'][0]['loss'],math.log(2),abs_tol=1e-4)
    assert read(d/'training/residual_verified.json')['status']=='passed'
    execute(d,evaluation_command(p,d/'evaluation_run',d/'training/checkpoint-12'),'evaluation')
    metrics=read(d/'evaluation_run/evaluation/summary.json')['metrics']
    parent=read(base/'parent_eval_check/summary.json')['metrics']
    gate=dict(pooled_improves=metrics['overall']['pooled_cer']<parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer']-parent['overall']['patient_macro_cer']<=.001,
        hard_regression_within_limit=metrics['hard']['pooled_cer']-parent['hard']['pooled_cer']<=.005)
    save(d/'paired_parent.json',compare(Path(p['initial_adapter_dev_predictions']['path']),d/'evaluation_run/evaluation/predictions.jsonl'))
    save(d/'results.json',dict(status='complete',metrics=metrics,gate_checks=gate,
        eligible_for_next_round_parent=all(gate.values()),actual_new_dpo_training=True,postprocessing_used=False))
    verify(root)


def supervise(base,root):
    run_retention(base,root,10.,smoke=True)
    jobs=[]
    for gpu,stage in enumerate(['pure','retention']):
        command=[sys.executable,str(Path(__file__).resolve()),stage,str(base)]
        with (root/(stage+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)))
        save(root/(stage+'.launch.json'),dict(pid=child.pid,gpu=gpu,command=command));jobs.append(child)
    codes=[p.wait() for p in jobs]
    assert codes==[0,0],codes
    save(root/'results.json',dict(status='complete',arms={a:read(root/a/'results.json') for a in ['pure','weight_10']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','pure','retention','supervise'])
    parser.add_argument('base',type=Path)
    args=parser.parse_args();base=args.base.resolve();root=base/'actual_pair_allocation_probe_v1'
    if args.stage=='prepare':prepare(base,root)
    elif args.stage=='pure':pure(base,root)
    elif args.stage=='retention':run_retention(base,root,10.)
    else:supervise(base,root)
