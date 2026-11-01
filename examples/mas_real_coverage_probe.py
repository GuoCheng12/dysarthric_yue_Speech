#!/usr/bin/env python3
"""MAS selects real retention coverage; matched audio-DPO arms test its value."""

import argparse
import ast
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_patient_retention_probe import request_mas
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file
from therapist_harness.dpo_training import group_training_rows, preference_key

TRAINER = REPO / 'finetune/scripts/qwen3_asr_coverage_kl_dpo.py'
ARMS = ['repeated_two', 'MAS_diverse']


class Coverage(BaseModel):
    model_config = ConfigDict(extra='forbid')
    focus_audio_count: Literal[12, 24, 48]
    candidate_keys: list[str] = Field(min_length=12, max_length=48)
    reason: str = Field(min_length=1, max_length=1500)
    hypothesis: str = Field(min_length=1, max_length=1000)


def verify(root):
    cfg = read(root/'protocol.json')
    assert all(sha256_file(Path(f)) == h for f,h in cfg['source_sha256'].items())
    return cfg


def choose(base, root):
    from openai.lib._pydantic import to_strict_json_schema
    catalog_path = base/'focus_retention_diversity_candidates_v1.json'
    catalog = read(catalog_path)
    assert all(sha256_file(Path(f)) == h for f,h in catalog['source_sha256'].items())
    p = RunProtocol.model_validate(read(base/'patient_weighted_reference_matched14_v1/protocol.json')['source_protocol'])
    root.mkdir(exist_ok=False)
    save(root/'choice_sources.json', {str(f):sha256_file(f) for f in [Path(__file__),
        catalog_path, base/'patient_retention_timing_probe_v1/results.json',
        base/'patient_weighted_reference_matched14_v1/MAS_reference/results.json',
        REPO/'examples/mas_patient_retention_probe.py']})
    request = dict(model=p.agent.model, reasoning={'effort':p.agent.reasoning_effort},
        store=False, stream=True, max_output_tokens=6000,
        input=[dict(role='system', content='''Select real-training content coverage for a MAS audio-DPO retention probe.
The primary method remains MAS -> TTS -> actual ASR negatives -> DPO. Same-parent
14-audio MAS reference/recipe selection achieved raw development CER20.703125%
versus random reference21.213942%; R-only20.793269%. MAS still regressed on hard
(697 versus688 edits) and on vlink247 (+16 edits), while other patients improved
(-22). No checkpoint was promoted. Spreading the existing two vlink247 retention
clips to every step removed gradient spikes but worsened CER to20.853365%, hard701.
There are413 vlink247 training recordings covering48 texts; the current real
retention pool has only2 clips for this patient. Choose12,24or48 distinct supplied
candidate keys, including BOTH original_focus_anchor clips, to test wider real
content coverage. Each supplied candidate represents one training transcript.
All new non-anchor clips were already positively exposed in initial R-only;
one original anchor was not, and remains shared between both arms. These are
training examples, not development examples. Duration is descriptive only.
Choose coverage based on the observed patient regression and linguistic variety,
not invented listening judgments. No audio admission or quality rule changes.
The matched control repeats the original2 clips; the diverse arm uses your set.
Both arms use8 focus exposures per step,96 focus exposures over12 steps, with
identical per-occurrence weight. Total focus KL weight is60, other patients60;
KL coefficient10,lr1e-4,beta0.1,seed42. The other94 real recordings keep all188
exposures in the same step/order/weight. Total real exposures284 in both arms;
the same14 MAS synthetic preference pairs and96 synthetic exposures are fixed.
Your chosen clips receive equal counts (8,4or2) over the96 focus slots.
Both arms start at initial R-only; frozen DPO/KL reference remains initial R-only.
Real labels supply teacher-forced histories for symmetric KL, not real CE.
This tests a new coverage hypothesis, with no guaranteed benefit. Return only
the structured selection, concise reason, and prospective hypothesis.'''),
            dict(role='user', content=json.dumps(catalog, ensure_ascii=False))],
        text={'format':{'type':'json_schema','name':'real_retention_coverage','strict':True,
            'schema':to_strict_json_schema(Coverage)}})
    save(root/'request.json',request)
    response=request_mas(p,request,root);assert response['status']=='completed'
    final=[x for x in response['output'] if x.get('type')=='message' and x.get('phase')=='final_answer']
    assert len(final)==1
    decision=Coverage.model_validate_json(''.join(x['text'] for x in final[0]['content'] if x['type']=='output_text'))
    keys=set(decision.candidate_keys); available={x['candidate_key']:x for x in catalog['candidates']}
    assert len(keys)==len(decision.candidate_keys)==decision.focus_audio_count and keys<=available.keys()
    assert {k for k,v in available.items() if v['original_focus_anchor']}<=keys
    save(root/'mas_decision.json',decision.model_dump(mode='json'))


def expected_exposures(real, synthetic, steps, allocation, coefficient):
    sys.path.insert(0,str(REPO/'finetune/scripts'))
    from qwen3_asr_scheduled_patient_kl_dpo import scheduled_windows
    groups=dict(real=group_training_rows(real,'sft')['real'],synthetic=group_training_rows(synthetic,'dpo')['synthetic'])
    windows=scheduled_windows(steps)(groups,24,len(allocation[0]),1.,42,synthetic_schedule=allocation)
    expected=[]
    for step in range(1,len(steps)+1):
        window=windows.next()
        assert windows.counts==dict(Counter(x['source'] for x in window))
        for row in window:
            expected.append(dict(step=step,source=row['source'],audio_id=row['_audio_group'],
                occurrence=row['_occurrence'],loss_weight=row['_loss_weight']*(coefficient if row['source']=='real' else 1),
                objective='real_symmetric_kl' if row['source']=='real' else 'dpo',preference_key=preference_key(row)))
    return expected


def compile_plan(base,root):
    assert all(sha256_file(Path(f))==h for f,h in read(root/'choice_sources.json').items())
    control=base/'patient_weighted_reference_matched14_v1/MAS_reference'
    p=read(control.parent/'protocol.json')['source_protocol']
    a=read(control/'training/run_protocol.json')['arguments']
    old=rows(control/'training/training_exposures.jsonl')
    real=rows(a['real_train_file']);synthetic=rows(a['synthetic_train_file'])
    allocation=read(a['synthetic_allocation_file'])['steps']
    catalog=read(base/'focus_retention_diversity_candidates_v1.json')
    decision=read(root/'mas_decision.json'); chosen=set(decision['candidate_keys'])
    candidates=[x for x in catalog['candidates'] if x['candidate_key'] in chosen]
    focus=sorted(x['audio_id'] for x in candidates if x['original_focus_anchor'])
    assert len(focus)==2 and len(real)==96 and len(synthetic)==14
    ordinary=[x for x in real if x['audio_id'] not in focus];assert len(ordinary)==94
    converted=[dict(source='real',audio_id=x['audio_id'],audio=x['audio'],audio_sha256=x['audio_sha256'],
        chosen=x['text'],prompt='',language='Cantonese') for x in candidates]
    for row in real:
        if row['audio_id'] in focus:assert row==next(x for x in converted if x['audio_id']==row['audio_id'])
    train_path=next(Path(f) for f in catalog['source_sha256'] if f.endswith('setting_d_train.jsonl'))
    training={str(Path(x['audio']).resolve()):x for x in rows(train_path)}
    dev=rows(p['feedback_split']['path']);dev_paths={str(Path(x['audio']).resolve()) for x in dev}
    dev_hashes={sha256_file(Path(x['audio'])) for x in dev};dev_ids={x['utt_id'] for x in dev}
    for row in ordinary+converted:
        path=str(Path(row['audio']).resolve());assert path in training and path not in dev_paths
        assert row['chosen']==training[path]['clean_gt']
        assert row['audio_id']==training[path]['utt_id'] and row['audio_id'] not in dev_ids
        assert sha256_file(Path(path))==row['audio_sha256'] and row['audio_sha256'] not in dev_hashes
    save(root/'isolation.json',dict(status='passed',real_training_pool=2117,development_feedback=511,
        checked_distinct_real_audio=len(ordinary+converted),paths_and_audio_hashes_disjoint=True,
        training_transcripts_exact=True,new_nonanchor_audio_already_positive_in_initial_parent=True,
        shared_original_unexposed_anchor=[x['audio_id'] for x in candidates if not x['positive_exposure_in_initial_R_only']]))
    all_expected={};arguments={};sources=[Path(__file__),TRAINER]
    for arm in ARMS:
        out=root/arm;out.mkdir()
        audio=real if arm=='repeated_two' else ordinary+converted
        jsonl(out/'real.jsonl',audio)
        ids=focus if arm=='repeated_two' else sorted(x['audio_id'] for x in candidates)
        order=list(ids);random.Random(42).shuffle(order)
        stream=(order*(96//len(order)));assert len(stream)==96
        steps=[]
        for step in range(1,13):
            nonfocus=[dict(audio_id=x['audio_id'],loss_weight=x['loss_weight']/10.) for x in old
                if x['step']==step and x['source']=='real' and x['audio_id'] not in focus]
            steps.append(nonfocus+[dict(audio_id=key,loss_weight=.0625) for key in stream[(step-1)*8:step*8]])
        save(out/'real_schedule.json',dict(steps=steps,focus_ids=ids,focus_exposures=96,real_exposures=284,
            focus_total_KL_weight=60,loss_weight_semantics='Coefficient before unchanged KL10; .625 full coefficient per focus occurrence.'))
        expected=expected_exposures(audio,synthetic,steps,allocation,10.)
        assert Counter(x['source'] for x in expected)=={'real':284,'synthetic':96}
        assert [x for x in expected if x['source']=='synthetic']==[x for x in old if x['source']=='synthetic']
        nonfocus=lambda data:[(x['step'],x['audio_id'],x['loss_weight'],x['preference_key']) for x in data if x['source']=='real' and x['audio_id'] not in set(ids+focus)]
        assert nonfocus(expected)==nonfocus(old)
        assert math.isclose(sum(x['loss_weight'] for x in expected if x['source']=='real'),120.,abs_tol=1e-8)
        save(out/'expected_exposures.json',expected);all_expected[arm]=expected
        arguments[arm]=dict(a,real_train_file=str(out/'real.jsonl'),output_dir=str(out/'training'),
            real_per_step=24,persist_teacher_cache=0,persist_optimizer_state=0,real_schedule=str(out/'real_schedule.json'))
        sources.extend([out/'real.jsonl',out/'real_schedule.json',out/'expected_exposures.json'])
    signature=lambda data:[(x['step'],x['source'],x['occurrence'],x['loss_weight']) for x in data]
    assert signature(all_expected[ARMS[0]])==signature(all_expected[ARMS[1]])
    engine=REPO/'finetune/scripts/qwen3_asr_kl_dpo_storage_v2.py';previous=engine.with_name('qwen3_asr_kl_dpo_probe.py')
    trees=[ast.parse(x.read_text()) for x in [previous,engine]]
    for name in ['transcript_distributions','symmetric_kl','probe']:
        fs=[next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
        assert ast.dump(fs[0])==ast.dump(fs[1])
    loops=[next(n for n in ast.walk(t) if isinstance(n,ast.With) and any(isinstance(v,ast.Constant) and v.value=='training_exposures.jsonl' for v in ast.walk(n))) for t in trees]
    assert ast.dump(loops[0])==ast.dump(loops[1])
    save(root/'cpu_smoke.json',dict(status='passed',actual_GPU_smoke=False,matched_all380_exposure_slots=True,
        ordinary188_step_order_weight_unchanged=True,synthetic96_records_identical=True,
        both_real_KL_total=120,focus_share=.5,numerical_helpers_and_full_optimizer_update_loop_AST_identical=True))
    sources += [engine,previous,train_path,Path(p['feedback_split']['path']),Path(a['real_train_file']),
        Path(a['synthetic_train_file']),Path(a['synthetic_allocation_file']),control/'training/run_protocol.json',
        control/'training/reference_log_probs.json',control/'training/training_exposures.jsonl',
        control/'training/real_teacher_cache.json',base/'parent_eval_check/summary.json']
    sources += [REPO/f for f in ['finetune/scripts/qwen3_asr_residual_kl_dpo.py',
        'finetune/scripts/qwen3_asr_residual_dpo.py','finetune/scripts/qwen3_asr_scheduled_patient_kl_dpo.py',
        'finetune/scripts/qwen3_asr_weights_only_patient_kl_dpo.py','finetune/scripts/qwen3_asr_audio_dpo.py',
        'src/therapist_harness/dpo_training.py','examples/evaluate_residual_asr.py','examples/mas_residual_dpo_probe.py']]
    sources += [root/n for n in ['choice_sources.json','request.json','response.json','mas_decision.json','isolation.json','cpu_smoke.json']]
    save(root/'protocol.json',dict(source_protocol=p,source_sha256={**read(root/'choice_sources.json'),**{str(f):sha256_file(f) for f in sources}},
        arguments=arguments,control=str(control),arms=ARMS,actual_new_generation=False,actual_new_sampling=False,
        actual_new_MAS_decision=True,actual_new_DPO_training=True,
        intervention='Focus real content coverage at matched focus budget, timing and exposure count; unchanged14 MAS synthetic pairs.',
        initial_parent_and_reference_same=True,real_CE=False,postprocessing=False))


def command(a):
    return [sys.executable,str(TRAINER)]+[item for key,value in a.items() if value is not None for item in ['--'+key,str(value)]]


def check_training(directory,expected=None):
    result=read(directory/'training/training_result.json');assert result['policy_changed'] and result['reference_files_unchanged']
    assert abs(result['history'][0]['weighted_losses']['real'])<1e-7
    assert math.isclose(result['history'][0]['weighted_losses']['synthetic'],math.log(2),abs_tol=1e-4)
    assert read(directory/'training/residual_verified.json')['status']=='passed'
    metadata=read(directory/'training/real_teacher_cache.json')
    assert metadata['tensor_data_persisted'] is False and metadata['cache_file_sha256'] is None
    assert metadata['actual_fresh_teacher_audio']==len(metadata['audio'])
    assert not (directory/'training/real_teacher_log_probs.pt').exists()
    assert not (Path(result['checkpoint'])/'optimizer.pt').exists()
    assert read(Path(result['checkpoint'])/'optimizer_omission.json')['optimizer_state_persisted'] is False
    if expected is not None:assert rows(directory/'training/training_exposures.jsonl')==expected
    return result,metadata


def smoke(root):
    cfg=verify(root);out=root/'smoke';out.mkdir()
    a=dict(cfg['arguments']['MAS_diverse']);real=rows(a['real_train_file'])
    real=real[:2]+real[-2:];synthetic=rows(a['synthetic_train_file'])[:2]
    jsonl(out/'real.jsonl',real);jsonl(out/'synthetic.jsonl',synthetic)
    allocation=[[x['audio_id'] for x in synthetic]]*2
    steps=[[dict(audio_id=x['audio_id'],loss_weight=.25) for x in real]]*2
    save(out/'allocation.json',dict(steps=allocation));save(out/'real_schedule.json',dict(steps=steps))
    a.update(real_train_file=str(out/'real.jsonl'),synthetic_train_file=str(out/'synthetic.jsonl'),
        synthetic_allocation_file=str(out/'allocation.json'),output_dir=str(out/'training'),
        real_schedule=str(out/'real_schedule.json'),max_steps=2,real_per_step=4,synthetic_per_step=2)
    execute(out,command(a),'training')
    result,metadata=check_training(out,expected_exposures(real,synthetic,steps,allocation,10.))
    assert result['history'][1]['weighted_losses']['real']>0
    save(out/'verified.json',dict(status='passed',actual_GPU_smoke=True,actual_steps=2,
        exact_exposure_schedule=True,initial_KL_zero_DPO_log2=True,second_step_real_KL_positive=True,
        frozen_parent_reference_verified=True,fresh_teacher_audio=len(metadata['audio']),
        teacher_tensor_hashes_only=True,adapter_weights_persisted=True,optimizer_hashes_only=True))
    verify(root)


def train(base,root,arm):
    cfg=verify(root);assert read(root/'smoke/verified.json')['status']=='passed'
    out=root/arm;execute(out,command(cfg['arguments'][arm]),'training')
    result,metadata=check_training(out,read(out/'expected_exposures.json'))
    old=read(Path(cfg['control'])/'training/real_teacher_cache.json')['audio']
    common=set(old)&set(metadata['audio'])
    assert len(common)==96
    for key in common:
        assert all(metadata['audio'][key][k]==old[key][k] for k in ['shape','tensor_sha256'])
    assert read(out/'training/reference_log_probs.json')['values']==read(Path(cfg['control'])/'training/reference_log_probs.json')['values']
    save(out/'verified.json',dict(status='passed',actual_training_exposures_match=True,
        all96_shared_teacher_tensors_identical=True,same14_DPO_reference_scores=True,actual_fresh_teacher_audio=len(metadata['audio'])))
    execute(out,evaluation_command(cfg['source_protocol'],out/'evaluation_run',Path(result['checkpoint'])),'evaluation')
    metrics=read(out/'evaluation_run/evaluation/summary.json')['metrics'];parent=read(base/'parent_eval_check/summary.json')['metrics']
    gate=dict(pooled_improves=metrics['overall']['pooled_cer']<parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer']-parent['overall']['patient_macro_cer']<=.001,
        hard_regression_within_limit=metrics['hard']['pooled_cer']-parent['hard']['pooled_cer']<=.005)
    save(out/'paired_parent.json',compare(base/'parent_eval_check/predictions.jsonl',out/'evaluation_run/evaluation/predictions.jsonl'))
    save(out/'results.json',dict(status='complete',metrics=metrics,gate_checks=gate,
        eligible_for_next_round_parent=all(gate.values()),actual_new_DPO_training=True,postprocessing_used=False))
    verify(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['choose','compile','smoke','train']);parser.add_argument('base',type=Path)
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();base=args.base.resolve();root=base/'real_coverage_dpo_probe_v1'
    if args.stage=='choose':choose(base,root)
    elif args.stage=='compile':compile_plan(base,root)
    elif args.stage=='smoke':smoke(root)
    else:
        assert args.arm is not None
        train(base,root,args.arm)
