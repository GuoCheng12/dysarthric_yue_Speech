#!/usr/bin/env python3
"""MAS chooses continuous retention timing without changing the real loss inventory."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, execute, read, rows, save
from mas_patient_retention_probe import request_mas
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file
from therapist_harness.dpo_training import group_training_rows, preference_key

sys.path.insert(0, str(REPO / 'finetune/scripts'))
from qwen3_asr_scheduled_patient_kl_dpo import scheduled_windows

TRAINER = REPO / 'finetune/scripts/qwen3_asr_scheduled_patient_kl_dpo.py'


class Timing(BaseModel):
    model_config = ConfigDict(extra='forbid')
    focus_audio_per_step: Literal[1, 2]
    reason: str = Field(min_length=1, max_length=600)
    hypothesis: str = Field(min_length=1, max_length=600)


def verify(root):
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == h for f, h in cfg['source_sha256'].items())
    return cfg


def smoke(real, synthetic, schedule, allocation, old_exposures, focus, coefficient):
    groups = dict(real=group_training_rows(real,'sft')['real'], synthetic=group_training_rows(synthetic,'dpo')['synthetic'])
    windows = scheduled_windows(schedule)(groups,16,8,1.,42,synthetic_schedule=allocation['steps'])
    actual=[]
    for step in range(1,13):
        window=windows.next()
        assert windows.counts==dict(Counter(x['source'] for x in window))
        for row in window:
            actual.append(dict(step=step,source=row['source'],audio_id=row['_audio_group'],occurrence=row['_occurrence'],
                loss_weight=row['_loss_weight']*(coefficient if row['source']=='real' else 1),
                objective='real_symmetric_kl' if row['source']=='real' else 'dpo',preference_key=preference_key(row)))
    before,after=Counter(),Counter()
    for record in old_exposures:
        if record['source']=='real':before[record['audio_id']]+=record['loss_weight']
    for record in actual:
        if record['source']=='real':after[record['audio_id']]+=record['loss_weight']
    assert before.keys()==after.keys() and all(math.isclose(before[k],after[k],abs_tol=1e-9) for k in before)
    ordinary=lambda data:[(x['step'],x['audio_id'],x['loss_weight'],x['preference_key']) for x in data if x['source']=='real' and x['audio_id'] not in focus]
    assert ordinary(actual)==ordinary(old_exposures)
    assert [x for x in actual if x['source']=='synthetic']==[x for x in old_exposures if x['source']=='synthetic']
    assert all(any(x['step']==s and x['audio_id'] in focus for x in actual) for s in range(1,13))
    return actual,dict(status='passed',actual_new_GPU_smoke=False,actual_new_CPU_smoke=True,
        all96_real_audio_total_loss_weights_unchanged=True,ordinary188_real_exposures_step_order_weight_unchanged=True,
        synthetic96_exposure_records_identical=True,counts=dict(Counter(x['source'] for x in actual)),
        total_real_KL_loss_weight=sum(after.values()),focus_share=sum(after[k] for k in focus)/sum(after.values()),
        objective_equivalence='At any fixed model, the summed real KL objective over the12 windows is identical, because every audio retains its total coefficient. SGD timing differs intentionally.',
        scope='Full actual12-window CPU schedule check; unchanged KL/DPO arithmetic reuses the recorded earlier actual GPU smoke.')


def prepare(base, root):
    from openai.lib._pydantic import to_strict_json_schema
    control=base/'patient_weighted_reference_matched14_v1/MAS_reference'
    original=read(control.parent/'protocol.json');p=RunProtocol.model_validate(original['source_protocol'])
    a=read(control/'training/run_protocol.json')['arguments'];old=rows(control/'training/training_exposures.jsonl')
    observation=read(base/'patient_retention_timing_diagnostic_v1.json')
    focus=sorted(observation['focus_ids']);assert len(focus)==2
    files=[Path(__file__),TRAINER,REPO/'finetune/scripts/qwen3_asr_weights_only_patient_kl_dpo.py',
        REPO/'finetune/scripts/qwen3_asr_shared_teacher_kl_dpo.py',REPO/'finetune/scripts/qwen3_asr_residual_kl_dpo.py',
        REPO/'finetune/scripts/qwen3_asr_kl_dpo_probe.py',REPO/'src/therapist_harness/dpo_training.py',
        REPO/'examples/mas_patient_retention_probe.py',REPO/'examples/evaluate_residual_asr.py',
        base/'patient_retention_timing_diagnostic_v1.json',control.parent/'protocol.json',control.parent/'cross_arm_verified.json',
        control/'training/run_protocol.json',control/'training/training_exposures.jsonl',control/'training/training_result.json',
        control/'training/reference_log_probs.json',control/'results.json',control/'target_response.json',
        Path(a['real_train_file']),Path(a['synthetic_train_file']),Path(a['synthetic_allocation_file']),
        Path(original['verified_teacher_cache']),Path(original['verified_teacher_metadata']),
        base/'residual_retention_probe_v1/smoke/verified.json']
    hashes={str(f):sha256_file(f) for f in files}
    root.mkdir(exist_ok=False);save(root/'input_sources.json',hashes)
    request=dict(model=p.agent.model,reasoning={'effort':p.agent.reasoning_effort},store=False,stream=True,max_output_tokens=4096,
        input=[dict(role='system',content='''Choose the timing of patient retention in a MAS audio-DPO training probe.
Current same-patient MAS reference/recipe selection gives raw development CER
20.703125% versus21.213942% for random selection, with matched14 audio,10 texts,
96 synthetic exposures,12steps,lr1e-4,beta0.1,KL10. R-only is20.793269%.
The MAS arm improves overall but hard has697 edits versus688; the fixed .5pp
hard limit is missed. Its focus patient has16 extra edits while other patients
have22 fewer. No inference postprocessing is used and no checkpoint was promoted.
The focus patient's50% total real-KL budget is concentrated in steps2,5,7; step7
contains both focus clips and the largest observed gradient norm. The final5
steps have no focus retention. This is an observed scheduling gap, not proof
that it causes regression. There are exactly2 existing focus training clips.
Choose1 or2 focus clips PER STEP for a prospective continuous-retention probe.
If1, the two clips alternate, each appearing6 times; if2, both appear at all12
steps. Each clip retains exactly its old total KL loss coefficient, divided
over its new occurrences. Every other real exposure stays in its original
step/order with unchanged weight. The same96 distinct real recordings remain.
Actual real exposures increase from192 to200 or212; total real KL weight and
the fixed-model summed objective do not increase. This tests timing, not added
data or stronger regularization. Synthetic data, negatives,96 exposure records,
lr,steps,beta,KL coefficient and seed remain unchanged. Both current control and
new branch start at initial R-only; DPO/KL reference also stays initial R-only.
No new generation, sampling, label edits, quality scores, CER admission thresholds
or listening claims. Return the clips-per-step choice and brief reason/hypothesis.
The outcome is unknown until the original full real-development gate is applied.'''),
            dict(role='user',content=json.dumps(dict(actual_step_observations=observation['steps'],
                focus_clip_labels=['A','B'],real_audio_pool=96,old_focus_exposures=4),ensure_ascii=False))],
        text={'format':{'type':'json_schema','name':'patient_retention_timing','strict':True,'schema':to_strict_json_schema(Timing)}})
    save(root/'request.json',request);response=request_mas(p,request,root);assert response['status']=='completed'
    final=[x for x in response['output'] if x.get('type')=='message' and x.get('phase')=='final_answer'];assert len(final)==1
    decision=Timing.model_validate_json(''.join(x['text'] for x in final[0]['content'] if x['type']=='output_text'))
    save(root/'mas_decision.json',decision.model_dump(mode='json'))
    count=decision.focus_audio_per_step;coefficient=a['kl_weight'];totals=Counter()
    for x in old:
        if x['source']=='real':totals[x['audio_id']]+=x['loss_weight']/coefficient
    steps=[]
    for step in range(1,13):
        ordinary=[dict(audio_id=x['audio_id'],loss_weight=x['loss_weight']/coefficient) for x in old
            if x['step']==step and x['source']=='real' and x['audio_id'] not in focus]
        chosen=focus if count==2 else [focus[(step-1)%2]]
        steps.append(ordinary+[dict(audio_id=key,loss_weight=totals[key]/(12 if count==2 else 6)) for key in chosen])
    save(root/'real_schedule.json',dict(steps=steps,focus_ids=focus,focus_audio_per_step=count,
        focus_exposures=12*count,real_exposures=188+12*count,old_real_exposures=192,
        loss_weight_semantics='Per-audio loss coefficient before multiplying the unchanged KL10; per-audio12-step total preserved.'))
    expected,report=smoke(rows(a['real_train_file']),rows(a['synthetic_train_file']),steps,
        read(a['synthetic_allocation_file']),old,set(focus),coefficient)
    save(root/'cpu_smoke.json',report);save(root/'expected_exposures.json',expected)
    for name in ['input_sources.json','request.json','response.json','mas_decision.json','real_schedule.json','cpu_smoke.json','expected_exposures.json']:
        hashes[str(root/name)]=sha256_file(root/name)
    assert all(sha256_file(Path(f))==h for f,h in hashes.items())
    save(root/'protocol.json',dict(source_protocol=p.model_dump(mode='json'),source_sha256=hashes,
        arguments=dict(a,output_dir=str(root/'training')),control=str(control),
        verified_teacher_cache=original['verified_teacher_cache'],verified_teacher_metadata=original['verified_teacher_metadata'],
        actual_new_generation=False,actual_new_sampling=False,actual_new_MAS_decision=True,actual_new_DPO_training=True,
        intervention='Redistribute the existing focus patient per-audio KL mass to every update; unchanged global per-audio objective, other real steps/order/weights and all synthetic exposure records.'))


def run(base,root):
    cfg=verify(root);assert read(root/'cpu_smoke.json')['status']=='passed'
    command=[sys.executable,str(TRAINER)]
    for key,value in cfg['arguments'].items():
        if value is not None:command.extend(['--'+key,str(value)])
    command.extend(['--real_schedule',str(root/'real_schedule.json'),'--verified_teacher_cache',cfg['verified_teacher_cache'],
        '--verified_teacher_metadata',cfg['verified_teacher_metadata']])
    execute(root,command,'training')
    result=read(root/'training/training_result.json');assert result['policy_changed'] and result['reference_files_unchanged']
    assert abs(result['history'][0]['weighted_losses']['real'])<1e-7
    assert math.isclose(result['history'][0]['weighted_losses']['synthetic'],math.log(2),abs_tol=1e-4)
    assert read(root/'training/residual_verified.json')['status']=='passed'
    assert rows(root/'training/training_exposures.jsonl')==read(root/'expected_exposures.json')
    assert read(root/'training/reference_log_probs.json')['values']==read(Path(cfg['control'])/'training/reference_log_probs.json')['values']
    assert read(root/'training/teacher_storage_reuse.json')['identical_actual_tensor_values_verified']==96
    save(root/'verified.json',dict(status='passed',all_actual_exposures_match_CPU_plan=True,same_actual_DPO_reference_scores=True,
        all96_fresh_teacher_tensors_verified=True,initial_KL_zero_DPO_log2=True,per_audio_total_real_KL_weight_preserved=True))
    execute(root,evaluation_command(cfg['source_protocol'],root/'evaluation_run',root/'training/checkpoint-12'),'evaluation')
    metrics=read(root/'evaluation_run/evaluation/summary.json')['metrics'];parent=read(base/'parent_eval_check/summary.json')['metrics']
    gate=dict(pooled_improves=metrics['overall']['pooled_cer']<parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer']-parent['overall']['patient_macro_cer']<=.001,
        hard_regression_within_limit=metrics['hard']['pooled_cer']-parent['hard']['pooled_cer']<=.005)
    save(root/'paired_parent.json',compare(base/'parent_eval_check/predictions.jsonl',root/'evaluation_run/evaluation/predictions.jsonl'))
    save(root/'paired_timing_control.json',compare(Path(cfg['control'])/'evaluation_run/evaluation/predictions.jsonl',root/'evaluation_run/evaluation/predictions.jsonl'))
    save(root/'results.json',dict(status='complete',metrics=metrics,gate_checks=gate,eligible_for_next_round_parent=all(gate.values()),
        actual_new_DPO_training=True,postprocessing_used=False))
    verify(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run']);parser.add_argument('base',type=Path)
    args=parser.parse_args();base=args.base.resolve();root=base/'patient_retention_timing_probe_v1'
    if args.stage=='prepare':prepare(base,root)
    else:run(base,root)
