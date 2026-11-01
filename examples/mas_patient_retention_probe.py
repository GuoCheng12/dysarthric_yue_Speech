#!/usr/bin/env python3
"""MAS allocates a fixed real-retention budget from actual patient responses."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, execute, read, rows, save
from mas_evaluation_report import compare, rows as evaluated_rows
from mas_residual_dpo_probe import evaluation_command
from therapist_harness.agent import serialize_response
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file
from therapist_harness.dpo_training import SourceBalancedWindows, group_training_rows

sys.path.insert(0, str(REPO / 'finetune/scripts'))
from qwen3_asr_patient_weighted_kl_dpo import weighted_window_type

TRAINER = REPO / 'finetune/scripts/qwen3_asr_patient_weighted_kl_dpo.py'


class Allocation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    patient_key: str
    focus_share: float = Field(ge=.1, le=.5)
    reason: str = Field(min_length=1, max_length=500)
    hypothesis: str = Field(min_length=1, max_length=500)


def request_mas(p, request, root):
    from httpx import Timeout
    from openai import OpenAI

    client = OpenAI(api_key=os.environ[p.agent.api_key_env], base_url=str(p.agent.base_url),
        max_retries=p.agent.max_retries, timeout=Timeout(float(p.agent.timeout_seconds), connect=20.),
        default_headers={'User-Agent':'curl/8.0'})
    start = time.monotonic()
    response = None
    try:
        with (root / 'api_events.jsonl').open('x') as log:
            with client.responses.create(**request) as stream:
                for event in stream:
                    record = dict(event_type=event.type, seconds=time.monotonic()-start,
                        sequence_number=getattr(event,'sequence_number',None))
                    log.write(json.dumps(record)+'\n');log.flush()
                    if event.type in {'response.completed','response.failed','response.incomplete'}:
                        response = serialize_response(event.response)
                        save(root / 'api_terminal.json', dict(**record, response_id=response['id'], status=response['status']))
                        break
    except Exception as error:
        save(root / 'api_failure.json', dict(exception_type=type(error).__name__, seconds=time.monotonic()-start,
            message=str(error)[:500]))
        raise
    finally:
        client.close()
    if response is None:
        raise RuntimeError('No actual terminal MAS response')
    save(root / 'response.json', response)
    return response


def cpu_smoke(real, synthetic, allocation, factors, focus_ids, alpha):
    groups = dict(real=group_training_rows(real,'sft')['real'], synthetic=group_training_rows(synthetic,'dpo')['synthetic'])
    plain = SourceBalancedWindows(groups,16,8,1.,42,synthetic_schedule=allocation['steps'])
    weighted = weighted_window_type(factors)(groups,16,8,1.,42,synthetic_schedule=allocation['steps'])
    identity = weighted_window_type({k:1. for k in factors})(groups,16,8,1.,42,synthetic_schedule=allocation['steps'])
    counts = Counter();real_mass = focus_mass = 0.
    for _ in range(12):
        before, after, unchanged = plain.next(), weighted.next(), identity.next()
        assert unchanged == before
        assert len(before) == len(after)
        for a,b in zip(before,after):
            assert {k:v for k,v in a.items() if k!='_loss_weight'} == {k:v for k,v in b.items() if k!='_loss_weight'}
            factor = factors[a['audio_id']] if a['source']=='real' else 1.
            assert math.isclose(b['_loss_weight'], a['_loss_weight']*factor, abs_tol=1e-12)
            counts[a['source']] += 1
            if a['source']=='real':
                real_mass += b['_loss_weight']
                if a['audio_id'] in focus_ids:focus_mass += b['_loss_weight']
    assert counts == {'real':192,'synthetic':96}
    assert math.isclose(real_mass,12.,abs_tol=1e-10)
    assert math.isclose(focus_mass/real_mass,alpha,abs_tol=1e-10)
    return dict(status='passed', actual_gpu_training=False, identity_factors_preserve_original_windows=True,
        unchanged_actual_audio_order_and_nonweight_fields=True, counts=dict(counts),
        real_loss_mass_before_KL_coefficient=real_mass, realized_focus_share=focus_mass/real_mass,
        scope='CPU smoke of the new weighting operation on all actual12 training windows; existing GPU KL/DPO smoke is separately reused.')


def prepare(base, root):
    from openai.lib._pydantic import to_strict_json_schema

    root.mkdir(exist_ok=False)
    control = base / 'strong_retention_probe_v1/weight_10'
    old = read(base / 'strong_retention_probe_v1/protocol.json')
    p = RunProtocol.model_validate(old['source_protocol'])
    a = read(control / 'training/run_protocol.json')['arguments']
    before_path = base / 'parent_eval_check/predictions.jsonl'
    after_path = control / 'evaluation_run/evaluation/predictions.jsonl'
    before = evaluated_rows(before_path);after = {x['utt_id']:x for x in evaluated_rows(after_path)}
    real = rows(a['real_train_file']);synthetic = rows(a['synthetic_train_file'])
    allocation = read(a['synthetic_allocation_file'])
    assert len(real)==96 and len(synthetic)==15
    real_counts = Counter(x['audio_id'].split('/')[0] for x in real)
    selected_ids = {x['audio_id'] for x in synthetic}
    inputs = rows(base / 'candidate_coverage_recovery_v2/sampling/validated_inputs.jsonl')
    reference_counts = Counter(Path(x['reference_wav']).parent.name for x in inputs if x['audio_id'] in selected_ids)
    patient_rows = defaultdict(list)
    for x in before:
        y=after[x['utt_id']]
        assert all(x[k]==y[k] for k in ['audio','speaker_id','clean_gt','normalized_reference_characters'])
        patient_rows[x['speaker_id']].append((x,y))
    keys, cards = {}, []
    for i,(speaker,data) in enumerate(sorted(patient_rows.items())):
        key=f'p{i:02d}';keys[key]=speaker
        old_edits=sum(x['edit_count'] for x,y in data);new_edits=sum(y['edit_count'] for x,y in data)
        chars=sum(x['normalized_reference_characters'] for x,y in data)
        cards.append(dict(patient_key=key,feedback_audio=len(data),reference_characters=chars,
            parent_edits=old_edits,current_edits=new_edits,edit_delta=new_edits-old_edits,
            parent_CER=old_edits/chars,current_CER=new_edits/chars,
            existing_real_retention_audio=real_counts[speaker],current_TTS_reference_audio=reference_counts[speaker],
            eligible_focus=new_edits>old_edits and real_counts[speaker]>0))
    save(root / 'patient_key_map.json',keys);save(root / 'patient_cards.json',cards)
    files = [Path(__file__),TRAINER,REPO/'finetune/scripts/qwen3_asr_shared_teacher_kl_dpo.py',
        REPO/'finetune/scripts/qwen3_asr_residual_kl_dpo.py',REPO/'finetune/scripts/qwen3_asr_kl_dpo_probe.py',
        REPO/'src/therapist_harness/dpo_training.py',REPO/'examples/evaluate_residual_asr.py',
        before_path,after_path,control/'training/run_protocol.json',control/'training/training_exposures.jsonl',
        control/'training/reference_log_probs.json',control/'results.json',control/'target_response.json',
        Path(a['real_train_file']),Path(a['synthetic_train_file']),Path(a['synthetic_allocation_file']),
        base/'strong_retention_probe_v1/protocol.json',base/'strong_retention_probe_v1/extended_verification.json',
        base/'patient_retention_gap_v1.json',base/'residual_retention_probe_v1/smoke/verified.json',
        Path(old['verified_teacher_cache']),Path(old['verified_teacher_metadata']),root/'patient_key_map.json',root/'patient_cards.json']
    hashes={str(f):sha256_file(f) for f in files}
    save(root/'input_sources.json',hashes)
    request=dict(model=p.agent.model,reasoning={'effort':p.agent.reasoning_effort},store=False,stream=True,max_output_tokens=4096,
        input=[dict(role='system',content='''Choose one patient to protect in a MAS ASR training-allocation probe from actual real-development responses.
Pick an eligible_focus patient from the supplied anonymous cards and assign it a
share between0.10 and0.50 of the existing real-retention KL loss budget. The share
must exceed its current share (existing_real_retention_audio/96). All96 real
training recordings remain, with exactly the same192 exposures and order; all
other recordings share the remaining budget uniformly per audio. Total real KL
weight over the run stays fixed. No extra data, generation, sampling, label edit,
test exclusion, CER admission cutoff, listening claim or new quality rating.
The synthetic part remains15 actual MAS-selected same-audio DPO pairs with96
exposures,12steps,lr1e-4,beta0.1,KL coefficient10,seed42. Both the current control
and this new branch start from the same original R-only, which also stays the
frozen DPO/KL reference. The existing real KL uses ground-truth prefixes, no real
cross entropy. Current DPO+KL10 recovers25 original i→yu target positions and has
123 insertions in their sentence cohort versus107 for R-only. Its pooled CER is
21.048678% versus20.793269%, despite improved patient-macro CER18.787513% versus
19.107915%. Use the actual edit contributions to balance protecting regression
against retaining successful corrections. Diagnosis/acoustics are not inferred
from identifiers. This is exploratory feedback, not sealed-test evidence. The
outcome is unknown until real evaluation; return a brief reason and hypothesis.'''),
            dict(role='user',content=json.dumps(dict(patient_cards=cards,real_pool_audio=96,
                current_uniform_real_weight_per_audio=1/96),ensure_ascii=False))],
        text={'format':{'type':'json_schema','name':'patient_retention_allocation','strict':True,
            'schema':to_strict_json_schema(Allocation)}})
    save(root/'request.json',request)
    response=request_mas(p,request,root)
    assert response['status']=='completed'
    final=[x for x in response['output'] if x.get('type')=='message' and x.get('phase')=='final_answer'];assert len(final)==1
    decision=Allocation.model_validate_json(''.join(x['text'] for x in final[0]['content'] if x['type']=='output_text'))
    card=next(x for x in cards if x['patient_key']==decision.patient_key);assert card['eligible_focus']
    speaker=keys[decision.patient_key];focus_ids={x['audio_id'] for x in real if x['audio_id'].split('/')[0]==speaker}
    n=len(focus_ids);alpha=decision.focus_share;assert alpha>n/96
    factors={x['audio_id']:(alpha*96/n if x['audio_id'] in focus_ids else (1-alpha)*96/(96-n)) for x in real}
    save(root/'mas_decision.json',decision.model_dump(mode='json'))
    save(root/'real_loss_factors.json',dict(factors=factors,focus_ids=sorted(focus_ids),focus_patient_key=decision.patient_key,
        focus_share=alpha,uniform_control_focus_share=n/96,normalization='mean factor across96 real audio=1'))
    save(root/'cpu_smoke.json',cpu_smoke(real,synthetic,allocation,factors,focus_ids,alpha))
    for f in ['input_sources.json','request.json','response.json','mas_decision.json','real_loss_factors.json','cpu_smoke.json']:
        hashes[str(root/f)]=sha256_file(root/f)
    assert all(sha256_file(Path(f))==h for f,h in hashes.items())
    save(root/'protocol.json',dict(source_protocol=p.model_dump(mode='json'),source_sha256=hashes,
        control=str(control),arguments=dict(a,output_dir=str(root/'training')),
        verified_teacher_cache=old['verified_teacher_cache'],verified_teacher_metadata=old['verified_teacher_metadata'],
        intervention='Only normalized per-audio real KL weights differ from completed KL10 control; same data,exposures,parent/reference and objective components.',
        actual_new_generation=False,actual_new_sampling=False,actual_new_dpo_training=True,
        acceptance='Existing raw pooled/macro/hard gate; no inference postprocessing or patient exclusion.'))
    print(dict(focus_patient_key=decision.patient_key,focus_share=alpha,real_audio=n,CPU_smoke='passed'),flush=True)


def run(base,root):
    cfg=read(root/'protocol.json');assert all(sha256_file(Path(f))==h for f,h in cfg['source_sha256'].items())
    assert read(root/'cpu_smoke.json')['status']=='passed'
    a=cfg['arguments'];command=[sys.executable,str(TRAINER)]
    for k,v in a.items():
        if v is not None:command.extend(['--'+k,str(v)])
    command.extend(['--real_loss_factors',str(root/'real_loss_factors.json'),
        '--verified_teacher_cache',cfg['verified_teacher_cache'],'--verified_teacher_metadata',cfg['verified_teacher_metadata']])
    execute(root,command,'training')
    result=read(root/'training/training_result.json');assert result['policy_changed'] and result['reference_files_unchanged']
    assert abs(result['history'][0]['weighted_losses']['real'])<1e-7
    assert math.isclose(result['history'][0]['weighted_losses']['synthetic'],math.log(2),abs_tol=1e-4)
    assert read(root/'training/residual_verified.json')['status']=='passed'
    old=Path(cfg['control'])/'training';actual=rows(root/'training/training_exposures.jsonl');control=rows(old/'training_exposures.jsonl')
    w=read(root/'real_loss_factors.json');factors=w['factors'];focus=set(w['focus_ids'])
    assert len(actual)==len(control)==288
    for x,y in zip(actual,control):
        assert {k:v for k,v in x.items() if k!='loss_weight'}=={k:v for k,v in y.items() if k!='loss_weight'}
        assert math.isclose(x['loss_weight'],y['loss_weight']*(factors[x['audio_id']] if x['source']=='real' else 1),abs_tol=1e-10)
    real=[x for x in actual if x['source']=='real'];total=sum(x['loss_weight'] for x in real)
    assert math.isclose(total,sum(x['loss_weight'] for x in control if x['source']=='real'),abs_tol=1e-9)
    assert math.isclose(sum(x['loss_weight'] for x in real if x['audio_id'] in focus)/total,w['focus_share'],abs_tol=1e-10)
    assert read(root/'training/reference_log_probs.json')['values']==read(old/'reference_log_probs.json')['values']
    assert read(root/'training/teacher_storage_reuse.json')['identical_actual_tensor_values_verified']==96
    save(root/'verified.json',dict(status='passed',same_actual_real_synthetic_order=True,same_total_real_KL_weight=True,
        realized_focus_share=w['focus_share'],actual_factor_application_verified=True,same_frozen_reference_scores=True,
        all96fresh_teacher_tensors_verified=True,initial_KL_zero_and_DPO_log2=True))
    p=cfg['source_protocol'];execute(root,evaluation_command(p,root/'evaluation_run',root/'training/checkpoint-12'),'evaluation')
    metrics=read(root/'evaluation_run/evaluation/summary.json')['metrics'];parent=read(base/'parent_eval_check/summary.json')['metrics']
    gate=dict(pooled_improves=metrics['overall']['pooled_cer']<parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer']-parent['overall']['patient_macro_cer']<=.001,
        hard_regression_within_limit=metrics['hard']['pooled_cer']-parent['hard']['pooled_cer']<=.005)
    save(root/'paired_parent.json',compare(Path(p['initial_adapter_dev_predictions']['path']),root/'evaluation_run/evaluation/predictions.jsonl'))
    save(root/'paired_uniform_control.json',compare(Path(cfg['control'])/'evaluation_run/evaluation/predictions.jsonl',root/'evaluation_run/evaluation/predictions.jsonl'))
    save(root/'results.json',dict(status='complete',metrics=metrics,gate_checks=gate,eligible_for_next_round_parent=all(gate.values()),
        actual_new_dpo_training=True,postprocessing_used=False))
    assert all(sha256_file(Path(f))==h for f,h in cfg['source_sha256'].items())
    print(dict(status='complete',overall=metrics['overall'],gate=gate),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run']);parser.add_argument('base',type=Path)
    args=parser.parse_args();base=args.base.resolve();root=base/'patient_retention_probe_v1'
    if args.stage=='prepare':prepare(base,root)
    else:run(base,root)
