#!/usr/bin/env python3
"""Matched reference/recipe comparison with real patient retention and audio DPO."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_candidate_coverage_probe import Choice
from mas_patient_retention_probe import request_mas, cpu_smoke
from mas_preference_target_feedback import actual_events
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from therapist_harness.preference_data import build_preference_pairs
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file

ARMS = ['MAS_reference', 'random_reference']
TRAINER = REPO / 'finetune/scripts/qwen3_asr_weights_only_patient_kl_dpo.py'
INSTRUCTIONS = '''Select one actual ASR error per audio for a matched MAS Cantonese audio-DPO probe.
All14 audio have passed the unchanged simple three-text coarse review. Their
intended targets remain chosen; word fidelity has not been established. Do not
change targets, invent errors, rerate admission, infer listening, or add a CER
threshold. Stable phone events supplied below describe textual substitutions.
Prefer useful localized corrections in the actual candidates, respecting each
assigned training purpose. A large bundle of errors is not automatically useful.
Primary real feedback is i→yu; protected m→n; counterbalance i→e. Observed real
errors include 鉗→鉛 (i→yu and m→n overlap, not independent mechanisms), 蠍→橛/決,
件→券, 繳→竄, 尖→專, 展→轉. Maintenance concerns z/easy and j/medium capability.
An earlier15-audio pilot with these purposes and patient-weighted real KL achieved
raw development CER20.658053% and20.733173%, versus R-only20.793269%; the first
passed the full pooled/macro/hard gate, while the repeat slightly missed the hard
limit. This new14-audio experiment is a separate matched comparison: same target
texts and logical cases, different reference/recipe choices during SG. A case
with all16 correct samples in one arm was removed from BOTH arms. Its5 exposures
were redistributed within the same primary purpose. No negative was invented.
Each supplied audio has16 actual initial-R-only samples. Choose exactly one
supplied negative_key per audio_key. If none fulfills its assigned role, state
that limitation and choose the best actual available error without claiming the
role is fulfilled. The shared96-exposure schedule is fixed. Both branches start
from initial R-only, also the frozen DPO/KL reference;12steps,lr1e-4,beta0.1,KL10.
The same96 real-training recordings supply192 KL exposures,50% of the real KL
budget assigned to the same MAS-selected focus patient. No real CE or inference
postprocessing is added. Return14 choices, brief reasons, a prospective hypothesis
and limitation. Benefit remains unknown until real evaluation.'''


class Decision14(BaseModel):
    model_config = ConfigDict(extra='forbid')
    choices: list[Choice] = Field(min_length=14, max_length=14)
    hypothesis: str = Field(min_length=1)
    limitation: str = Field(min_length=1)


def verify(root):
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == h for f, h in cfg['source_sha256'].items())
    return cfg


def prepare(base, root):
    plan_path = base / 'patient_weighted_reference_matched14_plan_v1.json'
    plan = read(plan_path)
    assert all(sha256_file(Path(f)) == h for f, h in plan['source_sha256'].items())
    prior = read(base / 'patient_retention_dose_probe_v1/protocol.json')
    p = prior['source_protocol'];common = set(plan['common_audio_ids'])
    root.mkdir(exist_ok=False)
    save(root / 'allocation.json', plan['allocation'])
    old_purposes = read(base / 'candidate_coverage_recovery_v2/purposes.json')
    purposes = [dict(x, exposures=plan['allocation']['counts'][x['audio_id']]) for x in old_purposes if x['audio_id'] in common]
    save(root / 'purposes.json', purposes)
    os.link(base / 'patient_retention_dose_probe_v1/real_loss_factors.json', root / 'real_loss_factors.json')
    files = [Path(__file__), plan_path, TRAINER, REPO / 'examples/mas_patient_retention_probe.py',
        REPO / 'examples/mas_preference_target_feedback.py', REPO / 'examples/mas_residual_dpo_probe.py',
        REPO / 'examples/evaluate_residual_asr.py', REPO / 'src/therapist_harness/preference_data.py',
        REPO / 'src/therapist_harness/dpo_training.py', REPO / 'finetune/scripts/qwen3_asr_patient_weighted_kl_dpo.py',
        REPO / 'finetune/scripts/qwen3_asr_shared_teacher_kl_dpo.py', REPO / 'finetune/scripts/qwen3_asr_residual_kl_dpo.py',
        REPO / 'finetune/scripts/qwen3_asr_kl_dpo_probe.py', REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py',
        base / 'patient_retention_dose_probe_v1/protocol.json', base / 'patient_retention_replication_v1/storage_smoke.json',
        base / 'patient_retention_replication_v1/replication_verified.json', base / 'residual_retention_probe_v1/smoke/verified.json',
        Path(prior['arguments']['real_train_file']), Path(prior['verified_teacher_cache']), Path(prior['verified_teacher_metadata']),
        root / 'allocation.json', root / 'purposes.json', root / 'real_loss_factors.json']
    inputs_by_arm = {}
    for arm, receipt_path in zip(ARMS, plan['independent_sampling_receipts']):
        out = root / arm;out.mkdir()
        source = Path(receipt_path).parent;receipt = read(Path(receipt_path))
        for name, field in [('candidates.jsonl','candidates_sha256'),('validated_inputs.jsonl','validated_inputs_sha256'),('greedy.jsonl','greedy_sha256')]:
            assert sha256_file(source / name) == receipt[field]
        assert receipt['current_asr_sha256'] == p['initial_adapter']['sha256']
        inputs = [x for x in rows(source/'validated_inputs.jsonl') if x['audio_id'] in common]
        candidates = [x for x in rows(source/'candidates.jsonl') if x['audio_id'] in common]
        greedy = [x for x in rows(source/'greedy.jsonl') if x['audio_id'] in common]
        assert len(inputs)==14 and len(candidates)==224 and len(greedy)==14
        assert all(x['review']['verdict']=='pass' and sha256_file(Path(x['audio']))==x['audio_sha256'] for x in inputs)
        pairs,audit = build_preference_pairs(inputs,candidates,p['initial_adapter']['sha256'],expected_samples=16,greedy_candidates=greedy)
        expected = [x for x in rows(Path(plan['arms'][arm])) if x['audio_id'] in common]
        assert sorted(pairs,key=lambda x:x['pair_id']) == sorted(expected,key=lambda x:x['pair_id'])
        jsonl(out/'inputs.jsonl',inputs);jsonl(out/'candidate_view.jsonl',candidates);jsonl(out/'greedy_view.jsonl',greedy)
        jsonl(out/'available_preferences.jsonl',pairs);save(out/'preference_audit.json',audit)
        save(out/'view_receipt.json',dict(actual_new_sampling=False,actual_source_receipt=str(receipt_path),
            actual_source_receipt_sha256=sha256_file(Path(receipt_path)),source_execution_draws=240,view_draws=224,
            common_audio_ids=sorted(common),view_sha256=sha256_file(out/'candidate_view.jsonl'),
            available_preferences_sha256=sha256_file(out/'available_preferences.jsonl')))
        inputs_by_arm[arm] = {x['audio_id']:x for x in inputs}
        files += [Path(receipt_path), source/'candidates.jsonl',source/'validated_inputs.jsonl',source/'greedy.jsonl'] + list(out.glob('*.json')) + list(out.glob('*.jsonl'))
    for key in common:
        assert inputs_by_arm[ARMS[0]][key]['target_text'] == inputs_by_arm[ARMS[1]][key]['target_text']
    save(root/'protocol.json',dict(source_protocol=p, source_sha256={**plan['source_sha256'],**{str(f):sha256_file(f) for f in files}},
        arguments=prior['arguments'],verified_teacher_cache=prior['verified_teacher_cache'],verified_teacher_metadata=prior['verified_teacher_metadata'],
        arms=ARMS,audio_per_arm=14,unique_target_texts=len({x['target_text'] for x in inputs_by_arm[ARMS[0]].values()}),
        actual_new_generation=False,actual_new_sampling=False,actual_new_MAS_selection=True,actual_new_DPO_training=True,
        interpretation=plan['interpretation'],negative_selection_stochasticity_not_eliminated=True,
        original15case_results='Separate pilot; original .50 share gate pass was not fully repeated. No parent promotion yet.'))


def choose(root, arm):
    from openai.lib._pydantic import to_strict_json_schema
    cfg=verify(root);out=root/arm;groups=defaultdict(list)
    for row in rows(out/'available_preferences.jsonl'):groups[row['audio_id']].append(row)
    purposes={x['audio_id']:x for x in read(root/'purposes.json')}
    greedy={x['audio_id']:x['prediction'] for x in rows(out/'greedy_view.jsonl')}
    keys,audio_keys,cards={},{},[]
    for i,audio_id in enumerate(sorted(groups)):
        group=groups[audio_id];audio_key=f'a{i:02d}';audio_keys[audio_key]=audio_id;options=[]
        for j,row in enumerate(group):
            key=f'{audio_key}.n{j:02d}';keys[key]=row
            events=[dict(level=e.level,intended_unit=e.reference_unit,observed_unit=e.hypothesis_unit,
                intended_character=e.reference_character,observed_character=e.hypothesis_character)
                for e in actual_events(row['chosen'],row['rejected']) if e.context_sensitive is False]
            options.append(dict(negative_key=key,rejected=row['rejected'],sample_indices=row['sample_indices'],observed_stable_phone_events=events))
        cards.append(dict(audio_key=audio_key,target=group[0]['chosen'],reference_text=group[0]['metadata']['review']['reference_text'],
            greedy=greedy[audio_id],purpose=purposes[audio_id]['purpose'],exposures=purposes[audio_id]['exposures'],negatives=options))
    p=RunProtocol.model_validate(cfg['source_protocol'])
    request=dict(model=p.agent.model,reasoning={'effort':p.agent.reasoning_effort},store=False,stream=True,max_output_tokens=12000,
        input=[dict(role='system',content=INSTRUCTIONS),dict(role='user',content=json.dumps(dict(audio_cards=cards),ensure_ascii=False))],
        text={'format':{'type':'json_schema','name':'matched14_negative_choice','strict':True,'schema':to_strict_json_schema(Decision14)}})
    save(out/'request.json',request)
    response=request_mas(p,request,out);assert response['status']=='completed'
    final=[x for x in response['output'] if x.get('type')=='message' and x.get('phase')=='final_answer'];assert len(final)==1
    decision=Decision14.model_validate_json(''.join(x['text'] for x in final[0]['content'] if x['type']=='output_text'))
    assert len({x.audio_key for x in decision.choices})==14 and {x.audio_key for x in decision.choices}==set(audio_keys)
    selected,mapping=[],[]
    for choice in decision.choices:
        row=keys[choice.negative_key];assert row['audio_id']==audio_keys[choice.audio_key]
        selected.append(dict(row,pair_weight=1.))
        mapping.append(dict(audio_id=row['audio_id'],source_pair_id=row['pair_id'],source_sample_indices=row['sample_indices'],reason=choice.reason))
    jsonl(out/'preferences.jsonl',selected);save(out/'mas_decision.json',decision.model_dump(mode='json'))
    save(out/'selection_receipt.json',dict(selections=mapping,changed_pair_fields=['pair_weight'],
        preferences_sha256=sha256_file(out/'preferences.jsonl'),actual_MAS_response_sha256=sha256_file(out/'response.json'),
        view_receipt_sha256=sha256_file(out/'view_receipt.json')))
    verify(root)


def smoke(root):
    cfg=verify(root);w=read(root/'real_loss_factors.json');real=rows(cfg['arguments']['real_train_file'])
    reports={}
    for arm in ARMS:
        out=root/arm;assert sha256_file(out/'preferences.jsonl')==read(out/'selection_receipt.json')['preferences_sha256']
        reports[arm]=cpu_smoke(real,rows(out/'preferences.jsonl'),read(root/'allocation.json'),w['factors'],set(w['focus_ids']),w['focus_share'])
    assert reports[ARMS[0]]==reports[ARMS[1]]
    requests=[read(root/arm/'request.json') for arm in ARMS]
    assert {k:v for k,v in requests[0].items() if k!='input'}=={k:v for k,v in requests[1].items() if k!='input'}
    assert requests[0]['input'][0]==requests[1]['input'][0]
    save(root/'smoke.json',dict(status='passed',actual_new_GPU_smoke=False,actual_new_CPU_smoke=True,arms=reports,
        previous_actual_GPU_smoke=str(root.parent/'residual_retention_probe_v1/smoke/verified.json'),
        scope='New complete12-window CPU exposure/weight check; unchanged residual KL/DPO engine reuses its documented actual GPU smoke. Actual runs additionally check initial zero KL/log2 DPO and frozen model integrity.'))
    files=[root/'smoke.json']+[root/arm/name for arm in ARMS for name in ['request.json','response.json','preferences.jsonl','selection_receipt.json']]
    save(root/'training_sources.json',{str(f):sha256_file(f) for f in files})


def train(base,root,arm):
    cfg=verify(root);assert read(root/'smoke.json')['status']=='passed'
    bound=read(root/'training_sources.json');assert all(sha256_file(Path(f))==h for f,h in bound.items())
    out=root/arm;a=dict(cfg['arguments'],synthetic_train_file=str(out/'preferences.jsonl'),
        synthetic_allocation_file=str(root/'allocation.json'),output_dir=str(out/'training'))
    command=[sys.executable,str(TRAINER)]
    for key,value in a.items():
        if value is not None:command.extend(['--'+key,str(value)])
    command.extend(['--real_loss_factors',str(root/'real_loss_factors.json'),'--verified_teacher_cache',cfg['verified_teacher_cache'],
        '--verified_teacher_metadata',cfg['verified_teacher_metadata']])
    execute(out,command,'training')
    result=read(out/'training/training_result.json');assert result['policy_changed'] and result['reference_files_unchanged']
    assert abs(result['history'][0]['weighted_losses']['real'])<1e-7
    assert math.isclose(result['history'][0]['weighted_losses']['synthetic'],math.log(2),abs_tol=1e-4)
    assert read(out/'training/residual_verified.json')['status']=='passed'
    assert read(out/'training/teacher_storage_reuse.json')['identical_actual_tensor_values_verified']==96
    exposures=rows(out/'training/training_exposures.jsonl')
    assert Counter(x['source'] for x in exposures)=={'real':192,'synthetic':96}
    previous=rows(base/'patient_retention_dose_probe_v1/training/training_exposures.jsonl')
    assert [x for x in exposures if x['source']=='real']==[x for x in previous if x['source']=='real']
    expected=[key for step in read(root/'allocation.json')['steps'] for key in step]
    assert [x['audio_id'] for x in exposures if x['source']=='synthetic']==expected
    save(out/'verified.json',dict(status='passed',initial_KL_zero_DPO_log2=True,real_exposures_identical_to15case_pilot=True,
        synthetic_exposures_match_new_shared_schedule=True,all96fresh_teacher_tensors_verified=True,frozen_base_reference_unchanged=True))
    execute(out,evaluation_command(cfg['source_protocol'],out/'evaluation_run',out/'training/checkpoint-12'),'evaluation')
    metrics=read(out/'evaluation_run/evaluation/summary.json')['metrics'];parent=read(base/'parent_eval_check/summary.json')['metrics']
    gate=dict(pooled_improves=metrics['overall']['pooled_cer']<parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer']-parent['overall']['patient_macro_cer']<=.001,
        hard_regression_within_limit=metrics['hard']['pooled_cer']-parent['hard']['pooled_cer']<=.005)
    save(out/'paired_parent.json',compare(base/'parent_eval_check/predictions.jsonl',out/'evaluation_run/evaluation/predictions.jsonl'))
    save(out/'results.json',dict(status='complete',metrics=metrics,gate_checks=gate,eligible_for_next_round_parent=all(gate.values()),
        actual_new_DPO_training=True,postprocessing_used=False))
    verify(root);assert all(sha256_file(Path(f))==h for f,h in bound.items())


def supervise(base,root):
    for stage in ['choose','train']:
        jobs=[]
        for gpu,arm in enumerate(ARMS):
            command=[sys.executable,str(Path(__file__).resolve()),stage,str(base),'--arm',arm]
            with (root/arm/f'{stage}.supervisor.log').open('x') as log:
                job=subprocess.Popen(command,cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
            save(root/arm/f'{stage}.launch.json',dict(pid=job.pid,gpu=gpu,command=command));jobs.append(job)
        codes=[job.wait() for job in jobs];assert codes==[0,0],codes
        if stage=='choose':smoke(root)
    exposure=[rows(root/arm/'training/training_exposures.jsonl') for arm in ARMS]
    # Different generated audio legitimately have different preference hashes.
    normalized=lambda seq:[{k:v for k,v in x.items() if k!='preference_key'} for x in seq]
    assert normalized(exposure[0])==normalized(exposure[1])
    for arm in ARMS:
        assert read(root/arm/'training/run_protocol.json')['reference_asr_sha256']==read(root/'protocol.json')['source_protocol']['initial_adapter']['sha256']
    save(root/'cross_arm_verified.json',dict(status='passed',same_actual_real_synthetic_ID_order_and_weights=True,
        different_actual_synthetic_audio_hashes_and_reference_scores_per_arm_expected=True,common_parent_frozen_reference=True,
        original_separate_sampling_receipts_preserved=True))
    save(root/'paired_reference_control.json',compare(root/'random_reference/evaluation_run/evaluation/predictions.jsonl',
        root/'MAS_reference/evaluation_run/evaluation/predictions.jsonl'))
    save(root/'results.json',dict(status='complete',arms={arm:read(root/arm/'results.json') for arm in ARMS},
        scope=read(root/'protocol.json')['interpretation'],parent_promotion_performed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','choose','smoke','train','supervise']);parser.add_argument('base',type=Path)
    parser.add_argument('--arm',choices=ARMS,default=ARMS[0]);args=parser.parse_args()
    base=args.base.resolve();root=base/'patient_weighted_reference_matched14_v1'
    if args.stage=='prepare':prepare(base,root)
    elif args.stage=='choose':choose(root,args.arm)
    elif args.stage=='smoke':smoke(root)
    elif args.stage=='train':train(base,root,args.arm)
    else:supervise(base,root)
