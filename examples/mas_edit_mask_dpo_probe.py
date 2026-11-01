#!/usr/bin/env python3
"""Matched15-pair full-sequence versus edit+EOS audio preference training."""

import argparse
from collections import Counter
import math
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file
from therapist_harness.dpo_training import group_training_rows, preference_key

sys.path.insert(0,str(REPO/'finetune/scripts'))
from qwen3_asr_patient_weighted_kl_dpo import weighted_window_type

ARMS=['full','edit_eos']
TRAINER=REPO/'finetune/scripts/qwen3_asr_edit_masked_kl_dpo.py'


def exposure_plan(a,real,synthetic,allocation,factors):
    groups=dict(real=group_training_rows(real,'sft')['real'],synthetic=group_training_rows(synthetic,'dpo')['synthetic'])
    windows=weighted_window_type(factors)(groups,a['real_per_step'],a['synthetic_per_step'],a['synthetic_weight'],
        a['seed'],synthetic_schedule=allocation['steps'])
    records=[]
    for step in range(1,a['max_steps']+1):
        for row in windows.next():
            records.append(dict(step=step,source=row['source'],audio_id=row['_audio_group'],occurrence=row['_occurrence'],
                loss_weight=row['_loss_weight']*(a['kl_weight'] if row['source']=='real' else 1),
                objective='real_symmetric_kl' if row['source']=='real' else 'dpo',preference_key=preference_key(row)))
    return records


def prepare(base,root):
    previous=base/'patient_retention_replication_v1';old=read(previous/'protocol.json')
    assert all(sha256_file(Path(f))==h for f,h in old['source_sha256'].items())
    a=read(previous/'training/run_protocol.json')['arguments'];real=rows(a['real_train_file']);synthetic=rows(a['synthetic_train_file'])
    allocation=read(a['synthetic_allocation_file']);weights=read(previous/'real_loss_factors.json')
    assert len(real)==96 and len(synthetic)==15
    expected=exposure_plan(a,real,synthetic,allocation,weights['factors'])
    assert expected==rows(previous/'training/training_exposures.jsonl')
    root.mkdir(exist_ok=False)
    save(root/'expected_exposures.json',expected)
    execute(root,[sys.executable,'-m','pytest','-q','tests/test_preference_token_mask.py','tests/test_audio_dpo_training.py'],'cpu_tests')
    save(root/'cpu_smoke.json',dict(status='passed',actual_new_GPU_smoke=False,
        exact288_exposure_records_replay=True,real96_synthetic15=True,
        mask_unit_and_existing_DPO_tests_passed=True,test_log=str(root/'cpu_tests.log')))
    arguments={}
    for arm in ARMS:
        out=root/arm;out.mkdir()
        arguments[arm]=dict(a,output_dir=str(out/'training'),persist_teacher_cache=0,persist_optimizer_state=0,
            real_loss_factors=str(previous/'real_loss_factors.json'),preference_token_mask=arm)
    files=[Path(__file__),TRAINER,previous/'protocol.json',previous/'replication_verified.json',
        previous/'training/run_protocol.json',previous/'training/training_result.json',previous/'training/reference_log_probs.json',
        previous/'training/real_teacher_cache.json',previous/'training/training_exposures.jsonl',previous/'real_loss_factors.json',
        base/'next_edit_masked_DPO_plan_v1.json',root/'expected_exposures.json',root/'cpu_smoke.json',root/'cpu_tests.log']
    files += [REPO/f for f in ['src/therapist_harness/preference_token_mask.py','tests/test_preference_token_mask.py',
        'tests/test_audio_dpo_training.py','src/therapist_harness/dpo_training.py','finetune/scripts/qwen3_asr_kl_dpo_storage_v2.py',
        'finetune/scripts/qwen3_asr_residual_kl_dpo.py','finetune/scripts/qwen3_asr_residual_dpo.py',
        'finetune/scripts/qwen3_asr_patient_weighted_kl_dpo.py','finetune/scripts/qwen3_asr_weights_only_patient_kl_dpo.py',
        'finetune/scripts/qwen3_asr_audio_dpo.py','examples/evaluate_residual_asr.py','examples/mas_residual_dpo_probe.py']]
    save(root/'protocol.json',dict(source_protocol=old['source_protocol'],
        source_sha256={**old['source_sha256'],**{str(f):sha256_file(f) for f in files}},
        arguments=arguments,previous=str(previous),arms=ARMS,
        actual_new_generation=False,actual_new_sampling=False,actual_new_MAS_decision=False,actual_new_DPO_training=True,
        intervention='Only synthetic preference score mask differs. Same15 actual MAS pairs,96 synthetic and192 real exposures, weights/order,initial R-only parent/reference,12steps,lr1e-4,beta.1,KL10,seed42.',
        scope='edit_eos changes gradient support and score scale; no length normalization or beta rescaling. It is a modified preference objective, not unchanged full-sequence DPO. No real CE or inference postprocessing.'))


def verify(root):
    cfg=read(root/'protocol.json');assert all(sha256_file(Path(f))==h for f,h in cfg['source_sha256'].items())
    return cfg


def command(a):
    return [sys.executable,str(TRAINER)]+[v for k,value in a.items() if value is not None for v in ['--'+k,str(value)]]


def check_training(out,expected,mode):
    result=read(out/'training/training_result.json');assert result['policy_changed'] and result['reference_files_unchanged']
    assert abs(result['history'][0]['weighted_losses']['real'])<1e-7
    assert math.isclose(result['history'][0]['weighted_losses']['synthetic'],math.log(2),abs_tol=1e-4)
    assert read(out/'training/residual_verified.json')['status']=='passed'
    assert rows(out/'training/training_exposures.jsonl')==expected
    teacher=read(out/'training/real_teacher_cache.json');assert not teacher['tensor_data_persisted'] and teacher['cache_file_sha256'] is None
    assert not (out/'training/real_teacher_log_probs.pt').exists()
    checkpoint=Path(result['checkpoint']);assert not (checkpoint/'optimizer.pt').exists()
    assert read(checkpoint/'optimizer_omission.json')['optimizer_state_persisted'] is False
    masks=read(out/'training/preference_token_masks.json');assert masks['mode']==mode
    reference=read(out/'training/reference_log_probs.json');assert reference['preference_token_mask']==mode
    assert reference['mask_manifest_sha256']==sha256_file(out/'training/preference_token_masks.json')
    counts=Counter()
    for row in masks['pairs'].values():
        for side in ['chosen','rejected']:
            ids=row[side+'_token_ids'];kept=row[side+'_scored_indices']
            assert kept and kept[-1]==len(ids)-1 and row['token_edit_distance']>0
            if mode=='full':assert kept==list(range(len(ids)))
            else:assert kept==row[side+'_edit_eos_indices']
            counts['full']+=len(ids);counts['scored']+=len(kept)
    if mode=='edit_eos':assert counts['scored']<counts['full']
    return result,teacher,dict(counts)


def smoke(root):
    cfg=verify(root);out=root/'smoke';out.mkdir();a=dict(cfg['arguments']['edit_eos'])
    real=rows(a['real_train_file'])[:4];synthetic=rows(a['synthetic_train_file'])[:2]
    jsonl(out/'real.jsonl',real);jsonl(out/'synthetic.jsonl',synthetic)
    allocation=dict(steps=[[x['audio_id'] for x in synthetic]]*2);weights={x['audio_id']:1. for x in real}
    save(out/'allocation.json',allocation);save(out/'weights.json',dict(factors=weights))
    a.update(output_dir=str(out/'training'),real_train_file=str(out/'real.jsonl'),synthetic_train_file=str(out/'synthetic.jsonl'),
        real_loss_factors=str(out/'weights.json'),synthetic_allocation_file=str(out/'allocation.json'),
        max_steps=2,real_per_step=4,synthetic_per_step=2)
    expected=exposure_plan(a,real,synthetic,allocation,weights);save(out/'expected_exposures.json',expected)
    execute(out,command(a),'training')
    result,teacher,counts=check_training(out,expected,'edit_eos')
    assert result['history'][1]['weighted_losses']['real']>0
    old=read(Path(cfg['previous'])/'training/real_teacher_cache.json')['audio']
    assert all(all(v[k]==old[key][k] for k in ['shape','tensor_sha256']) for key,v in teacher['audio'].items())
    save(out/'verified.json',dict(status='passed',actual_new_GPU_smoke=True,steps=2,
        initial_log2_DPO_and_zero_KL=True,actual_masks_bound_to_reference_scores=True,
        real_teacher_distributions_unchanged=True,full_teacher_history_and_EOS_preserved=True,
        residual_updated_and_frozen_base_reference_preserved=True,actual_synthetic_token_counts=counts))
    verify(root)


def train(base,root,arm):
    cfg=verify(root);assert read(root/'smoke/verified.json')['status']=='passed'
    out=root/arm;execute(out,command(cfg['arguments'][arm]),'training')
    result,teacher,counts=check_training(out,read(root/'expected_exposures.json'),arm)
    old=read(Path(cfg['previous'])/'training/real_teacher_cache.json')['audio']
    assert set(teacher['audio'])==set(old) and len(old)==96
    assert all(all(v[k]==old[key][k] for k in ['shape','tensor_sha256']) for key,v in teacher['audio'].items())
    if arm=='full':
        assert read(out/'training/reference_log_probs.json')['values']==read(Path(cfg['previous'])/'training/reference_log_probs.json')['values']
    save(out/'verified.json',dict(status='passed',actual288_exposures_exact=True,
        all96_real_teacher_distributions_identical=True,preference_token_mask=arm,
        actual_synthetic_token_counts=counts,reference_scores_recomputed_with_actual_mask=True,
        full_control_matches_original_reference_scores=True if arm=='full' else None))
    execute(out,evaluation_command(cfg['source_protocol'],out/'evaluation_run',Path(result['checkpoint'])),'evaluation')
    metrics=read(out/'evaluation_run/evaluation/summary.json')['metrics'];parent=read(base/'parent_eval_check/summary.json')['metrics']
    gate=dict(pooled_improves=metrics['overall']['pooled_cer']<parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer']-parent['overall']['patient_macro_cer']<=.001,
        hard_regression_within_limit=metrics['hard']['pooled_cer']-parent['hard']['pooled_cer']<=.005)
    save(out/'paired_parent.json',compare(base/'parent_eval_check/predictions.jsonl',out/'evaluation_run/evaluation/predictions.jsonl'))
    save(out/'results.json',dict(status='complete',preference_token_mask=arm,metrics=metrics,gate_checks=gate,
        eligible_for_next_round_parent=all(gate.values()),actual_new_DPO_training=True,postprocessing_used=False))
    verify(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','smoke','train'])
    parser.add_argument('base',type=Path);parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();base=args.base.resolve();root=base/'edit_mask_dpo_probe_v1'
    if args.stage=='prepare':prepare(base,root)
    elif args.stage=='smoke':smoke(root)
    else:
        assert args.arm is not None
        train(base,root,args.arm)
