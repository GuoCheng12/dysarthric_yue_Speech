#!/usr/bin/env python3
"""Independent training-seed checks of the completed15-pair MAS DPO recipe."""

import argparse
from collections import Counter
import math
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from therapist_harness.dpo_training import group_training_rows, preference_key
from therapist_harness.store import sha256_file

sys.path.insert(0,str(REPO/'finetune/scripts'))
from qwen3_asr_patient_weighted_kl_dpo import weighted_window_type

TRAINER=REPO/'finetune/scripts/qwen3_asr_weights_only_patient_kl_dpo.py'


def prepare(base,root):
    previous=base/'patient_retention_replication_v1';old=read(previous/'protocol.json')
    assert read(previous/'replication_verified.json')['status']=='passed'
    assert all(sha256_file(Path(f))==h for f,h in old['source_sha256'].items())
    a=read(previous/'training/run_protocol.json')['arguments'];w=read(previous/'real_loss_factors.json')
    real=rows(a['real_train_file']);synthetic=rows(a['synthetic_train_file'])
    allocation=read(a['synthetic_allocation_file'])['steps']
    groups=dict(real=group_training_rows(real,'sft')['real'],synthetic=group_training_rows(synthetic,'dpo')['synthetic'])
    root.mkdir(exist_ok=False);arguments={};expected_by_seed={}
    old_exposures=rows(previous/'training/training_exposures.jsonl')
    for seed in [42,43,44]:
        windows=weighted_window_type(w['factors'])(groups,16,8,1.,seed,synthetic_schedule=allocation)
        records=[]
        for step in range(1,13):
            for row in windows.next():
                records.append(dict(step=step,source=row['source'],audio_id=row['_audio_group'],
                    occurrence=row['_occurrence'],loss_weight=row['_loss_weight']*(10 if row['source']=='real' else 1),
                    objective='real_symmetric_kl' if row['source']=='real' else 'dpo',preference_key=preference_key(row)))
        assert Counter(x['source'] for x in records)=={'real':192,'synthetic':96}
        assert [x for x in records if x['source']=='synthetic']==[x for x in old_exposures if x['source']=='synthetic']
        inventory=lambda data:Counter((x['source'],x['audio_id'],x['loss_weight'],x['preference_key']) for x in data)
        assert inventory(records)==inventory(old_exposures)
        if seed==42:
            assert records==old_exposures
            continue
        out=root/f'seed_{seed}';out.mkdir();save(out/'expected_exposures.json',records)
        expected_by_seed[str(seed)]=str(out/'expected_exposures.json')
        arguments[str(seed)]=dict(a,seed=seed,output_dir=str(out/'training'))
    save(root/'cpu_smoke.json',dict(status='passed',actual_new_GPU_smoke=False,
        seed42_exactly_reproduces_recorded_actual_exposures=True,
        each_seed_real_audio=96,each_seed_synthetic_audio=15,real_exposures=192,synthetic_exposures=96,
        identical_per_audio_full_run_weights_and_counts=True,identical_synthetic_order=True,
        real_order_regenerated_by_training_seed=True,
        reused_actual_GPU_smoke=str(base/'residual_retention_probe_v1/smoke/verified.json'),
        reused_actual_same_engine12_step_training=str(previous/'replication_verified.json')))
    files=[Path(__file__),TRAINER,previous/'protocol.json',previous/'replication_verified.json',
        previous/'training/run_protocol.json',previous/'training/training_exposures.jsonl',
        previous/'training/reference_log_probs.json',previous/'real_loss_factors.json',
        base/'parent_eval_check/summary.json',root/'cpu_smoke.json']+[Path(x) for x in expected_by_seed.values()]
    save(root/'protocol.json',dict(source_protocol=old['source_protocol'],
        source_sha256={**old['source_sha256'],**{str(f):sha256_file(f) for f in files}},
        arguments=arguments,real_loss_factors=str(previous/'real_loss_factors.json'),
        verified_teacher_cache=old['verified_teacher_cache'],verified_teacher_metadata=old['verified_teacher_metadata'],
        seed42_comparator=str(previous),initial_parent='same original seed42 R-only for all arms',
        intervention='Training seed43/44 changes residual initialization and real shuffle. Same fixed MAS15 preferences, synthetic exposure order, per-audio inventory, optimizer and loss weights.',
        scope='Conditional training-seed reproducibility; not independent MAS generation/selection seeds or newly trained R-only parents.',
        actual_new_MAS_decision=False,actual_new_generation=False,actual_new_sampling=False,actual_new_DPO_training=True))


def verify(root):
    cfg=read(root/'protocol.json');assert all(sha256_file(Path(f))==h for f,h in cfg['source_sha256'].items())
    return cfg


def run(base,root,seed):
    cfg=verify(root);assert read(root/'cpu_smoke.json')['status']=='passed'
    out=root/f'seed_{seed}';a=cfg['arguments'][str(seed)]
    command=[sys.executable,str(TRAINER)]+[v for k,value in a.items() if value is not None for v in ['--'+k,str(value)]]
    command += ['--real_loss_factors',cfg['real_loss_factors'],'--verified_teacher_cache',cfg['verified_teacher_cache'],
        '--verified_teacher_metadata',cfg['verified_teacher_metadata']]
    execute(out,command,'training')
    result=read(out/'training/training_result.json');assert result['policy_changed'] and result['reference_files_unchanged']
    assert abs(result['history'][0]['weighted_losses']['real'])<1e-7
    assert math.isclose(result['history'][0]['weighted_losses']['synthetic'],math.log(2),abs_tol=1e-4)
    assert read(out/'training/residual_verified.json')['status']=='passed'
    assert rows(out/'training/training_exposures.jsonl')==read(out/'expected_exposures.json')
    reference=read(out/'training/reference_log_probs.json')['values']
    assert reference==read(Path(cfg['seed42_comparator'])/'training/reference_log_probs.json')['values']
    assert read(out/'training/teacher_storage_reuse.json')['identical_actual_tensor_values_verified']==96
    checkpoint=Path(result['checkpoint']);assert not (checkpoint/'optimizer.pt').exists()
    assert read(checkpoint/'optimizer_omission.json')['optimizer_state_persisted'] is False
    save(out/'verified.json',dict(status='passed',training_seed=seed,common_R_only_parent_and_frozen_reference=True,
        actual_exposures_match_seeded_plan=True,same15_DPO_reference_scores=True,
        all96_fresh_teacher_tensors_identical=True,initial_KL_zero_DPO_log2=True,exact_optimizer_resume_available=False))
    execute(out,evaluation_command(cfg['source_protocol'],out/'evaluation_run',checkpoint),'evaluation')
    metrics=read(out/'evaluation_run/evaluation/summary.json')['metrics'];parent=read(base/'parent_eval_check/summary.json')['metrics']
    gate=dict(pooled_improves=metrics['overall']['pooled_cer']<parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer']-parent['overall']['patient_macro_cer']<=.001,
        hard_regression_within_limit=metrics['hard']['pooled_cer']-parent['hard']['pooled_cer']<=.005)
    save(out/'paired_parent.json',compare(base/'parent_eval_check/predictions.jsonl',out/'evaluation_run/evaluation/predictions.jsonl'))
    save(out/'results.json',dict(status='complete',training_seed=seed,metrics=metrics,gate_checks=gate,
        eligible_for_next_round_parent=all(gate.values()),actual_new_DPO_training=True,postprocessing_used=False))
    verify(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('base',type=Path);parser.add_argument('--seed',type=int,choices=[43,44])
    args=parser.parse_args();base=args.base.resolve();root=base/'patient_retention_training_seeds_v1'
    if args.stage=='prepare':prepare(base,root)
    else:
        assert args.seed is not None
        run(base,root,args.seed)
