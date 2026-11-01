#!/usr/bin/env python3
"""Training-seed replication of the gate-passing edit+EOS MAS preference probe."""

import argparse
from collections import Counter
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save
from mas_edit_mask_dpo_probe import TRAINER, check_training, command, exposure_plan
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file


def prepare(base,root):
    previous=base/'edit_mask_dpo_probe_v1/edit_eos';old=read(previous.parent/'protocol.json')
    assert read(previous/'results.json')['eligible_for_next_round_parent']
    assert read(previous/'verified.json')['status']=='passed'
    assert all(sha256_file(Path(f))==h for f,h in old['source_sha256'].items())
    a=old['arguments']['edit_eos'];real=rows(a['real_train_file']);synthetic=rows(a['synthetic_train_file'])
    allocation=read(a['synthetic_allocation_file']);weights=read(a['real_loss_factors'])['factors']
    original=rows(previous/'training/training_exposures.jsonl')
    root.mkdir(exist_ok=False);arguments={}
    for seed in [42,43,44]:
        out=root/f'seed_{seed}';args=dict(a,seed=seed,output_dir=str(out/'training'))
        records=exposure_plan(args,real,synthetic,allocation,weights)
        inventory=lambda data:Counter((x['source'],x['audio_id'],x['loss_weight'],x['preference_key']) for x in data)
        assert inventory(records)==inventory(original)
        assert [x for x in records if x['source']=='synthetic']==[x for x in original if x['source']=='synthetic']
        if seed==42:
            assert records==original
            continue
        out.mkdir();save(out/'expected_exposures.json',records);arguments[str(seed)]=args
    save(root/'cpu_smoke.json',dict(status='passed',actual_new_GPU_smoke=False,
        seed42_matches_completed_exposure_records=True,same_all288_exposure_inventory_per_seed=True,
        synthetic_order_fixed=True,real_order_regenerated_by_training_seed=True,
        actual_GPU_smoke_reused=str(previous.parent/'smoke/verified.json'),
        same_engine_actual12_step_training_reused=str(previous/'verified.json')))
    files=[Path(__file__),TRAINER,REPO/'examples/mas_edit_mask_dpo_probe.py',previous.parent/'protocol.json',
        previous/'results.json',previous/'verified.json',previous/'training/training_exposures.jsonl',
        previous/'training/reference_log_probs.json',previous/'training/preference_token_masks.json',
        previous/'training/real_teacher_cache.json',root/'cpu_smoke.json']
    files += [root/f'seed_{seed}/expected_exposures.json' for seed in [43,44]]
    save(root/'protocol.json',dict(source_protocol=old['source_protocol'],previous=str(previous),
        source_sha256={**old['source_sha256'],**{str(f):sha256_file(f) for f in files}},arguments=arguments,
        actual_new_MAS_decision=False,actual_new_generation=False,actual_new_sampling=False,actual_new_DPO_training=True,
        scope='Fixed15 MAS preferences and synthetic order, common original seed42 R-only parent/reference. Training seed changes residual initialization and real shuffle. Not independent MAS generation seeds.',
        acceptance='Original pooled/macro/hard real-development gate, no inference postprocessing.'))


def verify(root):
    cfg=read(root/'protocol.json');assert all(sha256_file(Path(f))==h for f,h in cfg['source_sha256'].items())
    return cfg


def run(base,root,seed):
    cfg=verify(root);assert read(root/'cpu_smoke.json')['status']=='passed'
    out=root/f'seed_{seed}';previous=Path(cfg['previous']);a=cfg['arguments'][str(seed)]
    execute(out,command(a),'training')
    result,teacher,counts=check_training(out,read(out/'expected_exposures.json'),'edit_eos')
    assert teacher['audio']==read(previous/'training/real_teacher_cache.json')['audio']
    assert read(out/'training/reference_log_probs.json')['values']==read(previous/'training/reference_log_probs.json')['values']
    assert read(out/'training/preference_token_masks.json')==read(previous/'training/preference_token_masks.json')
    save(out/'verified.json',dict(status='passed',actual288_exposures_match_seeded_plan=True,
        same15_actual_masks_and_reference_scores=True,all96_real_teacher_distributions_identical=True,
        common_initial_parent_and_frozen_reference=True,training_seed=seed,actual_token_counts=counts))
    execute(out,evaluation_command(cfg['source_protocol'],out/'evaluation_run',Path(result['checkpoint'])),'evaluation')
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
    args=parser.parse_args();base=args.base.resolve();root=base/'edit_mask_training_seeds_v1'
    if args.stage=='prepare':prepare(base,root)
    else:
        assert args.seed is not None
        run(base,root,args.seed)
