#!/usr/bin/env python3
"""Paired training-seed checks of normalized-target edit+EOS preference training."""

import argparse
from collections import Counter
from pathlib import Path
import shutil

from mas_selection_dpo_probe import REPO, execute, read, rows, save
from mas_ASR_target_format_probe import copy_receipts
from mas_edit_mask_dpo_probe import TRAINER, command, check_training, exposure_plan
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file, sha256_directory

SCRATCH=Path('/tmp/qwen3_asr_mas_target_format_seeds_v1')


def prepare(base,root):
    previous=base/'ASR_target_format_dpo_probe_v1';old=read(previous/'protocol.json')
    assert all(sha256_file(Path(f))==h for f,h in old['source_sha256'].items())
    assert read(previous/'edit_eos/results.json')['eligible_for_next_round_parent']
    assert read(previous/'edit_eos/verified.json')['status']=='passed'
    a=old['arguments']['edit_eos'];real=rows(a['real_train_file']);synthetic=rows(a['synthetic_train_file'])
    allocation=read(a['synthetic_allocation_file']);weights=read(a['real_loss_factors'])['factors']
    original=rows(previous/'edit_eos/training_receipts/training_exposures.jsonl')
    root.mkdir(exist_ok=False);assert shutil.disk_usage('/tmp').free>300*1024**2
    SCRATCH.mkdir(exist_ok=False);arguments={}
    for seed in [42,43,44]:
        args=dict(a,seed=seed,output_dir=str(SCRATCH/f'seed_{seed}/training'))
        expected=exposure_plan(args,real,synthetic,allocation,weights)
        inventory=lambda data:Counter((x['source'],x['audio_id'],x['loss_weight'],x['preference_key']) for x in data)
        assert inventory(expected)==inventory(original)
        assert [x for x in expected if x['source']=='synthetic']==[x for x in original if x['source']=='synthetic']
        if seed==42:
            assert expected==original
            continue
        (root/f'seed_{seed}').mkdir();(SCRATCH/f'seed_{seed}').mkdir()
        save(root/f'seed_{seed}/expected_exposures.json',expected);arguments[str(seed)]=args
    save(root/'cpu_smoke.json',dict(status='passed',actual_new_GPU_smoke=False,
        all288_per_audio_exposure_counts_and_weights_preserved=True,synthetic_order_fixed=True,
        real_order_follows_training_seed=True,seed42_replays_actual_exposures=True,
        same_engine_actual_GPU_smoke=str(base/'edit_mask_dpo_probe_v1/smoke/verified.json'),
        actual12_step_normalized_training=str(previous/'edit_eos/verified.json')))
    files=[Path(__file__),TRAINER,REPO/'examples/mas_ASR_target_format_probe.py',previous/'protocol.json',
        previous/'edit_eos/results.json',previous/'edit_eos/verified.json',root/'cpu_smoke.json']
    files += [previous/'edit_eos/training_receipts'/n for n in ['reference_log_probs.json',
        'preference_token_masks.json','real_teacher_cache.json','training_exposures.jsonl']]
    for seed in [43,44]:
        files += [root/f'seed_{seed}/expected_exposures.json',base/f'edit_mask_training_seeds_v1/seed_{seed}/results.json',
            base/f'edit_mask_training_seeds_v1/seed_{seed}/evaluation_run/evaluation/predictions.jsonl']
    save(root/'protocol.json',dict(source_protocol=old['source_protocol'],previous=str(previous),
        source_sha256={**old['source_sha256'],**{str(f):sha256_file(f) for f in files}},arguments=arguments,
        actual_new_generation=False,actual_new_sampling=False,actual_new_MAS_decision=False,actual_new_DPO_training=True,
        storage=old['storage']|dict(checkpoint_root=str(SCRATCH),persistent_record_root=str(root)),
        scope='Same normalized15 MAS preferences and original seed42 R-only parent/reference; only training initialization and real shuffle follow seed43/44. Compare to the already completed same-seed raw-target edit_eos controls.',
        acceptance='Original pooled/macro/hard gate; no inference postprocessing or automatic promotion.'))


def verify(root):
    cfg=read(root/'protocol.json');assert all(sha256_file(Path(f))==h for f,h in cfg['source_sha256'].items())
    return cfg


def run(base,root,seed):
    cfg=verify(root);out=root/f'seed_{seed}';a=cfg['arguments'][str(seed)]
    execute(out,command(a),'training');training=Path(a['output_dir'])
    result,teacher,counts=check_training(training.parent,read(out/'expected_exposures.json'),'edit_eos')
    old=Path(cfg['previous'])/'edit_eos/training_receipts'
    assert teacher['audio']==read(old/'real_teacher_cache.json')['audio']
    assert read(training/'reference_log_probs.json')['values']==read(old/'reference_log_probs.json')['values']
    assert read(training/'preference_token_masks.json')==read(old/'preference_token_masks.json')
    checkpoint=Path(result['checkpoint']);checkpoint_sha=sha256_directory(checkpoint)
    copy_receipts(training,out)
    save(out/'verified.json',dict(status='passed',actual288_exposures_match=True,
        same15_normalized_targets_masks_and_reference_scores=True,all96_teacher_distributions_identical=True,
        common_initial_R_only_parent_and_reference=True,training_seed=seed,actual_token_counts=counts,
        checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,checkpoint_storage='temporary scratch'))
    execute(out,evaluation_command(cfg['source_protocol'],out/'evaluation_run',checkpoint),'evaluation')
    assert sha256_directory(checkpoint)==checkpoint_sha
    metrics=read(out/'evaluation_run/evaluation/summary.json')['metrics'];parent=read(base/'parent_eval_check/summary.json')['metrics']
    gate=dict(pooled_improves=metrics['overall']['pooled_cer']<parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer']-parent['overall']['patient_macro_cer']<=.001,
        hard_regression_within_limit=metrics['hard']['pooled_cer']-parent['hard']['pooled_cer']<=.005)
    save(out/'paired_parent.json',compare(base/'parent_eval_check/predictions.jsonl',out/'evaluation_run/evaluation/predictions.jsonl'))
    save(out/'paired_same_seed_raw_target.json',compare(base/f'edit_mask_training_seeds_v1/seed_{seed}/evaluation_run/evaluation/predictions.jsonl',out/'evaluation_run/evaluation/predictions.jsonl'))
    save(out/'results.json',dict(status='complete',training_seed=seed,metrics=metrics,gate_checks=gate,
        eligible_for_next_round_parent=all(gate.values()),actual_new_DPO_training=True,postprocessing_used=False,
        checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,checkpoint_storage='temporary scratch',
        persistent_checkpoint_exported=False))
    verify(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('base',type=Path);parser.add_argument('--seed',type=int,choices=[43,44])
    args=parser.parse_args();base=args.base.resolve();root=base/'ASR_target_format_training_seeds_v1'
    if args.stage=='prepare':prepare(base,root)
    else:
        assert args.seed is not None
        run(base,root,args.seed)
