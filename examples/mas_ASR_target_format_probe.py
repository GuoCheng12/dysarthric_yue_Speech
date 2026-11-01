#!/usr/bin/env python3
"""Declared terminal-punctuation label intervention with full/edit+EOS scoring."""

import argparse
from pathlib import Path
import shutil
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save
from mas_edit_mask_dpo_probe import TRAINER, command, check_training, exposure_plan
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from therapist_harness.text import text_normalize
from therapist_harness.store import sha256_file, sha256_directory

ARMS=['full','edit_eos']
SCRATCH=Path('/tmp/qwen3_asr_mas_target_format_v1')


def verify(root):
    cfg=read(root/'protocol.json')
    assert all(sha256_file(Path(f))==h for f,h in cfg['source_sha256'].items())
    return cfg


def prepare(base,root):
    previous=base/'edit_mask_dpo_probe_v1';old=read(previous/'protocol.json')
    assert all(sha256_file(Path(f))==h for f,h in old['source_sha256'].items())
    assert read(previous/'smoke/verified.json')['status']=='passed'
    data=base/'ASR_target_format_inputs_v1';receipt=read(data/'receipt.json')
    source=Path(receipt['source_preferences']);target=data/'preferences.jsonl'
    assert sha256_file(source)==receipt['source_sha256']
    assert sha256_file(target)==receipt['output_sha256']
    original,normalized=rows(source),rows(target)
    assert len(original)==len(normalized)==15
    for before,after in zip(original,normalized):
        assert after['chosen']==before['chosen'].rstrip('。！？!?')
        assert text_normalize(after['chosen'])==text_normalize(before['chosen'])
        assert text_normalize(after['chosen'])!=text_normalize(after['rejected'])
        assert after['metadata']['original_target_text']==before['chosen']
        assert {k:v for k,v in before.items() if k not in ['chosen','metadata']}=={k:v for k,v in after.items() if k not in ['chosen','metadata']}
        assert sha256_file(Path(after['audio']))==after['audio_sha256']
    assert shutil.disk_usage('/tmp').free>400*1024**2
    root.mkdir(exist_ok=False);SCRATCH.mkdir(exist_ok=False)
    arguments={};expected=None
    for arm in ARMS:
        out=root/arm;out.mkdir();scratch=SCRATCH/arm;scratch.mkdir()
        a=dict(old['arguments'][arm],synthetic_train_file=str(target),output_dir=str(scratch/'training'))
        records=exposure_plan(a,rows(a['real_train_file']),normalized,read(a['synthetic_allocation_file']),read(a['real_loss_factors'])['factors'])
        control=rows(previous/arm/'training/training_exposures.jsonl')
        assert len(records)==len(control)==288
        for before,after in zip(control,records):
            if before['source']=='real':assert before==after
            else:
                assert {k:v for k,v in before.items() if k!='preference_key'}=={k:v for k,v in after.items() if k!='preference_key'}
                assert before['preference_key']!=after['preference_key']
        if expected is None:expected=records
        else:assert expected==records
        arguments[arm]=a
    save(root/'expected_exposures.json',expected)
    save(root/'cpu_smoke.json',dict(status='passed',actual_new_GPU_smoke=False,
        all15_CER_normalized_targets_unchanged=True,all15_actual_negatives_and_audio_unchanged=True,
        exact192_real_exposure_records_preserved=True,synthetic96_identity_order_weights_preserved=True,
        synthetic_preference_keys_changed_with_chosen_text=True,
        reused_same_engine_actual_GPU_smoke=str(previous/'smoke/verified.json'),
        same_engine_full12_step_checks=[str(previous/arm/'verified.json') for arm in ARMS]))
    files=[Path(__file__),TRAINER,REPO/'examples/mas_edit_mask_dpo_probe.py',data/'receipt.json',target,
        previous/'protocol.json',previous/'smoke/verified.json',root/'expected_exposures.json',root/'cpu_smoke.json',
        base/'next_ASR_target_format_probe_plan_v1.json']
    for arm in ARMS:
        files += [previous/arm/n for n in ['results.json','verified.json','training/training_result.json',
            'training/training_exposures.jsonl','training/preference_token_masks.json',
            'training/reference_log_probs.json','training/real_teacher_cache.json']]
    save(root/'protocol.json',dict(source_protocol=old['source_protocol'],
        source_sha256={**old['source_sha256'],**{str(f):sha256_file(f) for f in files}},
        arguments=arguments,previous=str(previous),arms=ARMS,
        actual_new_generation=False,actual_new_sampling=False,actual_new_MAS_decision=False,actual_new_DPO_training=True,
        intervention='Remove only chosen terminal sentence punctuation, retaining raw SG target in metadata. Compare full/edit_eos scores with unchanged audio, actual negatives, real retention, exposures and optimizer settings.',
        storage=dict(checkpoint_root=str(SCRATCH),checkpoint_durability='temporary local scratch; not a persistent workspace checkpoint',
            persistent_record_root=str(root),small_training_receipts='byte-identical copies retained in each arm/training_receipts',
            model_export='A useful checkpoint may later be copied to workspace with a verified whole-directory hash and separate export receipt. No legacy artifact is removed.'),
        acceptance='Original pooled/macro/hard real-development gate; no inference postprocessing or immediate parent promotion.'))


def copy_receipts(training,out):
    destination=out/'training_receipts';destination.mkdir(exist_ok=False);copies=[]
    for source in sorted(training.iterdir()):
        if source.is_file() and source.suffix in {'.json','.jsonl'}:
            target=destination/source.name;shutil.copyfile(source,target)
            digest=sha256_file(source);assert sha256_file(target)==digest
            copies.append(dict(source=str(source),copy=str(target),sha256=digest))
    save(out/'training_receipt_copies.json',dict(status='complete',checkpoint_copied=False,
        source_training_directory=str(training),copies=copies))


def train(base,root,arm):
    cfg=verify(root);assert read(root/'cpu_smoke.json')['status']=='passed'
    out=root/arm;a=cfg['arguments'][arm];training=Path(a['output_dir'])
    execute(out,command(a),'training')
    result,teacher,counts=check_training(training.parent,read(root/'expected_exposures.json'),arm)
    previous=Path(cfg['previous'])/arm
    assert teacher['audio']==read(previous/'training/real_teacher_cache.json')['audio']
    masks=read(training/'preference_token_masks.json')
    reference=read(training/'reference_log_probs.json')
    assert set(masks['pairs'])==set(reference['values'])
    assert not (set(reference['values'])&set(read(previous/'training/reference_log_probs.json')['values']))
    checkpoint=Path(result['checkpoint']);checkpoint_sha=sha256_directory(checkpoint)
    copy_receipts(training,out)
    save(out/'verified.json',dict(status='passed',actual288_exposures_match=True,
        all96_real_teacher_distributions_identical=True,all15_preference_scores_recomputed_for_new_targets=True,
        initial_log2_DPO_and_zero_KL=True,actual_token_counts=counts,
        checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,checkpoint_storage='temporary scratch',
        persistent_checkpoint_exported=False))
    execute(out,evaluation_command(cfg['source_protocol'],out/'evaluation_run',checkpoint),'evaluation')
    assert sha256_directory(checkpoint)==checkpoint_sha
    metrics=read(out/'evaluation_run/evaluation/summary.json')['metrics'];parent=read(base/'parent_eval_check/summary.json')['metrics']
    gate=dict(pooled_improves=metrics['overall']['pooled_cer']<parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer']-parent['overall']['patient_macro_cer']<=.001,
        hard_regression_within_limit=metrics['hard']['pooled_cer']-parent['hard']['pooled_cer']<=.005)
    save(out/'paired_parent.json',compare(base/'parent_eval_check/predictions.jsonl',out/'evaluation_run/evaluation/predictions.jsonl'))
    save(out/'paired_raw_target_control.json',compare(previous/'evaluation_run/evaluation/predictions.jsonl',out/'evaluation_run/evaluation/predictions.jsonl'))
    save(out/'results.json',dict(status='complete',preference_token_mask=arm,metrics=metrics,gate_checks=gate,
        eligible_for_next_round_parent=all(gate.values()),actual_new_DPO_training=True,postprocessing_used=False,
        checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,checkpoint_storage='temporary scratch',
        persistent_checkpoint_exported=False))
    verify(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','train'])
    parser.add_argument('base',type=Path);parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();base=args.base.resolve();root=base/'ASR_target_format_dpo_probe_v1'
    if args.stage=='prepare':prepare(base,root)
    else:
        assert args.arm is not None
        train(base,root,args.arm)
