#!/usr/bin/env python3
"""Matched real-SFT controls for an auxiliary MAS synthetic-DPO signal."""

import argparse
import json
from pathlib import Path
import sys
import time

from mas_selection_dpo_probe import REPO,execute,jsonl,read,rows,save
from therapist_harness.store import sha256_file

TRAINER=REPO/'finetune/scripts/qwen3_asr_sft_dpo_probe.py'


def prepare(base,root):
    root.mkdir(exist_ok=False)
    p=read(base/'formal_mas96_v3/protocol.json')
    selected=rows(base/'real96_dpo_probe/inputs.jsonl')
    train={x['utt_id']:x for x in rows(p['real_train']['path'])}
    feedback=rows(p['feedback_split']['path'])
    feedback_ids={x['utt_id'] for x in feedback};feedback_paths={str(Path(x['audio']).resolve()) for x in feedback}
    real=[]
    for x in selected:
        assert x['utt_id'] in train and x['utt_id'] not in feedback_ids
        original=train[x['utt_id']];audio=str(Path(original['audio']).resolve())
        assert audio not in feedback_paths
        prefix='language Cantonese<asr_text>'
        assert original['text'].startswith(prefix)
        real.append(dict(source='real',audio_id=original['utt_id'],audio=audio,
            audio_sha256=sha256_file(Path(audio)),chosen=original['text'][len(prefix):],
            prompt=original.get('prompt',''),language='Cantonese'))
    assert len(real)==96 and len({x['audio_sha256'] for x in real})==96
    feedback_hashes={sha256_file(Path(x['audio'])) for x in feedback}
    assert not feedback_hashes.intersection(x['audio_sha256'] for x in real)
    jsonl(root/'real.jsonl',real)
    arms=dict(real_anchor=None,real_anchor_random=base/'formal_random96_control_v2',
        real_anchor_mas=base/'formal_mas96_v3/round_001')
    spec={}
    sources=[Path(__file__),TRAINER,REPO/'finetune/scripts/qwen3_asr_audio_dpo.py',
        REPO/'src/therapist_harness/dpo_training.py',REPO/'finetune/scripts/evaluate_asr_checkpoint.py',
        base/'real96_dpo_probe/inputs.jsonl',Path(p['real_train']['path']),Path(p['feedback_split']['path']),root/'real.jsonl']
    for name,d in arms.items():
        (root/name).mkdir();spec[name]=None
        if d:
            allocation=d/('allocation.json' if name.endswith('random') else 'training_allocation.json')
            spec[name]=dict(preferences=str(d/'sampling/preferences.jsonl'),allocation=str(allocation),
                receipt=str(d/'sampling/sampling_receipt.json'))
            sources += [Path(x) for x in spec[name].values()]
    save(root/'protocol.json',dict(source_protocol=p,arms=spec,steps=12,real_per_step=8,synthetic_per_step=8,
        synthetic_weight=.05,beta=.1,lr=1e-5,seed=42,
        objective='mean real token CE + 0.05 * mean synthetic audio DPO; real-only control uses CE alone',
        hypothesis='Real acoustic supervision may preserve hard-patient recognition while synthetic preferences supply a small corrective update.',
        criterion='MAS hybrid must improve on BOTH untouched R-only and the matched real-SFT anchor; compare random hybrid to isolate MAS decisions.',
        scope='Independent mixed-objective probe, kept separate from current synthetic-only formal loop',
        real_selection='Same fixed-seed random 96 of 2117 as the earlier real diagnostic, including all-correct audio; no negative construction for real SFT',
        isolation=dict(train_pool=len(train),feedback_count=len(feedback),real_audio_hash_overlap=0),
        synthetic_source='Existing actual parent-sampled preference manifests and separate receipts reused without resampling',
        source_sha256={str(f):sha256_file(f) for f in sources}))


def verify(root):
    cfg=read(root/'protocol.json')
    assert all(sha256_file(Path(f))==digest for f,digest in cfg['source_sha256'].items())
    return cfg


def command(cfg,root,out,*,arm=None,smoke=False):
    p=cfg['source_protocol']
    result=[sys.executable,str(TRAINER),'--model_path',p['base_model_dir'],
        '--policy_adapter_path',p['initial_adapter']['path'],'--reference_adapter_path',p['initial_adapter']['path'],
        '--real_train_file',str(root/('smoke_real.jsonl' if smoke else 'real.jsonl')),
        '--output_dir',str(out/'training'),'--max_steps','1' if smoke else '12',
        '--real_per_step','1' if smoke else '8','--synthetic_per_step','1' if smoke else '8',
        '--synthetic_weight','.05','--lr','1e-5','--beta','.1','--seed','42']
    if smoke:result+=['--synthetic_train_file',str(root/'smoke_preferences.jsonl'),
        '--synthetic_allocation_file',str(root/'smoke_allocation.json')]
    elif arm:result+=['--synthetic_train_file',arm['preferences'],'--synthetic_allocation_file',arm['allocation']]
    return result


def smoke(root):
    cfg=verify(root);d=root/'smoke';d.mkdir(exist_ok=False)
    source=cfg['arms']['real_anchor_mas'];key=read(source['allocation'])['steps'][0][0]
    selected=[x for x in rows(source['preferences']) if x['audio_id']==key]
    assert selected
    jsonl(root/'smoke_real.jsonl',rows(root/'real.jsonl')[:1]);jsonl(root/'smoke_preferences.jsonl',selected)
    save(root/'smoke_allocation.json',dict(steps=[[key]]))
    execute(d,command(cfg,root,d,smoke=True),'training')
    result=read(d/'training/training_result.json');terms=result['history'][0]['weighted_losses']
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert abs(terms['synthetic']-.05*__import__('math').log(2))<1e-4
    exposure=rows(d/'training/training_exposures.jsonl')
    assert sum(x['loss_weight'] for x in exposure if x['source']=='real')==1
    assert abs(sum(x['loss_weight'] for x in exposure if x['source']=='synthetic')-.05)<1e-6
    save(d/'verified.json',dict(status='passed',real_exposures=1,synthetic_exposures=1,
        initial_dpo_is_log2=True,per_source_weights_correct=True,training=result))


def run(root):
    cfg=verify(root);assert read(root/'smoke/verified.json')['status']=='passed'
    p=cfg['source_protocol'];real_order=None;results={}
    for name,arm in cfg['arms'].items():
        d=root/name;started=time.time()
        execute(d,command(cfg,root,d,arm=arm),'training')
        exposure=rows(d/'training/training_exposures.jsonl')
        order=[(x['step'],x['occurrence'],x['audio_id'],x['loss_weight']) for x in exposure if x['source']=='real']
        assert len(order)==96
        if real_order is None:real_order=order
        else:assert order==real_order
        if arm:
            actual={}
            for x in exposure:
                if x['source']=='synthetic':actual[(x['step'],x['occurrence'])]=x['audio_id']
            expected={(i+1,j):audio for i,step in enumerate(read(arm['allocation'])['steps']) for j,audio in enumerate(step)}
            assert actual==expected
        execute(d,[sys.executable,str(REPO/'finetune/scripts/evaluate_asr_checkpoint.py'),
            '--base-model-path',p['base_model_dir'],'--adapter-path',str(d/'training/checkpoint-12'),
            '--split-jsonl',p['feedback_split']['path'],'--out-dir',str(d/'evaluation'),
            '--batch-size','8','--critical-cer-threshold','.5','--max-new-tokens','256'],'evaluation')
        results[name]=dict(metrics=read(d/'evaluation/summary.json')['metrics'],elapsed_seconds=time.time()-started)
        save(d/'results.json',results[name]);print(json.dumps(dict(arm=name,**results[name])),flush=True)
        verify(root)
    save(root/'results.json',dict(status='complete',arms=results,real_exposure_order_identical=True,
        synthetic_exposures_match_source_schedules=True,real_exposures_per_arm=96,reference='frozen initial R-only'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','smoke','run'])
    parser.add_argument('base',type=Path);args=parser.parse_args();base=args.base.resolve();root=base/'real_anchor_probe'
    if args.stage=='prepare':prepare(base,root)
    elif args.stage=='smoke':smoke(root)
    else:run(root)
