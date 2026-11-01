#!/usr/bin/env python3
"""Decompose final-checkpoint real retention KL into text and EOS positions."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

import torch

from mas_selection_dpo_probe import REPO, read, rows, save
from therapist_harness.dpo_training import AudioPreferenceCollator, move_batch
from therapist_harness.store import sha256_file, sha256_directory

sys.path.insert(0,str(REPO/'finetune/scripts'))
from qwen3_asr_residual_dpo import load_merged_parent, add_residual, frozen_base_digest
from qwen3_asr_kl_dpo_storage_v2 import transcript_distributions
from qwen3_asr_sft import patch_outer_forward


def measure(training,teacher_path,output):
    protocol=read(training/'run_protocol.json');a=protocol['arguments']
    checkpoint=Path(read(training/'training_result.json')['checkpoint'])
    metadata=read(training/'real_teacher_cache.json');data=rows(a['real_train_file'])
    weights=Counter()
    for x in rows(training/'training_exposures.jsonl'):
        if x['source']=='real':weights[x['audio_id']]+=x['loss_weight']
    assert set(weights)=={x['audio_id'] for x in data}==set(metadata['audio'])
    files=[Path(__file__),teacher_path,Path(a['real_train_file']),training/'run_protocol.json',
        training/'real_teacher_cache.json',training/'training_exposures.jsonl',training/'training_result.json',
        REPO/'finetune/scripts/qwen3_asr_residual_dpo.py',REPO/'finetune/scripts/qwen3_asr_kl_dpo_storage_v2.py',
        REPO/'src/therapist_harness/dpo_training.py']
    hashes={str(f):sha256_file(f) for f in files};checkpoint_sha=sha256_directory(checkpoint)
    cache=torch.load(teacher_path,map_location='cpu',weights_only=True)
    for key in weights:
        value=cache[key];expected=metadata['audio'][key]
        assert list(value.shape)==expected['shape']
        assert hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()==expected['tensor_sha256']
    assert sha256_directory(Path(a['reference_adapter_path']))==protocol['reference_asr_sha256']
    torch.manual_seed(a['seed']);device=torch.device('cuda:0');dtype=torch.bfloat16
    wrapper=load_merged_parent(a['model_path'],a['reference_adapter_path'],device,dtype)
    patch_outer_forward(wrapper.model)
    model=add_residual(wrapper.model,a['reference_adapter_path'],checkpoint)
    model.requires_grad_(False);model.eval();model.config.use_cache=False
    expected_base=read(training/'reference_runtime.json')['merged_base_tensor_sha256']
    assert frozen_base_digest(model)==expected_base
    collator=AudioPreferenceCollator(wrapper.processor);eos=wrapper.processor.tokenizer.eos_token_id
    period=wrapper.processor.tokenizer.encode('。',add_special_tokens=False);assert len(period)==1
    measurements=[]
    with torch.inference_mode():
        for row in data:
            batch=move_batch(collator([row],include_rejected=False),device,dtype)
            labels=batch['labels'][0];ids=labels[labels.ne(-100)]
            assert ids[-1].item()==eos and len(ids)>1
            student=transcript_distributions(model,batch);teacher=cache[row['audio_id']].to(device)
            assert student.shape==teacher.shape
            terms=.5*((student.exp()-teacher.exp())*(student-teacher)).sum(-1)
            length=len(terms)
            measurements.append(dict(audio_id=row['audio_id'],chosen=row['chosen'],tokens=length,
                weight=weights[row['audio_id']],all_token_mean_KL=terms.mean().item(),
                text_contribution=terms[:-1].sum().item()/length,EOS_contribution=terms[-1].item()/length,
                EOS_position_KL=terms[-1].item(),
                reference_P_EOS_at_end=teacher[-1,eos].exp().item(),policy_P_EOS_at_end=student[-1,eos].exp().item(),
                reference_P_period_at_end=teacher[-1,period[0]].exp().item(),policy_P_period_at_end=student[-1,period[0]].exp().item()))
    keys=[k for k in measurements[0] if k not in ['audio_id','chosen','tokens','weight']]
    total=sum(x['weight'] for x in measurements)
    weighted={k:sum(x['weight']*x[k] for x in measurements)/total for k in keys}
    uniform={k:sum(x[k] for x in measurements)/len(measurements) for k in keys}
    assert abs(weighted['all_token_mean_KL']-weighted['text_contribution']-weighted['EOS_contribution'])<1e-6
    assert sha256_directory(checkpoint)==checkpoint_sha
    assert frozen_base_digest(model)==expected_base
    assert all(sha256_file(Path(f))==h for f,h in hashes.items())
    save(output,dict(status='complete',checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
        source_sha256=hashes,real_audio=len(measurements),EOS_token_id=eos,period_token_id=period[0],
        patient_weighted=weighted,uniform=uniform,audio=measurements,
        scope='Post-training checkpoint teacher-forced diagnostic on real-training recordings, no updates or new greedy decodes. Full-run realized KL coefficients supply weights. EOS contribution uses the original per-audio full-token denominator; not comparable as an identical sample to the logged pre-update final-step loss.'))
    print(weighted,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('training',type=Path)
    parser.add_argument('teacher_cache',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();measure(args.training.resolve(),args.teacher_cache.resolve(),args.output.resolve())
