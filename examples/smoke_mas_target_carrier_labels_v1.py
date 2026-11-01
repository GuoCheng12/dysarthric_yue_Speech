#!/usr/bin/env python3
"""Actual processor/label checks plus shifted-token gradient isolation."""
import json
from pathlib import Path
import sys
sys.path.insert(0,'/tmp')
import mas_real_even_kl_gpu_smoke_v1 as original
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
from therapist_harness.dpo_training import transcript_log_probs
from mas_target_carrier_mask_v1 import TargetCarrierCollator, targeted_labels
import prepare_mas_target_carrier_dpo_v1 as run


def main():
    torch=original.torch;root=run.ROOT;plan=run.read(root/'training_plan/plan.json')
    assert all(original.sha256_file(Path(p))==h for p,h in plan['source_sha256'].items())
    out=root/'label_smoke';out.mkdir(exist_ok=False)
    processor=Qwen3ASRProcessor.from_pretrained('/data/qwen3-asr/models/Qwen3-ASR-1.7B',local_files_only=True)
    targets=run.read(root/'focus_token_plan.json')['pairs']
    prefs=run.rows(root/'synthetic_preferences.jsonl')
    target_rows=[r for r in prefs if original.preference_key(r) in targets]
    anchor=next(r for r in prefs if original.preference_key(r) not in targets)
    real=run.rows(plan['data_paths']['real_preferences'])[0]
    examples=target_rows+[anchor,real]
    legacy=original.RealSyntheticEditCollator(processor,{})
    full=TargetCarrierCollator(processor,{},targets,False)
    focus=TargetCarrierCollator(processor,{},targets,True)
    changed=[]
    for start in range(0,len(examples),4):
        group=examples[start:start+4]
        a,b,c=legacy(group),full(group),focus(group)
        assert set(a)==set(b)==set(c)
        assert all(torch.equal(a[k],b[k]) for k in a)
        assert all(torch.equal(a[k],c[k]) for k in a if k!='labels')
        for i,row in enumerate(group):
            key=original.preference_key(row)
            assert legacy.audit[key]==full.audit[key]
            if key not in targets:
                assert legacy.audit[key]==focus.audit[key]
                assert torch.equal(a['labels'][[i,i+len(group)]],c['labels'][[i,i+len(group)]])
            else:
                record=focus.audit[key]
                assert record['chosen_scored_indices']==targets[key]['chosen']['scored_indices']
                assert record['rejected_scored_indices']==targets[key]['rejected']['scored_indices']
                changed.append(not torch.equal(a['labels'][[i,i+len(group)]],c['labels'][[i,i+len(group)]]))
    assert len(changed)==14 and any(changed)
    teacher=run.rows(plan['data_paths']['real_KL'])[:1]
    a,b,c=[x(teacher,include_rejected=False) for x in [legacy,full,focus]]
    assert all(torch.equal(a[k],b[k]) and torch.equal(a[k],c[k]) for k in a)
    # Independent mathematical check: causal next-token shift, EOS and padding.
    labels=torch.tensor([[-100,-100,2,3,4,5,-100,-100],[-100,-100,2,6,4,5,-100,-100]])
    fixture={s:dict(token_ids=t,scored_indices=[1,3]) for s,t in [('chosen',[2,3,4,5]),('rejected',[2,6,4,5])]}
    selected=targeted_labels(labels,[0,1],fixture)
    logits=torch.zeros((2,8,8),requires_grad=True)
    logp,counts=transcript_log_probs(logits,selected);(-logp.sum()).backward()
    assert counts.tolist()==[2,2]
    active=logits.grad.abs().sum(-1).ne(0)
    expected=torch.zeros((2,8),dtype=torch.bool);expected[:,[2,4]]=True
    assert torch.equal(active,expected)
    result=dict(passed=True,actual_target_pairs=14,changed_pair_masks=sum(changed),
        original_full_edit_labels_exact=True,full_audio_text_inputs_identical=True,
        real_and_anchor_edit_labels_exact=True,full_real_KL_labels_exact=True,
        causal_shift_EOS_padding_gradient_check=True,
        policy_or_reference_updated=False,actual_ASR_decoding=False,
        source_sha256={str(Path(__file__)):original.sha256_file(Path(__file__)),str(root/'training_plan/plan.json'):original.sha256_file(root/'training_plan/plan.json')},
        focus_audit=focus.audit)
    original.write_immutable_json(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['focus_audit','source_sha256']}),flush=True)


if __name__=='__main__':main()
