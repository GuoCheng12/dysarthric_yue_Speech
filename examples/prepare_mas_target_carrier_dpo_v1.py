#!/usr/bin/env python3
"""Freeze identical actual odd-content pairs for full-edit vs target scoring."""
from collections import Counter
from pathlib import Path
import sys
import json

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'examples')]
from therapist_harness.store import sha256_file,write_immutable_json,write_immutable_bytes
from therapist_harness.dpo_training import preference_key
from therapist_harness.preference_token_mask import edit_positions
from mas_target_carrier_mask_v1 import token_plan

B=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_design_exploration_20260908_v1')
PREVIOUS=B/'semantic_context_matched_DPO_v1'
ROOT=B/'target_carrier_scoring_DPO_v1'
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def main():
    from transformers import AutoTokenizer
    ROOT.mkdir(exist_ok=False)
    old=read(PREVIOUS/'training_plan/plan.json')
    preflight=read(PREVIOUS/'next_target_mask_preflight.json')
    assert all(sha256_file(Path(p))==h for p,h in preflight['sources'].items())
    views=rows(PREVIOUS/'next_target_mask_pair_view.jsonl')
    assert len(views)==14 and len({x['pair']['audio_id'] for x in views})==5
    source_pairs={preference_key(x):x for x in rows(PREVIOUS/'MAS/training_preferences.jsonl')}
    assert all(source_pairs[x['preference_key']]==x['pair'] for x in views)
    meta={x['sample_id']:x for x in read(PREVIOUS/'MAS/role_sidecar.json')['samples']}
    anchors={k for k,v in meta.items() if v['role'] in ['protection','maintenance']}
    anchor_pairs=[x for x in source_pairs.values() if x['audio_id'] in anchors]
    assert len(anchors)==8 and len(anchor_pairs)==28
    prefs=[x['pair'] for x in views]+anchor_pairs
    assert len(prefs)==len({preference_key(x) for x in prefs})==42
    write_immutable_bytes(ROOT/'synthetic_preferences.jsonl',(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in prefs)).encode())
    tokenizer=AutoTokenizer.from_pretrained('/data/qwen3-asr/models/Qwen3-ASR-1.7B',local_files_only=True)
    assert tokenizer.is_fast
    plans={x['preference_key']:token_plan(tokenizer,x['pair'],x['target_events']) for x in views}
    checks=[]
    for key,plan in plans.items():
        a,b=plan['chosen'],plan['rejected']
        left,right,_=edit_positions(a['token_ids'][:-1],b['token_ids'][:-1])
        for side,target,edits in [('chosen',a,left),('rejected',b,right)]:
            old_indices=edits+[len(target['token_ids'])-1]
            checks.append(dict(pair_key=key,side=side,all_edit_tokens=len(old_indices),target_tokens=len(target['scored_indices']),
                strict_subset=set(target['scored_indices'])<set(old_indices),
                subset=set(target['scored_indices'])<=set(old_indices),target_surfaces=[x['surface'] for x in target['tokens']]))
    assert any(x['strict_subset'] for x in checks)
    write_immutable_json(ROOT/'focus_token_plan.json',dict(pairs=plans,checks=checks,
        semantics='Target-character overlapping tokenizer offsets plusEOS. Full text/audio conditioning. Real and anchor editEOS unchanged. This changes scoring, not SG review.',
        tokenizer_class=type(tokenizer).__name__,eos_token_id=tokenizer.eos_token_id))
    ids=sorted({x['pair']['audio_id'] for x in views})
    target_order=[ids[i%len(ids)] for i in range(24)]
    protection=sorted(k for k in anchors if meta[k]['role']=='protection')
    maintenance=sorted(k for k in anchors if meta[k]['role']=='maintenance')
    old_dpo=read(PREVIOUS/'training_plan/expected_DPO_exposures.json')['MAS']
    real=[[x for x in w if x['source']=='real'] for w in old_dpo]
    windows=[];selections=[]
    for step in range(1,5):
        selected=target_order[(step-1)*6:step*6]+[protection[step-1],maintenance[step-1]]
        selections+=selected;synthetic=[]
        for i,sid in enumerate(selected):
            group=[x for x in prefs if x['audio_id']==sid]
            assert group
            synthetic += [dict(audio_id=sid,source='synthetic',objective='dpo',step=step,occurrence=(step-1)*8+i,
                preference_key=preference_key(x),loss_weight=.25/8/len(group)) for x in group]
        assert abs(sum(x['loss_weight'] for x in synthetic)-.25)<1e-12
        windows.append(real[step-1]+synthetic)
    files=[Path(__file__),REPO/'examples/mas_target_carrier_mask_v1.py',REPO/'examples/train_mas_target_carrier_dpo_v1.py',
        REPO/'examples/smoke_mas_target_carrier_labels_v1.py',PREVIOUS/'training_plan/plan.json',
        PREVIOUS/'training_plan/expected_DPO_exposures.json',PREVIOUS/'training_plan/expected_common_KL_exposures.json',
        PREVIOUS/'training/MAS/reference_binding.json',PREVIOUS/'MAS_reflection/reflection.json',
        PREVIOUS/'next_target_mask_preflight.json',PREVIOUS/'next_target_mask_pair_view.jsonl',PREVIOUS/'next_minimal_plan.zh.md',
        ROOT/'synthetic_preferences.jsonl',ROOT/'focus_token_plan.json']
    hashes={**old['source_sha256'],**{str(f):sha256_file(f) for f in files}}
    assert all(sha256_file(Path(p))==h for p,h in hashes.items())
    plan=ROOT/'training_plan';plan.mkdir()
    write_immutable_json(plan/'plan.json',dict(parent=old['parent'],source_sha256=hashes,
        data_paths=dict(real_KL=old['data_paths']['real_KL'],real_preferences=old['data_paths']['real_preferences'],
            synthetic_preferences={m:str(ROOT/'synthetic_preferences.jsonl') for m in ['MAS','fixed']}),
        source_sampling_receipts=old['source_sampling_receipts'],selections={m:selections for m in ['MAS','fixed']},
        train_modes=['MAS','fixed'],mode_labels=dict(MAS='target-character token+EOS scoring',fixed='all-edit token+EOS scoring'),
        settings=old['settings'],gate=old['gate'],
        actual_target_pairs=14,actual_target_audio=5,synthetic_unique_audio=13,synthetic_distinct_pairs=42,
        control='EXACTsameactualpairs,audios,visitorder,weights,initialparentandfrozenreference. Only14targetpairsscoredpositionstreatmentdiffers. BothscopesretainEOS. Freshsame-subsetall-editcontrol; previous1408notmatchedcontrol.',
        original_receipts='Preserved real/newnatural/newodd/sharedanchor receipts. Newnatural is inherited context only, not sampled anew or used in these training preferences.',
        admission='Priornewoddness-awarethree-textreviewretained; targetspansareASRpreference-loss metadata, notphoneme/wordaudioaudit orSGadmissionthreshold.'))
    write_immutable_json(plan/'expected_DPO_exposures.json',{m:windows for m in ['MAS','fixed']})
    write_immutable_json(plan/'expected_common_KL_exposures.json',read(PREVIOUS/'training_plan/expected_common_KL_exposures.json'))
    print(json.dumps(dict(pairs=14,audio=5,SG_pairs=42,SG_audio=13,visits=Counter(target_order),
        changed_side_masks=sum(x['strict_subset'] for x in checks),non_subset_masks=sum(not x['subset'] for x in checks))),flush=True)


if __name__=='__main__':main()
