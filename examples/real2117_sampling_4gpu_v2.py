#!/usr/bin/env python3
"""Resume interrupted real-ASR draws with per-record provenance on four GPUs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import real2117_dpo_kl_v1 as old
from therapist_harness.preference_data import build_preference_pairs, sample_seed
from therapist_harness.store import sha256_file, sha256_directory, write_immutable_json

PREVIOUS=old.ROOT
ROOT=PREVIOUS/'resource_4gpu_v2'
read,rows,jsonl=old.read,old.rows,old.jsonl


def prepare():
    old.verify()
    ROOT.mkdir(exist_ok=False)
    inputs=rows(PREVIOUS/'real_inputs.jsonl');by_id={r['audio_id']:r for r in inputs}
    recovered=[];origins=[];bindings=[];common=None
    base_sha=sha256_directory(Path(old.BASE))
    for shard in range(2):
        directory=PREVIOUS/f'sampling_shard{shard}'
        config=read(directory/'sampling_config.json')
        assert config['current_asr_sha256']==old.PARENT_SHA and config['base_model_sha256']==base_sha
        assert config['source_sha256']==sha256_file(old.REPO/'examples/sample_asr_preferences.py')
        assert config['preference_code_sha256']==sha256_file(old.REPO/'src/therapist_harness/preference_data.py')
        assert config['input_manifest_sha256']==sha256_file(PREVIOUS/f'inputs_shard{shard}.jsonl')
        assert rows(directory/'validated_inputs.jsonl')==inputs[shard::2]
        settings={k:config[k] for k in ['base_model_sha256','current_asr_sha256','samples_per_audio','seed','temperature','top_p','top_k','max_new_tokens','language','prompt','do_sample','real_train_sha256','feedback_sha256']}
        if common is None:common=settings
        else:assert settings==common
        path=directory/'candidates.jsonl';file_hash=sha256_file(path)
        discarded_tail=0
        for lineno,line in enumerate(path.read_bytes().splitlines(keepends=True),1):
            if not line.endswith(b'\n'):
                discarded_tail=len(line);continue
            record=json.loads(line);row=by_id[record['audio_id']]
            assert record['audio_id'] in {r['audio_id'] for r in inputs[shard::2]}
            assert record['audio_sha256']==row['audio_sha256']==sha256_file(Path(row['audio']))
            assert record['current_asr_sha256']==old.PARENT_SHA
            assert record['seed']==sample_seed(42,record['audio_id'],record['sample_index'])
            assert record['language']=='Cantonese' and record['prompt']=='' and 0<=record['sample_index']<4
            tokens=record['generated_token_ids']
            assert record['ended_with_eos']==bool(tokens and tokens[-1] in {151645,151643})
            assert record['reached_max_new_tokens']==(len(tokens)>=256)
            recovered.append(record)
            origins.append(dict(sample_id=record['sample_id'],source_path=str(path),source_sha256=file_hash,line=lineno))
        bindings.append(dict(directory=str(directory),config_sha256=sha256_file(directory/'sampling_config.json'),
            validated_inputs_sha256=sha256_file(directory/'validated_inputs.jsonl'),candidates_sha256=file_hash,
            normal_completion_receipt_exists=(directory/'sampling_receipt.json').exists(),discarded_incomplete_tail_bytes=discarded_tail))
    assert len({x['sample_id'] for x in recovered})==len(recovered)
    for path in ['real_inputs.jsonl']:
        jsonl(ROOT/path,inputs)
    for shard in range(4):jsonl(ROOT/f'inputs_shard{shard}.jsonl',inputs[shard::4])
    jsonl(ROOT/'recovered_candidates.jsonl',recovered);jsonl(ROOT/'recovered_origins.jsonl',origins)
    counts=Counter(x['audio_id'] for x in recovered)
    write_immutable_json(ROOT/'migration.json',dict(schema='interrupted_sampling_recovery_v2',created_utc=datetime.now(timezone.utc).isoformat(),
        old_root=str(PREVIOUS),original_protocol_sha256=sha256_file(PREVIOUS/'protocol.json'),
        original_bindings=bindings,common_settings=common,recovered_draws=len(recovered),complete_audio=sum(v==4 for v in counts.values()),
        partial_audio=sum(v<4 for v in counts.values()),remaining_draws=8468-len(recovered),
        original_end_verification='Unavailable: old workers exited without completion receipts. Current files match their recorded start identities; individual durable records checked.',
        greedy_recovery='Old greedy records were held in memory and are absent. Regenerate greedy only, write each result immediately; retain all valid stochastic draws.',
        recovered_candidates_sha256=sha256_file(ROOT/'recovered_candidates.jsonl'),recovered_origins_sha256=sha256_file(ROOT/'recovered_origins.jsonl'),
        sampler_sha256=sha256_file(Path(__file__)),original_sources=read(PREVIOUS/'protocol.json')['source_sha256']))
    print(json.dumps(dict(stage='recovery_prepared',recovered_draws=len(recovered),complete_audio=sum(v==4 for v in counts.values()),remaining_draws=8468-len(recovered))),flush=True)


def verify():
    m=read(ROOT/'migration.json')
    assert sha256_file(Path(__file__))==m['sampler_sha256']
    assert all(sha256_file(Path(p))==h for p,h in m['original_sources'].items())
    assert sha256_file(ROOT/'recovered_candidates.jsonl')==m['recovered_candidates_sha256']
    assert sha256_file(ROOT/'recovered_origins.jsonl')==m['recovered_origins_sha256']
    assert sha256_directory(Path(old.PARENT))==old.PARENT_SHA
    return m


def sample(shard, smoke=False):
    m=verify()
    import torch
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import normalize_audio_input,parse_asr_output
    from transformers import set_seed
    before_base=sha256_directory(Path(old.BASE));assert before_base==m['common_settings']['base_model_sha256']
    recovered={(r['audio_id'],r['sample_index']):r for r in rows(ROOT/'recovered_candidates.jsonl')}
    origins={r['sample_id']:r for r in rows(ROOT/'recovered_origins.jsonl')}
    inputs=rows(ROOT/f'inputs_shard{shard}.jsonl')
    if smoke:
        ids=list(dict.fromkeys(r['audio_id'] for r in recovered.values()))
        chosen={ids[0],ids[-1]}
        inputs=[r for r in rows(ROOT/'real_inputs.jsonl') if r['audio_id'] in chosen]
    out=ROOT/('sampling_smoke' if smoke else f'sampling_shard{shard}');out.mkdir(exist_ok=False)
    jsonl(out/'validated_inputs.jsonl',inputs)
    config=dict(schema='resumable_sampling_config_v2',shard=shard,settings=m['common_settings'],
        inputs_sha256=sha256_file(out/'validated_inputs.jsonl'),sampler_sha256=sha256_file(Path(__file__)),migration_sha256=sha256_file(ROOT/'migration.json'))
    write_immutable_json(out/'sampling_config.json',config)
    wrapper=Qwen3ASRModel.from_pretrained(old.BASE,dtype=torch.bfloat16,device_map='cuda:0',max_inference_batch_size=1,max_new_tokens=256)
    wrapper.model=PeftModel.from_pretrained(wrapper.model,old.PARENT,is_trainable=False).merge_and_unload()
    wrapper.model.eval();wrapper.model.requires_grad_(False)
    generated=reused=matches=0;t0=time.time()
    with (out/'candidates.jsonl').open('x') as cfile,(out/'greedy.jsonl').open('x') as gfile,(out/'sample_origins.jsonl').open('x') as ofile:
        for row_index,row in enumerate(inputs):
            assert sha256_file(Path(row['audio']))==row['audio_sha256']
            wav=normalize_audio_input(row['audio'])
            prompt=wrapper._build_text_prompt(context='',force_language='Cantonese')
            features=wrapper.processor(text=[prompt],audio=[wav],return_tensors='pt',padding=True)
            features=features.to(wrapper.model.device).to(wrapper.model.dtype)
            prefix_length=features['input_ids'].shape[1]
            with torch.inference_mode():normal=wrapper.model.generate(**features,do_sample=False,num_beams=1,max_new_tokens=256)
            tokens=normal.sequences[0,prefix_length:].tolist()
            raw=wrapper.processor.batch_decode([tokens],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            _,text=parse_asr_output(raw,user_language='Cantonese')
            greedy=dict(audio_id=row['audio_id'],audio_sha256=row['audio_sha256'],current_asr_sha256=old.PARENT_SHA,
                prediction=text,raw_output=raw,language='Cantonese',prompt='',generated_token_ids=tokens,
                ended_with_eos=bool(tokens and tokens[-1] in {151645,151643}),reached_max_new_tokens=len(tokens)>=256,do_sample=False,num_beams=1)
            gfile.write(json.dumps(greedy,ensure_ascii=False)+'\n');gfile.flush()
            for index in range(4):
                key=(row['audio_id'],index);sample_id=f"{row['audio_id']}::{index}"
                if key in recovered and not smoke:
                    candidate=recovered[key];origin=dict(origins[sample_id],kind='recovered_interrupted_record');reused+=1
                else:
                    seed=sample_seed(42,row['audio_id'],index);set_seed(seed)
                    with torch.inference_mode():result=wrapper.model.generate(**features,do_sample=True,temperature=.8,top_p=.95,top_k=0,max_new_tokens=256)
                    tokens=result.sequences[0,prefix_length:].tolist()
                    raw=wrapper.processor.batch_decode([tokens],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                    language,text=parse_asr_output(raw,user_language='Cantonese')
                    candidate=dict(schema_version='asr_sample_v1',audio_id=row['audio_id'],sample_id=sample_id,sample_index=index,
                        seed=seed,audio_sha256=row['audio_sha256'],current_asr_sha256=old.PARENT_SHA,raw_output=raw,prediction=text,
                        language='Cantonese',detected_language=language,prompt='',generated_token_ids=tokens,
                        ended_with_eos=bool(tokens and tokens[-1] in {151645,151643}),reached_max_new_tokens=len(tokens)>=256)
                    origin=dict(sample_id=sample_id,kind='new_draw',sampler_sha256=config['sampler_sha256']);generated+=1
                    if smoke and key in recovered:
                        assert candidate==recovered[key],sample_id
                        matches+=1
                cfile.write(json.dumps(candidate,ensure_ascii=False)+'\n');cfile.flush()
                ofile.write(json.dumps(origin,ensure_ascii=False)+'\n');ofile.flush()
            print(json.dumps(dict(stage='sampling',shard=shard,audio=row_index+1,total=len(inputs),generated_draws=generated,reused_draws=reused,seconds=time.time()-t0)),flush=True)
    assert sha256_directory(Path(old.PARENT))==old.PARENT_SHA and sha256_directory(Path(old.BASE))==before_base
    verify()
    receipt=dict(schema='resumed_sampling_shard_receipt_v2',status='complete',shard=shard,config=config,audio_count=len(inputs),
        generated_draws=generated,recovered_draws=reused,candidate_count=generated+reused,greedy_count=len(inputs),
        current_execution_model_files_verified_unchanged=True,original_interrupted_execution_not_relabelled_complete=True,
        artifacts={name:sha256_file(out/name) for name in ['candidates.jsonl','greedy.jsonl','sample_origins.jsonl','validated_inputs.jsonl']})
    write_immutable_json(out/'sampling_receipt.json',receipt)
    if smoke:write_immutable_json(out/'result.json',dict(passed=True,matched_old_records_exactly=matches,new_verification_draws=generated,not_used_for_training=True))


def merge():
    m=verify();inputs=rows(ROOT/'real_inputs.jsonl');candidates=[];greedy=[];receipts=[];seen=set();reused=0
    original={(x['audio_id'],x['sample_index']):x for x in rows(ROOT/'recovered_candidates.jsonl')}
    for shard in range(4):
        out=ROOT/f'sampling_shard{shard}';r=read(out/'sampling_receipt.json')
        assert r['status']=='complete' and r['config']['settings']==m['common_settings']
        assert all(sha256_file(out/name)==h for name,h in r['artifacts'].items())
        assert rows(out/'validated_inputs.jsonl')==inputs[shard::4]
        values=rows(out/'candidates.jsonl');sources=rows(out/'sample_origins.jsonl')
        assert len(values)==len(sources)==4*len(inputs[shard::4])
        for value,source in zip(values,sources):
            key=(value['audio_id'],value['sample_index']);assert key not in seen;seen.add(key)
            assert source['sample_id']==value['sample_id']
            if key in original:
                assert value==original[key] and source['kind']=='recovered_interrupted_record';reused+=1
            else:assert source['kind']=='new_draw'
        candidates+=values;greedy+=rows(out/'greedy.jsonl')
        receipts.append(dict(path=str(out/'sampling_receipt.json'),sha256=sha256_file(out/'sampling_receipt.json')))
    assert len(candidates)==8468 and reused==m['recovered_draws']
    pairs,audit=build_preference_pairs(inputs,candidates,old.PARENT_SHA,expected_samples=4,greedy_candidates=greedy)
    order={x['audio_id']:i for i,x in enumerate(inputs)};candidates.sort(key=lambda x:(order[x['audio_id']],x['sample_index']))
    jsonl(ROOT/'candidates.jsonl',candidates);jsonl(ROOT/'preferences.jsonl',pairs)
    write_immutable_json(ROOT/'preference_audit.json',audit)
    write_immutable_json(ROOT/'sampling_shard_index.json',dict(schema='four_shard_recovered_sampling_index_v2',source_receipts=receipts,
        migration_sha256=sha256_file(ROOT/'migration.json'),audio_count=2117,candidate_count=8468,recovered_draws=reused,
        newly_generated_draws=8468-reused,preferences_sha256=sha256_file(ROOT/'preferences.jsonl'),
        complete_disjoint_coverage=True,original_partial_receipts_preserved=True))
    old.ROOT=ROOT;old.emit('preferences_ready',audio=2117,eligible_audio=audit['paired_audio_count'],pairs=len(pairs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','sample','smoke','merge']);p.add_argument('--shard',type=int,default=0)
    a=p.parse_args()
    if a.stage in ['sample','smoke']:sample(a.shard,smoke=a.stage=='smoke')
    else:globals()[a.stage]()
