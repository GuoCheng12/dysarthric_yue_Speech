#!/usr/bin/env python3
"""Three model replicas sum globally weighted gradients once per original window."""
import os
if 'LOCAL_RANK' in os.environ:
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2').split(',')
    os.environ['CUDA_VISIBLE_DEVICES']=visible[int(os.environ['LOCAL_RANK'])]

import argparse
from collections import Counter,defaultdict
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import random
import time
import real2117_sampling_4gpu_v2 as sampling

old=sampling.old
ROOT=sampling.ROOT
old.ROOT=ROOT
read,rows=old.read,old.rows
write_json=old.write_immutable_json


def main(smoke=False):
    execution=read(ROOT/'training_execution_v3.json')
    assert all(old.sha256_file(Path(p))==h for p,h in execution['source_sha256'].items())
    plan=old.verify();torch,rt,set_seed=old.runtime()
    assert old.sha256_directory(Path(old.BASE))==read(ROOT/'migration.json')['common_settings']['base_model_sha256']
    import torch.distributed as dist
    torch.cuda.set_device(0)
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);assert world==3
    out=ROOT/('smoke_3gpu_v3' if smoke else 'training')
    if rank==0:out.mkdir(exist_ok=False)
    real=rows(ROOT/'real_inputs.jsonl')
    if smoke:
        source=old.B/'real96_dpo_probe/sampling';receipt=read(source/'sampling_receipt.json')
        assert receipt['status']=='complete' and receipt['current_asr_sha256']==old.PARENT_SHA
        assert old.sha256_file(source/'candidates.jsonl')==receipt['candidates_sha256']
        prefs=rows(source/'preferences.jsonl');ids=list(dict.fromkeys(x['audio_id'] for x in prefs))[:5]
        prefs=[r for r in prefs if r['audio_id'] in ids];real=[r for r in real if r['audio_id'] in ids]
        if rank==0:write_json(out/'source.json',dict(path=str(source),receipt_sha256=old.sha256_file(source/'sampling_receipt.json'),preferences_sha256=old.sha256_file(source/'preferences.jsonl'),discarded_smoke_only=True))
    else:
        passed=read(ROOT/'smoke_3gpu_v3/result.json')
        assert passed['passed'] and passed['trainer_sha256']==old.sha256_file(Path(__file__))
        prefs=rows(ROOT/'preferences.jsonl')
        assert old.sha256_file(ROOT/'preferences.jsonl')==read(ROOT/'sampling_shard_index.json')['preferences_sha256']
    by_id={r['audio_id']:r for r in real};groups=defaultdict(list)
    for p in prefs:
        assert p['source']=='real' and p['current_asr_sha256']==old.PARENT_SHA
        assert p['audio_id'] in by_id and p['chosen']==by_id[p['audio_id']]['chosen']
        assert p['audio_sha256']==by_id[p['audio_id']]['audio_sha256']
        groups[p['audio_id']].append(p)
    assert groups
    for group in groups.values():assert math.isclose(sum(p['pair_weight'] for p in group),1.)
    for r in real:assert old.sha256_file(Path(r['audio']))==r['audio_sha256']
    set_seed(42);reference,processor=rt.load_model(old.BASE,old.PARENT,False)
    reference_sha=rt.parameter_digest(reference);base_sha=rt.frozen_base_digest(reference)
    collator=rt.RealSyntheticEditCollator(processor,{})
    device,dtype=torch.device('cuda:0'),torch.bfloat16
    cache={};t0=time.time()
    with torch.no_grad():
        assigned=prefs[rank::world]
        for i,p in enumerate(assigned):
            values,_=rt.batch_log_probs(reference,rt.move_batch(collator([p]),device,dtype))
            cache[rt.preference_key(p)]=values.detach().cpu().tolist()
            if i%100==0:print(json.dumps(dict(stage='reference_scores',rank=rank,done=i+1,total=len(assigned),seconds=time.time()-t0)),flush=True)
    # PEFT must load unsharded adapters before a process group is initialized.
    set_seed(42);policy,_=rt.load_model(old.BASE,old.PARENT,True)
    assert rt.frozen_base_digest(policy)==base_sha
    initial=rt.parameter_digest(policy,trainable_only=True)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    dist.barrier(device_ids=[0])
    collected=[None]*world
    dist.all_gather_object(collected,dict(scores=cache,masks=collator.audit))
    all_masks={};cache={}
    for data in collected:
        assert not set(cache).intersection(data['scores'])
        cache.update(data['scores']);all_masks.update(data['masks'])
    assert len(cache)==len(prefs)
    cache={k:torch.tensor(v,dtype=torch.float32) for k,v in cache.items()}
    if rank==0:write_json(out/'reference_scores.json',dict(parent_sha256=old.PARENT_SHA,reference_parameter_sha256=reference_sha,
        scores={k:v.tolist() for k,v in cache.items()},masks=all_masks,disjoint_rank_scoring=True))
    initial_all=[None]*world;dist.all_gather_object(initial_all,initial);assert len(set(initial_all))==1
    policy.train();policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    parameters=[p for p in policy.parameters() if p.requires_grad]
    assert parameters and all(p.dtype==torch.float32 for p in parameters)
    optimizer=torch.optim.AdamW(parameters,lr=plan['lr'],weight_decay=0.)
    steps_per_epoch=1 if smoke else 265;epochs=2 if smoke else 3
    counts=Counter();step=0;history=[];equivalence=[]
    exposures=(out/f'training_exposures_rank{rank}.jsonl').open('x')

    def backward_audio(audio_id,epoch,record=True):
        r=by_id[audio_id];terms=Counter()
        batch=rt.move_batch(collator([r],include_rejected=False),device,dtype)
        with torch.no_grad():teacher=rt.transcript_distributions(reference,batch)
        student=rt.transcript_distributions(policy,batch)
        if smoke and step==0:assert torch.equal(student.detach(),teacher)
        kl=rt.symmetric_kl(student,teacher)
        weight=10.*steps_per_epoch/len(real)
        (kl*weight).backward();terms['KL_weighted']=kl.item()*weight
        if record:
            counts[epoch,'KL',audio_id]+=1
            exposures.write(json.dumps(dict(epoch=epoch+1,step=step+1,objective='real_KL',audio_id=audio_id,weight=weight,rank=rank))+'\n')
        del batch,teacher,student,kl
        for pair in groups.get(audio_id,[]):
            key=rt.preference_key(pair)
            logs,_=rt.batch_log_probs(policy,rt.move_batch(collator([pair]),device,dtype))
            assert collator.audit[key]==all_masks[key]
            ref=cache[key].to(device)
            if smoke and step==0:assert torch.equal(logs.detach(),ref)
            dpo=rt.dpo_losses(logs[:1],logs[1:],ref[:1],ref[1:],.1)[0]
            weight=pair['pair_weight']*steps_per_epoch/len(groups)
            (dpo*weight).backward();terms['DPO']+=dpo.item()*weight
            if record:
                counts[epoch,'DPO',key]+=1
                exposures.write(json.dumps(dict(epoch=epoch+1,step=step+1,objective='real_DPO',audio_id=audio_id,preference_key=key,weight=weight,rank=rank))+'\n')
        return terms

    def flat_gradient():
        return torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in parameters])

    def assign_gradient(flat,used):
        offset=0
        for p,active in zip(parameters,used):
            p.grad=flat[offset:offset+p.numel()].view_as(p) if active else None
            offset+=p.numel()
        assert offset==flat.numel()

    for epoch in range(epochs):
        ids=list(by_id);random.Random(42+epoch).shuffle(ids)
        for j in range(steps_per_epoch):
            chunk=ids if smoke else ids[j*8:(j+1)*8]
            assert chunk
            local=chunk[rank::world]
            optimizer.zero_grad(set_to_none=True);terms=Counter();started=time.time()
            for audio_id in local:terms.update(backward_audio(audio_id,epoch))
            # Each original global loss term already has its global weight.
            # SUM, not DDP's average: adding ranks must not divide the objective.
            reduced=flat_gradient();dist.all_reduce(reduced,op=dist.ReduceOp.SUM)
            usage=torch.tensor([p.grad is not None for p in parameters],device=device,dtype=torch.int32)
            dist.all_reduce(usage,op=dist.ReduceOp.MAX);used=usage.cpu().tolist()
            assert torch.isfinite(reduced).all()
            stats=torch.tensor([terms['DPO'],terms['KL_weighted']],device=device,dtype=torch.float64)
            dist.all_reduce(stats,op=dist.ReduceOp.SUM)
            if smoke:
                if rank==0:
                    optimizer.zero_grad(set_to_none=True);serial_terms=Counter()
                    for audio_id in chunk:serial_terms.update(backward_audio(audio_id,epoch,record=False))
                    serial=flat_gradient()
                    assert used==[int(p.grad is not None) for p in parameters]
                    difference=(serial-reduced).abs().max().item()
                    relative=((serial-reduced).norm()/serial.norm().clamp_min(1e-12)).item()
                    assert torch.allclose(serial,reduced,atol=2e-6,rtol=2e-4),(difference,relative)
                    assert relative<1e-5,relative
                    assert math.isclose(sum(serial_terms.values()),stats.sum().item(),abs_tol=1e-6)
                    equivalence.append(dict(step=step+1,max_absolute_gradient_difference=difference,relative_gradient_error=relative,
                        serial_loss=sum(serial_terms.values()),three_rank_loss=stats.sum().item(),global_audio=len(chunk),allocation=[len(chunk[x::world]) for x in range(world)]))
                    del serial
                dist.barrier(device_ids=[0])
                if step==0:
                    assert abs(stats[0].item()-math.log(2))<1e-6 and stats[1].item()==0
            assign_gradient(reduced,used)
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(norm)
            optimizer.step();step+=1;exposures.flush()
            item=dict(step=step,epoch=epoch+(j+1)/steps_per_epoch,loss=stats.sum().item(),DPO=stats[0].item(),
                KL_weighted=stats[1].item(),gradient_norm=norm.item(),seconds=time.time()-started,training_gpus=world)
            if rank==0:
                history.append(item)
                with (out/'loss.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
                old.emit('smoke_loss' if smoke else 'training_loss',**item)
                if not smoke and step in plan['evaluation_steps']:
                    binding=old.checkpoint(policy,optimizer,step,item['epoch'],out/f'checkpoint-{step}',rt,torch)
                    write_json(ROOT/'eval_queue'/f'{step:06}.json',binding)
                    old.emit('evaluation_queued',step=step,epoch=item['epoch'])
            if not smoke and step in plan['evaluation_steps']:dist.barrier(device_ids=[0])
        collected_counts=[None]*world;dist.all_gather_object(collected_counts,dict(counts))
        total=Counter()
        for c in collected_counts:total.update(c)
        assert all(total[epoch,'KL',x]==1 for x in by_id)
        assert all(total[epoch,'DPO',rt.preference_key(p)]==1 for p in prefs)
    exposures.close()
    assert rt.parameter_digest(reference)==reference_sha
    assert rt.frozen_base_digest(policy)==base_sha
    final=rt.parameter_digest(policy,trainable_only=True);assert final!=initial
    finals=[None]*world;dist.all_gather_object(finals,final);assert len(set(finals))==1
    old.verify()
    assert all(old.sha256_file(Path(p))==h for p,h in execution['source_sha256'].items())
    if rank==0:
        combined=[]
        for worker in range(world):combined+=rows(out/f'training_exposures_rank{worker}.jsonl')
        # Preserve rank receipts and supply an index; do not invent serial order.
        write_json(out/'exposure_index.json',dict(rank_files=[dict(path=str(out/f'training_exposures_rank{x}.jsonl'),sha256=old.sha256_file(out/f'training_exposures_rank{x}.jsonl')) for x in range(world)],
            total_terms=len(combined),global_window_order_unchanged=True,within_window_partition='chunk[rank::3]',exact_once_per_epoch=True))
        result=dict(passed=True,steps=step,epochs=epochs,training_gpus=3,real_KL_unique=len(real),eligible_DPO_audio=len(groups),distinct_pairs=len(prefs),
            initial_trainable_sha256=initial,final_trainable_sha256=final,all_rank_final_weights_identical=True,
            frozen_reference_unchanged=True,frozen_policy_base_unchanged=True,exact_once_per_epoch_KL_and_each_pair=True,
            trainer_sha256=old.sha256_file(Path(__file__)),history=history,serial_gradient_comparison=equivalence,
            gradient_reduction='SUM of already globally weighted losses; no world-size division',
            resource_policy='Full float32 reference KL recomputed per audio; no whole-pool CPU tensor cache')
        if smoke:result['checkpoint']=old.checkpoint(policy,optimizer,step,2.,out/'checkpoint-2',rt,torch)
        write_json(out/'result.json',result)
        old.emit('smoke_3gpu_complete' if smoke else 'training_complete',steps=step,passed=True)
    dist.barrier(device_ids=[0]);dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','train']);a=p.parse_args();main(smoke=a.stage=='smoke')
