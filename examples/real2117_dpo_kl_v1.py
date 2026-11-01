#!/usr/bin/env python3
"""Full real-pool DPO + streaming frozen-reference KL, seed42, three epochs."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
B = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_design_exploration_20260908_v1')
ROOT = B / 'real2117_DPO_KL_seed42_3epochs_20260910_v1'
PY = '/data/qwen3-asr/venvs/qwen3-asr/bin/python'
TRAIN = Path('/data/qwen3-asr/finetune/data_setting_d_v1/setting_d_train.jsonl')
DEV = Path('/data/qwen3-asr/finetune/data_setting_d_merged_test_v1/setting_d_merged_test.jsonl')
BASE = '/data/qwen3-asr/models/Qwen3-ASR-1.7B'
PARENT = '/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_utility_diagnosis_v1/additive/real_w0/seed_42/training/checkpoint-50'
PARENT_SHA = '8c3e944f7ee56fde2471748e5228eeb0fc21df0a1a51b6c29805eb4726043db6'
sys.path[:0] = [str(REPO/'src'), str(REPO/'examples'), str(REPO/'finetune/scripts'), '/tmp', '/tmp/mas_gradient_deterministic_v2']
from therapist_harness.store import sha256_file, sha256_directory, write_immutable_json


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    path = Path(path)
    data = gzip.decompress(path.read_bytes()).decode() if path.suffix == '.gz' else path.read_text()
    return [json.loads(x) for x in data.splitlines() if x.strip()]


def jsonl(path, values):
    with Path(path).open('x') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def emit(stage, **values):
    value = dict(utc=datetime.now(timezone.utc).isoformat(), stage=stage, **values)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    with (ROOT/'events.jsonl').open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def verify():
    plan = read(ROOT/'protocol.json')
    assert all(sha256_file(Path(p)) == h for p, h in plan['source_sha256'].items())
    assert sha256_directory(Path(PARENT)) == PARENT_SHA
    return plan


def prepare():
    from therapist_harness.preference_data import validate_sampling_inputs
    train, dev = rows(TRAIN), rows(DEV)
    assert len(train) == 2117 and len(dev) == 511
    inputs = validate_sampling_inputs([dict(r, source='real', audio_id=r['utt_id'],
        target_text=r['clean_gt'], chosen=r['clean_gt'], language='Cantonese') for r in train], train, dev)
    assert sha256_directory(Path(PARENT)) == PARENT_SHA
    jsonl(ROOT/'real_inputs.jsonl', inputs)
    for shard in range(2):
        jsonl(ROOT/f'inputs_shard{shard}.jsonl', inputs[shard::2])
    scripts = [Path(__file__), REPO/'examples/sample_asr_preferences.py',
        Path('/tmp/mas_real_even_kl_gpu_smoke_v1.py'),
        Path('/tmp/mas_gradient_deterministic_v2/deterministic_qwen_positions_v1.py'),
        REPO/'examples/mas_real_headroom_selection_dpo_v1.py',
        REPO/'finetune/scripts/evaluate_asr_checkpoint.py',
        REPO/'finetune/scripts/qwen3_asr_residual_dpo.py',
        REPO/'finetune/scripts/qwen3_asr_audio_dpo.py',
        REPO/'finetune/scripts/qwen3_asr_kl_dpo_storage_v2.py',
        REPO/'src/therapist_harness/dpo_training.py',
        REPO/'src/therapist_harness/preference_token_mask.py',
        REPO/'src/therapist_harness/preference_data.py', TRAIN, DEV,
        ROOT/'real_inputs.jsonl', ROOT/'inputs_shard0.jsonl', ROOT/'inputs_shard1.jsonl',
        B/'parent_eval_check/predictions.jsonl']
    plan = dict(execution_version='real2117_DPO_KL_seed42_3epochs_v1', base=BASE,
        parent=PARENT, parent_sha256=PARENT_SHA, DPO_reference='frozen initial R-only ASR',
        train_audio=2117, evaluation_audio=511, seed=42, epochs=3,
        nominal_audio_per_step=8, steps_per_epoch=265, total_steps=795,
        evaluation_steps=[e*265+53*k for e in range(3) for k in range(1,6)],
        epoch_definition='Every2117real input visited once; every distinct usable preference once, equal total weight per eligible audio; every2117real KL audio once. All-correct inputs contribute KL only.',
        loss='edit-token+EOS SUM DPO(beta0.1) +10*full-transcript+EOS symmetric KL to initial R-only; no real CE',
        normalization='At each of265windows: DPO pair_weight*265/eligible_audio_count; KL10*265/2117. Average window loss over a whole epoch is exactly the mean per eligible DPO audio plus10*mean per real KL audio, including final partial window.',
        lr=1e-5, beta=.1, kl_weight=10., clip=1., weight_decay=0.,
        optimizer='fresh AdamW on zero-output residual LoRA over merged R-only; constant lr1e-5 for3epochs, no outcome-based adjustment',
        sampling=dict(draws_per_audio=4, temperature=.8, top_p=.95, top_k=0,
            max_new_tokens=256, language='Cantonese', prompt='', include_greedy=True,
            shards='disjoint alternating input indices; original per-audio RNG independent of shard'),
        KL_storage='No whole-pool teacher cache; reference computed on GPU0 per real audio, detached, discarded after backward.32GiB container memory limit.',
        checkpoint_policy='Save all15evaluation adapters and optimizer states; evaluation does not stop/change3epoch schedule. No automatic promotion.',
        GPUs='Two sampling shards in parallel, then GPU0 train+frozen reference and GPU1 sequential checkpoint evaluation.',
        source_sha256={str(p):sha256_file(p) for p in scripts})
    write_immutable_json(ROOT/'protocol.json', plan)
    emit('prepared', real_audio=len(inputs), epochs=3, evaluation_steps=plan['evaluation_steps'])


def merge():
    from therapist_harness.preference_data import build_preference_pairs, sample_seed
    plan = verify(); inputs = rows(ROOT/'real_inputs.jsonl')
    all_candidates, all_greedy, all_inputs, receipts = [], [], [], []
    common = None
    for shard in range(2):
        d = ROOT/f'sampling_shard{shard}'; p = d/'sampling_receipt.json'; r = read(p)
        assert r['status']=='complete' and r['current_asr_sha256']==PARENT_SHA
        for name,key in [('candidates.jsonl','candidates_sha256'),('validated_inputs.jsonl','validated_inputs_sha256'),('greedy.jsonl','greedy_sha256')]:
            assert sha256_file(d/name)==r[key]
        keys = ['current_asr_sha256','base_model_sha256','samples_per_audio','seed','temperature','top_p','top_k','max_new_tokens','language','prompt','do_sample','source_sha256','preference_code_sha256','real_train_sha256','feedback_sha256']
        settings = {k:r[k] for k in keys}
        if common is None: common=settings
        else: assert settings==common
        assert settings['samples_per_audio']==4 and settings['seed']==42
        shard_inputs=rows(d/'validated_inputs.jsonl')
        assert shard_inputs==inputs[shard::2]
        candidates=rows(d/'candidates.jsonl')
        assert len(candidates)==len(shard_inputs)*4
        assert all(x['seed']==sample_seed(42,x['audio_id'],x['sample_index']) for x in candidates)
        all_inputs+=shard_inputs;all_candidates+=candidates;all_greedy+=rows(d/'greedy.jsonl')
        receipts.append(dict(path=str(p),sha256=sha256_file(p),audio_count=len(shard_inputs),candidate_count=len(candidates)))
    assert len({x['audio_id'] for x in all_inputs})==len(inputs)==2117
    pairs,audit=build_preference_pairs(inputs,all_candidates,PARENT_SHA,expected_samples=4,greedy_candidates=all_greedy)
    order={r['audio_id']:i for i,r in enumerate(inputs)}
    all_candidates.sort(key=lambda x:(order[x['audio_id']],x['sample_index']))
    jsonl(ROOT/'candidates.jsonl',all_candidates);jsonl(ROOT/'preferences.jsonl',pairs)
    write_immutable_json(ROOT/'preference_audit.json',audit)
    write_immutable_json(ROOT/'sampling_shard_index.json',dict(schema='multi_shard_sampling_index_v1',
        source_receipts=receipts,common_settings=common,audio_count=2117,candidate_count=len(all_candidates),
        preferences_sha256=sha256_file(ROOT/'preferences.jsonl'),candidates_sha256=sha256_file(ROOT/'candidates.jsonl'),
        complete_disjoint_coverage=True,synthetic_samples=0,
        provenance='Two original sampling receipts retained. This is an aggregation index, not a fabricated single-run receipt.',
        cap_reached=sum(bool(x['reached_max_new_tokens']) for x in all_candidates)))
    emit('preferences_ready', audio=2117, eligible_audio=audit['paired_audio_count'], pairs=len(pairs), all_correct=audit['all_correct_audio_count'])


def runtime():
    import torch
    from transformers import set_seed
    import mas_real_even_kl_gpu_smoke_v1 as rt
    from deterministic_qwen_positions_v1 import install
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    install();set_seed(42)
    return torch, rt, set_seed


def checkpoint(policy, optimizer, step, epoch, directory, rt, torch):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    directory.mkdir(parents=True,exist_ok=False)
    policy.save_pretrained(directory,safe_serialization=True)
    actual=load_file(str(directory/'adapter_model.safetensors'))
    state=get_peft_model_state_dict(policy)
    assert set(actual)==set(state) and all(torch.equal(actual[k],v.detach().cpu()) for k,v in state.items())
    write_immutable_json(directory/'residual_composition.json',dict(base_model=BASE,parent_adapter=PARENT,parent_adapter_sha256=PARENT_SHA,residual_merged=False))
    optimizer_path=directory.parent/f'optimizer_step{step}.pt'
    torch.save(optimizer.state_dict(),optimizer_path)
    binding=dict(step=step,epoch=epoch,path=str(directory),sha256=sha256_directory(directory),
        trainable_sha256=rt.parameter_digest(policy,trainable_only=True),parent_sha256=PARENT_SHA,
        optimizer_path=str(optimizer_path),optimizer_sha256=sha256_file(optimizer_path),exact_state_roundtrip=True)
    return binding


def residual_digest(model, torch):
    """Same ordering/bytes as the trainer digest, independent of inference grad flags."""
    digest=hashlib.sha256()
    for name,value in model.named_parameters():
        if '.lora_' in name:
            digest.update(name.encode())
            digest.update(value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def train(smoke=False):
    plan=verify();torch,rt,set_seed=runtime()
    out=ROOT/('smoke' if smoke else 'training');out.mkdir(exist_ok=False)
    real=rows(ROOT/'real_inputs.jsonl')
    if smoke:
        source=B/'real96_dpo_probe/sampling';receipt=read(source/'sampling_receipt.json')
        assert receipt['status']=='complete' and receipt['current_asr_sha256']==PARENT_SHA
        assert sha256_file(source/'candidates.jsonl')==receipt['candidates_sha256']
        prefs=rows(source/'preferences.jsonl')
        ids=list(dict.fromkeys(x['audio_id'] for x in prefs))[:2]
        prefs=[r for r in prefs if r['audio_id'] in ids]
        real=[r for r in real if r['audio_id'] in ids]
        write_immutable_json(out/'input_source.json',dict(path=str(source),sampling_receipt_sha256=sha256_file(source/'sampling_receipt.json'),preferences_sha256=sha256_file(source/'preferences.jsonl'),note='Actual earlier real samples from same initial R-only, used only for discarded smoke updates. Formal2117sampling remains fresh.'))
    else:
        assert read(ROOT/'smoke/result.json')['passed']
        assert read(ROOT/'smoke/result.json')['trainer_sha256']==sha256_file(Path(__file__))
        prefs=rows(ROOT/'preferences.jsonl')
        assert sha256_file(ROOT/'preferences.jsonl')==read(ROOT/'sampling_shard_index.json')['preferences_sha256']
    by_id={r['audio_id']:r for r in real};groups=defaultdict(list)
    for r in prefs:
        assert r['current_asr_sha256']==PARENT_SHA and r['source']=='real'
        assert r['audio_id'] in by_id and r['audio_sha256']==by_id[r['audio_id']]['audio_sha256']
        assert r['chosen']==by_id[r['audio_id']]['chosen']
        groups[r['audio_id']].append(r)
    assert groups
    for group in groups.values():assert math.isclose(sum(r['pair_weight'] for r in group),1.)
    assert all(sha256_file(Path(r['audio']))==r['audio_sha256'] for r in real)
    device,dtype=torch.device('cuda:0'),torch.bfloat16
    set_seed(42);reference,processor=rt.load_model(BASE,PARENT,False)
    ref_digest=rt.parameter_digest(reference);ref_base=rt.frozen_base_digest(reference)
    collator=rt.RealSyntheticEditCollator(processor,{})
    cache={};started=time.time()
    with torch.no_grad():
        for i,r in enumerate(prefs):
            v,_=rt.batch_log_probs(reference,rt.move_batch(collator([r]),device,dtype))
            cache[rt.preference_key(r)]=v.detach().cpu()
            if i%100==0:emit('reference_scores',pairs=i+1,total=len(prefs),seconds=time.time()-started)
    write_immutable_json(out/'reference_scores.json',dict(parent_sha256=PARENT_SHA,
        reference_parameter_sha256=ref_digest,scores={k:v.tolist() for k,v in cache.items()},masks=collator.audit))
    set_seed(42);policy,_=rt.load_model(BASE,PARENT,True)
    assert rt.frozen_base_digest(policy)==ref_base
    initial=rt.parameter_digest(policy,trainable_only=True)
    assert residual_digest(policy,torch)==initial
    if smoke:
        import mas_real_headroom_selection_dpo_v1 as live
        expected=rows(B/'parent_eval_check/predictions.jsonl')[:8]
        prediction=live.wrapper_for(policy,processor).transcribe(audio=[r['audio'] for r in expected],language=['Cantonese']*8)
        assert [x.text for x in prediction]==[x['prediction'] for x in expected]
        policy.config.use_cache=False
    policy.train();policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    parameters=[p for p in policy.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=plan['lr'],weight_decay=0.)
    if smoke:
        r=prefs[0];logs,_=rt.batch_log_probs(policy,rt.move_batch(collator([r]),device,dtype));ref=cache[rt.preference_key(r)].to(device)
        assert torch.equal(logs.detach(),ref)
        loss=rt.dpo_losses(logs[:1],logs[1:],ref[:1],ref[1:],.1)[0]
        assert abs(loss.item()-math.log(2))<1e-6
        loss.backward();assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in parameters)
        optimizer.zero_grad(set_to_none=True)
        batch=rt.move_batch(collator([real[0]],include_rejected=False),device,dtype)
        with torch.no_grad():teacher=rt.transcript_distributions(reference,batch)
        student=rt.transcript_distributions(policy,batch)
        assert torch.equal(student.detach(),teacher)
        zero_kl=rt.symmetric_kl(student,teacher);assert zero_kl.item()==0
        zero_kl.backward();assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in parameters)
        del logs,ref,loss,teacher,student,zero_kl,batch
        optimizer.zero_grad(set_to_none=True)
    steps_per_epoch=1 if smoke else plan['steps_per_epoch'];epochs=2 if smoke else 3
    visits=Counter();pair_visits=Counter();step=0;history=[]
    with (out/'training_exposures.jsonl').open('x') as exposures:
        for epoch in range(epochs):
            ids=list(by_id);random.Random(42+epoch).shuffle(ids)
            for j in range(steps_per_epoch):
                chunk=ids if smoke else ids[j*8:(j+1)*8]
                assert chunk
                optimizer.zero_grad(set_to_none=True);terms=Counter();t0=time.time()
                for audio_id in chunk:
                    r=by_id[audio_id]
                    batch=rt.move_batch(collator([r],include_rejected=False),device,dtype)
                    with torch.no_grad():teacher=rt.transcript_distributions(reference,batch)
                    assert not teacher.requires_grad
                    student=rt.transcript_distributions(policy,batch)
                    kl=rt.symmetric_kl(student,teacher)
                    weight=plan['kl_weight']*steps_per_epoch/len(real)
                    (kl*weight).backward();terms['KL_weighted']+=kl.item()*weight
                    visits[(epoch,'KL',audio_id)]+=1
                    exposures.write(json.dumps(dict(epoch=epoch+1,step=step+1,objective='real_KL',audio_id=audio_id,weight=weight))+'\n')
                    del teacher,student,kl,batch
                    for pair in groups.get(audio_id,[]):
                        key=rt.preference_key(pair)
                        logs,_=rt.batch_log_probs(policy,rt.move_batch(collator([pair]),device,dtype))
                        ref=cache[key].to(device)
                        dpo=rt.dpo_losses(logs[:1],logs[1:],ref[:1],ref[1:],plan['beta'])[0]
                        weight=pair['pair_weight']*steps_per_epoch/len(groups)
                        (dpo*weight).backward();terms['DPO']+=dpo.item()*weight
                        pair_visits[(epoch,key)]+=1
                        exposures.write(json.dumps(dict(epoch=epoch+1,step=step+1,objective='real_DPO',audio_id=audio_id,preference_key=key,weight=weight))+'\n')
                        del logs,ref,dpo
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                assert torch.isfinite(norm) and all(math.isfinite(v) for v in terms.values())
                optimizer.step();step+=1;exposures.flush()
                item=dict(step=step,epoch=epoch+(j+1)/steps_per_epoch,loss=sum(terms.values()),
                    DPO=terms['DPO'],KL_weighted=terms['KL_weighted'],gradient_norm=norm.item(),seconds=time.time()-t0)
                history.append(item)
                with (out/'loss.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
                emit('smoke_loss' if smoke else 'training_loss',**item)
                if not smoke and step in plan['evaluation_steps']:
                    binding=checkpoint(policy,optimizer,step,item['epoch'],out/f'checkpoint-{step}',rt,torch)
                    write_immutable_json(ROOT/'eval_queue'/f'{step:06}.json',binding)
                    emit('evaluation_queued',step=step,epoch=item['epoch'])
            assert all(visits[epoch,'KL',x]==1 for x in by_id)
            assert all(pair_visits[epoch,rt.preference_key(p)]==1 for p in prefs)
    assert rt.parameter_digest(reference)==ref_digest
    assert rt.frozen_base_digest(policy)==ref_base
    assert rt.parameter_digest(policy,trainable_only=True)!=initial
    verify()
    result=dict(passed=True,steps=step,epochs=epochs,initial_trainable_sha256=initial,
        final_trainable_sha256=rt.parameter_digest(policy,trainable_only=True),
        frozen_reference_unchanged=True,frozen_policy_base_unchanged=True,
        real_KL_unique=len(by_id),eligible_DPO_audio=len(groups),distinct_pairs=len(prefs),
        exact_once_per_epoch_KL_and_each_pair=True,trainer_sha256=sha256_file(Path(__file__)),
        history=history,reference_storage='streaming full float32 vocabulary KL; no whole-pool tensor cache')
    if smoke:
        binding=checkpoint(policy,optimizer,step,2.,out/'checkpoint-2',rt,torch)
        result.update(initial_DPO_ln2=True,initial_KL_and_gradient_zero=True,initial8_predictions_match_Ronly=True,
            actual_optimizer_steps=2,checkpoint=binding)
    write_immutable_json(out/'result.json',result)
    emit('smoke_complete' if smoke else 'training_complete',steps=step,passed=True)


def evaluate():
    verify();torch,rt,set_seed=runtime()
    import mas_real_headroom_selection_dpo_v1 as live
    from peft import set_peft_model_state_dict,get_peft_model_state_dict
    from safetensors.torch import load_file
    policy,processor=rt.load_model(BASE,PARENT,False)
    frozen=rt.frozen_base_digest(policy)
    out=ROOT/'evaluations';out.mkdir(exist_ok=False)
    baseline=out/'baseline';baseline.mkdir()
    live.evaluate_live(policy,processor,baseline,BASE,PARENT)
    before=rows(B/'parent_eval_check/predictions.jsonl');after=rows(baseline/'evaluation/predictions.jsonl.gz')
    assert [(x['utt_id'],x['prediction']) for x in before]==[(x['utt_id'],x['prediction']) for x in after]
    emit('baseline_verified',audio=511,errors=1384,CER_percent=100*1384/6656)
    done=set()
    while len(done)<15:
        pending=[p for p in sorted((ROOT/'eval_queue').glob('*.json')) if p.name not in done]
        if not pending:
            if (ROOT/'failure.json').exists():raise RuntimeError('Other stage failed; ending evaluator')
            time.sleep(5);continue
        for path in pending:
            job=read(path);cp=Path(job['path']);assert sha256_directory(cp)==job['sha256']
            state=load_file(str(cp/'adapter_model.safetensors'))
            set_peft_model_state_dict(policy,state)
            actual=get_peft_model_state_dict(policy)
            assert set(actual)==set(state) and all(torch.equal(v.detach().cpu(),state[k]) for k,v in actual.items())
            assert residual_digest(policy,torch)==job['trainable_sha256']
            dest=out/f"step-{job['step']}";dest.mkdir()
            live.evaluate_live(policy,processor,dest,BASE,PARENT)
            d=read(dest/'comparison.json');groups=d['comparison']['groups'];metrics={}
            for label,key in [('overall','overall'),('hard','difficulty:hard'),('vlink247','vlink247'),('others','other_patients')]:
                metrics[label]=dict(CER_percent=100*groups[key]['after_pooled'],delta_Ronly_pp=groups[key]['delta_pp'])
            event=dict(step=job['step'],epoch=job['epoch'],metrics=metrics,passes_original_gate=d['passes_original_gate'],checkpoint=str(cp))
            write_immutable_json(dest/'checkpoint_binding.json',job)
            assert rt.frozen_base_digest(policy)==frozen and sha256_directory(cp)==job['sha256']
            emit('evaluation_complete',**event);done.add(path.name)
    verify();write_immutable_json(ROOT/'evaluation_complete.json',dict(evaluations=15,status='complete'))


def start_process(command, gpu, log):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with (ROOT/log).open('x') as f:
        return subprocess.Popen(command,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)


def supervise():
    verify();assert read(ROOT/'smoke/result.json')['passed']
    jobs=[]
    try:
        for shard in range(2):
            command=[PY,str(REPO/'examples/sample_asr_preferences.py'),'sample','--inputs',str(ROOT/f'inputs_shard{shard}.jsonl'),
                '--real-train',str(TRAIN),'--feedback',str(DEV),'--out-dir',str(ROOT/f'sampling_shard{shard}'),
                '--base-model',BASE,'--current-adapter',PARENT,'--expected-parent-sha256',PARENT_SHA,
                '--samples','4','--seed','42','--temperature','0.8','--top-p','0.95','--max-new-tokens','256','--include-greedy']
            jobs.append(start_process(command,shard,f'sampling_gpu{shard}.log'))
        emit('sampling_started',pids=[j.pid for j in jobs],audio=2117,draws=8468,greedy=2117)
        while any(j.poll() is None for j in jobs):
            assert all(j.poll() in (None,0) for j in jobs),'Sampling shard failed'
            time.sleep(5)
        assert all(j.returncode==0 for j in jobs)
        merge()
        jobs=[start_process([PY,str(Path(__file__)),'train'],0,'training.log'),
              start_process([PY,str(Path(__file__)),'evaluate'],1,'evaluation.log')]
        emit('DPO_and_evaluator_started',pids=[j.pid for j in jobs])
        while any(j.poll() is None for j in jobs):
            assert all(j.poll() in (None,0) for j in jobs),'Training or evaluator failed'
            time.sleep(5)
        assert all(j.returncode==0 for j in jobs)
        write_immutable_json(ROOT/'complete.json',dict(status='complete',epochs=3,evaluations=15,seed=42))
        emit('experiment_complete',epochs=3,evaluations=15)
    except BaseException as error:
        for job in jobs:
            if job.poll() is None:job.terminate()
        write_immutable_json(ROOT/'failure.json',dict(error=repr(error),utc=datetime.now(timezone.utc).isoformat()))
        emit('failed',error=repr(error));raise


def status():
    print('Real2117 DPO + KL | seed42 | 3 epochs | 15 evaluations')
    for shard in range(2):
        path=ROOT/f'sampling_shard{shard}/candidates.jsonl'
        count=sum(1 for _ in path.open()) if path.exists() else 0
        print(f'GPU{shard} sampling: {count//4}/{1059 if shard==0 else 1058} audio; {count} draws')
    path=ROOT/'training/loss.jsonl'
    print('\nTraining loss:')
    if path.exists():
        for line in path.read_text().splitlines()[-6:]:print(line)
    else:print('Waiting for preparation / sampling / frozen-reference scores. No DPO update yet.')
    print('\nRecent stage/evaluation events:')
    path=ROOT/'events.jsonl'
    if path.exists():
        events=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        events=[x for x in events if x['stage']!='training_loss']
        for row in events[-6:]:print(json.dumps(row,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','merge','smoke','train','evaluate','supervise','status'])
    args=parser.parse_args()
    if args.stage=='smoke':train(smoke=True)
    elif args.stage=='train':train()
    else:globals()[args.stage]()
