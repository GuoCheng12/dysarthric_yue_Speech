#!/usr/bin/env python3
"""Independent real-SFT + synthetic-DPO diagnostic; not the synthetic-only loop."""

import argparse
from collections import Counter
import gc
import json
from pathlib import Path

import torch

from qwen3_asr_audio_dpo import (
    AudioPreferenceCollator, SourceBalancedWindows, artifact_hashes,
    batch_log_probs, dpo_losses, frozen_reference_log_probs, group_training_rows,
    load_adapter, move_batch, parameter_digest, preference_key, sha256,
    sha256_directory, write_json,
)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['model_path','policy_adapter_path','reference_adapter_path','real_train_file','output_dir']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--synthetic_train_file')
    parser.add_argument('--synthetic_allocation_file')
    parser.add_argument('--max_steps',type=int,default=12)
    parser.add_argument('--real_per_step',type=int,default=8)
    parser.add_argument('--synthetic_per_step',type=int,default=8)
    parser.add_argument('--synthetic_weight',type=float,default=.05)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--beta',type=float,default=.1)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.max_steps<=0 or args.real_per_step<=0 or args.lr<=0:
        parser.error('positive step count, real exposure count and learning rate required')
    if bool(args.synthetic_train_file)!=bool(args.synthetic_allocation_file):
        parser.error('synthetic data and its fixed exposure schedule are required together')
    from transformers import set_seed
    from qwen3_asr_sft import copy_required_hf_files_for_qwen_asr

    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    real=read_rows(args.real_train_file)
    synthetic=read_rows(args.synthetic_train_file) if args.synthetic_train_file else []
    assert all(x['source']=='real' and 'rejected' not in x for x in real)
    assert all(x['source']=='synthetic' for x in synthetic)
    parent_hash=sha256_directory(Path(args.policy_adapter_path))
    audio_hashes={}
    for row in real+synthetic:
        path=str(Path(row['audio']).resolve())
        if path not in audio_hashes:audio_hashes[path]=sha256(path)
        assert row['audio_sha256']==audio_hashes[path]
        if row['source']=='synthetic':assert row['current_asr_sha256']==parent_hash
    groups=dict(real=group_training_rows(real,'sft')['real'],
        synthetic=group_training_rows(synthetic,'dpo')['synthetic'] if synthetic else {})
    allocation=json.loads(Path(args.synthetic_allocation_file).read_text()) if synthetic else None
    if allocation:assert len(allocation['steps'])==args.max_steps
    windows=SourceBalancedWindows(groups,args.real_per_step,args.synthetic_per_step,
        args.synthetic_weight,args.seed,synthetic_schedule=allocation['steps'] if allocation else None)
    sources=[Path(__file__),Path(__file__).with_name('qwen3_asr_audio_dpo.py'),
        Path(__file__).parents[2]/'src/therapist_harness/dpo_training.py',Path(args.real_train_file)]
    if synthetic:sources += [Path(args.synthetic_train_file),Path(args.synthetic_allocation_file)]
    protocol=dict(arguments=vars(args),objective='mean real per-audio token CE + lambda * mean synthetic per-audio DPO',
        scope='Independent mixed-objective diagnostic, not synthetic-only DPO',
        source_sha256={str(p):sha256(p) for p in sources},
        reference_identity='frozen initial R-only ASR; distinct from TTS reference speech',
        reference_asr_sha256=sha256_directory(Path(args.reference_adapter_path)),
        policy_parent_sha256=parent_hash,reference_artifacts=artifact_hashes(args.reference_adapter_path),
        model_artifacts=artifact_hashes(args.model_path),audio_sha256=audio_hashes,
        real_audio=len(groups['real']),synthetic_audio=len(groups['synthetic']),
        score_mask='transcript + EOS only; chat, audio and forced-language prefix masked',
        real_label='original real-training continuation, with forced-language prefix separated',
        optimizer='fresh AdamW, constant lr, zero weight decay, clip norm 1, dropout disabled')
    write_json(output/'run_protocol.json',protocol)
    device=torch.device('cuda:0');dtype=torch.bfloat16
    set_seed(args.seed);cache={}
    if synthetic:
        reference,processor=load_adapter(args.model_path,args.reference_adapter_path,device,dtype,False,True)
        collator=AudioPreferenceCollator(processor)
        before={n:parameter_digest(m) for n,m in reference.named_modules() if n.endswith(('lora_A','lora_B'))}
        assert before and reference.active_adapters
        for index,row in enumerate(synthetic):
            logs=frozen_reference_log_probs(reference,move_batch(collator([row]),device,dtype))
            cache[preference_key(row)]=logs.float().cpu().tolist()
            if index%25==0:print(json.dumps(dict(reference_pairs=index+1,total=len(synthetic))),flush=True)
        after={n:parameter_digest(m) for n,m in reference.named_modules() if n.endswith(('lora_A','lora_B'))}
        assert before==after
        write_json(output/'reference_log_probs.json',dict(values=cache,adapter_tensor_hashes=before,
            reference_unchanged=True,trainable_parameters=sum(p.numel() for p in reference.parameters() if p.requires_grad),
            protocol_sha256=sha256(output/'run_protocol.json')))
        del reference,processor,collator,logs;gc.collect();torch.cuda.empty_cache()
    set_seed(args.seed)
    policy,processor=load_adapter(args.model_path,args.policy_adapter_path,device,dtype,True,True)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    collator=AudioPreferenceCollator(processor)
    parameters=[p for p in policy.parameters() if p.requires_grad]
    policy_before=parameter_digest(policy,trainable_only=True)
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=0)

    @torch.no_grad()
    def probe():
        policy.eval();result=[]
        for row in real[:1]+synthetic[:1]:
            logs,counts=batch_log_probs(policy,move_batch(collator([row],include_rejected=row['source']=='synthetic'),device,dtype))
            value=dict(source=row['source'],audio_id=row['audio_id'],chosen_logp=logs[0].item(),chosen_nll=(-logs[0]/counts[0]).item())
            if row['source']=='synthetic':
                ref=torch.tensor(cache[preference_key(row)],device=device)
                value.update(margin=(logs[0]-logs[1]).item(),dpo_loss=dpo_losses(logs[:1],logs[1:],ref[:1],ref[1:],args.beta).item())
            result.append(value)
        policy.train();return result

    before_probe=probe();history=[]
    with (output/'training_exposures.jsonl').open('x') as handle:
        for step in range(1,args.max_steps+1):
            optimizer.zero_grad(set_to_none=True);window=windows.next();terms=Counter()
            for row in window:
                is_dpo=row['source']=='synthetic'
                logs,counts=batch_log_probs(policy,move_batch(collator([row],include_rejected=is_dpo),device,dtype))
                if is_dpo:
                    ref=torch.tensor(cache[preference_key(row)],device=device)
                    loss=dpo_losses(logs[:1],logs[1:],ref[:1],ref[1:],args.beta)[0]
                else:loss=-logs[0]/counts[0]
                weighted=loss*row['_loss_weight']
                if not torch.isfinite(weighted):raise RuntimeError('nonfinite mixed-objective loss')
                weighted.backward();terms[row['source']]+=weighted.item()
                handle.write(json.dumps(dict(step=step,source=row['source'],audio_id=row['_audio_group'],
                    occurrence=row['_occurrence'],loss_weight=row['_loss_weight'],
                    objective='dpo' if is_dpo else 'sft',preference_key=preference_key(row)))+'\n')
            grad=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(grad):raise RuntimeError('nonfinite mixed-objective gradient')
            optimizer.step();handle.flush()
            item=dict(step=step,weighted_losses=dict(terms),loss=sum(terms.values()),gradient_norm=grad.item(),
                audio_exposures=windows.counts);history.append(item);print(json.dumps(item),flush=True)
    after_probe=probe();policy_after=parameter_digest(policy,trainable_only=True)
    checkpoint=output/f'checkpoint-{args.max_steps}'
    policy.save_pretrained(checkpoint,safe_serialization=True)
    copy_required_hf_files_for_qwen_asr(args.model_path,str(checkpoint))
    write_json(checkpoint/'trainer_state.json',dict(global_step=args.max_steps,log_history=history,
        objective=protocol['objective'],reference_asr_sha256=protocol['reference_asr_sha256']))
    torch.save(optimizer.state_dict(),checkpoint/'optimizer.pt')
    assert artifact_hashes(args.reference_adapter_path)==protocol['reference_artifacts']
    assert all(sha256(path)==digest for path,digest in protocol['source_sha256'].items())
    write_json(output/'training_result.json',dict(status='complete',checkpoint=str(checkpoint),history=history,
        reference_files_unchanged=True,policy_changed=policy_before!=policy_after,
        policy_trainable_sha256_before=policy_before,policy_trainable_sha256_after=policy_after,
        probe_before=before_probe,probe_after=after_probe,protocol_sha256=sha256(output/'run_protocol.json')))


if __name__=='__main__':main()
