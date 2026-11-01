"""Hold real samples, real loss weight, updates and RNG fixed; add TTS loss."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import sys

from tts_increment_control import REPO, ROOT as CONTROL, TEST, rows, read, jsonl
from tts_increment_control import training_command, sha256_file, write_immutable_json as save
import qwen3_asr_lora_sft as training
import torch
from torch.utils.data import SequentialSampler
from train_m0_small_dose_pair import Monitor

ROOT = CONTROL.parent/'tts_utility_diagnosis_v1/additive'
MAS = CONTROL.parent/'real2117_mas576_hardref_v1'


def repeated_permutations(size, count, seed):
    rng=random.Random(seed); result=[]
    while len(result)<count:
        order=list(range(size));rng.shuffle(order);result.extend(order)
    return result[:count]


def make_order(real, synthetic, seed, steps):
    real_order=repeated_permutations(len(real),32*steps,seed)
    synth_order=repeated_permutations(len(synthetic),8*steps,seed+100000)
    ordered=[]; index=[]
    for step in range(steps):
        for domain, source, order, size in [('real',real,real_order,32),('synthetic',synthetic,synth_order,8)]:
            for source_index in order[step*size:(step+1)*size]:
                ordered.append(source[source_index])
                index.append({'step':step+1,'domain':domain,'source_index':source_index,'audio':source[source_index]['audio']})
    return ordered,index


class DomainCollator:
    def __init__(self, collator, synthetic_audio):
        self.collator=collator;self.synthetic_audio=set(synthetic_audio)

    def __call__(self, features):
        domains={f['audio'] in self.synthetic_audio for f in features}
        assert len(domains)==1, 'Each microbatch must have one domain'
        return {**self.collator(features),'_synthetic':next(iter(domains))}


class AdditiveTrainer(training.QwenASRLoRATrainer):
    synthetic_weight=.25

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.domain_losses=[];self.gradient_checks=[]

    def _get_train_sampler(self,train_dataset=None):
        return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        inputs=dict(inputs);synthetic=inputs.pop('_synthetic')
        loss,outputs=super().compute_loss(model,inputs,True,num_items_in_batch)
        size=inputs['labels'].shape[0]
        denominator=size if num_items_in_batch is None else num_items_in_batch
        # Sum over the accumulation window gives L_real + lambda * L_synthetic.
        factor=denominator*(self.synthetic_weight/8 if synthetic else 1/32)
        self.domain_losses.append({'step':self.state.global_step+1,'domain':'synthetic' if synthetic else 'real',
            'mean_loss':float(loss.detach())*denominator/size,'rows':size})
        result=loss*factor
        return (result,outputs) if return_outputs else result

    def training_step(self,model,inputs,num_items_in_batch=None):
        synthetic=inputs['_synthetic']
        # Synthetic dropout must not advance the RNG used by the next real batch.
        devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []
        context=torch.random.fork_rng(devices=devices) if synthetic else nullcontext()
        measure=synthetic and self.synthetic_weight>0 and self.state.global_step in (0,9,24,49)
        params=[p for p in model.parameters() if p.requires_grad and p.grad is not None] if measure else []
        before=[p.grad.detach().float().clone() for p in params]
        with context:
            result=super().training_step(model,inputs,num_items_in_batch)
        if measure:
            real_sq=synth_sq=dot=0.
            for p,a in zip(params,before):
                b=p.grad.detach().float()-a
                real_sq+=float(a.square().sum());synth_sq+=float(b.square().sum());dot+=float((a*b).sum())
            self.gradient_checks.append({'step':self.state.global_step+1,'real_norm':real_sq**.5,
                'weighted_synthetic_norm':synth_sq**.5,'cosine':dot/(real_sq*synth_sq)**.5 if real_sq*synth_sq else None})
        return result


def prepare(args):
    directory=args.root/args.arm/f'seed_{args.seed}'
    assert not directory.exists()
    real=rows(CONTROL/'real_q/train.jsonl')
    synthetic=rows(args.synthetic_file)
    if len(synthetic)==2693:synthetic=synthetic[2117:]
    assert len(real)==2117 and synthetic
    assert not {r['audio'] for r in real}&{r['audio'] for r in synthetic}
    ordered,index=make_order(real,synthetic,args.seed,args.steps)
    jsonl(directory/'train.jsonl',ordered);save(directory/'index.json',index)
    command=training_command(args.root,args.arm,args.seed)
    changes={'--train_file':str(directory/'train.jsonl'),'--grad_acc':'5'}
    for k,v in changes.items():command[command.index(k)+1]=v
    source_paths=[Path(__file__),Path(training.__file__),REPO/'finetune/official/qwen3_asr_sft.py',
        REPO/'finetune/scripts/train_m0_small_dose_pair.py',REPO/'finetune/scripts/train_setting_d_m0.py',
        REPO/'src/therapist_harness/evidence.py',REPO/'src/therapist_harness/text.py']
    save(directory/'protocol.json',{'seed':args.seed,'arm':args.arm,'steps':args.steps,
        'synthetic_weight':args.weight,'real_per_step':32,'synthetic_per_step':8,
        'objective':'mean(real32) + weight*mean(synthetic8)',
        'zero_weight_control':'same synthetic forward/backward with zero loss; RNG isolated',
        'real_order_seed':args.seed,'synthetic_order_seed':args.seed+100000,
        'real_rows':2117,'synthetic_rows':len(synthetic),'command':command,
        'selection':'fixed endpoint, exploratory existing511; no checkpoint selection',
        'source_sha256':{str(p):sha256_file(p) for p in source_paths},
        'input_sha256':{str(p):sha256_file(p) for p in [CONTROL/'real_q/train.jsonl',args.synthetic_file,TEST]},
        'audio_sha256':{r['audio']:sha256_file(Path(r['audio'])) for r in real+synthetic},
        'train_sha256':sha256_file(directory/'train.jsonl')})
    return directory


def train(directory):
    cfg=read(directory/'protocol.json')
    for field in ('source_sha256','input_sha256','audio_sha256'):
        for path,digest in cfg[field].items():assert sha256_file(Path(path))==digest,path
    assert sha256_file(directory/'train.jsonl')==cfg['train_sha256']
    synthetic={r['audio'] for r in read(directory/'index.json') if r['domain']=='synthetic'}

    class FixedMonitor(Monitor):
        def on_step_end(self,args,state,control,**kwargs):
            control.should_save=state.global_step==cfg['steps']
            control.should_training_stop=state.global_step>=cfg['steps']
            return control

    monitor=FixedMonitor(directory);monitor.end_step=cfg['steps']
    monitor.state['endpoint_step']=cfg['steps'];monitor.publish()

    class ExperimentTrainer(AdditiveTrainer):
        synthetic_weight=cfg['synthetic_weight']
        def __init__(self,*args,**kwargs):
            kwargs['args'].save_only_model=True
            kwargs['data_collator']=DomainCollator(kwargs['data_collator'],synthetic)
            super().__init__(*args,**kwargs)
            monitor.trainer=self;self.add_callback(monitor)

    training.QwenASRLoRATrainer=ExperimentTrainer
    sys.argv=cfg['command'][1:]
    try:
        training.main()
        assert [r['step'] for r in monitor.history]==[cfg['steps']]
        state=read(directory/f"training/checkpoint-{cfg['steps']}/trainer_state.json")
        assert state['global_step']==cfg['steps']
        logs=monitor.trainer.domain_losses
        for step in range(1,cfg['steps']+1):
            assert sum(r['rows'] for r in logs if r['step']==step and r['domain']=='real')==32
            assert sum(r['rows'] for r in logs if r['step']==step and r['domain']=='synthetic')==8
        save(directory/'result.json',{'seed':cfg['seed'],'arm':cfg['arm'],'step':cfg['steps'],
            'metrics':monitor.history[0]['metrics'],'train_log':state['log_history'],
            'domain_losses':logs,'gradient_checks':monitor.trainer.gradient_checks})
        monitor.state['status']='complete';monitor.publish()
    except Exception as error:
        monitor.state.update(status='failed',error=f'{type(error).__name__}: {error}');monitor.publish();raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--arm',required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--weight',type=float,default=.25)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--synthetic-file',type=Path,default=MAS/'mas_hard_reference/train.jsonl')
    args=p.parse_args()
    train(prepare(args))
