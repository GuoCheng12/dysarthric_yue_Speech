"""Remove conflicting synthetic gradient and cap its norm relative to real."""
import argparse
from pathlib import Path
import sys

import torch
import tts_additive_control as base


def protect_real_gradient(real,synthetic,cap=.25):
    real_sq=sum(float(r.square().sum()) for r in real)
    synth_sq=sum(float(s.square().sum()) for s in synthetic)
    dot=sum(float((r*s).sum()) for r,s in zip(real,synthetic))
    coefficient=min(0.,dot)/real_sq if real_sq else 0.
    projected=[s-coefficient*r for r,s in zip(real,synthetic)]
    projected_sq=sum(float(s.square().sum()) for s in projected)
    scale=min(1.,cap*(real_sq/projected_sq)**.5) if projected_sq else 1.
    return [r+scale*s for r,s in zip(real,projected)],{
        'cosine_before':dot/(real_sq*synth_sq)**.5 if real_sq*synth_sq else None,
        'synthetic_to_real_norm_before':(synth_sq/real_sq)**.5 if real_sq else None,
        'projection_coefficient':coefficient,'synthetic_scale':scale,
        'synthetic_to_real_norm_after':scale*(projected_sq/real_sq)**.5 if real_sq else None}


class ProtectedTrainer(base.AdditiveTrainer):
    def training_step(self,model,inputs,num_items_in_batch=None):
        if not inputs['_synthetic']:
            return super().training_step(model,inputs,num_items_in_batch)
        params=[p for p in model.parameters() if p.requires_grad and p.grad is not None]
        real=[p.grad.detach().float().clone() for p in params]
        result=super().training_step(model,inputs,num_items_in_batch)
        synthetic=[p.grad.detach().float()-r for p,r in zip(params,real)]
        combined,record=protect_real_gradient(real,synthetic)
        for p,g in zip(params,combined):p.grad.copy_(g.to(p.grad.dtype))
        if not hasattr(self,'protection_trace'):self.protection_trace=[]
        self.protection_trace.append({'step':self.state.global_step+1,**record})
        # Persist before evaluation so the actual interventions remain inspectable.
        base.save(Path(self.args.output_dir).parent/f"gradient_control/step_{self.state.global_step+1}.json",self.protection_trace[-1])
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=base.ROOT)
    p.add_argument('--arm',default='mas_hard_protected_w025')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--weight',type=float,default=.25)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--synthetic-file',type=Path,default=base.MAS/'mas_hard_reference/train.jsonl')
    args=p.parse_args();directory=base.prepare(args)
    base.save(directory/'intervention.json',{'scope':'synthetic gradient only',
        'rule':'if dot(gR,gS)<0 subtract conflicting projection; cap remaining norm at .25*norm(gR)',
        'optimizer':'same AdamW; rule acts before normal global clipping and optimizer step',
        'command':sys.argv,'source_sha256':base.sha256_file(Path(__file__)),
        'interpretation':'joint projection+norm-cap intervention; training-batch direction, not held-out guarantee'})
    base.AdditiveTrainer=ProtectedTrainer
    base.train(directory)
