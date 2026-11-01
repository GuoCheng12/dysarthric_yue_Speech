import sys
from pathlib import Path

import accelerate.utils.other as accelerate_other
import pytest
import torch
from torch import nn
from transformers import TrainingArguments

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from tts_additive_control import AdditiveTrainer,make_order
from tts_scope_control import LanguageSyntheticTrainer
from tts_audio_scope_control import AudioSyntheticTrainer


def test_same_real_order_independent_of_synthetic_pool():
    real=[{'audio':f'r{i}'} for i in range(71)]
    a,ai=make_order(real,[{'audio':f'a{i}'} for i in range(37)],42,4)
    b,bi=make_order(real,[{'audio':f'b{i}'} for i in range(11)],42,4)
    assert [r for r,i in zip(a,ai) if i['domain']=='real']==[r for r,i in zip(b,bi) if i['domain']=='real']
    for step in range(1,5):
        assert sum(i['step']==step and i['domain']=='real' for i in ai)==32
        assert sum(i['step']==step and i['domain']=='synthetic' for i in ai)==8


class Toy(nn.Module):
    def __init__(self,weight):
        super().__init__();self.weight=nn.Parameter(weight.clone())
    def forward(self,x,**kwargs):return {'logits':x@self.weight}


@pytest.mark.parametrize('weight',[0.,.1,.25])
def test_actual_accumulation_matches_additive_objective(tmp_path,monkeypatch,weight):
    monkeypatch.setattr(accelerate_other,'is_deepspeed_available',lambda:False)
    torch.manual_seed(3)
    x=torch.randn(40,7,3);labels=torch.randint(0,5,(40,7));labels[:,0]=-100
    for i in range(40):labels[i,2+i%5:]=-100
    initial=torch.randn(3,5);reference=Toy(initial);model=Toy(initial)
    logits=reference(x=x)['logits']
    losses=torch.stack([nn.functional.cross_entropy(logits[i,:-1],labels[i,1:],ignore_index=-100) for i in range(40)])
    (losses[:32].mean()+weight*losses[32:].mean()).backward()
    trainer=AdditiveTrainer(model=model,args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,
        per_device_train_batch_size=8,gradient_accumulation_steps=5,max_grad_norm=0,report_to='none'))
    trainer.synthetic_weight=weight
    trainer.current_gradient_accumulation_steps=5
    batches=[{'x':x[i:i+8],'labels':labels[i:i+8],'_synthetic':i==32} for i in range(0,40,8)]
    for batch in batches:trainer.training_step(model,batch,num_items_in_batch=40)
    torch.testing.assert_close(model.weight.grad,reference.weight.grad,atol=2e-7,rtol=2e-6)


def test_synthetic_batch_does_not_advance_real_dropout_rng(tmp_path,monkeypatch):
    monkeypatch.setattr(accelerate_other,'is_deepspeed_available',lambda:False)
    class DropToy(Toy):
        def forward(self,x,**kwargs):return super().forward(nn.functional.dropout(x,p=.5,training=True))
    model=DropToy(torch.randn(3,5))
    trainer=AdditiveTrainer(model=model,args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,
        gradient_accumulation_steps=5,report_to='none'))
    trainer.current_gradient_accumulation_steps=5
    batch={'x':torch.randn(8,7,3),'labels':torch.randint(0,5,(8,7)),'_synthetic':True}
    before=torch.random.get_rng_state().clone()
    trainer.training_step(model,batch,num_items_in_batch=40)
    assert torch.equal(before,torch.random.get_rng_state())


@pytest.mark.parametrize('trainer_class,frozen_name,updated_name',[
    (LanguageSyntheticTrainer,'audio_tower','language'),
    (AudioSyntheticTrainer,'language','audio_tower'),
])
def test_synthetic_scope_preserves_accumulated_real_gradient(tmp_path,monkeypatch,trainer_class,frozen_name,updated_name):
    monkeypatch.setattr(accelerate_other,'is_deepspeed_available',lambda:False)
    class TwoParts(nn.Module):
        def __init__(self):
            super().__init__();self.audio_tower=nn.Linear(3,3);self.language=nn.Linear(3,5)
        def forward(self,x,**kwargs):return {'logits':self.language(self.audio_tower(x))}
    model=TwoParts()
    trainer=trainer_class(model=model,args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,
        gradient_accumulation_steps=5,report_to='none'))
    trainer.current_gradient_accumulation_steps=5
    batch={'x':torch.randn(8,7,3),'labels':torch.randint(0,5,(8,7)),'_synthetic':False}
    trainer.training_step(model,batch,num_items_in_batch=40)
    frozen=getattr(model,frozen_name).weight;updated=getattr(model,updated_name).weight
    frozen_before=frozen.grad.clone();updated_before=updated.grad.clone()
    trainer.training_step(model,{**batch,'_synthetic':True},num_items_in_batch=40)
    assert torch.equal(frozen_before,frozen.grad)
    assert not torch.equal(updated_before,updated.grad)
    assert all(p.requires_grad for p in model.parameters())
