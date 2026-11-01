"""Token-weighted accumulated gradients must match one full-batch CE."""
import sys
from pathlib import Path
import pytest
import torch
import accelerate.utils.other as other
from transformers import TrainingArguments

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from run_fresh_lora_sft_grid_20260910 import RecipeTrainer, EpochSampler


@pytest.mark.parametrize('count',[5,13])
def test_token_mean_across_unequal_microbatches(tmp_path,monkeypatch,count):
    monkeypatch.setattr(other,'is_deepspeed_available',lambda:False)
    torch.manual_seed(19)
    x=torch.randn(count,8,3);labels=torch.randint(0,5,(count,8));labels[:,0]=-100
    for i in range(count):labels[i,2+i%6:]=-100
    class Model(torch.nn.Module):
        def __init__(self,weights):super().__init__();self.weight=torch.nn.Parameter(weights.clone())
        def forward(self,x):return {'logits':x@self.weight}
    weights=torch.randn(3,5);reference=Model(weights)
    torch.nn.functional.cross_entropy(reference(x)['logits'][:,:-1].reshape(-1,5),
        labels[:,1:].reshape(-1),ignore_index=-100).backward()
    for size in (1,4,8):
        model=Model(weights)
        trainer=RecipeTrainer(model=model,objective='token',args=TrainingArguments(
            output_dir=str(tmp_path/str(size)),use_cpu=True,report_to='none',gradient_accumulation_steps=8))
        batches=[{'x':x[i:i+size],'labels':labels[i:i+size]} for i in range(0,count,size)]
        batches,denom=trainer.get_batch_samples(iter(batches),len(batches),torch.device('cpu'))
        assert denom==labels[:,1:].ne(-100).sum().item()
        trainer.current_gradient_accumulation_steps=len(batches)
        for batch in batches:trainer.training_step(model,batch,num_items_in_batch=denom)
        torch.testing.assert_close(model.weight.grad,reference.weight.grad,atol=2e-7,rtol=2e-6)


def test_epoch_sampler_consumes_each_record_once():
    sampler=EpochSampler([[1,2,0],[2,0,1],[0,1,2]])
    for epoch,expected in enumerate(sampler.orders):
        sampler.set_epoch(epoch)
        assert list(sampler)==expected and len(set(sampler))==3
