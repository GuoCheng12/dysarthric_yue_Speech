import accelerate.utils.other as accelerate_other
import pytest
import torch
from torch import nn
from transformers import TrainingArguments

from finetune.scripts.qwen3_asr_lora_sft import QwenASRLoRATrainer


class ToyCausalModel(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weight = nn.Parameter(weights.clone())

    def forward(self, x, **kwargs):
        return {"logits": x @ self.weight}


@pytest.mark.parametrize("count", [32, 29, 5])
def test_microbatch_partitions_preserve_utterance_mean(tmp_path, monkeypatch, count):
    monkeypatch.setattr(accelerate_other, "is_deepspeed_available", lambda: False)
    torch.manual_seed(7)
    x = torch.randn(count, 8, 3)
    labels = torch.randint(0, 5, (count, 8))
    labels[:, 0] = -100
    for i in range(count):
        labels[i, 2 + i % 6:] = -100
    weights = torch.randn(3, 5)
    reference = ToyCausalModel(weights)
    # Explicit legacy 1 x N objective: each sentence has its own mean CE.
    losses = [nn.functional.cross_entropy(
        reference(x=x[i:i+1])["logits"][:, :-1].reshape(-1, 5),
        labels[i:i+1, 1:].reshape(-1), ignore_index=-100,
    ) for i in range(count)]
    torch.stack(losses).mean().backward()
    for microbatch in (1, 8, 16, 32):
        model = ToyCausalModel(weights)
        trainer = QwenASRLoRATrainer(model=model, args=TrainingArguments(
            output_dir=str(tmp_path / str(microbatch)), use_cpu=True,
            per_device_train_batch_size=microbatch,
            gradient_accumulation_steps=max(1, 32 // microbatch),
            max_grad_norm=0, report_to="none", disable_tqdm=True,
        ))
        batches = [{"x": x[i:i+microbatch], "labels": labels[i:i+microbatch]}
                   for i in range(0, count, microbatch)]
        batches, denominator = trainer.get_batch_samples(iter(batches), len(batches), torch.device("cpu"))
        assert denominator == count
        trainer.current_gradient_accumulation_steps = len(batches)
        for batch in batches:
            trainer.training_step(model, batch, num_items_in_batch=denominator)
        torch.testing.assert_close(model.weight.grad, reference.weight.grad, atol=2e-7, rtol=2e-6)


def test_eval_loss_and_empty_target(tmp_path, monkeypatch):
    monkeypatch.setattr(accelerate_other, "is_deepspeed_available", lambda: False)
    model = ToyCausalModel(torch.randn(3, 5))
    trainer = QwenASRLoRATrainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to="none"))
    data = {"x": torch.randn(2, 4, 3), "labels": torch.tensor([[-100, 1, -100, -100], [-100, 2, 3, 4]])}
    loss, output = trainer.compute_loss(model, data, return_outputs=True)
    expected = torch.stack([trainer.compute_loss(model, {k:v[i:i+1] for k,v in data.items()}) for i in range(2)]).mean()
    torch.testing.assert_close(loss, expected)
    assert output["logits"].shape == (2, 4, 5)
    data["labels"][:] = -100
    with pytest.raises(ValueError, match="supervised token"):
        trainer.compute_loss(model, data)
