import accelerate.utils.other as accelerate_other
import torch
from torch import nn
from transformers import TrainingArguments

from finetune.scripts.qwen3_asr_lora_sft import QwenASRLoRATrainer


class _ToyMeanLossModel(nn.Module):
    """Mean-loss toy model whose kwargs reproduce Qwen's forward signature."""

    def __init__(self, initial_weight):
        super().__init__()
        self.weight = nn.Parameter(initial_weight.clone())

    def forward(self, x, labels=None, **kwargs):
        return {"loss": ((x @ self.weight - labels) ** 2).mean()}


def _make_trainer(model, output_dir, monkeypatch):
    # Match the production guard for this environment's unusable DeepSpeed install.
    monkeypatch.setattr(accelerate_other, "is_deepspeed_available", lambda: False)
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        max_grad_norm=0,
        use_cpu=True,
        report_to="none",
        disable_tqdm=True,
    )
    return QwenASRLoRATrainer(model=model, args=args)


def _accumulate(initial_weight, x, labels, output_dir, monkeypatch, old_behavior=False):
    model = _ToyMeanLossModel(initial_weight)
    trainer = _make_trainer(model, output_dir, monkeypatch)
    assert trainer.model_accepts_loss_kwargs is False
    if old_behavior:
        # Reproduce the pre-fix inference caused by Qwen's kwargs signature.
        trainer.model_accepts_loss_kwargs = True

    microbatches = [
        {"x": x[index : index + 1], "labels": labels[index : index + 1]}
        for index in range(x.shape[0])
    ]
    batches, num_items = trainer.get_batch_samples(
        iter(microbatches), len(microbatches), torch.device("cpu")
    )
    assert (int(num_items) == 32) if old_behavior else (num_items is None)
    trainer.current_gradient_accumulation_steps = len(batches)
    for batch in batches:
        trainer.training_step(model, batch, num_items_in_batch=num_items)
    return model


def test_grad_acc_32_matches_batch_32_mean_loss(tmp_path, monkeypatch):
    torch.manual_seed(7)
    x = torch.randn(32, 3, dtype=torch.float64)
    labels = torch.randn(32, dtype=torch.float64)
    initial_weight = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64)

    reference = _ToyMeanLossModel(initial_weight)
    reference(x=x, labels=labels)["loss"].backward()
    accumulated = _accumulate(
        initial_weight, x, labels, tmp_path / "fixed", monkeypatch
    )
    old_behavior = _accumulate(
        initial_weight, x, labels, tmp_path / "old", monkeypatch, old_behavior=True
    )

    torch.testing.assert_close(
        accumulated.weight.grad, reference.weight.grad, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        old_behavior.weight.grad, reference.weight.grad * 32, rtol=1e-12, atol=1e-12
    )

    # SGD makes gradient scale observable; AdamW's first step is nearly scale invariant.
    learning_rate = 0.05
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=learning_rate)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=learning_rate)
    reference_optimizer.step()
    accumulated_optimizer.step()
    torch.testing.assert_close(
        accumulated.weight, reference.weight, rtol=1e-12, atol=1e-12
    )
