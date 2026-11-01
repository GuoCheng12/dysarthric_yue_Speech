import random

import accelerate.utils.other as accelerate_other
import numpy as np
import pytest
import torch
from torch import nn
from transformers import TrainerCallback, TrainingArguments, default_data_collator, set_seed

from therapist_harness.curriculum_v2_training import (BlockDataset, ContinuousCourseTrainer,
    measure_losses, preserve_rng, state_digest)


class ToyASR(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 7))
        self.dropout = nn.Dropout(.2)

    def forward(self, x, **kwargs):
        return {"logits": self.dropout(x) @ self.weight}


def train_toy(directory, barrier):
    set_seed(42)
    features = {f"R{i:04d}": {"x": torch.randn(6, 3), "labels": torch.tensor([-100, 1, 2, 3, 4, -100])}
                for i in range(32)}
    manifest = [{"position": i, "id": f"R{i:04d}"} for i in range(32)]
    dataset = BlockDataset(features, 32)
    dataset.install(manifest[:16] if barrier else manifest)
    model = ToyASR()
    trainer = ContinuousCourseTrainer(model=model, args=TrainingArguments(
        output_dir=str(directory), use_cpu=True, per_device_train_batch_size=2,
        gradient_accumulation_steps=4, max_steps=4, warmup_steps=1, learning_rate=.01,
        dataloader_num_workers=0, save_strategy="no", report_to="none", disable_tqdm=True,
        remove_unused_columns=False, logging_steps=1), train_dataset=dataset, data_collator=default_data_collator)
    observed = []

    class Barrier(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                assert dataset.read_positions == list(range(16)), "data were prefetched across barrier"
                before = (state_digest(trainer.optimizer.state_dict()), state_digest(trainer.lr_scheduler.state_dict()))
                with preserve_rng():
                    random.random()
                    np.random.rand(20)
                    torch.rand(100)
                    losses = measure_losses(trainer, features)
                    assert len(losses) == 32
                    dataset.install(manifest[16:])
                assert before == (state_digest(trainer.optimizer.state_dict()), state_digest(trainer.lr_scheduler.state_dict()))
                observed.append(state.global_step)
    if barrier:
        trainer.add_callback(Barrier())
    trainer.train()
    assert dataset.read_positions == list(range(32))
    assert trainer.state.global_step == 4 and trainer.lr_scheduler.last_epoch == 4
    assert all(int(v["step"]) == 4 for v in trainer.optimizer.state.values())
    assert observed == ([2] if barrier else [])
    return {"weights": state_digest(model.state_dict()), "optimizer": state_digest(trainer.optimizer.state_dict()),
            "scheduler": state_digest(trainer.lr_scheduler.state_dict()), "rng": state_digest(torch.get_rng_state())}


def test_live_assessment_barrier_preserves_exact_training_trajectory(tmp_path, monkeypatch):
    monkeypatch.setattr(accelerate_other, "is_deepspeed_available", lambda: False)
    torch.set_num_threads(1)
    uninterrupted = train_toy(tmp_path / "plain", False)
    paused = train_toy(tmp_path / "paused", True)
    assert uninterrupted == paused


def test_unmaterialized_course_and_overwrites_fail():
    ds = BlockDataset({"R0001": {"x": 1}, "R0002": {"x": 2}}, 2)
    ds.install([{"position": 0, "id": "R0001"}])
    with pytest.raises(AssertionError, match="prefetched"):
        ds[1]
    ds.install([{"position": 1, "id": "R0002"}])
    assert ds[1]["x"] == 2
    with pytest.raises(AssertionError, match="immutable"):
        ds.install([{"position": 1, "id": "R0001"}])
