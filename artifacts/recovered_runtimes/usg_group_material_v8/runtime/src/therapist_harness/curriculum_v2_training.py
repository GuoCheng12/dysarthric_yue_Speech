"""Single-process stage barriers: no future-batch prefetch or optimizer reset."""

from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from finetune.scripts.qwen3_asr_lora_sft import QwenASRLoRATrainer


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def state_digest(value):
    digest = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            digest.update(str((x.dtype, tuple(x.shape))).encode())
            digest.update(x.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=repr):
                digest.update(repr(k).encode())
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            digest.update(str(len(x)).encode())
            for v in x:
                visit(v)
        else:
            digest.update(repr(x).encode())
    visit(value)
    return digest.hexdigest()


class BlockDataset(Dataset):
    def __init__(self, features, total_rows=3200):
        self.features = features
        self.plan = [None] * total_rows
        self.read_positions = []

    def install(self, manifest):
        for row in manifest:
            p = row["position"]
            assert self.plan[p] is None, "a materialized training position is immutable"
            assert row["id"] in self.features
            self.plan[p] = dict(row)

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        row = self.plan[index]
        assert row is not None, f"future course was prefetched before its decision: position {index}"
        self.read_positions.append(index)
        return self.features[row["id"]]


class ContinuousCourseTrainer(QwenASRLoRATrainer):
    def get_train_dataloader(self):
        assert self.args.world_size == 1 and self.args.dataloader_num_workers == 0
        # Accelerate's DataLoaderShard looks one batch ahead. A raw single-device
        # loader avoids that; Trainer._prepare_inputs still moves tensors to CUDA.
        return DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size,
                          shuffle=False, num_workers=0, drop_last=False,
                          collate_fn=self.data_collator, pin_memory=self.args.dataloader_pin_memory,
                          generator=torch.Generator().manual_seed(731))


def measure_losses(trainer, features):
    """One eval-mode loss per UNIQUE training recording; no RNG/training side effects."""
    ids = sorted(features)
    was_training = trainer.model.training
    result = {}
    try:
        with preserve_rng(), torch.no_grad():
            trainer.model.eval()
            for start in range(0, len(ids), 8):
                keys = ids[start:start+8]
                batch = trainer._prepare_inputs(trainer.data_collator([features[k] for k in keys]))
                with trainer.compute_loss_context_manager():
                    _, output = trainer.compute_loss(trainer.model, batch, return_outputs=True)
                labels = batch["labels"][:, 1:].contiguous()
                logits = output["logits"][:, :-1].float().contiguous()
                losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                         ignore_index=-100, reduction="none").view_as(labels)
                values = losses.sum(1) / labels.ne(-100).sum(1)
                assert torch.isfinite(values).all(), "nonfinite dropout-off loss"
                result.update(zip(keys, values.cpu().tolist()))
    finally:
        trainer.model.train(was_training)
    return result
