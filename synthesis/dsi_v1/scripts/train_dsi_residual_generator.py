#!/usr/bin/env python3
"""Train a deterministic DSI V1 residual-mel generator from Step B features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    feature_manifest: str
    out_dir: str
    seed: int
    device: str
    batch_size: int
    epochs: int
    max_steps: int
    lr: float
    weight_decay: float
    hidden_dim: int
    num_layers: int
    dropout: float
    smooth_weight: float
    num_workers: int
    train_limit: int
    dev_limit: int
    train_eval_limit: int
    overfit_n: int
    eval_every: int
    save_every: int
    grad_clip: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--smooth-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=0)
    parser.add_argument("--train-eval-limit", type=int, default=64)
    parser.add_argument("--overfit-n", type=int, default=0, help="Train and evaluate on the same N train rows.")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def select_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("split") == split and row.get("status") == "generated"]


def deterministic_limit(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:limit]


def build_patient_vocab(rows: list[dict[str, str]]) -> dict[str, int]:
    patients = sorted({row.get("patient_id", "") for row in rows})
    return {patient: idx for idx, patient in enumerate(patients)}


class ResidualFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, str]], patient_vocab: dict[str, int]) -> None:
        self.rows = rows
        self.patient_vocab = patient_vocab

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        payload = torch.load(row["feature_path"], map_location="cpu", weights_only=False)
        norm_mel = payload["norm_mel"].float()
        norm_ssl = payload["norm_ssl"].float()
        residual_mel = payload["residual_mel"].float()
        patient_id = row.get("patient_id", "")
        return {
            "utt_id": row.get("utt_id", ""),
            "patient_id": patient_id,
            "patient_idx": self.patient_vocab[patient_id],
            "norm_mel": norm_mel,
            "norm_ssl": norm_ssl,
            "residual_mel": residual_mel,
            "length": norm_mel.shape[0],
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(int(item["length"]) for item in items)
    batch_size = len(items)
    norm_mel = torch.zeros(batch_size, max_len, 80, dtype=torch.float32)
    norm_ssl = torch.zeros(batch_size, max_len, 768, dtype=torch.float32)
    residual = torch.zeros(batch_size, max_len, 80, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    patient_idx = torch.zeros(batch_size, dtype=torch.long)
    utt_ids: list[str] = []
    patient_ids: list[str] = []
    for batch_idx, item in enumerate(items):
        length = int(item["length"])
        norm_mel[batch_idx, :length] = item["norm_mel"]
        norm_ssl[batch_idx, :length] = item["norm_ssl"]
        residual[batch_idx, :length] = item["residual_mel"]
        mask[batch_idx, :length] = True
        patient_idx[batch_idx] = int(item["patient_idx"])
        utt_ids.append(item["utt_id"])
        patient_ids.append(item["patient_id"])
    return {
        "utt_ids": utt_ids,
        "patient_ids": patient_ids,
        "patient_idx": patient_idx,
        "norm_mel": norm_mel,
        "norm_ssl": norm_ssl,
        "residual_mel": residual,
        "mask": mask,
    }


class ResidualConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=padding, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        return residual + y


class DeterministicResidualGenerator(nn.Module):
    def __init__(
        self,
        patient_count: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.mel_proj = nn.Sequential(nn.Linear(80, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.ssl_proj = nn.Sequential(nn.Linear(768, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.patient_emb = nn.Embedding(patient_count, hidden_dim)
        self.in_proj = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        layers = []
        for layer_idx in range(num_layers):
            dilation = 2 ** (layer_idx % 4)
            layers.append(ResidualConvBlock(hidden_dim=hidden_dim, dilation=dilation, dropout=dropout))
        self.blocks = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 80)

    def forward(self, norm_mel: torch.Tensor, norm_ssl: torch.Tensor, patient_idx: torch.Tensor) -> torch.Tensor:
        mel_h = self.mel_proj(norm_mel)
        ssl_h = self.ssl_proj(norm_ssl)
        patient_h = self.patient_emb(patient_idx).unsqueeze(1).expand(-1, norm_mel.shape[1], -1)
        x = self.in_proj(torch.cat([mel_h, ssl_h, patient_h], dim=-1))
        x = self.blocks(x)
        return self.out_proj(self.out_norm(x))


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(pred.dtype)
    denom = mask_f.sum().clamp_min(1.0) * pred.shape[-1]
    return torch.sum(torch.abs(pred - target) * mask_f) / denom


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    pred_delta = pred[:, 1:] - pred[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    delta_mask = mask[:, 1:] & mask[:, :-1]
    return masked_l1(pred_delta, target_delta, delta_mask)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    smooth_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_l1 = masked_l1(pred, target, mask)
    smooth_l1 = masked_smooth_l1(pred, target, mask)
    return residual_l1 + smooth_weight * smooth_l1, residual_l1, smooth_l1


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        "patient_idx": batch["patient_idx"].to(device),
        "norm_mel": batch["norm_mel"].to(device),
        "norm_ssl": batch["norm_ssl"].to(device),
        "residual_mel": batch["residual_mel"].to(device),
        "mask": batch["mask"].to(device),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    smooth_weight: float,
) -> dict[str, float]:
    model.eval()
    total_frames = 0.0
    total_loss = 0.0
    total_l1 = 0.0
    total_smooth = 0.0
    pred_abs_sum = 0.0
    target_abs_sum = 0.0
    value_count = 0.0
    example_count = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        pred = model(batch["norm_mel"], batch["norm_ssl"], batch["patient_idx"])
        loss, residual_l1, smooth_l1 = compute_loss(pred, batch["residual_mel"], batch["mask"], smooth_weight)
        frames = float(batch["mask"].sum().item())
        total_frames += frames
        total_loss += float(loss.item()) * frames
        total_l1 += float(residual_l1.item()) * frames
        total_smooth += float(smooth_l1.item()) * frames
        mask_f = batch["mask"].unsqueeze(-1).to(pred.dtype)
        pred_abs_sum += float(torch.sum(torch.abs(pred) * mask_f).item())
        target_abs_sum += float(torch.sum(torch.abs(batch["residual_mel"]) * mask_f).item())
        value_count += frames * pred.shape[-1]
        example_count += len(batch["utt_ids"])
    denom = max(total_frames, 1.0)
    value_denom = max(value_count, 1.0)
    return {
        "loss": total_loss / denom,
        "residual_l1": total_l1 / denom,
        "smooth_l1": total_smooth / denom,
        "prediction_abs_mean": pred_abs_sum / value_denom,
        "target_abs_mean": target_abs_sum / value_denom,
        "frames": total_frames,
        "examples": float(example_count),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    patient_vocab: dict[str, int],
    epoch: int,
    step: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "patient_vocab": patient_vocab,
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
        },
        path,
    )


def make_loader(
    rows: list[dict[str, str]],
    patient_vocab: dict[str, int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[dict[str, Any]]:
    dataset = ResidualFeatureDataset(rows, patient_vocab)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    config = TrainConfig(**vars(args))
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(Path(args.feature_manifest))
    generated_rows = [row for row in rows if row.get("status") == "generated"]
    train_rows = select_split(rows, "train")
    dev_rows = select_split(rows, "dev")
    if args.train_limit > 0:
        train_rows = deterministic_limit(train_rows, args.train_limit, args.seed)
    if args.dev_limit > 0:
        dev_rows = deterministic_limit(dev_rows, args.dev_limit, args.seed + 1)
    if args.overfit_n > 0:
        train_rows = deterministic_limit(train_rows, args.overfit_n, args.seed)
        dev_rows = list(train_rows)
    if not train_rows:
        raise ValueError("No training rows selected")
    if not dev_rows:
        raise ValueError("No dev rows selected")

    patient_vocab = build_patient_vocab(generated_rows)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_eval_rows = train_rows
    if args.overfit_n <= 0 and args.train_eval_limit > 0:
        train_eval_rows = deterministic_limit(train_rows, args.train_eval_limit, args.seed + 2)

    train_loader = make_loader(train_rows, patient_vocab, args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_eval_loader = make_loader(train_eval_rows, patient_vocab, args.batch_size, shuffle=False, num_workers=args.num_workers)
    dev_loader = make_loader(dev_rows, patient_vocab, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = DeterministicResidualGenerator(
        patient_count=len(patient_vocab),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    (out_dir / "run_config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "patient_vocab.json").write_text(json.dumps(patient_vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_path = out_dir / "metrics.csv"
    metric_fields = [
        "event",
        "epoch",
        "step",
        "elapsed_seconds",
        "train_loss",
        "train_residual_l1",
        "train_smooth_l1",
        "train_prediction_abs_mean",
        "train_target_abs_mean",
        "train_examples",
        "dev_loss",
        "dev_residual_l1",
        "dev_smooth_l1",
        "dev_prediction_abs_mean",
        "dev_target_abs_mean",
        "dev_examples",
        "best_dev_residual_l1",
        "checkpoint",
    ]
    metric_rows: list[dict[str, Any]] = []
    best_dev_l1 = math.inf
    global_step = 0
    started = time.time()

    def run_eval(epoch: int, step: int, event: str) -> None:
        nonlocal best_dev_l1
        train_metrics = evaluate(model, train_eval_loader, device, args.smooth_weight)
        dev_metrics = evaluate(model, dev_loader, device, args.smooth_weight)
        checkpoint = ""
        if dev_metrics["residual_l1"] < best_dev_l1:
            best_dev_l1 = dev_metrics["residual_l1"]
            checkpoint = "best_dev.pt"
            save_checkpoint(out_dir / checkpoint, model, optimizer, config, patient_vocab, epoch, step, dev_metrics)
        row = {
            "event": event,
            "epoch": epoch,
            "step": step,
            "elapsed_seconds": f"{time.time() - started:.3f}",
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_residual_l1": f"{train_metrics['residual_l1']:.6f}",
            "train_smooth_l1": f"{train_metrics['smooth_l1']:.6f}",
            "train_prediction_abs_mean": f"{train_metrics['prediction_abs_mean']:.6f}",
            "train_target_abs_mean": f"{train_metrics['target_abs_mean']:.6f}",
            "train_examples": int(train_metrics["examples"]),
            "dev_loss": f"{dev_metrics['loss']:.6f}",
            "dev_residual_l1": f"{dev_metrics['residual_l1']:.6f}",
            "dev_smooth_l1": f"{dev_metrics['smooth_l1']:.6f}",
            "dev_prediction_abs_mean": f"{dev_metrics['prediction_abs_mean']:.6f}",
            "dev_target_abs_mean": f"{dev_metrics['target_abs_mean']:.6f}",
            "dev_examples": int(dev_metrics["examples"]),
            "best_dev_residual_l1": f"{best_dev_l1:.6f}",
            "checkpoint": checkpoint,
        }
        metric_rows.append(row)
        write_csv(metrics_path, metric_rows, metric_fields)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "train_rows": len(train_rows),
                "dev_rows": len(dev_rows),
                "train_eval_rows": len(train_eval_rows),
                "patient_count": len(patient_vocab),
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    run_eval(epoch=0, step=0, event="initial_eval")

    stop_training = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["norm_mel"], batch["norm_ssl"], batch["patient_idx"])
            loss, _, _ = compute_loss(pred, batch["residual_mel"], batch["mask"], args.smooth_weight)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                run_eval(epoch=epoch, step=global_step, event="step_eval")
            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(
                    out_dir / f"checkpoint_step_{global_step}.pt",
                    model,
                    optimizer,
                    config,
                    patient_vocab,
                    epoch,
                    global_step,
                    {"event": "periodic_save"},
                )
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break
        run_eval(epoch=epoch, step=global_step, event="epoch_eval")
        if stop_training:
            break

    final_metrics = metric_rows[-1] if metric_rows else {}
    save_checkpoint(out_dir / "last.pt", model, optimizer, config, patient_vocab, epoch, global_step, final_metrics)
    (out_dir / "done.json").write_text(
        json.dumps(
            {
                "status": "done",
                "elapsed_seconds": time.time() - started,
                "steps": global_step,
                "epochs_completed": epoch,
                "best_dev_residual_l1": best_dev_l1,
                "best_checkpoint": str(out_dir / "best_dev.pt"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
