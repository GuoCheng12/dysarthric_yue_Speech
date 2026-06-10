#!/usr/bin/env python3
"""LoRA SFT entrypoint for Qwen3-ASR.

This mirrors the official Qwen3-ASR SFT script, but wraps the underlying
Qwen3-ASR model with PEFT LoRA adapters before passing it to Trainer.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import GenerationConfig, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "finetune" / "official"))

from qwen3_asr_sft import (  # noqa: E402
    CastFloatInputsTrainer,
    DataCollatorForQwen3ASRFinetuning,
    MakeEveryCheckpointInferableCallback,
    find_latest_checkpoint,
    make_preprocess_fn_prefix_only,
    patch_outer_forward,
)

try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency `peft`. Install it in the training environment, "
        "for example: pip install -U peft"
    ) from exc


DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR LoRA SFT")

    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-lora-sft-out")

    p.add_argument("--sr", type=int, default=16000)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_acc", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--log_steps", type=int, default=5)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)

    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=69)
    p.add_argument("--save_total_limit", type=int, default=3)

    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32)
    p.add_argument(
        "--lora_scale",
        type=float,
        default=None,
        help="Optional LoRA scaling lambda. When set, lora_alpha = lora_rank * lora_scale.",
    )
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default=DEFAULT_TARGET_MODULES)
    p.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    p.add_argument("--dry_run_model_setup", type=int, default=0)

    return p.parse_args()


def parse_target_modules(value: str):
    value = (value or "").strip()
    if not value:
        return DEFAULT_TARGET_MODULES.split(",")
    if value == "all-linear":
        return "all-linear"
    return [item.strip() for item in value.split(",") if item.strip()]


def count_targeted_linear_modules(model, target_modules):
    if target_modules == "all-linear":
        return sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear))

    matched = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(name.endswith(target) for target in target_modules):
            matched.append(name)
    return len(matched)


def write_lora_metadata(path: Path, args, target_modules, matched_count, model):
    path.mkdir(parents=True, exist_ok=True)
    trainable = 0
    total = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    metadata = {
        "method": "LoRA",
        "base_model": args.model_path,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_scale": args.lora_alpha / args.lora_rank,
        "lora_dropout": args.lora_dropout,
        "lora_bias": args.lora_bias,
        "lora_target_modules": target_modules,
        "matched_linear_module_count": matched_count,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total if total else None,
    }
    (path / "lora_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if not args.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")
    if args.lora_scale is not None:
        if args.lora_scale <= 0:
            raise ValueError("--lora_scale must be positive")
        args.lora_alpha = args.lora_rank * args.lora_scale

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    if hasattr(model, "config"):
        model.config.use_cache = False

    target_modules = parse_target_modules(args.lora_target_modules)
    matched_count = count_targeted_linear_modules(model, target_modules)
    if matched_count == 0:
        raise RuntimeError(
            "LoRA target modules matched 0 Linear layers. "
            f"Requested target_modules={target_modules!r}. "
            "Try --lora_target_modules all-linear or inspect model.named_modules()."
        )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    write_lora_metadata(Path(args.output_dir), args, target_modules, matched_count, model)
    print(f"[lora] target_modules={target_modules}")
    print(f"[lora] matched_linear_module_count={matched_count}")
    print(f"[lora] metadata={Path(args.output_dir) / 'lora_run_metadata.json'}")
    if args.dry_run_model_setup == 1:
        print("[dry_run_model_setup] LoRA model setup succeeded; exiting before dataset/training.")
        return

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            **({"validation": args.eval_file} if args.eval_file else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=args.sr)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=(args.pin_memory == 1),
        dataloader_persistent_workers=(args.persistent_workers == 1),
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        do_eval=bool(args.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args.model_path)],
    )

    resume_from = (args.resume_from or "").strip()
    if not resume_from and args.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
