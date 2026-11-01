#!/usr/bin/env python3
"""Source-balanced audio DPO (or matched SFT) from an existing ASR LoRA parent.

The fixed R-only reference is loaded, scored without gradients, and unloaded
before the policy is loaded. Cached values are bound to audio bytes, raw target
texts, prompt/language, tokenizer/base files, and the exact reference adapter.
There is no disabled-adapter/base-model substitute for the DPO reference.
"""

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "finetune" / "official"))

import torch

from therapist_harness.dpo_training import (
    AudioPreferenceCollator, SourceBalancedWindows, batch_log_probs, dpo_losses,
    frozen_reference_log_probs, group_training_rows, move_batch, preference_key,
)
from therapist_harness.store import sha256_directory


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_hashes(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"model/adapter must be a local directory: {directory}")
    # Explicit top-level model artifacts only; no secrets, logs, or nested checkpoints.
    files = [p for p in directory.iterdir() if p.is_file() and (
        p.suffix in {".safetensors", ".json", ".model"} or p.name in {"merges.txt", "vocab.txt", "pytorch_model.bin", "adapter_model.bin"})]
    return {p.name: sha256(p) for p in sorted(files)}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parameter_digest(model, *, trainable_only=False):
    digest = hashlib.sha256()
    for name, tensor in model.named_parameters():
        if trainable_only and not tensor.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def load_adapter(model_path, adapter_path, device, dtype, trainable, disable_dropout):
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel
    from transformers import GenerationConfig
    from qwen3_asr_sft import patch_outer_forward

    wrapper = Qwen3ASRModel.from_pretrained(model_path, dtype=dtype, device_map=None)
    model = wrapper.model
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    if disable_dropout:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    if trainable:
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
    return model.to(device), wrapper.processor


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--policy_adapter_path", required=True)
    parser.add_argument("--reference_adapter_path", required=True,
                        help="Frozen initial R-only ASR adapter, held fixed across every round")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--objective", choices=["dpo", "sft"], default="dpo")
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1, help="Pairs (DPO) or audio (SFT) per forward")
    parser.add_argument("--real_per_step", type=int, default=32)
    parser.add_argument("--synthetic_per_step", type=int, default=8)
    parser.add_argument("--synthetic_weight", type=float, default=0.05)
    parser.add_argument("--synthetic_allocation_file", help="Explicit audio exposure schedule; loss remains unchanged")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable_dropout", type=int, choices=[0, 1], default=1)
    parser.add_argument("--gradient_checkpointing", type=int, choices=[0, 1], default=1)
    parser.add_argument("--probe_pairs", type=int, default=2,
                        help="Fixed first preference rows measured before and after training")
    args = parser.parse_args(argv)
    if args.max_steps <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("max_steps, batch_size and lr must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    from transformers import set_seed
    from qwen3_asr_sft import copy_required_hf_files_for_qwen_asr

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("audio preference training currently supports one process")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_protocol.json").exists():
        raise FileExistsError("output already contains a bound training run; use a fresh directory")
    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float32
    rows = [json.loads(line) for line in Path(args.train_file).read_text().splitlines() if line.strip()]
    policy_parent_hash = sha256_directory(Path(args.policy_adapter_path))
    audio_hashes = {}
    for row in rows:
        if args.objective == "dpo" and row.get("current_asr_sha256") != policy_parent_hash:
            raise ValueError("DPO rejected samples must be bound to the current policy parent adapter")
        path = str(Path(row["audio"]).resolve())
        if path not in audio_hashes:
            audio_hashes[path] = sha256(path)
        if row.get("audio_sha256") and row["audio_sha256"] != audio_hashes[path]:
            raise ValueError(f"audio content changed: {path}")
        row["audio"], row["audio_sha256"] = path, audio_hashes[path]
    groups = group_training_rows(rows, args.objective)
    allocation = None
    if args.synthetic_allocation_file:
        allocation=json.loads(Path(args.synthetic_allocation_file).read_text())
        if len(allocation['steps'])!=args.max_steps:
            raise ValueError('allocation must specify every optimizer step')
    rows = [row for source in groups.values() for group in source.values() for row in group]
    protocol = {
        "schema_version": 1, "arguments": vars(args),
        "objective": "mean_real_audio_loss + synthetic_weight * mean_synthetic_audio_loss",
        "dpo_log_probs": "sum transcript + EOS, mask chat/audio/forced language/padding",
        "sft_loss": "per-audio mean transcript + EOS token NLL",
        "reference_identity": "frozen_initial_R_only_ASR_adapter_not_TTS_reference_speech",
        "reference_asr_sha256": sha256_directory(Path(args.reference_adapter_path)),
        "dropout": "runtime dropout disabled for both models" if args.disable_dropout else "adapter-configured dropout",
        "optimizer_lineage": "new AdamW optimizer each invocation; parent adapter weights continued",
        "scheduler": "constant", "training_file_sha256": sha256(args.train_file),
        "model_artifacts": artifact_hashes(args.model_path),
        "policy_parent_sha256": policy_parent_hash,
        "policy_parent_artifacts": artifact_hashes(args.policy_adapter_path),
        "reference_artifacts": artifact_hashes(args.reference_adapter_path),
        "audio_sha256": audio_hashes,
        "audio_counts": {s: len(g) for s, g in groups.items()},
        "row_counts": dict(Counter(r["source"] for r in rows)),
        **({'synthetic_allocation_sha256':sha256(args.synthetic_allocation_file)} if allocation else {}),
    }
    write_json(output / "run_protocol.json", protocol)
    cache = {}
    if args.objective == "dpo":
        reference, processor = load_adapter(args.model_path, args.reference_adapter_path, device, dtype, False, args.disable_dropout)
        collator = AudioPreferenceCollator(processor, args.sr)
        # Hash adapter tensors before/after inference; all base tensors also have requires_grad=False.
        reference_before = {name: parameter_digest(module) for name, module in reference.named_modules()
                            if name.endswith(("lora_A", "lora_B"))}
        if not reference_before or not reference.active_adapters:
            raise RuntimeError("reference must have active R-only LoRA weights, not a bare base model")
        for start in range(0, len(rows), args.batch_size):
            selected = rows[start:start + args.batch_size]
            batch = move_batch(collator(selected), device, dtype)
            values = frozen_reference_log_probs(reference, batch).float().cpu().tolist()
            for index, row in enumerate(selected):
                cache[preference_key(row)] = [values[index], values[index + len(selected)]]
            print(f"[reference] {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)
        reference_after = {name: parameter_digest(module) for name, module in reference.named_modules()
                           if name.endswith(("lora_A", "lora_B"))}
        if reference_before != reference_after:
            raise RuntimeError("frozen R-only reference adapter changed during inference")
        write_json(output / "reference_log_probs.json", {
            "protocol_sha256": sha256(output / "run_protocol.json"),
            "adapter_tensor_hashes": reference_before, "reference_unchanged": True,
            "active_adapters": reference.active_adapters,
            "trainable_parameters": sum(p.numel() for p in reference.parameters() if p.requires_grad),
            "values": cache,
        })
        del reference, processor, collator, batch
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Reference construction must not change dropout or sampler RNG for the policy.
    set_seed(args.seed)
    policy, processor = load_adapter(args.model_path, args.policy_adapter_path, device, dtype, True, args.disable_dropout)
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    collator = AudioPreferenceCollator(processor, args.sr)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("policy has no trainable adapter parameters")
    policy_before = parameter_digest(policy, trainable_only=True)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    windows = SourceBalancedWindows(groups, args.real_per_step, args.synthetic_per_step, args.synthetic_weight, args.seed,
        synthetic_schedule=allocation['steps'] if allocation else None)

    @torch.no_grad()
    def measure_probe():
        if args.objective != "dpo":
            return []
        policy.eval()
        values = []
        for row in rows[:args.probe_pairs]:
            logs, _ = batch_log_probs(policy, move_batch(collator([row]), device, dtype))
            reference_logs = cache[preference_key(row)]
            values.append({"key": preference_key(row), "policy_chosen": logs[0].item(),
                           "policy_rejected": logs[1].item(),
                           "policy_margin": (logs[0] - logs[1]).item(),
                           "dpo_loss": dpo_losses(logs[:1], logs[1:],
                               torch.tensor([reference_logs[0]], device=device),
                               torch.tensor([reference_logs[1]], device=device), args.beta).item()})
        policy.train()
        return values

    before_probe = measure_probe()
    history = []
    exposure_handle = (output / "training_exposures.jsonl").open("w", encoding="utf-8")
    for step in range(1, args.max_steps + 1):
        window = windows.next()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for start in range(0, len(window), args.batch_size):
            selected = window[start:start + args.batch_size]
            batch = move_batch(collator(selected, include_rejected=args.objective == "dpo"), device, dtype)
            logs, counts = batch_log_probs(policy, batch)
            if args.objective == "dpo":
                reference_values = torch.tensor([cache[preference_key(r)] for r in selected], device=device)
                losses = dpo_losses(logs[:len(selected)], logs[len(selected):],
                                    reference_values[:, 0], reference_values[:, 1], args.beta)
            else:
                losses = -logs / counts
            weights = torch.tensor([r["_loss_weight"] for r in selected], device=device)
            loss = (losses * weights).sum()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            total_loss += loss.detach().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("nonfinite policy gradient")
        optimizer.step()
        row = {"step": step, "loss": total_loss, "gradient_norm": grad_norm.item(),
               "audio_exposures": windows.counts, "pair_or_sft_row_exposures": dict(Counter(r["source"] for r in window))}
        history.append(row)
        for sample in window:
            exposure_handle.write(json.dumps({"step": step, "source": sample["source"],
                "audio_id": sample["_audio_group"], "occurrence": sample["_occurrence"],
                "preference_key": preference_key(sample), "loss_weight": sample["_loss_weight"]}, ensure_ascii=False) + "\n")
        exposure_handle.flush()
        print(json.dumps(row), flush=True)
    exposure_handle.close()
    after_probe = measure_probe()
    policy_after = parameter_digest(policy, trainable_only=True)
    checkpoint = output / f"checkpoint-{args.max_steps}"
    policy.save_pretrained(checkpoint, safe_serialization=True)
    copy_required_hf_files_for_qwen_asr(args.model_path, str(checkpoint))
    write_json(checkpoint / "trainer_state.json", {
        "global_step": args.max_steps, "max_steps": args.max_steps,
        "log_history": history, "objective": args.objective,
        "reference_asr_sha256": protocol["reference_asr_sha256"],
    })
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    reference_unchanged = artifact_hashes(args.reference_adapter_path) == protocol["reference_artifacts"]
    if not reference_unchanged:
        raise RuntimeError("frozen R-only reference files changed during policy training")
    write_json(output / "training_result.json", {
        "status": "complete", "checkpoint": str(checkpoint), "history": history,
        "reference_files_unchanged": reference_unchanged,
        "policy_trainable_sha256_before": policy_before, "policy_trainable_sha256_after": policy_after,
        "policy_changed": policy_before != policy_after,
        "probe_before": before_probe, "probe_after": after_probe,
        "protocol_sha256": sha256(output / "run_protocol.json"),
        "checkpoint_artifacts": artifact_hashes(checkpoint),
    })


if __name__ == "__main__":
    main()
