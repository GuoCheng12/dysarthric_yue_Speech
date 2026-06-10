#!/usr/bin/env python3
import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel
from transformers import GenerationConfig


def import_sft_module(path):
    spec = importlib.util.spec_from_file_location("qwen3_asr_sft_official", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cuda_snapshot(tag):
    free, total = torch.cuda.mem_get_info()
    return {
        "tag": tag,
        "free_gib": round(free / 1024**3, 3),
        "total_gib": round(total / 1024**3, 3),
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "max_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }


def prepare_features(rows, processor, sft_mod):
    fn = sft_mod.make_preprocess_fn_prefix_only(processor)
    features = []
    for row in rows:
        ex = {"prompt": row.get("prompt", ""), "audio": row["audio"], "text": row["text"]}
        features.append(fn(ex))
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/qwen3-asr/models/Qwen3-ASR-1.7B")
    parser.add_argument("--train-jsonl", default="/data/qwen3-asr/finetune/data/e2_train.jsonl")
    parser.add_argument("--sft-script", default="/data/qwen3-asr/finetune/official/qwen3_asr_sft.py")
    parser.add_argument("--batch-sizes", default="1,2,4,6,8")
    parser.add_argument("--sample-pool", type=int, default=32)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.backends.cuda.matmul.allow_tf32 = True

    sft_mod = import_sft_module(args.sft_script)
    rows = read_jsonl(args.train_jsonl)
    rows = sorted(rows, key=lambda r: float(r.get("duration") or 0), reverse=True)[: args.sample_pool]
    print(json.dumps({
        "kind": "probe_samples",
        "rows": len(rows),
        "max_duration": max(float(r.get("duration") or 0) for r in rows),
        "min_duration": min(float(r.get("duration") or 0) for r in rows),
        "sample_utts": [r["utt_id"] for r in rows[:5]],
    }, ensure_ascii=False), flush=True)

    use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor
    sft_mod.patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.to("cuda")
    model.train()
    collator = sft_mod.DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=args.sr)
    features = prepare_features(rows, processor, sft_mod)

    print(json.dumps({"kind": "after_model_load", **cuda_snapshot("after_model_load")}, ensure_ascii=False), flush=True)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        batch_features = features[:bs]
        result = {"kind": "sft_forward_backward_probe", "batch_size": bs}
        try:
            t0 = time.time()
            batch = collator(batch_features)
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to("cuda")
                    if v.is_floating_point():
                        batch[k] = batch[k].to(dtype=getattr(model, "dtype", torch.bfloat16))
            t_collate = time.time()
            model.zero_grad(set_to_none=True)
            out = model(**batch)
            loss = out.loss
            t_forward = time.time()
            loss.backward()
            torch.cuda.synchronize()
            t_backward = time.time()
            result.update({
                "ok": True,
                "loss": round(float(loss.detach().cpu()), 6),
                "collate_sec": round(t_collate - t0, 3),
                "forward_sec": round(t_forward - t_collate, 3),
                "backward_sec": round(t_backward - t_forward, 3),
                "total_sec": round(t_backward - t0, 3),
                **cuda_snapshot("after_backward"),
            })
            model.zero_grad(set_to_none=True)
            del batch, out, loss
        except RuntimeError as exc:
            torch.cuda.synchronize()
            result.update({
                "ok": False,
                "error": str(exc).splitlines()[0],
                **cuda_snapshot("after_error"),
            })
            try:
                model.zero_grad(set_to_none=True)
            except Exception:
                pass
            torch.cuda.empty_cache()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            break
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
