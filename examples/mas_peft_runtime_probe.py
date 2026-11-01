#!/usr/bin/env python3
"""Compare actual unmerged ASR inference with the same checkpoints' merged runs."""

import argparse
from collections import Counter
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from mas_selection_dpo_probe import REPO, read, save
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file, sha256_directory

sys.path.insert(0, str(REPO / 'finetune/scripts'))
import evaluate_asr_checkpoint as evaluator


def main(args):
    import torch
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel

    base = args.base.resolve()
    root = base / 'peft_runtime_probe_v1' / args.arm
    root.mkdir(parents=True, exist_ok=False)
    p = read(base / 'lexical24_loop_v1/protocol.json')
    if args.arm == 'parent':
        adapter = Path(p['initial_adapter']['path'])
        merged_predictions = Path(p['initial_adapter_dev_predictions']['path'])
    else:
        adapter = base / 'lexical24_loop_v1/round_001/mas_dpo/training/checkpoint-4'
        merged_predictions = base / 'lexical24_loop_v1/round_001/mas_dpo/evaluation/predictions.jsonl'
    assert adapter.is_dir() and merged_predictions.is_file()
    split = Path(p['feedback_split']['path'])
    sources = [Path(__file__), Path(evaluator.__file__), split, merged_predictions,
               base / 'lexical24_loop_v1/protocol.json']
    hashes = {str(f): sha256_file(f) for f in sources}
    adapter_hash = sha256_directory(adapter)
    save(root / 'protocol.json', dict(source_sha256=hashes, adapter_path=str(adapter),
        adapter_sha256=adapter_hash, base_model_path=p['base_model_dir'],
        mode='active unmerged PEFT LoRA; bf16 base; default PEFT adapter dtype promotion',
        language='Cantonese', context='', batch_size=8, max_new_tokens=256,
        actual_training=False, postprocessing=False, arm=args.arm,
        scope='Numerical execution diagnostic for the same actual parent and DPO weights. Compare DPO with its own unmerged parent; do not count a parent runtime gain as DPO benefit.'))
    started = time.monotonic()
    wrapper = Qwen3ASRModel.from_pretrained(p['base_model_dir'], dtype=torch.bfloat16,
        device_map='cuda:0', max_inference_batch_size=8, max_new_tokens=256)
    wrapper.model = PeftModel.from_pretrained(wrapper.model, str(adapter), is_trainable=False)
    wrapper.model.eval()
    assert wrapper.model.active_adapters and not any(x.requires_grad for x in wrapper.model.parameters())
    save(root / 'runtime.json', dict(load_seconds=time.monotonic() - started,
        active_adapters=wrapper.model.active_adapters,
        adapter_tensor_dtypes=dict(Counter(str(v.dtype) for k, v in wrapper.model.named_parameters() if 'lora_' in k)),
        base_tensor_dtypes=dict(Counter(str(v.dtype) for k, v in wrapper.model.named_parameters() if 'lora_' not in k)),
        merged=False))
    # Reuse the maintained evaluator without modifying its frozen source.
    evaluator.parse_args = lambda: SimpleNamespace(base_model_path=Path(p['base_model_dir']),
        adapter_path=adapter, split_jsonl=split, out_dir=root / 'evaluation',
        language='Cantonese', batch_size=8, max_new_tokens=256, critical_cer_threshold=.5)
    evaluator.load_model = lambda config: wrapper
    evaluator.main()
    assert all(sha256_file(Path(f)) == digest for f, digest in hashes.items())
    assert sha256_directory(adapter) == adapter_hash
    save(root / 'paired_merged.json', compare(merged_predictions, root / 'evaluation/predictions.jsonl'))
    save(root / 'results.json', dict(status='complete', actual_training=False,
        postprocessing=False, total_seconds=time.monotonic() - started,
        metrics=read(root / 'evaluation/summary.json')['metrics']))
    print(read(root / 'results.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('--arm', required=True, choices=['parent', 'dpo'])
    main(parser.parse_args())
