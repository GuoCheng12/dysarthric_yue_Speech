#!/usr/bin/env python3
"""Evaluate a residual DPO checkpoint on its bound merged initial R-only base."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

from mas_selection_dpo_probe import REPO, read, save
from therapist_harness.store import sha256_file, sha256_directory

sys.path.insert(0, str(REPO / 'finetune/scripts'))
import evaluate_asr_checkpoint as evaluator
from qwen3_asr_residual_dpo import load_merged_parent, add_residual


def main():
    import torch
    from transformers import set_seed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--initial-adapter', type=Path, required=True)
    parser.add_argument('--residual-adapter', type=Path)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    initial_hash = sha256_directory(args.initial_adapter)
    files = [Path(__file__), Path(evaluator.__file__),
             REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py', args.split]
    hashes = {str(f): sha256_file(f) for f in files}
    residual_hash = sha256_directory(args.residual_adapter) if args.residual_adapter else None
    if args.residual_adapter:
        composition = read(args.residual_adapter / 'residual_composition.json')
        assert composition['parent_adapter_sha256'] == initial_hash
        assert composition['base_model'] == args.base_model
    save(args.out / 'runtime.json', dict(initial_adapter=str(args.initial_adapter),
        initial_adapter_sha256=initial_hash, base_model=args.base_model,
        residual_adapter=str(args.residual_adapter) if args.residual_adapter else None,
        residual_adapter_sha256=residual_hash, zero_initial_residual=args.residual_adapter is None,
        original_R_only_merged=True, residual_merged=False, postprocessing=False, source_sha256=hashes))
    set_seed(42)
    wrapper = load_merged_parent(args.base_model, str(args.initial_adapter), 'cuda:0', torch.bfloat16)
    wrapper.model = add_residual(wrapper.model, str(args.initial_adapter), args.residual_adapter)
    wrapper.model.requires_grad_(False)
    wrapper.model.eval()
    evaluator.load_model = lambda config: wrapper
    evaluator.parse_args = lambda: SimpleNamespace(base_model_path=Path(args.base_model),
        adapter_path=args.residual_adapter, split_jsonl=args.split, out_dir=args.out / 'evaluation',
        batch_size=8, max_new_tokens=256, language='Cantonese', critical_cer_threshold=.5)
    evaluator.main()
    assert all(sha256_file(Path(f)) == digest for f, digest in hashes.items())
    assert sha256_directory(args.initial_adapter) == initial_hash
    if args.residual_adapter:
        assert sha256_directory(args.residual_adapter) == residual_hash
    save(args.out / 'verified.json', dict(status='passed', model_files_unchanged=True))


if __name__ == '__main__':
    main()
