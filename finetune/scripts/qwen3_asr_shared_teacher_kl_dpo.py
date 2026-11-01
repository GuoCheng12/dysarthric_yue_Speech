#!/usr/bin/env python3
"""Persist a freshly computed, identical frozen teacher by linking its archive."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

import qwen3_asr_residual_kl_dpo as residual


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--verified_teacher_cache', type=Path, required=True)
    parser.add_argument('--verified_teacher_metadata', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index('--output_dir')+1])
    metadata = json.loads(args.verified_teacher_metadata.read_text())
    digest = residual.trainer.sha256(args.verified_teacher_cache)
    assert digest == metadata['cache_file_sha256']
    assert metadata['conditional_history'] == 'real training ground-truth prefixes, never development labels'
    reference_path = Path(remaining[remaining.index('--reference_adapter_path')+1])
    assert residual.trainer.sha256_directory(reference_path) == metadata['frozen_reference_sha256']
    source_hashes = {str(p.resolve()): residual.trainer.sha256(p) for p in
        [Path(__file__), args.verified_teacher_cache, args.verified_teacher_metadata]}
    original_save = torch.save
    reused = []

    def save_tensor_data(value, path, *positional, **kwargs):
        if Path(path).resolve() != (output / 'real_teacher_log_probs.pt').resolve():
            return original_save(value, path, *positional, **kwargs)
        assert value.keys() == metadata['audio'].keys()
        for key, tensor in value.items():
            expected = metadata['audio'][key]
            assert tensor.device.type == 'cpu' and tensor.dtype == torch.float32
            assert list(tensor.shape) == expected['shape']
            assert hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() == expected['tensor_sha256']
        os.link(args.verified_teacher_cache, path)
        assert residual.trainer.sha256(path) == digest
        reused.append(True)
        residual.trainer.write_json(output / 'teacher_storage_reuse.json', dict(
            source_sha256=source_hashes, source_cache=str(args.verified_teacher_cache.resolve()),
            actual_fresh_teacher_computation=True, identical_actual_tensor_values_verified=len(value),
            actual_training_uses_fresh_in_memory_values=True, reused_archive_sha256=digest,
            duplicate_archive_bytes_avoided=args.verified_teacher_cache.stat().st_size,
            unchanged_optimizer_checkpoint_serialization=True))

    sys.argv = [sys.argv[0]] + remaining
    torch.save = save_tensor_data
    try:
        residual.main()
    finally:
        torch.save = original_save
    assert reused == [True]
    assert all(residual.trainer.sha256(Path(p)) == h for p, h in source_hashes.items())


if __name__ == '__main__':
    main()
