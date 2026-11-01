#!/usr/bin/env python3
"""Real transcript CE and optional synthetic edit+EOS DPO on merged R-only."""

import argparse
from pathlib import Path
import sys

import torch

import qwen3_asr_sft_dpo_probe as engine
from qwen3_asr_residual_dpo import load_merged_parent, add_residual, frozen_base_digest
from therapist_harness.preference_token_mask import collator_type


def main():
    from qwen3_asr_sft import patch_outer_forward
    from transformers import GenerationConfig

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--preference_token_mask', choices=['full', 'edit_eos'], required=True)
    parser.add_argument('--expected_frozen_base_sha256', required=True)
    extra, remaining = parser.parse_known_args()
    value = lambda flag: remaining[remaining.index(flag) + 1]
    parent = value('--policy_adapter_path')
    assert Path(parent).resolve() == Path(value('--reference_adapter_path')).resolve()
    output = Path(value('--output_dir'))
    hashes = {str(p.resolve()): engine.sha256(p) for p in [
        Path(__file__), Path(engine.__file__), Path(__file__).with_name('qwen3_asr_residual_dpo.py'),
        Path(__file__).parents[1] / 'official/qwen3_asr_sft.py',
        Path(__file__).parents[2] / 'src/therapist_harness/preference_token_mask.py']}
    composition = dict(parent_adapter=parent,
        parent_adapter_sha256=engine.sha256_directory(Path(parent)),
        base_model=value('--model_path'),
        optimizer_lineage='Fresh zero-output residual LoRA and AdamW; initial R-only is merged once into the frozen base.',
        starting_function='Merged initial R-only, matching candidate sampling and canonical parent evaluation.',
        reference='Frozen merged initial R-only with a frozen zero-output residual; distinct from TTS reference speech.',
        evaluation='Load merged initial R-only, then this residual unmerged.',
        objective='Mean real per-audio full transcript+EOS CE plus optional weighted synthetic edit+EOS DPO.',
        preference_token_mask=extra.preference_token_mask)
    loaded = {}; audit = {}
    original_write = engine.write_json
    original_torch_save = torch.save

    def loader(model_path, parent_path, device, dtype, trainable, disable_dropout):
        wrapper = load_merged_parent(model_path, parent_path, device, dtype)
        patch_outer_forward(wrapper.model)
        wrapper.model.generation_config = GenerationConfig.from_model_config(wrapper.model.config)
        wrapper.model.config.use_cache = False
        model = add_residual(wrapper.model, parent_path)
        if disable_dropout:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
        if trainable:
            model.train()
        else:
            model.requires_grad_(False)
            model.eval()
        digest = frozen_base_digest(model)
        assert digest == extra.expected_frozen_base_sha256
        if trainable:
            loaded.update(policy=model, policy_base_sha256=digest)
        else:
            loaded['reference_base_sha256'] = digest
        original_write(output / ('policy_initial_runtime.json' if trainable else 'reference_runtime.json'),
            dict(merged_base_tensor_sha256=digest, zero_residual=True,
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        return model, wrapper.processor

    def write_record(path, record):
        name = Path(path).name
        if name == 'run_protocol.json':
            record = dict(record, residual_composition=composition,
                source_sha256={**record['source_sha256'], **hashes},
                score_mask=dict(real='Full transcript + EOS, per-audio token mean CE',
                    synthetic=extra.preference_token_mask + ', summed scores; no length normalization'),
                reference_identity=composition['reference'],
                optimizer_storage='Optimizer state deliberately omitted; no resume from this checkpoint.')
        if name == 'reference_log_probs.json':
            original_write(output / 'preference_token_masks.json', dict(mode=extra.preference_token_mask, pairs=audit))
            record = dict(record, mask_manifest_sha256=engine.sha256(output / 'preference_token_masks.json'))
        if name == 'training_result.json':
            assert frozen_base_digest(loaded['policy']) == loaded['policy_base_sha256']
            assert all(engine.sha256(Path(p)) == h for p, h in hashes.items())
            checkpoint = Path(record['checkpoint'])
            original_write(checkpoint / 'residual_composition.json', composition)
            original_write(output / 'residual_verified.json', dict(status='passed',
                frozen_base_unchanged=True, zero_initial_residual=True,
                reference_base_matches_canonical=loaded.get('reference_base_sha256') == extra.expected_frozen_base_sha256
                    if 'reference_base_sha256' in loaded else None,
                policy_base_matches_canonical=True, checkpoint_requires_residual_loader=True))
            record = dict(record, frozen_base_unchanged=True, optimizer_state_persisted=False)
        original_write(path, record)

    def save_without_optimizer(obj, path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).name == 'optimizer.pt':
            original_write(Path(path).with_name('optimizer_omission.json'), dict(
                optimizer_state_persisted=False, reason='Declared fresh-run experiment; preserve adapter and training receipts only.'))
            return
        return original_torch_save(obj, path, *args, **kwargs)

    engine.load_adapter = loader
    engine.AudioPreferenceCollator = collator_type(extra.preference_token_mask, audit)
    engine.write_json = write_record
    torch.save = save_without_optimizer
    sys.argv = [sys.argv[0]] + remaining
    try:
        engine.main()
    finally:
        torch.save = original_torch_save


if __name__ == '__main__':
    main()
