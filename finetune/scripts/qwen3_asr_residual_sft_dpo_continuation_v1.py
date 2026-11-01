#!/usr/bin/env python3
"""Continue a learned residual policy with the initial R-only DPO reference fixed."""

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
    initial = value('--reference_adapter_path')
    warm = Path(parent).resolve() != Path(initial).resolve()
    initial_hash = engine.sha256_directory(Path(initial))
    parent_hash = engine.sha256_directory(Path(parent))
    if warm:
        composition_before = __import__('json').loads((Path(parent) / 'residual_composition.json').read_text())
        assert composition_before['parent_adapter_sha256'] == initial_hash
        assert Path(composition_before['base_model']).resolve() == Path(value('--model_path')).resolve()
    output = Path(value('--output_dir'))
    hashes = {str(p.resolve()): engine.sha256(p) for p in [
        Path(__file__), Path(engine.__file__), Path(__file__).with_name('qwen3_asr_residual_dpo.py'),
        Path(__file__).parents[1] / 'official/qwen3_asr_sft.py',
        Path(__file__).parents[2] / 'src/therapist_harness/preference_token_mask.py']}
    composition = dict(parent_adapter=initial,
        parent_adapter_sha256=initial_hash,
        previous_policy_adapter=parent, previous_policy_adapter_sha256=parent_hash,
        learned_residual_continued=warm,
        base_model=value('--model_path'),
        optimizer_lineage='Continue the current residual weights with fresh AdamW; initial R-only is merged once into the frozen base.',
        starting_function='Merged initial R-only plus the current learned residual unmerged, matching residual candidate sampling.',
        reference='Frozen merged initial R-only with a frozen zero-output residual; distinct from TTS reference speech.',
        evaluation='Load merged initial R-only, then this residual unmerged.',
        objective='Mean real per-audio full transcript+EOS CE plus optional weighted synthetic edit+EOS DPO.',
        preference_token_mask=extra.preference_token_mask)
    loaded = {}; audit = {}
    original_write = engine.write_json
    original_torch_save = torch.save

    def loader(model_path, parent_path, device, dtype, trainable, disable_dropout):
        wrapper = load_merged_parent(model_path, initial, device, dtype)
        patch_outer_forward(wrapper.model)
        wrapper.model.generation_config = GenerationConfig.from_model_config(wrapper.model.config)
        wrapper.model.config.use_cache = False
        if trainable and warm:
            from peft import PeftModel
            model = PeftModel.from_pretrained(wrapper.model, parent_path, is_trainable=True)
        else:
            model = add_residual(wrapper.model, initial)
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
            dict(merged_base_tensor_sha256=digest, zero_residual=not (trainable and warm),
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
            assert engine.sha256_directory(Path(parent)) == parent_hash
            assert all(engine.sha256(Path(p)) == h for p, h in hashes.items())
            checkpoint = Path(record['checkpoint'])
            original_write(checkpoint / 'residual_composition.json', composition)
            original_write(output / 'residual_verified.json', dict(status='passed',
                frozen_base_unchanged=True, zero_initial_residual=not warm,
                learned_residual_continued=warm, fixed_initial_R_only_reference=True,
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
