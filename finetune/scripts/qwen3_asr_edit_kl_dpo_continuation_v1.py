#!/usr/bin/env python3
"""Continue an unmerged residual while retaining the initial R-only KL/DPO teacher."""

import argparse
import json
from pathlib import Path
import sys

import torch

import qwen3_asr_edit_masked_kl_dpo as masked
import qwen3_asr_residual_kl_dpo as residual
from qwen3_asr_residual_dpo import load_merged_parent, add_residual, frozen_base_digest


def continue_training(expected_base):
    from peft import PeftModel
    from qwen3_asr_sft import patch_outer_forward
    from transformers import GenerationConfig

    engine = residual.trainer
    parser = argparse.ArgumentParser(add_help=False)
    for name in ['model_path','policy_adapter_path','reference_adapter_path','output_dir']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max_steps', type=int, required=True)
    args, _ = parser.parse_known_args()
    initial = Path(args.reference_adapter_path).resolve()
    parent = Path(args.policy_adapter_path).resolve()
    assert parent != initial, 'This entry explicitly continues a learned residual.'
    initial_hash = engine.sha256_directory(initial); parent_hash = engine.sha256_directory(parent)
    composition = json.loads((parent / 'residual_composition.json').read_text())
    assert composition['parent_adapter_sha256'] == initial_hash
    assert Path(composition['base_model']).resolve() == Path(args.model_path).resolve()
    previous_result = parent.parent / 'training_result.json'
    previous = json.loads(previous_result.read_text())
    assert Path(previous['checkpoint']).resolve() == parent and previous['policy_changed']
    expected_parameters = previous['policy_trainable_sha256_after']
    output = Path(args.output_dir)
    paths = [Path(__file__), Path(masked.__file__), Path(engine.__file__), previous_result,
        parent / 'residual_composition.json', Path(__file__).with_name('qwen3_asr_residual_dpo.py')]
    hashes = {str(p):engine.sha256(p) for p in paths}
    execution = dict(source_sha256=hashes, parent_adapter=str(initial), parent_adapter_sha256=initial_hash,
        previous_policy_adapter=str(parent), previous_policy_adapter_sha256=parent_hash,
        base_model=args.model_path, learned_residual_continued=True,
        expected_starting_trainable_sha256=expected_parameters,
        starting_function='Initial R-only merged once into frozen bf16 base plus the learned current residual, kept unmerged.',
        optimizer_lineage='Continue current residual A/B weights with fresh AdamW; previous optimizer state is not resumed.',
        reference='Frozen merged INITIAL R-only with frozen zero-output residual for both DPO and real KL. Not the current learned policy or TTS reference speech.',
        evaluation='Merge initial R-only once, then load this residual unmerged.',
        real_history='Original real-training ground-truth prefixes, full transcript+EOS symmetric KL; no real CE.',
        initial_loss='Warm policy need not have zero real KL or log(2) DPO; report the actual initial losses.')
    loaded = {}; original_write = engine.write_json

    def write_record(path, value):
        name = Path(path).name
        if name == 'run_protocol.json':
            value = dict(value, residual_execution=execution, reference_identity=execution['reference'],
                source_sha256={**value['source_sha256'], **hashes})
            original_write(output / 'residual_execution.json', execution)
        if name == 'training_result.json':
            assert value['policy_trainable_sha256_before'] == expected_parameters
            assert frozen_base_digest(loaded['policy']) == expected_base
            assert engine.sha256_directory(initial) == initial_hash
            assert engine.sha256_directory(parent) == parent_hash
            original_write(Path(value['checkpoint']) / 'residual_composition.json', execution)
            original_write(output / 'residual_verified.json', dict(status='passed',
                learned_residual_continued=True, starting_parameters_equal_previous_endpoint=True,
                fixed_initial_R_only_DPO_and_KL_reference=True, frozen_base_unchanged=True,
                previous_policy_files_unchanged=True, zero_initial_residual=False,
                checkpoint_requires_residual_loader=True))
        original_write(path, value)

    def loader(model_path, adapter_path, device, dtype, trainable, disable_dropout):
        assert Path(adapter_path).resolve() == (parent if trainable else initial)
        wrapper = load_merged_parent(model_path, str(initial), device, dtype)
        patch_outer_forward(wrapper.model)
        wrapper.model.generation_config = GenerationConfig.from_model_config(wrapper.model.config)
        wrapper.model.config.use_cache = False
        model = (PeftModel.from_pretrained(wrapper.model, str(parent), is_trainable=True)
                 if trainable else add_residual(wrapper.model, str(initial)))
        if disable_dropout:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout): module.p = 0.0
        if trainable:
            model.train(); loaded['policy'] = model
        else:
            model.requires_grad_(False); model.eval()
        digest = frozen_base_digest(model); assert digest == expected_base
        original_write(output / ('policy_initial_runtime.json' if trainable else 'reference_runtime.json'),
            dict(merged_base_tensor_sha256=digest, zero_residual=not trainable,
                learned_residual_loaded=trainable,
                trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad)))
        return model, wrapper.processor

    engine.write_json = write_record; engine.load_adapter = loader
    engine.main()
    assert all(engine.sha256(Path(p)) == h for p,h in hashes.items())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--expected_frozen_base_sha256', required=True)
    extra, remaining = parser.parse_known_args(); sys.argv = [sys.argv[0]] + remaining
    residual.main = lambda: continue_training(extra.expected_frozen_base_sha256)
    masked.main()
