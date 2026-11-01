#!/usr/bin/env python3
"""Reuse the KL/DPO objective with the evaluated merged R-only starting function."""

import argparse
from pathlib import Path

import torch

import qwen3_asr_kl_dpo_probe as trainer
from qwen3_asr_residual_dpo import load_merged_parent, add_residual, frozen_base_digest


def main():
    from qwen3_asr_sft import patch_outer_forward
    from transformers import GenerationConfig

    parser = argparse.ArgumentParser(add_help=False)
    for name in ['model_path', 'policy_adapter_path', 'reference_adapter_path', 'output_dir']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max_steps', type=int, default=12)
    args, _ = parser.parse_known_args()
    assert Path(args.policy_adapter_path).resolve() == Path(args.reference_adapter_path).resolve()
    output = Path(args.output_dir)
    paths = [Path(__file__), Path(trainer.__file__), Path(__file__).with_name('qwen3_asr_residual_dpo.py')]
    hashes = {str(p): trainer.sha256(p) for p in paths}
    execution = dict(source_sha256=hashes, parent_adapter=args.policy_adapter_path,
        parent_adapter_sha256=trainer.sha256_directory(Path(args.policy_adapter_path)),
        base_model=args.model_path, objective='synthetic audio DPO plus real-training symmetric KL',
        starting_function='Original R-only merged into bf16 base, plus new zero-output residual LoRA',
        optimizer_lineage='New residual LoRA and fresh AdamW; merged R-only base is frozen',
        reference='Fixed original merged R-only, with frozen zero-output residual; same teacher for DPO and real KL',
        evaluation='Merge original R-only once, then load this residual without merging it',
        real_history='Original real-training ground-truth prefixes; unchanged existing KL objective, no real CE or development training')
    loaded = {}
    original_write = trainer.write_json

    def write_record(path, value):
        if Path(path).name == 'run_protocol.json':
            value = dict(value, residual_execution=execution, reference_identity=execution['reference'])
            original_write(output / 'residual_execution.json', execution)
        original_write(path, value)

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
        if trainable:
            assert digest == loaded['reference_base_hash']
            loaded.update(policy=model, policy_base_hash=digest)
        else:
            loaded['reference_base_hash'] = digest
        trainer.write_json(output / ('policy_initial_runtime.json' if trainable else 'reference_runtime.json'),
            dict(merged_base_tensor_sha256=digest, zero_residual=True,
                 trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad)))
        return model, wrapper.processor

    trainer.write_json = write_record
    trainer.load_adapter = loader
    trainer.main()
    assert frozen_base_digest(loaded['policy']) == loaded['policy_base_hash']
    assert all(trainer.sha256(Path(p)) == digest for p, digest in hashes.items())
    trainer.write_json(output / f'checkpoint-{args.max_steps}/residual_composition.json', execution)
    trainer.write_json(output / 'residual_verified.json', dict(status='passed',
        merged_parent_base_identical_across_policy_reference=True, merged_parent_base_unchanged=True,
        zero_initial_residual=True, checkpoint_requires_residual_loader=True))


if __name__ == '__main__':
    main()
