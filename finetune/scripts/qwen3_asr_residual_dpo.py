#!/usr/bin/env python3
"""DPO a fresh residual LoRA on the frozen, merged R-only inference model."""

import hashlib
from pathlib import Path

import torch

import qwen3_asr_audio_dpo as trainer


def load_merged_parent(model_path, parent_path, device, dtype, batch_size=8, max_tokens=256):
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel

    wrapper = Qwen3ASRModel.from_pretrained(model_path, dtype=dtype, device_map=str(device),
        max_inference_batch_size=batch_size, max_new_tokens=max_tokens)
    wrapper.model = PeftModel.from_pretrained(wrapper.model, parent_path,
        is_trainable=False).merge_and_unload()
    wrapper.model.requires_grad_(False)
    wrapper.model.eval()
    return wrapper


def add_residual(model, parent_path, residual_path=None):
    from peft import LoraConfig, PeftModel, get_peft_model

    if residual_path is not None:
        return PeftModel.from_pretrained(model, str(residual_path), is_trainable=False)
    config = LoraConfig.from_pretrained(parent_path)
    assert config.bias == 'none' and not config.modules_to_save
    config.inference_mode = False
    config.init_lora_weights = True
    model = get_peft_model(model, config)
    tensors = [v for k, v in model.named_parameters() if '.lora_B.' in k]
    assert tensors and all(torch.count_nonzero(v).item() == 0 for v in tensors)
    return model


def frozen_base_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if '.lora_' in name:
            continue
        assert not value.requires_grad
        digest.update(name.encode())
        digest.update(value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    from qwen3_asr_sft import patch_outer_forward
    from transformers import GenerationConfig

    args = trainer.parse_args()
    assert args.objective == 'dpo'
    assert Path(args.policy_adapter_path).resolve() == Path(args.reference_adapter_path).resolve()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path(trainer.__file__), trainer.REPO_ROOT / 'src/therapist_harness/dpo_training.py']
    hashes = {str(p): trainer.sha256(p) for p in paths}
    execution = dict(source_sha256=hashes, parent_adapter=args.policy_adapter_path,
        parent_adapter_sha256=trainer.sha256_directory(Path(args.policy_adapter_path)),
        base_model=args.model_path, objective='unchanged standard audio DPO',
        starting_function='bf16 base plus merged frozen initial R-only, exactly as candidate sampling and original evaluator',
        optimizer_lineage='New zero-output residual LoRA and new AdamW. Original R-only LoRA is already merged into frozen base; its A/B parameters are not continued.',
        reference='The same merged initial R-only plus a frozen zero-output residual. All reference parameters are frozen.',
        evaluation='Merge original R-only once, then load the new residual without merging it. This residual checkpoint cannot be evaluated directly on bare Qwen base.',
        interpretation='Changes parameterization as well as aligns runtime; a gain would not alone isolate the cause of the old merged/unmerged discrepancy.')
    trainer.write_json(output / 'residual_execution.json', execution)
    loaded = {}

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
        base_hash = frozen_base_digest(model)
        if trainable:
            assert base_hash == loaded['reference_base_hash']
            loaded.update(policy=model, policy_base_hash=base_hash)
        else:
            loaded['reference_base_hash'] = base_hash
        trainer.write_json(output / ('policy_initial_runtime.json' if trainable else 'reference_runtime.json'),
            dict(merged_base_tensor_sha256=base_hash, zero_residual=True,
                 trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad)))
        return model, wrapper.processor

    original_write = trainer.write_json

    def write_record(path, value):
        if Path(path).name == 'run_protocol.json':
            value = dict(value, optimizer_lineage=execution['optimizer_lineage'],
                reference_identity=execution['reference'], residual_execution=execution)
        original_write(path, value)

    trainer.write_json = write_record
    trainer.load_adapter = loader
    trainer.main()
    assert frozen_base_digest(loaded['policy']) == loaded['policy_base_hash']
    assert all(trainer.sha256(Path(p)) == digest for p, digest in hashes.items())
    checkpoint = output / f'checkpoint-{args.max_steps}'
    trainer.write_json(checkpoint / 'residual_composition.json', execution)
    trainer.write_json(output / 'residual_verified.json', dict(status='passed',
        merged_parent_base_identical_across_policy_reference=True, merged_parent_base_unchanged=True,
        zero_initial_residual=True, checkpoint_requires_residual_loader=True))


if __name__ == '__main__':
    main()
