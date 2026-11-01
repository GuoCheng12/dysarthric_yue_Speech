#!/usr/bin/env python3
"""Warm DPO with initial R-only reference and separately frozen current-parent KL."""

import argparse
import gc
from pathlib import Path
import sys

import torch

import qwen3_asr_edit_kl_dpo_continuation_v1 as warm
import qwen3_asr_kl_dpo_storage_v2 as engine


def install_current_teacher(args, expected_base):
    from peft import PeftModel
    from transformers import GenerationConfig
    from qwen3_asr_sft import patch_outer_forward

    parent = Path(args.policy_adapter_path).resolve(); initial = Path(args.reference_adapter_path).resolve()
    assert parent != initial
    parent_hash = engine.sha256_directory(parent); initial_hash = engine.sha256_directory(initial)
    output = Path(args.output_dir)
    sources = [Path(__file__), Path(warm.__file__), Path(engine.__file__), parent / 'residual_composition.json']
    hashes = {str(p): engine.sha256(p) for p in sources}
    original_distribution = engine.transcript_distributions; original_write = engine.write_json
    state = dict(phase='teacher_cache', model=None, reference_object=None, before=None)

    def adapter_digests(model):
        return {n: engine.parameter_digest(m) for n, m in model.named_modules() if n.endswith(('lora_A', 'lora_B'))}

    def distribution(model, batch):
        if state['phase'] != 'teacher_cache':
            return original_distribution(model, batch)
        assert not any(p.requires_grad for p in model.parameters())
        if state['model'] is None:
            wrapper = warm.load_merged_parent(args.model_path, str(initial), model.device, model.dtype)
            patch_outer_forward(wrapper.model)
            wrapper.model.generation_config = GenerationConfig.from_model_config(wrapper.model.config)
            wrapper.model.config.use_cache = False
            teacher = PeftModel.from_pretrained(wrapper.model, str(parent), is_trainable=False)
            for module in teacher.modules():
                if isinstance(module, torch.nn.Dropout): module.p = 0.0
            teacher.requires_grad_(False); teacher.eval()
            assert warm.frozen_base_digest(teacher) == expected_base
            state.update(model=teacher, reference_object=id(model), before=adapter_digests(teacher))
            original_write(output / 'current_KL_teacher_runtime.json', dict(status='loaded',
                checkpoint=str(parent), checkpoint_sha256=parent_hash, initial_R_only_sha256=initial_hash,
                current_residual_merged=False, merged_base_tensor_sha256=expected_base,
                trainable_parameters=sum(p.numel() for p in teacher.parameters() if p.requires_grad),
                initial_DPO_reference_is_separate_object=True, adapter_tensor_hashes=state['before']))
        assert id(model) == state['reference_object']
        return original_distribution(state['model'], batch)

    def write_record(path, value):
        name = Path(path).name
        reference_description = 'DPO: frozen initial R-only. Real KL: a separate frozen current round parent. Neither is TTS reference speech.'
        if name in {'run_protocol.json', 'trainer_state.json', 'residual_execution.json', 'residual_composition.json'}:
            value = dict(value, objective='real symmetric KL to frozen CURRENT round parent + synthetic edit+EOS DPO against frozen INITIAL R-only',
                real_KL_teacher_checkpoint=str(parent), real_KL_teacher_sha256=parent_hash,
                initial_R_only_DPO_reference_sha256=initial_hash,
                retention_anchor_intervention='Only the real-KL teacher changes; DPO reference remains initial R-only. Full real GT-prefix transcript+EOS distributions are cached before any update.',
                initial_loss='Current-parent KL starts at zero; warm DPO against initialR-only need not equal log(2).')
            if name == 'run_protocol.json':
                value['source_sha256'] = {**value['source_sha256'], **hashes}
                value['reference_identity'] = reference_description
                value['residual_execution'] = dict(value['residual_execution'], reference=reference_description,
                    real_KL_teacher_sha256=parent_hash, initial_loss=value['initial_loss'])
            if name in {'residual_execution.json', 'residual_composition.json'}:
                value['reference'] = reference_description
        if name == 'residual_verified.json':
            value = dict(value)
            value.pop('fixed_initial_R_only_DPO_and_KL_reference')
            value.update(fixed_initial_R_only_DPO_reference=True, fixed_current_parent_KL_reference=True)
        if name == 'training_result.json':
            value = dict(value, real_KL_teacher_sha256=parent_hash, initial_R_only_DPO_reference_sha256=initial_hash)
        if name == 'real_teacher_cache.json':
            teacher = state['model']; assert teacher is not None
            assert adapter_digests(teacher) == state['before']
            assert warm.frozen_base_digest(teacher) == expected_base
            assert engine.sha256_directory(parent) == parent_hash
            value = dict(value, frozen_reference_sha256=parent_hash,
                teacher_kind='frozen_current_round_parent', initial_R_only_DPO_reference_sha256=initial_hash,
                teacher_parameters_unchanged=True, teacher_frozen_before_first_policy_update=True)
            original_write(output / 'current_KL_teacher_verified.json', dict(status='passed',
                current_teacher_frozen=True, current_teacher_parameters_unchanged=True,
                initial_DPO_reference_object_unchanged_by_KL=True, actual_teacher_audio=value['actual_fresh_teacher_audio'],
                current_teacher_checkpoint_sha256=parent_hash, initial_DPO_reference_sha256=initial_hash))
            state.update(model=None, phase='policy'); del teacher; gc.collect(); torch.cuda.empty_cache()
        original_write(path, value)

    engine.transcript_distributions = distribution; engine.write_json = write_record
    return hashes


if __name__ == '__main__':
    extra = argparse.ArgumentParser(add_help=False); extra.add_argument('--expected_frozen_base_sha256', required=True)
    known, remaining = extra.parse_known_args(); sys.argv = [sys.argv[0]] + remaining
    parser = argparse.ArgumentParser(add_help=False)
    for name in ['model_path', 'policy_adapter_path', 'reference_adapter_path', 'output_dir']:
        parser.add_argument('--' + name, required=True)
    args, _ = parser.parse_known_args()
    hashes = install_current_teacher(args, known.expected_frozen_base_sha256)
    warm.residual.main = lambda: warm.continue_training(known.expected_frozen_base_sha256)
    warm.masked.main()
    assert all(engine.sha256(Path(f)) == h for f, h in hashes.items())
