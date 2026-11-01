#!/usr/bin/env python3
"""Run the existing audio DPO trainer with an explicitly recorded LoRA scope."""

import argparse
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file

import qwen3_asr_audio_dpo as trainer


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--train_component', required=True, choices=['all', 'decoder'])
    scope, remaining = parser.parse_known_args()
    args = trainer.parse_args(remaining)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {str(p): trainer.sha256(p) for p in [Path(__file__), Path(trainer.__file__),
        trainer.REPO_ROOT / 'src/therapist_harness/dpo_training.py']}
    trainer.write_json(output / 'scope_execution.json', dict(train_component=scope.train_component,
        source_sha256=hashes, objective_changed=False,
        intervention='Before optimizer construction, set requires_grad=False for audio_tower LoRA when decoder scope is selected. Preserve its existing R-only adapter values; do not disable those adapters.'))
    original_loader = trainer.load_adapter

    def load_adapter(*values):
        model, processor = original_loader(*values)
        trainable = values[4]
        if trainable:
            names = {}
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    assert '.lora_' in name
                    if scope.train_component == 'decoder' and '.audio_tower.' in name:
                        parameter.requires_grad_(False)
                    if parameter.requires_grad:
                        assert '.thinker.model.' in name or '.thinker.audio_tower.' in name
                    names[name] = dict(trainable=parameter.requires_grad, numel=parameter.numel())
            assert names and any(x['trainable'] for x in names.values())
            if scope.train_component == 'decoder':
                assert all(not x['trainable'] for n, x in names.items() if '.audio_tower.' in n)
            trainer.write_json(output / 'actual_parameter_scope.json', names)
        return model, processor

    trainer.load_adapter = load_adapter
    trainer.main(remaining)
    before = load_file(str(Path(args.policy_adapter_path) / 'adapter_model.safetensors'))
    after = load_file(str(output / f'checkpoint-{args.max_steps}/adapter_model.safetensors'))
    assert before.keys() == after.keys()
    counts, changed = Counter(), Counter()
    for name in before:
        group = 'audio' if '.thinker.audio_tower.' in name else 'decoder'
        assert group == 'audio' or '.thinker.model.' in name
        counts[group] += 1
        changed[group] += int(not torch.equal(before[name], after[name]))
    assert changed['decoder'] > 0
    assert changed['audio'] == 0 if scope.train_component == 'decoder' else changed['audio'] > 0
    assert all(trainer.sha256(Path(p)) == digest for p, digest in hashes.items())
    trainer.write_json(output / 'scope_verified.json', dict(status='passed', train_component=scope.train_component,
        adapter_tensor_counts=dict(counts), changed_adapter_tensors=dict(changed),
        audio_values_exactly_match_parent=changed['audio'] == 0, decoder_updated=changed['decoder'] > 0))


if __name__ == '__main__':
    main()
