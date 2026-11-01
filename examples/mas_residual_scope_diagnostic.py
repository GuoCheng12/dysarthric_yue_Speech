#!/usr/bin/env python3
"""Localize a completed DPO update; these are ablations, not trained candidates."""

import argparse
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

from mas_selection_dpo_probe import REPO, read, save
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file, sha256_directory

sys.path.insert(0, str(REPO / 'finetune/scripts'))
import evaluate_asr_checkpoint as evaluator
from qwen3_asr_residual_dpo import load_merged_parent, add_residual, frozen_base_digest


def residual_hash(model):
    import torch
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if '.lora_' not in name: continue
        digest.update(name.encode())
        digest.update(value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def main(base):
    import torch
    from transformers import set_seed

    root = base / 'adaptive96_residual_scope_diagnostic_v1'
    source = base / 'adaptive96_matched_DPO_recovery_v2/MAS_reference'
    source_result = read(source / 'results.json')
    cfg = read(source.parent / 'protocol.json'); p = cfg['source_protocol']
    checkpoint = Path(source_result['checkpoint']); initial = Path(p['initial_adapter']['path'])
    assert sha256_directory(checkpoint) == source_result['checkpoint_sha256']
    assert sha256_directory(initial) == p['initial_adapter']['sha256']
    composition = read(checkpoint / 'residual_composition.json')
    assert composition['parent_adapter_sha256'] == p['initial_adapter']['sha256']
    root.mkdir(exist_ok=False)
    sources = [Path(__file__), Path(evaluator.__file__), REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py',
               base / 'adaptive96_residual_scope_plan_v1.json', source / 'results.json',
               source.parent / 'protocol.json', Path(p['feedback_split']['path'])]
    hashes = {str(f):sha256_file(f) for f in sources}
    save(root / 'protocol.json', dict(source_checkpoint=str(checkpoint), source_checkpoint_sha256=source_result['checkpoint_sha256'],
        initial_adapter=str(initial), initial_adapter_sha256=p['initial_adapter']['sha256'],
        base_model=p['base_model_dir'], source_sha256=hashes, actual_new_training=False,
        interpretation='Post-training residual-component ablations of the completed fresh MAS DPO checkpoint. No parameter update, new DPO candidate, checkpoint promotion, or inference text postprocessing. Zero only the excluded LoRA B matrices in memory; retain the merged R-only base.',
        scopes=['audio','language']))
    set_seed(42)
    wrapper = load_merged_parent(p['base_model_dir'], str(initial), 'cuda:0', torch.bfloat16)
    wrapper.model = add_residual(wrapper.model, str(initial), checkpoint)
    model = wrapper.model; model.requires_grad_(False); model.eval()
    parameters = dict(model.named_parameters())
    full_digest = residual_hash(model); base_digest = frozen_base_digest(model)
    b_weights = {name:value.detach().cpu().clone() for name,value in parameters.items() if '.lora_B.' in name}
    assert len(b_weights) == 268
    results = {}
    for scope in ['audio','language']:
        out = root / scope; out.mkdir()
        kept = []; suppressed = []
        with torch.no_grad():
            for name, value in b_weights.items():
                is_audio = '.thinker.audio_tower.' in name
                retain = is_audio if scope == 'audio' else not is_audio
                if retain:
                    parameters[name].copy_(value); kept.append(name)
                    assert torch.equal(parameters[name].detach().cpu(),value)
                else:
                    parameters[name].zero_(); suppressed.append(name)
                    assert torch.count_nonzero(parameters[name]).item() == 0
        assert kept and suppressed and len(kept) + len(suppressed) == len(b_weights)
        scoped_digest = residual_hash(model)
        save(out / 'runtime.json', dict(scope=scope, retained_B_names=kept, suppressed_B_names=suppressed,
            original_residual_sha256=full_digest, in_memory_scoped_residual_sha256=scoped_digest,
            merged_R_only_base_sha256=base_digest, checkpoint_file_sha256=source_result['checkpoint_sha256'],
            actual_training=False, adapter_files_modified=False, audio_and_language_boundaries='Parameter names containing .thinker.audio_tower. versus the remaining actual residual parameters.'))
        evaluator.load_model = lambda config: wrapper
        evaluator.parse_args = lambda: SimpleNamespace(base_model_path=Path(p['base_model_dir']),
            adapter_path=checkpoint, split_jsonl=Path(p['feedback_split']['path']), out_dir=out / 'evaluation',
            batch_size=8, max_new_tokens=256, language='Cantonese', critical_cer_threshold=.5)
        evaluator.main()
        assert residual_hash(model) == scoped_digest
        summary = read(out / 'evaluation/summary.json')
        save(out / 'paired_parent.json', compare(base / 'parent_eval_check/predictions.jsonl', out / 'evaluation/predictions.jsonl'))
        save(out / 'paired_full_DPO.json', compare(source / 'evaluation_run/evaluation/predictions.jsonl', out / 'evaluation/predictions.jsonl'))
        results[scope] = dict(metrics=summary['metrics'], actual_new_training=False, eligible_for_parent=False)
        save(out / 'results.json', results[scope])
    with torch.no_grad():
        for name,value in b_weights.items(): parameters[name].copy_(value)
    assert residual_hash(model) == full_digest and frozen_base_digest(model) == base_digest
    assert sha256_directory(checkpoint) == source_result['checkpoint_sha256']
    assert sha256_directory(initial) == p['initial_adapter']['sha256']
    assert all(sha256_file(Path(f)) == h for f,h in hashes.items())
    save(root / 'verified.json', dict(status='passed', original_residual_restored_exactly=True,
        merged_R_only_base_parameters_unchanged=True, original_checkpoint_files_unchanged=True,
        actual_optimizer_steps=0, actual_training=False))
    save(root / 'results.json', dict(status='complete', scopes=results, actual_new_training=False,
        parent_promoted=False, scope='Component-localization diagnosis only; a new training-scope experiment is required before calling any ablation improvement a DPO training gain.'))
    print({name:x['metrics']['overall']['pooled_cer'] for name,x in results.items()},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('base',type=Path)
    args = parser.parse_args(); main(args.base.resolve())
