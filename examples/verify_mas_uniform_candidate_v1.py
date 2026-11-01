#!/usr/bin/env python3
"""Reload the newly passing native residual and verify exact model and predictions."""
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mas_synthetic_utility_allocation_dpo_v1 as run


def main():
    from peft import PeftModel
    from transformers import GenerationConfig, set_seed
    from qwen3_asr_sft import patch_outer_forward
    start = time.time()
    out = run.ROOT / 'uniform'
    record = run.read(out / 'passing_weight_storage.json')
    trained = run.read(out / 'training_result.json')
    assert record['durably_archived'] and record['exact_native_state_roundtrip']
    checkpoint = Path(record['checkpoint'])
    assert run.sha256_directory(checkpoint) == record['sha256']
    recipe = record['load_recipe']
    run.torch.use_deterministic_algorithms(True)
    run.torch.backends.cudnn.deterministic = True
    run.torch.backends.cudnn.benchmark = False
    run.live.install(); set_seed(42)
    wrapper = run.probe.runtime.load_merged_parent(recipe['base_model'], recipe['merge_initial_R_only'],
                                                  'cuda:0', run.torch.bfloat16)
    patch_outer_forward(wrapper.model)
    model = PeftModel.from_pretrained(wrapper.model, str(checkpoint), is_trainable=True)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    assert run.parameter_digest(model, trainable_only=True) == trained['final_residual_sha256']
    assert run.frozen_base_digest(model) == trained['unchanged_base_sha256']
    predictions = run.live.gzip_rows(out / 'evaluation/predictions.jsonl.gz')[:8]
    actual = run.live.wrapper_for(model, wrapper.processor).transcribe(
        audio=[r['audio'] for r in predictions], language=['Cantonese'] * 8)
    assert [r.text for r in actual] == [r['prediction'] for r in predictions]
    result = dict(status='passed_native_checkpoint_reload', model_parameter_and_base_digests_exact=True,
        first8_predictions_exact=True, decoded_audio_ids=[r['sample_id'] if 'sample_id' in r else r.get('utt_id') for r in predictions],
        checkpoint_sha256=record['sha256'], residual_sha256=trained['final_residual_sha256'],
        original_full511_predictions=str(out / 'evaluation/predictions.jsonl.gz'),
        promoted=False, seconds=time.time() - start, source_sha256={str(Path(__file__)): run.sha256_file(Path(__file__))})
    run.write_immutable_json(out / 'reload_verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
