#!/usr/bin/env python3
"""Matched new-recipe DPO with fresh frozen-initial scores and exact real replay."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'examples'), '/tmp']
import mas_real_even_kl_gpu_smoke_v1 as runtime
import mas_real_headroom_selection_dpo_v1 as live
from mas_real_even_kl_gpu_smoke_v1 import (
    B, read, rows, load_model, RealSyntheticEditCollator, torch, sha256_file,
    sha256_directory, write_immutable_json, preference_key, move_batch, batch_log_probs,
    dpo_losses, transcript_distributions, symmetric_kl, parameter_digest, frozen_base_digest,
)

DATA = B / 'contextual_recipe_DPO24_v1'
ROOT = DATA / 'training'
PLAN = DATA / 'training_plan'
OLD = Path('/tmp/mas_deterministic_matched_DPO_v1')


def main():
    from transformers import set_seed
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'fixed', 'MAS'], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    live.install()
    start = time.time()
    plan = read(PLAN / 'plan.json')
    mode = 'MAS' if args.mode == 'smoke' else args.mode
    dpo = read(PLAN / 'expected_DPO_exposures.json')[mode]
    kl = read(PLAN / 'expected_common_KL_exposures.json')
    windows = [[e for e in kl if e['step'] == step] + dpo[step - 1] for step in range(1, 5)]
    used = {e['preference_key'] for w in dpo for e in w}
    all_prefs = rows(plan['data_paths']['real_preferences']) + rows(plan['data_paths']['synthetic_preferences'][mode])
    preferences = [r for r in all_prefs if preference_key(r) in used]
    assert len(preferences) == len(used)
    real = rows(plan['data_paths']['real_KL'])
    files = [Path(__file__), Path(live.__file__), Path(runtime.__file__),
        Path('/tmp/mas_gradient_deterministic_v2/deterministic_qwen_positions_v1.py'),
        REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py',
        REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py',
        REPO / 'finetune/scripts/qwen3_asr_kl_dpo_storage_v2.py',
        REPO / 'src/therapist_harness/dpo_training.py',
        REPO / 'src/therapist_harness/preference_token_mask.py',
        PLAN / 'plan.json', PLAN / 'expected_DPO_exposures.json', PLAN / 'expected_common_KL_exposures.json',
        OLD / 'R_MAS_DPO_KL/reference_log_probs.json', OLD / 'R_MAS_DPO_KL/preference_token_masks.json',
        OLD / 'R_DPO_KL/real_teacher_cache.json', OLD / 'R_DPO_KL/training_result.json',
        OLD / 'R_DPO_KL/training_exposures.jsonl']
    hashes = {**plan['source_sha256'], **{str(p): sha256_file(p) for p in files}}
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    audio_hashes = {r['audio']: r['audio_sha256'] for r in preferences + real}
    assert all(sha256_file(Path(p)) == h for p, h in audio_hashes.items())
    old_args = read(runtime.CONTROL / 'run_protocol.json')['arguments']
    base, parent = old_args['model_path'], old_args['policy_adapter_path']
    assert sha256_directory(Path(parent)) == plan['parent']
    out = ROOT / args.mode
    out.mkdir(parents=True, exist_ok=False)
    write_immutable_json(out / 'protocol.json', dict(mode=args.mode, source_sha256=hashes,
        audio_sha256=audio_hashes, parent=parent, parent_sha256=plan['parent'], base=base,
        selections=plan['selections'][mode], source_sampling_receipts=plan['source_sampling_receipts'],
        execution='zero optimizer smoke' if args.mode == 'smoke' else 'four optimizer steps plus full511 evaluation',
        planned_formal_settings=dict(seed=42, steps=4, lr=1e-4, beta=.1, KL_weight=10.,
            real_DPO_audio_occurrences=32, synthetic_occurrences=32, synthetic_unique_audio=len(set(plan['selections'][mode])),
            synthetic_weight=.25, KL_visits=256, r=16, alpha=4, clip=1., weight_decay=0.),
        score='Original edit+EOS DPO; full GT+EOS KL.',
        DPO_reference='Fresh scores computed before any update at function-identical initial R-only; actual separate frozen model equivalence checked on real and both SG arms in smoke. Separate from TTS reference speech.',
        KL_reference='Distributions computed once at function-identical initial R-only policy before any optimizer step, then detached; exact old teacher hashes checked.',
        SG_review='Unchanged simple three-text coarse review; selected target is intended, not word-verified.',
        retention='Failed weights omitted with evaluation retained. Passing adapter saved natively to durable /data with exact state roundtrip; no automatic promotion.'))
    if args.mode == 'smoke':
        preferences = [next(r for r in preferences if r['source'] == 'real')]
        for arm in ['MAS', 'fixed']:
            keys = {e['preference_key'] for w in read(PLAN / 'expected_DPO_exposures.json')[arm] for e in w if e['source'] == 'synthetic'}
            preferences.append(next(r for r in rows(plan['data_paths']['synthetic_preferences'][arm]) if preference_key(r) in keys))
        audio_hashes.update({r['audio']: r['audio_sha256'] for r in preferences})
        assert all(sha256_file(Path(p)) == h for p, h in audio_hashes.items())
        real = real[:1]
    else:
        smoke = read(ROOT / 'smoke/result.json')
        assert smoke['passed'] and smoke['trainer_sha256'] == sha256_file(Path(__file__))
    old_cache = read(OLD / 'R_MAS_DPO_KL/reference_log_probs.json')['values']
    old_masks = read(OLD / 'R_MAS_DPO_KL/preference_token_masks.json')['pairs']
    independent = {}
    device, dtype = torch.device('cuda:0'), torch.bfloat16
    if args.mode == 'smoke':
        set_seed(42)
        reference, processor = load_model(base, parent, False)
        reference_digest = parameter_digest(reference)
        collator = RealSyntheticEditCollator(processor, {})
        with torch.no_grad():
            for r in preferences:
                values, _ = batch_log_probs(reference, move_batch(collator([r]), device, dtype))
                independent[preference_key(r)] = values.detach().float().cpu()
        assert parameter_digest(reference) == reference_digest
        del reference, processor, collator, values
        gc.collect(); torch.cuda.empty_cache()
    set_seed(42)
    policy, processor = load_model(base, parent, True)
    initial, frozen = parameter_digest(policy, trainable_only=True), frozen_base_digest(policy)
    old = read(OLD / 'R_DPO_KL/training_result.json')
    assert initial == old['policy_trainable_sha256_before'] and frozen == old['frozen_base_sha256']
    collator = RealSyntheticEditCollator(processor, {})
    device, dtype = torch.device('cuda:0'), torch.bfloat16
    teacher, cache = {}, {}
    policy.eval()
    with torch.no_grad():
        for r in preferences:
            values, _ = batch_log_probs(policy, move_batch(collator([r]), device, dtype))
            key = preference_key(r); cache[key] = values.detach().float().cpu()
            if r['source'] == 'real':
                assert cache[key].tolist() == old_cache[key] and collator.audit[key] == old_masks[key]
            if args.mode == 'smoke':
                assert torch.equal(cache[key], independent[key])
        for r in real:
            teacher[r['audio_id']] = transcript_distributions(policy,
                move_batch(collator([r], include_rejected=False), device, dtype)).cpu()
    old_teacher = read(OLD / 'R_DPO_KL/real_teacher_cache.json')['audio']
    teacher_records = {k: dict(shape=list(v.shape), dtype=str(v.dtype),
        tensor_sha256=hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()) for k, v in teacher.items()}
    assert all(v == old_teacher[k] for k, v in teacher_records.items())
    assert parameter_digest(policy, trainable_only=True) == initial
    write_immutable_json(out / 'reference_binding.json', dict(scores={k: v.tolist() for k, v in cache.items()},
        teacher=teacher_records, fresh_scores_before_optimizer=True, independent_frozen_model_smoke=str(ROOT / 'smoke/result.json'), real_cache_source=str(OLD / 'R_MAS_DPO_KL/reference_log_probs.json'),
        exact_initial_parameter_sha256=initial, unchanged_initial_parent_sha256=plan['parent']))
    if args.mode != 'smoke':
        expected_smoke = read(ROOT / 'smoke/result.json')['exact_reference_scores']
        for key in cache.keys() & expected_smoke.keys():
            assert cache[key].tolist() == expected_smoke[key]
    policy.train(); policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = [p for p in policy.parameters() if p.requires_grad]
    assert len(parameters) == 536
    if args.mode == 'smoke':
        checks = []
        for r in preferences:
            policy.zero_grad(set_to_none=True)
            logp, _ = batch_log_probs(policy, move_batch(collator([r]), device, dtype))
            ref = cache[preference_key(r)].to(device)
            assert torch.equal(logp.detach(), ref)
            loss = dpo_losses(logp[:1], logp[1:], ref[:1], ref[1:], .1)[0]
            assert abs(loss.item() - .6931471805599453) < 1e-6
            loss.backward()
            norm = sum(p.grad.float().square().sum().item() for p in parameters if p.grad is not None) ** .5
            assert norm > 0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
            checks.append(dict(audio_id=r['audio_id'], source=r['source'], gradient_norm=norm))
        assert all(collator.audit[preference_key(r)] == old_masks[preference_key(r)] for r in preferences if r['source'] == 'real')
        policy.zero_grad(set_to_none=True)
        expected = rows(live.BASELINE)[:8]
        decoded = live.wrapper_for(policy, processor).transcribe(audio=[r['audio'] for r in expected], language=['Cantonese'] * 8)
        assert [r.text for r in decoded] == [r['prediction'] for r in expected]
        assert parameter_digest(policy, trainable_only=True) == initial and frozen_base_digest(policy) == frozen
        assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
        write_immutable_json(out / 'result.json', dict(passed=True, checks=checks, optimizer_steps=0,
            actual_checked_audio_sha256=audio_hashes, frozen_reference_parameter_sha256=reference_digest,
            trainer_sha256=sha256_file(Path(__file__)), exact_reference_scores={k: v.tolist() for k,v in cache.items()}, exact_historical_real_reference_and_KL=True, fresh_SG_reference_matches_separate_frozen_model=True,
            first8_initial_predictions_identical=True, seconds=time.time() - start))
        return
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.)
    by_key, by_id = {preference_key(r): r for r in preferences}, {r['audio_id']: r for r in real}
    history = []
    with (out / 'training_exposures.jsonl').open('x') as f:
        for step, window in enumerate(windows, 1):
            optimizer.zero_grad(set_to_none=True)
            losses = Counter()
            for e in window:
                if e['objective'] == 'real_symmetric_kl':
                    r = by_id[e['audio_id']]
                    values = transcript_distributions(policy, move_batch(collator([r], include_rejected=False), device, dtype))
                    loss = symmetric_kl(values, teacher[r['audio_id']].to(device))
                    name = 'real_symmetric_kl'
                else:
                    r = by_key[e['preference_key']]
                    logp, _ = batch_log_probs(policy, move_batch(collator([r]), device, dtype))
                    ref = cache[e['preference_key']].to(device)
                    loss = dpo_losses(logp[:1], logp[1:], ref[:1], ref[1:], .1)[0]
                    name = r['source'] + '_dpo'
                weighted = loss * e['loss_weight']
                assert torch.isfinite(weighted)
                weighted.backward(); losses[name] += weighted.item()
                f.write(json.dumps(e) + '\n')
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            assert torch.isfinite(norm)
            optimizer.step(); f.flush()
            history.append(dict(step=step, weighted_losses=dict(losses), gradient_norm=norm.item()))
            print(json.dumps(history[-1]), flush=True)
    exposures = rows(out / 'training_exposures.jsonl')
    assert [e for e in exposures if e['source'] == 'real'] == rows(OLD / 'R_DPO_KL/training_exposures.jsonl')
    assert all(collator.audit[preference_key(r)] == old_masks[preference_key(r)] for r in preferences if r['source'] == 'real')
    assert frozen_base_digest(policy) == frozen
    final = parameter_digest(policy, trainable_only=True)
    assert final != initial
    write_immutable_json(out / 'training_result.json', dict(history=history, seconds=time.time() - start,
        initial_residual_sha256=initial, final_residual_sha256=final, unchanged_base_sha256=frozen,
        same_real_and_KL_exposure_as_original_control=True, actual_masks=collator.audit))
    del optimizer, teacher, values, loss, weighted, logp, ref
    policy.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    eval_start = time.time()
    passed = live.evaluate_live(policy, processor, out, base, parent)
    storage = dict(weights_saved=False, reason='Failed original gate; preregistered policy')
    if passed:
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        checkpoint = out / 'checkpoint-4'
        policy.save_pretrained(checkpoint, safe_serialization=True)
        saved = load_file(str(checkpoint / 'adapter_model.safetensors'))
        state = get_peft_model_state_dict(policy)
        assert set(saved) == set(state) and all(torch.equal(saved[k], v.detach().cpu()) for k, v in state.items())
        storage = dict(weights_saved=True, checkpoint=str(checkpoint), durably_archived=True,
            exact_native_state_roundtrip=True, sha256=sha256_directory(checkpoint),
            load_recipe=dict(base_model=base, merge_initial_R_only=parent, then_load_residual_adapter=str(checkpoint)))
        write_immutable_json(out / 'passing_weight_storage.json', storage)
    assert sha256_directory(Path(parent)) == plan['parent']
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    write_immutable_json(out / 'result.json', dict(status='training_and_evaluation_complete',
        passes_original_gate=passed, storage=storage, promoted=False,
        evaluation_seconds=time.time() - eval_start, total_seconds=time.time() - start))
    print(json.dumps(dict(status='complete', mode=args.mode, passed=passed)), flush=True)


if __name__ == '__main__':
    main()
