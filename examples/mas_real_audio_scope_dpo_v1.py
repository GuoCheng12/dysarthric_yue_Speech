#!/usr/bin/env python3
"""Original strict real DPO with only audio-tower residual LoRA trainable."""
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
sys.path[:0] = [str(REPO / 'examples'), str(REPO / 'src'), '/tmp']
import mas_real_headroom_selection_dpo_v1 as live
import mas_real_even_kl_gpu_smoke_v1 as runtime
from mas_real_even_kl_gpu_smoke_v1 import (
    read, rows, load_model, RealSyntheticEditCollator, torch, sha256_file, sha256_directory,
    write_immutable_json, preference_key, move_batch, batch_log_probs, dpo_losses,
    transcript_distributions, symmetric_kl, parameter_digest, frozen_base_digest,
)

ROOT = Path('/tmp/mas_real_audio_scope_DPO_v1')
PLAN = Path('/tmp/mas_deterministic_matched_DPO_v1/plan')
CONTROL = Path('/tmp/mas_deterministic_matched_DPO_v1/R_DPO_KL')
DECISION = Path('/tmp/mas_real_headroom_selection_DPO_v1/MAS_reflection/decision.json')


def text_lora_digest(model):
    digest = hashlib.sha256()
    count = 0
    for name, value in model.named_parameters():
        if '.lora_' in name and 'audio_tower' not in name:
            digest.update(name.encode())
            digest.update(value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
            count += 1
    assert count == 392
    return digest.hexdigest()


def freeze_text(model):
    selected, frozen = [], []
    for name, value in model.named_parameters():
        if value.requires_grad:
            assert '.lora_' in name
            if 'audio_tower' in name:
                selected.append(name)
            else:
                value.requires_grad_(False)
                frozen.append(name)
    assert len(selected) == 144 and len(frozen) == 392
    return dict(trainable_audio=selected, frozen_text=frozen)


def main():
    from transformers import set_seed
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    live.install()
    start = time.time()
    original = read(PLAN / 'plan.json')
    assert read(DECISION)['first_action'] == 'real_audio_scope_DPO'
    old_args = read(runtime.CONTROL / 'run_protocol.json')['arguments']
    base, parent = old_args['model_path'], old_args['policy_adapter_path']
    assert sha256_directory(Path(parent)) == original['parent']
    dpo = read(PLAN / 'expected_DPO_exposures.json')['R_DPO_KL']
    kl = read(PLAN / 'expected_common_KL_exposures.json')
    windows = [[e for e in kl if e['step'] == step] + dpo[step - 1] for step in range(1, 5)]
    used = {e['preference_key'] for w in dpo for e in w}
    preferences = [r for r in rows(original['data_paths']['real_preferences']) if preference_key(r) in used]
    real = rows(original['data_paths']['real_KL'])
    assert len(preferences) == len(used) == 91
    assert len({r['audio_id'] for r in preferences}) == 32
    assert all(r['source'] == 'real' for r in preferences + real)
    hashes = original['source_sha256'].copy()
    files = [Path(__file__), Path(runtime.__file__), Path(live.__file__), DECISION,
        CONTROL / 'training_result.json', CONTROL / 'reference_log_probs.json',
        CONTROL / 'real_teacher_cache.json', CONTROL / 'preference_token_masks.json',
        CONTROL / 'training_exposures.jsonl', PLAN / 'plan.json',
        PLAN / 'expected_DPO_exposures.json', PLAN / 'expected_common_KL_exposures.json']
    hashes.update({str(p): sha256_file(p) for p in files})
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    audio_hashes = {r['audio']: r['audio_sha256'] for r in preferences + real}
    assert all(sha256_file(Path(p)) == h for p, h in audio_hashes.items())
    out = ROOT / ('smoke' if args.smoke else 'audio_only')
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(source_sha256=hashes, audio_sha256=audio_hashes, parent=parent,
        parent_sha256=original['parent'], base_model=base, reference='frozen initial R-only ASR',
        settings=dict(seed=42, steps=4, real_DPO_audio_occurrences=32, unique_pairs=91, real_KL_visits=256,
                      beta=.1, KL_weight=10., lr=1e-4, r=16, alpha=4, clip=1., weight_decay=0.),
        intervention='Freeze only text-decoder residual LoRA; train audio-tower residual LoRA. Original full-scope real control reused.',
        score='Original edit+EOS DPO, full GT+EOS KL. Not newer mean-score/worst-beam data.',
        source_sampling_receipts=original['source_sampling_receipts'],
        selected_sampling_source='historical_real only; synthetic receipts above retained as parent-plan provenance, no synthetic data used',
        comparator=str(CONTROL), current_MAS_decision=str(DECISION),
        SG_review='unchanged; no SG used in this bounded diagnostic',
        retention='Evaluate live final model. Failed new weights omitted with full traces retained; passing weights saved losslessly and durability declared. No automatic promotion.')
    write_immutable_json(out / 'protocol.json', protocol)
    if args.smoke:
        preferences = preferences[:2]
        real = real[:1]
    else:
        smoke = read(ROOT / 'smoke/result.json')
        assert smoke['status'] == 'passed_audio_scope_GPU_no_update'
        assert smoke['trainer_sha256'] == sha256_file(Path(__file__))
    set_seed(42)
    reference, processor = load_model(base, parent, False)
    before_ref = parameter_digest(reference)
    frozen_base = frozen_base_digest(reference)
    assert frozen_base == read(CONTROL / 'training_result.json')['frozen_base_sha256']
    audit, cache, teacher = {}, {}, {}
    collator = RealSyntheticEditCollator(processor, audit)
    device, dtype = torch.device('cuda:0'), torch.bfloat16
    with torch.no_grad():
        for i, row in enumerate(preferences):
            cache[preference_key(row)] = batch_log_probs(reference, move_batch(collator([row]), device, dtype))[0].cpu()
            if i % 40 == 0:
                print(json.dumps(dict(stage='reference', pairs=i + 1, total=len(preferences))), flush=True)
        for row in real:
            teacher[row['audio_id']] = transcript_distributions(reference,
                move_batch(collator([row], include_rejected=False), device, dtype)).cpu()
    previous_scores = read(CONTROL / 'reference_log_probs.json')['values']
    previous_masks = read(CONTROL / 'preference_token_masks.json')['pairs']
    previous_teacher = read(CONTROL / 'real_teacher_cache.json')['audio']
    teacher_metadata = {k: dict(shape=list(v.shape), dtype=str(v.dtype),
        tensor_sha256=hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()) for k, v in teacher.items()}
    assert all(v.tolist() == previous_scores[k] and audit[k] == previous_masks[k] for k, v in cache.items())
    assert all(v == previous_teacher[k] for k, v in teacher_metadata.items())
    assert parameter_digest(reference) == before_ref
    write_immutable_json(out / 'reference_scores.json', dict(values={k: v.tolist() for k, v in cache.items()},
        masks=audit, teacher_metadata=teacher_metadata, exact_original_reference_scores_masks_and_KL=True))
    del reference, processor, collator
    gc.collect(); torch.cuda.empty_cache()
    set_seed(42)
    policy, processor = load_model(base, parent, True)
    initial_all = parameter_digest(policy, trainable_only=True)
    assert initial_all == read(CONTROL / 'training_result.json')['policy_trainable_sha256_before']
    assert frozen_base_digest(policy) == frozen_base
    initial_text = text_lora_digest(policy)
    scope = freeze_text(policy)
    initial_audio = parameter_digest(policy, trainable_only=True)
    write_immutable_json(out / 'scope.json', dict(scope=scope, initial_full_residual_sha256=initial_all,
        initial_audio_sha256=initial_audio, initial_text_sha256=initial_text,
        functional_initialization='Same full zero-output residual as old control; only requires_grad flags changed.'))
    parameters = [p for p in policy.parameters() if p.requires_grad]
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    collator = RealSyntheticEditCollator(processor, audit)
    if args.smoke:
        checks = []
        for row in preferences:
            policy.zero_grad(set_to_none=True)
            logp, _ = batch_log_probs(policy, move_batch(collator([row]), device, dtype))
            ref = cache[preference_key(row)].to(device)
            assert torch.equal(logp.detach(), ref)
            loss = dpo_losses(logp[:1], logp[1:], ref[:1], ref[1:], .1)[0]
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
            assert all(p.grad is None for n, p in policy.named_parameters() if n in scope['frozen_text'])
            norm = sum(p.grad.float().square().sum().item() for p in parameters if p.grad is not None) ** .5
            assert norm > 0
            checks.append(dict(audio_id=row['audio_id'], loss=loss.item(), audio_gradient_norm=norm))
        policy.zero_grad(set_to_none=True)
        expected = rows(live.BASELINE)[:8]
        decoded = live.wrapper_for(policy, processor).transcribe(audio=[r['audio'] for r in expected], language=['Cantonese'] * 8)
        assert [r.text for r in decoded] == [r['prediction'] for r in expected]
        assert parameter_digest(policy, trainable_only=True) == initial_audio
        assert text_lora_digest(policy) == initial_text and frozen_base_digest(policy) == frozen_base
        write_immutable_json(out / 'result.json', dict(status='passed_audio_scope_GPU_no_update', checks=checks,
            trainer_sha256=sha256_file(Path(__file__)), optimizer_steps=0, initial_first8_predictions_exact=True,
            frozen_text_has_no_gradients=True, exact_old_reference_scores_and_masks=True, seconds=time.time() - start))
        return
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.)
    by_key = {preference_key(r): r for r in preferences}
    by_id = {r['audio_id']: r for r in real}
    history = []
    with (out / 'training_exposures.jsonl').open('x') as f:
        for step, window in enumerate(windows, 1):
            optimizer.zero_grad(set_to_none=True)
            terms = Counter()
            for e in window:
                if e['objective'] == 'real_symmetric_kl':
                    row = by_id[e['audio_id']]
                    values = transcript_distributions(policy, move_batch(collator([row], include_rejected=False), device, dtype))
                    loss = symmetric_kl(values, teacher[row['audio_id']].to(device))
                    term = 'real_symmetric_kl'
                else:
                    row = by_key[e['preference_key']]
                    logp, _ = batch_log_probs(policy, move_batch(collator([row]), device, dtype))
                    ref = cache[e['preference_key']].to(device)
                    loss = dpo_losses(logp[:1], logp[1:], ref[:1], ref[1:], .1)[0]
                    term = 'real_dpo'
                weighted = loss * e['loss_weight']
                assert torch.isfinite(weighted)
                weighted.backward()
                terms[term] += weighted.item()
                f.write(json.dumps(e) + '\n')
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            assert torch.isfinite(norm)
            assert all(p.grad is None for n, p in policy.named_parameters() if n in scope['frozen_text'])
            optimizer.step(); f.flush()
            entry = dict(step=step, weighted_losses=dict(terms), gradient_norm=norm.item())
            history.append(entry); print(json.dumps(entry), flush=True)
    final_audio = parameter_digest(policy, trainable_only=True)
    assert final_audio != initial_audio
    assert text_lora_digest(policy) == initial_text
    assert frozen_base_digest(policy) == frozen_base
    assert rows(out / 'training_exposures.jsonl') == rows(CONTROL / 'training_exposures.jsonl')
    assert audit == read(out / 'reference_scores.json')['masks']
    write_immutable_json(out / 'training_result.json', dict(status='complete', history=history,
        initial_full_residual_sha256=initial_all, initial_audio_sha256=initial_audio, final_audio_sha256=final_audio,
        unchanged_text_sha256=initial_text, unchanged_base_sha256=frozen_base,
        exact_old_exposures_reference_masks_and_KL=True, seconds=time.time() - start))
    del optimizer, teacher, values, loss, weighted
    policy.zero_grad(set_to_none=True)
    gc.collect(); torch.cuda.empty_cache()
    eval_start = time.time()
    passed = live.evaluate_live(policy, processor, out, base, parent)
    storage = live.retain_passing_weights(policy, out) if passed else dict(weights_saved=False, reason='Failed original gate; preregistered policy')
    assert sha256_directory(Path(parent)) == original['parent']
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    write_immutable_json(out / 'result.json', dict(status='training_and_evaluation_complete',
        passes_original_gate=passed, storage=storage, promoted=False,
        evaluation_seconds=time.time() - eval_start, total_seconds=time.time() - start))
    print(json.dumps(dict(status='complete', passed=passed)), flush=True)


if __name__ == '__main__':
    main()
