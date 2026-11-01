#!/usr/bin/env python3
"""Measure per-audio SG DPO transfer to fixed, training-only GT anchors; no update."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path[:0] = ['/tmp', '/tmp/mas_gradient_deterministic_v2']
import mas_real_even_kl_gpu_smoke_v1 as runtime
from mas_real_even_kl_gpu_smoke_v1 import (
    B, REPO, torch, read, rows, load_model, RealSyntheticEditCollator, move_batch,
    batch_log_probs, dpo_losses, preference_key, parameter_digest, frozen_base_digest,
    sha256_file, sha256_directory, write_immutable_json,
)
from deterministic_qwen_positions_v1 import install

ROOT = B / 'synthetic_audio_utility_probe_v1'
OLD = Path('/tmp/mas_deterministic_matched_DPO_v1')
ANCHORS = Path('/tmp/mas_fresh_real_anchor64_v1')
SCOPE_RESULT = Path('/tmp/mas_real_audio_scope_DPO_v1/results.json')


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def relation(x, y):
    x, y = x.double(), y.double()
    dot = torch.dot(x, y).item()
    nx, ny = x.norm().item(), y.norm().item()
    return dict(dot=dot, cosine=dot / (nx * ny) if nx and ny else None)


def main():
    from transformers import set_seed
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    install()
    start = time.time()
    assert not read(SCOPE_RESULT)['passes_original_gate']
    plan = read(OLD / 'plan/plan.json')
    assert all(sha256_file(Path(p)) == h for p, h in plan['source_sha256'].items())
    windows = read(OLD / 'plan/expected_DPO_exposures.json')['R_MAS_DPO_KL']
    exposures = [e for w in windows for e in w if e['source'] == 'synthetic']
    audio_order = list(dict.fromkeys(e['audio_id'] for e in exposures))
    used_keys = {e['preference_key'] for e in exposures}
    grouped = defaultdict(list)
    for row in rows(plan['data_paths']['synthetic_preferences']):
        if preference_key(row) in used_keys:
            grouped[row['audio_id']].append(row)
    assert len(audio_order) == len(grouped) == 22 and sum(map(len, grouped.values())) == 52
    anchor_receipt = read(ANCHORS / 'receipt.json')
    assert sha256_file(ANCHORS / 'real.jsonl') == anchor_receipt['output_sha256']
    anchors = rows(ANCHORS / 'real.jsonl')[:32]
    assert len({r['audio_id'] for r in anchors}) == 32
    anchor_sets = dict(A=anchors[:8], B=anchors[8:16], C=anchors[16:32])
    metadata = {}
    for aid in audio_order:
        group = grouped[aid]
        r = group[0]
        assert len({x['rejected'] for x in group}) == len(group)
        assert all(x['chosen'] == r['chosen'] and x['audio_sha256'] == r['audio_sha256'] for x in group)
        review = r['metadata']['review']
        assert review['verdict'] == 'pass' and review['candidate_audio_sha256'] == r['audio_sha256']
        metadata[aid] = dict(audio_id=aid, audio_sha256=r['audio_sha256'], target=r['chosen'],
            reference_audio_sha256=r['metadata']['reference_audio_sha256'],
            reference_text=review['reference_text'], review_transcription=review['asr_transcription'],
            role=aid.split('.r1.')[1].split('.')[0], actual_rejected=[x['rejected'] for x in group],
            pair_keys=[preference_key(x) for x in group],
            chosen_status='intended_target_after_coarse_review_not_word_verified')
    # This control is fixed without observing any gradients; match 8 distinct audios x 4 visits.
    random_order = np.random.default_rng(42).permutation(len(audio_order)).tolist()
    uniform, content_count = [], Counter()
    for i in random_order:
        aid = audio_order[i]
        if content_count[metadata[aid]['target']] >= 2:
            continue
        uniform.append(aid); content_count[metadata[aid]['target']] += 1
        if len(uniform) == 8:
            break
    uniform.sort(key=audio_order.index)
    files = [Path(__file__), Path(runtime.__file__), SCOPE_RESULT, ANCHORS / 'real.jsonl',
        ANCHORS / 'receipt.json', OLD / 'plan/plan.json', OLD / 'plan/expected_DPO_exposures.json',
        OLD / 'R_MAS_DPO_KL/reference_log_probs.json', OLD / 'R_MAS_DPO_KL/preference_token_masks.json',
        OLD / 'R_DPO_KL/training_result.json',
        Path('/tmp/mas_gradient_deterministic_v2/deterministic_qwen_positions_v1.py')]
    hashes = {**plan['source_sha256'], **{str(p): sha256_file(p) for p in files}}
    audio_hashes = {r['audio']: r['audio_sha256'] for r in anchors + [r for g in grouped.values() for r in g]}
    assert all(sha256_file(Path(p)) == h for p, h in audio_hashes.items())
    out = ROOT / ('smoke' if args.smoke else 'probe')
    out.mkdir(parents=True, exist_ok=False)
    write_immutable_json(out / 'protocol.json', dict(source_sha256=hashes, audio_sha256=audio_hashes,
        parent_sha256=plan['parent'], anchors={k: [r['audio_id'] for r in v] for k, v in anchor_sets.items()},
        anchor_pool_receipt=anchor_receipt, audio_order=audio_order, metadata=metadata,
        uniform_control_audio=uniform, optimizer_steps=0, model_saved=False,
        gradient_objective='Per-audio average of original edit+EOS DPO beta0.1 over exactly its historically trained pairs. Full residual LoRA, same initial R-only. Anchors mean full GT+EOS token NLL per audio.',
        selection_contract='MAS may select 8 distinct existing audios, at most2 per target text; at least4 reference recordings and2 content roles. Same32 real DPO occurrences and256 KL visits; both synthetic subsets use8 audios x4 visits and weight0.25. Selection membership is the intervention; real/KL order stays identical. No adjacent loss/lr/freeze scan.',
        proxy_use='A/B are selection evidence; C is an unobserved-in-MAS-request diagnostic reserved until selection is frozen. No 511 dev gradient or content used here. Positive gradient alignment is a training proxy, not CER or acoustic truth.',
        activation='Actual MAS Reflection chooses train_selected_subset or stop_current_pool after seeing A/B and metadata. Do not train a subset whose aggregate dot is nonpositive to either A or B. No individual SG-quality rejection or CER threshold is introduced.',
        SG_review='Existing coarse three-text decisions and intended chosen/rejected preserved. Utility scheduling does not change synthesis-failure labels.',
        smoke=args.smoke))
    if args.smoke:
        anchor_sets = {k: v[:1] for k, v in anchor_sets.items() if k != 'C'}
        audio_order = audio_order[:2]
    else:
        smoke = read(ROOT / 'smoke/result.json')
        assert smoke['status'] == 'passed_no_update_per_audio_utility_smoke'
        assert smoke['trainer_sha256'] == sha256_file(Path(__file__))
    original = read(runtime.CONTROL / 'run_protocol.json')['arguments']
    base, parent = original['model_path'], original['policy_adapter_path']
    assert sha256_directory(Path(parent)) == plan['parent']
    set_seed(42)
    policy, processor = load_model(base, parent, True)
    initial, frozen = parameter_digest(policy, trainable_only=True), frozen_base_digest(policy)
    old = read(OLD / 'R_DPO_KL/training_result.json')
    assert initial == old['policy_trainable_sha256_before'] and frozen == old['frozen_base_sha256']
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    members = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    assert len(members) == 536
    audio_elements = sum(p.numel() for n, p in members if 'audio_tower' in n)
    # Explicitly concatenate audio then text so module relations and whole-vector relations agree.
    members.sort(key=lambda item: 'audio_tower' not in item[0])
    audit, input_records = {}, []
    collator = RealSyntheticEditCollator(processor, audit)
    scores = read(OLD / 'R_MAS_DPO_KL/reference_log_probs.json')['values']
    masks = read(OLD / 'R_MAS_DPO_KL/preference_token_masks.json')['pairs']
    device, dtype = torch.device('cuda:0'), torch.bfloat16
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()

    def gradient(name, examples, is_dpo):
        policy.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
        loss_value, began = 0., time.time()
        for r in examples:
            batch = move_batch(collator([r], include_rejected=is_dpo), device, dtype)
            logp, counts = batch_log_probs(policy, batch)
            if is_dpo:
                key = preference_key(r)
                ref = torch.tensor(scores[key], device=device, dtype=torch.float32)
                assert torch.equal(logp.detach().float(), ref) and audit[key] == masks[key]
                loss = dpo_losses(logp[:1], logp[1:], ref[:1], ref[1:], .1)[0]
            else:
                loss = -logp[0] / counts[0]
            weighted = loss / len(examples)
            assert torch.isfinite(weighted)
            weighted.backward(); loss_value += weighted.item()
            input_records.append(dict(objective=name, audio_id=r['audio_id'], audio_sha256=r['audio_sha256'],
                labels_sha256=tensor_sha(batch['labels']), scored_token_counts=counts.tolist()))
        vector = torch.cat([torch.zeros(p.numel(), dtype=torch.float32) if p.grad is None
            else p.grad.detach().float().cpu().reshape(-1) for n, p in members])
        assert torch.isfinite(vector).all() and vector.norm() > 0
        summary = dict(loss=loss_value, norm=vector.double().norm().item(), sha256=tensor_sha(vector),
                       microbatches=len(examples), seconds=time.time() - began)
        print(json.dumps(dict(objective_complete=name, **summary)), flush=True)
        return vector, summary

    anchor_vectors, anchor_summary = {}, {}
    for key, examples in anchor_sets.items():
        anchor_vectors[key], anchor_summary[key] = gradient('anchor_' + key, examples, False)
    candidate_vectors, candidates = [], []
    for aid in audio_order:
        vector, summary = gradient(aid, grouped[aid], True)
        candidate_vectors.append(vector)
        relations = {key: dict(full=relation(vector, anchor),
            audio=relation(vector[:audio_elements], anchor[:audio_elements]),
            text=relation(vector[audio_elements:], anchor[audio_elements:]))
            for key, anchor in anchor_vectors.items()}
        candidates.append(dict(metadata=metadata[aid], gradient=summary, relations=relations))
    matrix = torch.stack(candidate_vectors).double()
    gram = (matrix @ matrix.T).tolist()
    assert parameter_digest(policy, trainable_only=True) == initial and frozen_base_digest(policy) == frozen
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    assert sha256_directory(Path(parent)) == plan['parent']
    status = 'passed_no_update_per_audio_utility_smoke' if args.smoke else 'complete_per_audio_synthetic_utility_probe'
    result = dict(status=status, trainer_sha256=sha256_file(Path(__file__)), anchors=anchor_summary,
        candidates=candidates, audio_order=audio_order, gradient_gram=gram, gram_order='audio_order',
        unchanged_trainable_sha256=initial, unchanged_frozen_base_sha256=frozen,
        optimizer_steps=0, weights_saved=False, input_receipts=input_records, seconds=time.time() - start,
        exact_old_reference_scores_and_masks=True)
    write_immutable_json(out / 'result.json', result)
    print(json.dumps(dict(status=status, audio=len(audio_order), seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
