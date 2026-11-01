#!/usr/bin/env python3
"""Real-audio character adaptation of N-best MWER, plus a matched CE control."""
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
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'examples'), '/tmp']
import mas_sequence_objective_pilot_v1 as common
from mas_sequence_objective_pilot_v1 import (
    runtime, read, rows, torch, sha256_file, sha256_directory, write_immutable_json,
    load_model, parameter_digest, frozen_base_digest, move_batch, batch_log_probs,
    AudioPreferenceCollator, install, wrapper_for, evaluate_live, retain_passing_weights,
)
from therapist_harness.asr_sequence_objectives import character_costs, minimum_character_risk
from therapist_harness.store import write_immutable_bytes

ROOT = Path('/tmp/mas_character_risk_v2')
TRAIN = Path('/data/qwen3-asr/finetune/data_setting_d_v1/setting_d_train.jsonl')
SOURCE = runtime.B / 'real96_dpo_probe/sampling/validated_inputs.jsonl'
EOS = {151645, 151643}


def select_inputs():
    source = {r['audio_id']: r for r in rows(SOURCE)}
    exposure = read(common.PLAN / 'expected_DPO_exposures.json')['R_MAS_DPO_KL']
    order = []
    for window in exposure:
        for e in window:
            if e['source'] == 'real' and e['audio_id'] not in order:
                order.append(e['audio_id'])
    selected = [{**source[k], 'chosen': source[k]['target_text']} for k in order]
    assert len(selected) == 32
    train = {r['utt_id']: r for r in rows(TRAIN)}
    dev = rows(common.DEV)
    dev_ids = {r['utt_id'] for r in dev}
    dev_paths = {r['audio'] for r in dev}
    dev_hashes = {sha256_file(Path(r['audio'])) for r in dev}
    for r in selected:
        assert r['audio_id'] in train and r['audio_id'] not in dev_ids
        assert r['audio'] == train[r['audio_id']]['audio'] and r['audio'] not in dev_paths
        assert r['chosen'] == train[r['audio_id']]['clean_gt']
        assert sha256_file(Path(r['audio'])) == r['audio_sha256'] and r['audio_sha256'] not in dev_hashes
    return selected


def prefix_features(wrapper, row):
    from qwen_asr.inference.utils import normalize_audio_input
    prompt = wrapper._build_text_prompt(context='', force_language='Cantonese')
    features = wrapper.processor(text=[prompt], audio=[normalize_audio_input(row['audio'])],
                                 return_tensors='pt', padding=True)
    assert features['attention_mask'].eq(1).all()
    return prompt, dict(features)


def exact_candidate_batch(prefix, tokens):
    """Keep the actual generated token sequence, including its first EOS."""
    assert tokens and tokens[-1] in EOS and not any(t in EOS for t in tokens[:-1])
    completion = torch.tensor([tokens], dtype=prefix['input_ids'].dtype)
    full = torch.cat([prefix['input_ids'], completion], dim=1)
    labels = full.clone()
    labels[:, :prefix['input_ids'].shape[1]] = -100
    return {**prefix, 'input_ids': full, 'attention_mask': torch.ones_like(full), 'labels': labels}


def sample_one(wrapper, row):
    from qwen_asr.inference.utils import parse_asr_output
    prompt, features = prefix_features(wrapper, row)
    length = features['input_ids'].shape[1]
    gpu = move_batch(features, torch.device('cuda:0'), torch.bfloat16)
    started = time.time()
    with torch.inference_mode():
        generated = wrapper.model.generate(**gpu, do_sample=False, num_beams=4,
            num_return_sequences=4, length_penalty=0., early_stopping=True, max_new_tokens=256)
    assert generated.sequences.shape[0] == 4
    records, unique = [], []
    seen = set()
    for rank, sequence in enumerate(generated.sequences):
        assert torch.equal(sequence[:length].cpu(), features['input_ids'][0])
        all_tokens = sequence[length:].tolist()
        end = next((i for i, t in enumerate(all_tokens) if t in EOS), None)
        tokens = all_tokens if end is None else all_tokens[:end + 1]
        raw = wrapper.processor.batch_decode([tokens], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        language, prediction = parse_asr_output(raw, user_language='Cantonese')
        key = tuple(tokens)
        valid = end is not None
        record = dict(rank=rank, generated_token_ids=tokens, raw_output=raw, prediction=prediction,
            language=language, ended_with_eos=valid, truncated=not valid,
            duplicate_token_sequence=key in seen,
            character_errors=character_costs(row['chosen'], [prediction], normalized=False)[0])
        records.append(record)
        if valid and key not in seen:
            unique.append(record)
        seen.add(key)
    return dict(audio_id=row['audio_id'], audio_sha256=row['audio_sha256'],
        prompt=prompt, prefix_token_ids=features['input_ids'][0].tolist(), all_four=records,
        unique_valid_candidates=unique, seconds=time.time() - started)


def prepare_protocol(out, stage, selected):
    old = read(runtime.CONTROL / 'run_protocol.json')['arguments']
    base, parent = old['model_path'], old['policy_adapter_path']
    files = [Path(__file__), Path(common.__file__), Path(common.objectives.__file__),
        SOURCE, TRAIN, common.DEV, common.PLAN / 'expected_DPO_exposures.json',
        Path('/tmp/mas_gradient_deterministic_v2/deterministic_qwen_positions_v1.py')]
    hashes = {str(p): sha256_file(p) for p in files}
    out.mkdir(parents=True, exist_ok=False)
    write_immutable_bytes(out / 'inputs.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected).encode())
    protocol = dict(stage=stage, parent=parent, base_model=base, parent_sha256=sha256_directory(Path(parent)),
        source_sha256=hashes, audio=len(selected), steps=4, real_per_step=8, seed=42,
        selection='Same 32 real audio identities and first-occurrence order as prior real DPO. Historical error-enriched sample; no new dev-based selection. Fresh diagnostic anchors excluded.',
        decode=dict(num_beams=4, num_return_sequences=4, length_penalty=0., early_stopping=True,
                    do_sample=False, max_new_tokens=256, language='Cantonese', context=''),
        candidates='Fresh beam4 from common initial R-only. Deduplicate exact token sequences. Keep actual token IDs through first EOS. Truncated candidates excluded only from risk; preserve all four records. Ground truth is never inserted into N-best.',
        refresh='One parent N-best generation, then one 32-audio epoch. This is a fixed-list MWER approximation; candidates are rescored differentiably at each current policy.',
        cost='Unnormalized character edit COUNT under unchanged evaluation text normalization. Character-level MWER adaptation, not per-utterance CER averaging.',
        risk='sum softmax(raw sequence logp) * (character edits - unweighted N-best mean edits)',
        CE='0.01 times summed GT transcript+EOS negative log probability, equal in both arms',
        optimizer=dict(lr=1e-4, weight_decay=0., gradient_clip=1., r=16, alpha=4),
        empty_or_equal_cost='Zero risk gradient; retain GT CE, no manufactured negatives',
        review='No new SG review rules; this pilot uses real training audio only',
        split_isolation='All selected IDs, paths and hashes checked against full 2117 training pool and 511 dev',
        paper='https://arxiv.org/abs/1712.01818', storage_policy='Same bounded live-evaluation policy as IPO/DPO pilot; no failed new-pilot weight persistence.')
    write_immutable_json(out / 'protocol.json', protocol)
    return protocol


def main():
    from transformers import set_seed
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['sample-smoke', 'sample', 'train-smoke', 'ce', 'mcer'], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    install(); set_seed(42)
    selected = select_inputs()
    start = time.time()
    out = ROOT / args.stage
    if args.stage == 'sample-smoke':
        selected = selected[:2]
    protocol = prepare_protocol(out, args.stage, selected)
    if args.stage.startswith('sample'):
        model, processor = load_model(protocol['base_model'], protocol['parent'], False)
        initial = parameter_digest(model)
        wrapper = wrapper_for(model, processor)
        results = []
        with (out / 'nbest.jsonl').open('x') as f:
            for r in selected:
                sampled = sample_one(wrapper, r)
                results.append(sampled)
                f.write(json.dumps(sampled, ensure_ascii=False) + '\n'); f.flush()
                print(json.dumps(dict(sampled_audio=len(results), total=len(selected),
                    unique_valid=len(sampled['unique_valid_candidates']),
                    costs=[v['character_errors'] for v in sampled['unique_valid_candidates']])), flush=True)
        assert parameter_digest(model) == initial
        write_immutable_json(out / 'result.json', dict(status='complete_beam4_sampling', audio=len(results),
            source_model=protocol['parent_sha256'], nbest_sha256=sha256_file(out / 'nbest.jsonl'),
            source_files_unchanged=all(sha256_file(Path(p)) == h for p, h in protocol['source_sha256'].items()),
            parents_unchanged=True, seconds=time.time() - start))
        return
    sample_receipt = read(ROOT / 'sample/result.json')
    assert sample_receipt['status'] == 'complete_beam4_sampling'
    assert sample_receipt['source_model'] == protocol['parent_sha256']
    assert sample_receipt['nbest_sha256'] == sha256_file(ROOT / 'sample/nbest.jsonl')
    sampled = {r['audio_id']: r for r in rows(ROOT / 'sample/nbest.jsonl')}
    assert set(sampled) == {r['audio_id'] for r in selected}
    if args.stage in {'ce', 'mcer'}:
        assert read(ROOT / 'train-smoke/result.json')['status'] == 'passed_actual_GPU_risk_backward_no_update'
    model, processor = load_model(protocol['base_model'], protocol['parent'], True)
    initial = parameter_digest(model, trainable_only=True)
    frozen = frozen_base_digest(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    # Prompt building itself does not require changing the model's train mode.
    from qwen_asr import Qwen3ASRModel
    wrapper = Qwen3ASRModel('transformers', model, processor)
    collator = AudioPreferenceCollator(processor)
    device, dtype = torch.device('cuda:0'), torch.bfloat16
    parameters = [v for v in model.parameters() if v.requires_grad]
    optimizer = None if args.stage == 'train-smoke' else torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.)
    if args.stage == 'train-smoke':
        variable = [r for r in selected if len({c['character_errors'] for c in sampled[r['audio_id']]['unique_valid_candidates']}) > 1]
        assert variable, 'No variable-risk N-best list in this pilot; do not claim risk training signal.'
        selected = variable[:2]
    history = []
    with (out / 'training_audio.jsonl').open('x') as f:
        for step, begin in enumerate(range(0, len(selected), 8), 1):
            model.zero_grad(set_to_none=True)
            window = selected[begin:begin + 8]
            terms = Counter()
            for row in window:
                sample = sampled[row['audio_id']]
                assert sample['audio_sha256'] == row['audio_sha256']
                gt_logp, _ = batch_log_probs(model, move_batch(collator([row], include_rejected=False), device, dtype))
                ce = -.01 * gt_logp[0]
                risk, expectation = ce * 0, ce.detach() * 0
                costs, q_values = [], []
                if args.stage != 'ce' and sample['unique_valid_candidates']:
                    prompt, prefix = prefix_features(wrapper, row)
                    assert prefix['input_ids'][0].tolist() == sample['prefix_token_ids']
                    logps = []
                    for c in sample['unique_valid_candidates']:
                        lp, count = batch_log_probs(model, move_batch(exact_candidate_batch(prefix, c['generated_token_ids']), device, dtype))
                        assert count.item() == len(c['generated_token_ids'])
                        logps.append(lp[0]); costs.append(c['character_errors'])
                    risk, expectation, q = minimum_character_risk(torch.stack(logps), costs)
                    q_values = q.detach().cpu().tolist()
                    if args.stage == 'train-smoke':
                        direct = torch.autograd.grad(risk, logps, retain_graph=True)
                        assert any(v.abs().item() > 0 for v in direct)
                        assert direct[costs.index(min(costs))].item() < 0
                        assert direct[costs.index(max(costs))].item() > 0
                loss = (ce + risk) / len(window)
                assert torch.isfinite(loss)
                loss.backward()
                terms['CE'] += ce.item() / len(window)
                terms['centered_risk'] += risk.item() / len(window)
                record = dict(step=step, audio_id=row['audio_id'], audio_sha256=row['audio_sha256'],
                    loss_weight=1 / len(window), CE_coefficient=.01, risk_coefficient=int(args.stage != 'ce'),
                    candidate_costs=costs, probabilities=q_values, expected_character_errors=expectation.item())
                f.write(json.dumps(record) + '\n'); f.flush()
                del ce, risk, expectation, loss, gt_logp
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            assert torch.isfinite(norm) and norm.item() > 0
            if optimizer is not None:
                optimizer.step()
            entry = dict(step=step, terms=dict(terms), gradient_norm=norm.item())
            history.append(entry); print(json.dumps(entry), flush=True)
    assert frozen_base_digest(model) == frozen
    if args.stage == 'train-smoke':
        assert parameter_digest(model, trainable_only=True) == initial
        write_immutable_json(out / 'result.json', dict(status='passed_actual_GPU_risk_backward_no_update',
            unchanged_parameters=True, optimizer_steps=0, exact_generated_tokens_scored=True,
            cost_sensitive_gradient_sign_verified=True, seconds=time.time() - start))
        return
    final = parameter_digest(model, trainable_only=True)
    assert final != initial
    write_immutable_json(out / 'training_result.json', dict(status='complete', history=history,
        initial_parameter_sha256=initial, final_parameter_sha256=final, frozen_base_sha256=frozen,
        nbest_receipt=sample_receipt, seconds=time.time() - start))
    del optimizer
    model.zero_grad(set_to_none=True)
    gc.collect(); torch.cuda.empty_cache()
    passed = bool(evaluate_live(model, processor, out, protocol['base_model'], protocol['parent']))
    storage = retain_passing_weights(model, out) if passed else dict(weights_saved=False, reason='Failed original gate; new pilot policy')
    assert all(sha256_file(Path(p)) == h for p, h in protocol['source_sha256'].items())
    assert sha256_directory(Path(protocol['parent'])) == protocol['parent_sha256']
    write_immutable_json(out / 'result.json', dict(status='training_and_evaluation_complete',
        passes_original_gate=passed, storage=storage, promoted=False, seconds=time.time() - start))
    print(json.dumps(dict(status='complete', arm=args.stage, passes_original_gate=passed)), flush=True)


if __name__ == '__main__':
    main()
