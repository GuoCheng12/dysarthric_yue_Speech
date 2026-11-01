#!/usr/bin/env python3
"""Matched mean-score DPO/IPO pilot on the existing real + MAS curriculum.

Evaluate the live trained model before releasing it. Failed pilot weights are
not retained; receipts, predictions, and parameter digests always are. Passing
weights are serialized losslessly, with storage durability stated explicitly.
"""
import argparse
from collections import Counter
import gc
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'finetune/scripts'), '/tmp',
                '/tmp/mas_gradient_deterministic_v2']
import mas_real_even_kl_gpu_smoke_v1 as runtime
from mas_real_even_kl_gpu_smoke_v1 import (
    read, rows, load_model, torch, sha256_file, sha256_directory,
    write_immutable_json, preference_key, move_batch, batch_log_probs,
    transcript_distributions, symmetric_kl, parameter_digest, frozen_base_digest,
)
from therapist_harness.dpo_training import AudioPreferenceCollator
from therapist_harness.asr_sequence_objectives import mean_preference_loss
import therapist_harness.asr_sequence_objectives as objectives
from deterministic_qwen_positions_v1 import install
import evaluate_asr_checkpoint as evaluator
from evaluate_deterministic_ram_asr_v1 import write_outputs
import mas_evaluation_report as report

ROOT = Path('/tmp/mas_IPO_DPO_mean_v1')
PLAN = Path('/tmp/mas_deterministic_matched_DPO_v1/plan')
DEV = Path('/data/qwen3-asr/finetune/data_setting_d_merged_test_v1/setting_d_merged_test.jsonl')
BASELINE = runtime.B / 'parent_eval_check/predictions.jsonl'


def wrapper_for(model, processor):
    from qwen_asr import Qwen3ASRModel
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True
    wrapper = Qwen3ASRModel('transformers', model, processor,
                            max_inference_batch_size=8, max_new_tokens=256)
    wrapper.dtype = torch.bfloat16
    return wrapper


def gzip_rows(path):
    path = Path(path)
    if path.suffix == '.gz':
        return [json.loads(x) for x in gzip.decompress(path.read_bytes()).decode().splitlines()]
    return rows(path)


def evaluate_live(policy, processor, out, base, parent):
    wrapper = wrapper_for(policy, processor)
    evaluator.load_model = lambda config: wrapper
    evaluator.parse_args = lambda: SimpleNamespace(base_model_path=Path(base), adapter_path=None,
        split_jsonl=DEV, out_dir=out / 'evaluation', batch_size=8, max_new_tokens=256,
        language='Cantonese', critical_cer_threshold=.5)
    evaluator.write_outputs = write_outputs
    evaluator.main()
    # adapter_path=None means no on-disk residual was loaded; this binding records
    # the actual in-process residual, rather than claiming a nonexistent file.
    write_immutable_json(out / 'live_evaluation_binding.json', dict(
        base_model=base, initial_R_only=parent, residual_merged=False,
        live_parameter_digest=parameter_digest(policy, trainable_only=True),
        source='trained residual in this process; no checkpoint reload',
        decoding='unchanged evaluator: greedy Cantonese, empty context, batch8, cap256'))
    report.rows = gzip_rows
    comparison = report.compare(BASELINE, out / 'evaluation/predictions.jsonl.gz')
    overall, hard = comparison['groups']['overall'], comparison['groups']['difficulty:hard']
    passed = (overall['after_pooled'] < overall['before_pooled'] and
              overall['after_patient_macro'] - overall['before_patient_macro'] <= .001 and
              hard['after_pooled'] - hard['before_pooled'] <= .005)
    write_immutable_json(out / 'comparison.json', dict(comparison=comparison, passes_original_gate=bool(passed)))
    return passed


def retain_passing_weights(model, out):
    """Exact native safetensors bytes; /dev parts are explicitly volatile tmpfs.

    Only new passing candidates use this bounded storage. No old artifact is
    removed. A candidate with any tmpfs part cannot be promoted as durable.
    """
    import numpy as np
    from peft import get_peft_model_state_dict
    from safetensors.torch import save
    state = {k: v.detach().contiguous().cpu() for k, v in get_peft_model_state_dict(model).items()}
    raw = save(state, metadata={'format': 'pt'})
    padding = (-len(raw)) % 4
    shuffled = np.frombuffer(raw + b'\0' * padding, dtype=np.uint8).reshape(-1, 4).T.copy().tobytes()
    packed = lzma.compress(shuffled, preset=3)
    available = [Path('/tmp'), Path('/dev')]
    capacity = sum(max(0, shutil.disk_usage(p).free - 2 * 1024**2) for p in available)
    assert len(packed) <= capacity, 'Insufficient space for a passing candidate; do not promote.'
    parts, offset = [], 0
    for root in available:
        take = min(len(packed) - offset, max(0, shutil.disk_usage(root).free - 2 * 1024**2))
        if take == 0:
            continue
        path = root / ('mas_IPO_DPO_mean_v1_' + out.name + '.weights.part' + str(len(parts)))
        data = packed[offset:offset + take]
        with path.open('xb') as f:
            f.write(data)
        parts.append(dict(path=str(path), bytes=take, sha256=sha256_file(path), volatile_tmpfs=root == Path('/dev')))
        offset += take
        if offset == len(packed):
            break
    restored = lzma.decompress(b''.join(Path(x['path']).read_bytes() for x in parts))
    restored = np.frombuffer(restored, dtype=np.uint8).reshape(4, -1).T.copy().tobytes()[:len(raw)]
    assert restored == raw
    record = dict(format='byte_shuffle4_lzma_native_safetensors_v1', raw_bytes=len(raw),
        raw_sha256=hashlib.sha256(raw).hexdigest(), compressed_bytes=len(packed), parts=parts,
        lossless_roundtrip_verified=True, durably_archived=not any(p['volatile_tmpfs'] for p in parts),
        promoted=False, adapter_config=model.peft_config['default'].to_dict())
    record['adapter_config'] = {k: sorted(v) if isinstance(v, set) else v
                                for k, v in record['adapter_config'].items()}
    write_immutable_json(out / 'passing_weight_storage.json', record)
    return record


def prepare(out, mode):
    original = read(PLAN / 'plan.json')
    hashes = original['source_sha256'].copy()
    hashes.update({str(p): sha256_file(p) for p in [Path(__file__), Path(objectives.__file__),
        Path(runtime.__file__), Path(evaluator.__file__), Path(report.__file__), PLAN / 'plan.json',
        PLAN / 'expected_DPO_exposures.json', PLAN / 'expected_common_KL_exposures.json', DEV, BASELINE]})
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    old = read(runtime.CONTROL / 'run_protocol.json')['arguments']
    base, parent = old['model_path'], old['policy_adapter_path']
    assert sha256_directory(Path(parent)) == original['parent']
    preferences = rows(original['data_paths']['real_preferences']) + rows(original['data_paths']['synthetic_preferences'])
    real = rows(original['data_paths']['real_KL'])
    dpo = read(PLAN / 'expected_DPO_exposures.json')['R_MAS_DPO_KL']
    kl = read(PLAN / 'expected_common_KL_exposures.json')
    windows = [[r for r in kl if r['step'] == step] + dpo[step - 1] for step in range(1, 5)]
    used = {e['preference_key'] for w in windows for e in w if e['objective'] == 'dpo'}
    preferences = [r for r in preferences if preference_key(r) in used]
    assert len(used) == len(preferences)
    audio_hashes = {r['audio']: r['audio_sha256'] for r in preferences + real}
    assert all(sha256_file(Path(p)) == h for p, h in audio_hashes.items())
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(mode=mode, source_sha256=hashes, audio_sha256=audio_hashes,
        parent=parent, parent_sha256=original['parent'], base_model=base,
        source_sampling_receipts=original['source_sampling_receipts'],
        settings=dict(seed=42, steps=4, lr=1e-4, beta=1., KL_weight=10.,
            real_audio_occurrences=32, synthetic_audio_occurrences=32, synthetic_weight=.25,
            KL_visits=256, r=16, alpha=4, gradient_clip=1., weight_decay=0.),
        score='Full transcript + EOS mean token log probabilities in BOTH arms; no edit mask.',
        DPO='-logsigmoid(beta * (mean log policy/ref chosen - mean log policy/ref rejected))',
        IPO='0.5 * (beta * margin - 0.5)^2; positive rescaling of IPO, equal initial DPO derivative',
        beta_choice='1.0 on mean scores, fixed before outcomes; prior edit-sum beta0.1 is a different scoring convention.',
        storage_policy='Evaluate final model in-process. Failed new pilot weights omitted explicitly; all predictions, exposures, digests retained. Passing weights losslessly archived, tmpfs portions labelled volatile and no promotion until durable.',
        DPO_reference='Frozen initial R-only ASR, distinct from TTS reference speech',
        SG_review='unchanged existing coarse three-text review',
        interpretation='Two-arm objective pilot on the SAME mixed curriculum. Does not by itself isolate MAS incremental value; a winning objective needs a matched real-only control.')
    write_immutable_json(out / 'protocol.json', protocol)
    return protocol, preferences, real, windows


def main():
    from transformers import set_seed
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'dpo', 'ipo'], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    install()
    start = time.time()
    out = ROOT / args.mode
    protocol, preferences, real, windows = prepare(out, args.mode)
    base, parent = protocol['base_model'], protocol['parent']
    if args.mode != 'smoke':
        smoke = read(ROOT / 'smoke/result.json')
        assert smoke['status'] == 'passed_actual_GPU_no_update'
        assert smoke['trainer_sha256'] == sha256_file(Path(__file__))
    else:
        preferences = [next(r for r in preferences if r['source'] == s) for s in ['real', 'synthetic']]
        real = real[:1]
    set_seed(42)
    reference, processor = load_model(base, parent, False)
    collator = AudioPreferenceCollator(processor)
    device, dtype = torch.device('cuda:0'), torch.bfloat16
    before_ref = parameter_digest(reference)
    frozen = frozen_base_digest(reference)
    cache, counts, teacher = {}, {}, {}
    with torch.no_grad():
        for i, row in enumerate(preferences):
            score, n = batch_log_probs(reference, move_batch(collator([row]), device, dtype))
            key = preference_key(row)
            cache[key], counts[key] = score.cpu(), n.cpu()
            if i % 50 == 0:
                print(json.dumps(dict(stage='reference', pairs=i + 1, total=len(preferences))), flush=True)
        for row in real:
            teacher[row['audio_id']] = transcript_distributions(reference,
                move_batch(collator([row], include_rejected=False), device, dtype)).cpu()
    assert parameter_digest(reference) == before_ref
    write_immutable_json(out / 'reference_scores.json', dict(
        scores={k: v.tolist() for k, v in cache.items()}, counts={k: v.tolist() for k, v in counts.items()},
        frozen_parent=protocol['parent_sha256'], unchanged=True))
    del reference, processor, collator
    gc.collect(); torch.cuda.empty_cache()
    set_seed(42)
    policy, processor = load_model(base, parent, True)
    initial = parameter_digest(policy, trainable_only=True)
    assert frozen_base_digest(policy) == frozen
    collator = AudioPreferenceCollator(processor)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = [v for v in policy.parameters() if v.requires_grad]
    if args.mode == 'smoke':
        checks = []
        for row in preferences:
            gradients = []
            for mode in ['dpo', 'ipo']:
                policy.zero_grad(set_to_none=True)
                logp, n = batch_log_probs(policy, move_batch(collator([row]), device, dtype))
                ref = cache[preference_key(row)].to(device)
                assert torch.equal(logp.detach(), ref)
                assert torch.equal(n.cpu(), counts[preference_key(row)])
                loss, margin = mean_preference_loss(logp, ref, n, objective=mode, beta=1.)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
                digest = hashlib.sha256()
                norm = 0.
                for p in parameters:
                    if p.grad is not None:
                        digest.update(p.grad.detach().cpu().numpy().tobytes())
                        norm += p.grad.float().square().sum().item()
                assert norm > 0 and margin.item() == 0
                gradients.append(digest.hexdigest())
            assert gradients[0] == gradients[1]
            checks.append(dict(audio_id=row['audio_id'], source=row['source'], identical_initial_full_gradients=True))
        policy.zero_grad(set_to_none=True)
        expected = rows(BASELINE)[:8]
        decoded = wrapper_for(policy, processor).transcribe(audio=[r['audio'] for r in expected], language=['Cantonese'] * 8)
        assert [r.text for r in decoded] == [r['prediction'] for r in expected]
        assert parameter_digest(policy, trainable_only=True) == initial
        assert frozen_base_digest(policy) == frozen
        write_immutable_json(out / 'result.json', dict(status='passed_actual_GPU_no_update', checks=checks,
            trainer_sha256=sha256_file(Path(__file__)), unchanged_parameters=True, optimizer_steps=0,
            same_initial_reference_scores=True, initial_live_eval_first8_exact=True, seconds=time.time() - start))
        print('actual_GPU_smoke_passed', flush=True)
        return
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0)
    by_key = {preference_key(r): r for r in preferences}
    by_id = {r['audio_id']: r for r in real}
    history = []
    with (out / 'exposures.jsonl').open('x') as handle:
        for step, window in enumerate(windows, 1):
            optimizer.zero_grad(set_to_none=True)
            terms, margins = Counter(), []
            for e in window:
                if e['objective'] == 'real_symmetric_kl':
                    row = by_id[e['audio_id']]
                    values = transcript_distributions(policy, move_batch(collator([row], include_rejected=False), device, dtype))
                    loss = symmetric_kl(values, teacher[row['audio_id']].to(device))
                    term = 'real_KL'
                else:
                    row = by_key[e['preference_key']]
                    logp, n = batch_log_probs(policy, move_batch(collator([row]), device, dtype))
                    ref = cache[e['preference_key']].to(device)
                    assert torch.equal(n.cpu(), counts[e['preference_key']])
                    loss, margin = mean_preference_loss(logp, ref, n, objective=args.mode, beta=1.)
                    margins.append(margin.detach().item())
                    term = row['source'] + '_' + args.mode
                weighted = loss * e['loss_weight']
                assert torch.isfinite(weighted)
                weighted.backward()
                terms[term] += weighted.item()
                handle.write(json.dumps(e) + '\n')
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            assert torch.isfinite(norm)
            optimizer.step(); handle.flush()
            item = dict(step=step, weighted_losses=dict(terms), gradient_norm=norm.item(),
                        mean_margin=sum(margins) / len(margins), maximum_margin=max(margins))
            history.append(item)
            print(json.dumps(item), flush=True)
    final = parameter_digest(policy, trainable_only=True)
    assert final != initial and frozen_base_digest(policy) == frozen
    assert rows(out / 'exposures.jsonl') == [e for w in windows for e in w]
    write_immutable_json(out / 'training_result.json', dict(status='complete', steps=4, history=history,
        policy_trainable_sha256_before=initial, policy_trainable_sha256_after=final,
        frozen_base_sha256=frozen, real_synthetic_and_KL_exposures_verified=True,
        seconds=time.time() - start, strict_deterministic_algorithms=True))
    del optimizer, teacher, values, loss, weighted
    policy.zero_grad(set_to_none=True)
    gc.collect(); torch.cuda.empty_cache()
    eval_start = time.time()
    passed = evaluate_live(policy, processor, out, base, parent)
    storage = retain_passing_weights(policy, out) if passed else dict(
        weights_saved=False, reason='Failed original gate; preregistered new-pilot retention policy')
    assert sha256_directory(Path(parent)) == protocol['parent_sha256']
    assert all(sha256_file(Path(p)) == h for p, h in protocol['source_sha256'].items())
    write_immutable_json(out / 'result.json', dict(status='training_and_evaluation_complete',
        passes_original_gate=passed, storage=storage, promoted=False,
        evaluation_seconds=time.time() - eval_start, total_seconds=time.time() - start))
    print(json.dumps(dict(status='complete', arm=args.mode, passes_original_gate=passed)), flush=True)


if __name__ == '__main__':
    main()
