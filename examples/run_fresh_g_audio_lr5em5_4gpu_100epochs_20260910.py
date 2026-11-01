"""G LR 5e-5 for one hundred full epochs; four GPUs, bounded checkpoint storage."""
import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/qwen3_asr_fresh_G_lr5em5_4gpu_seed42_100epochs_20260910_v1')
PREVIOUS = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/qwen3_asr_fresh_lora_sft_seed42_EFGH_20260910_v1')
NAME = 'G_audio_lr5em5_4gpu_100epochs'
OUT = ROOT / NAME
WORLD = 4
EPOCHS = 100
MILESTONE_EPOCHS = {3, 10, 30, 60, 100}
TOTAL_STEPS = 67 * EPOCHS
EVAL_STEPS = [epoch*67 + offset for epoch in range(EPOCHS) for offset in (14,27,40,54,67)]
os.environ['TMPDIR'] = str(ROOT / 'tmp')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('MKL_NUM_THREADS', '3')

import torch
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup, set_seed
import run_fresh_lora_sft_efgh_20260910 as old

CFG = dict(old.CONFIGS['G_audio_lr5em5'], lr=5e-5)
read, save, sha, event = old.read, old.save, old.sha, old.event


def prepare():
    assert shutil.disk_usage(ROOT.parent).free > 0.95 * 10**9
    previous = read(PREVIOUS / 'protocol.json')
    for key in ('source_sha256', 'input_sha256', 'base_files_sha256'):
        assert all(sha(p) == h for p, h in previous[key].items()), key
    audio_hashes = read(PREVIOUS / 'audio_sha256.json')
    assert all(sha(p) == h for p, h in audio_hashes.items())
    ROOT.mkdir(exist_ok=False)
    (ROOT / 'tmp').mkdir()
    for name in ('train', 'dev', 'smoke_audio', 'audio_sha256'):
        shutil.copyfile(PREVIOUS / f'{name}.json', ROOT / f'{name}.json')
    orders = []
    for epoch in range(EPOCHS):
        indices = list(range(2117))
        old.random.Random(42+epoch).shuffle(indices)
        orders.append(indices)
    assert orders[:3] == read(PREVIOUS/'orders.json')
    preceding_ten = ROOT.parent/'qwen3_asr_fresh_G_lr2em4_4gpu_seed42_10epochs_20260910_v1'
    assert orders[:10] == read(preceding_ten/'orders.json')
    save(ROOT/'orders.json', orders)
    sources = [Path(__file__).resolve(), Path(__file__).with_name('run_fresh_g_audio_lr2em4_4gpu_10epochs_20260910.py').resolve(), Path(__file__).with_name('run_fresh_g_audio_lr2em4_4gpu_20260910.py').resolve(), Path(old.__file__).resolve(),
               *[Path(p) for p in previous['source_sha256'] if p != old.__file__]]
    sources = list(dict.fromkeys(sources))
    for p in sources:
        dst = ROOT / 'sources' / p.relative_to(old.REPO)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    save(ROOT / 'protocol.json', dict(
        base_model=str(old.BASE), initial_adapter=None, resume_from=None,
        config=CFG, rank=16, dropout=.05, seed=42, epochs=EPOCHS, total_steps=TOTAL_STEPS,
        real_audio=2117, dev_audio=511, global_batch=32, microbatch=4,
        world_size=4, microbatches_per_rank_full_step=2, final_epoch_batch=5,
        optimizer='AdamW torch default (same as prior Trainer); betas=(0.9,0.999), eps=1e-8, weight_decay=0',
        schedule='linear over 6700 steps; same 5 warmup steps', warmup_steps=5, max_grad_norm=1,
        checkpoint_retention='All 500 evaluations retained; immutable epoch 3/10/30/60/100 adapters, rolling best adapter, rolling latest epoch adapter with optimizer/scheduler and rank RNG states. Previous experiments are not modified.',
        evaluation_steps=EVAL_STEPS,
        objective='Maintained utterance-mean CE; local loss sums / actual GLOBAL audio count; SUM gradients across ranks, no additional divide.',
        training_distribution='Original contiguous microbatches of 4 assigned round-robin to ranks; last 5 = rank0:4, rank1:1, rank2/3:0; no padding or duplicate audio.',
        numerical_difference='Global optimizer batches, microbatch contents and initialization match G; rank RNG streams use 42+rank, so dropout and reduction rounding are not bitwise identical to single GPU.',
        primary_evaluation='Original contiguous batches of 8 distributed round-robin; BF16 merged, greedy Cantonese, empty context, cap256; exact base restore.',
        comparison=dict(root=str(PREVIOUS), config='G_audio_lr5em5', lr=5e-5,
                        protocol_sha256=sha(PREVIOUS / 'protocol.json'),
                        intended_change='LR 5e-5, one hundred epochs from identical fresh initialization; linear decay horizon 6700 steps. First ten epoch orders match the preceding LR2e-4 run. Both LR and horizon differ; do not attribute results to LR alone.',
                        previous_ten_epoch_run=str(preceding_ten),
                        previous_ten_epoch_protocol_sha256=sha(preceding_ten/'protocol.json')),
        source_sha256={str(p): sha(p) for p in sources},
        input_sha256={str(ROOT / f'{n}.json'): sha(ROOT / f'{n}.json')
                      for n in ('train', 'dev', 'orders', 'smoke_audio', 'audio_sha256')},
        base_files_sha256=previous['base_files_sha256'], disk_estimate_additional_GB=0.8))
    event(ROOT, 'prepared', config=NAME, world_size=WORLD, epochs=EPOCHS, audio=2117,
          global_batch=32, steps=TOTAL_STEPS, evaluation_steps=EVAL_STEPS)


def gather(value):
    values = [None] * WORLD
    dist.all_gather_object(values, value)
    return values


def flat_gradient(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in params])


def install_gradient(params, flat):
    offset = 0
    for p in params:
        p.grad = flat[offset:offset+p.numel()].view_as(p)
        offset += p.numel()
    assert offset == flat.numel()


def backward(wrapper, data, rank, distributed=True):
    """Use the existing loss directly; keep the original four-audio microbatches."""
    params = [p for p in wrapper.model.parameters() if p.requires_grad]
    for p in params:
        p.grad = None
    collate = old.collator_for(wrapper, trace=True)
    total = torch.zeros((), device='cuda', dtype=torch.float32)
    observed = []
    for batch_index, start in enumerate(range(0, len(data), 4)):
        if distributed and batch_index % WORLD != rank:
            continue
        batch = data[start:start+4]
        inputs = collate(batch)
        observed.extend((start+i, uid) for i, uid in enumerate(inputs.pop('_audit_ids')))
        inputs = {k: v.to('cuda', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                  for k, v in inputs.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            # This maintained method has no dependency on Trainer instance state.
            loss = old.QwenASRLoRATrainer.compute_loss(
                None, wrapper.model, inputs, num_items_in_batch=len(data))
        loss.backward()
        total += loss.detach()
        del inputs, loss
    gradient = flat_gradient(params)
    if distributed:
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        observed = sorted(item for part in gather(observed) for item in part)
    assert observed == [(i, r['utt_id']) for i, r in enumerate(data)]
    assert torch.isfinite(gradient).all() and torch.isfinite(total)
    install_gradient(params, gradient)
    return float(total), gradient, [uid for _, uid in observed]


def all_metrics(predictions):
    def metric(xs):
        return dict(audio=len(xs), errors=sum(x['errors'] for x in xs),
                    chars=sum(x['chars'] for x in xs),
                    CER_percent=100*sum(x['errors'] for x in xs)/sum(x['chars'] for x in xs))
    return dict(**metric(predictions),
                vlink247=metric([x for x in predictions if x['speaker_id']=='vlink247']),
                other=metric([x for x in predictions if x['speaker_id']!='vlink247']),
                **{b: metric([x for x in predictions if x['zero_shot_bucket']==b])
                   for b in ('easy', 'medium', 'hard')})


def evaluate(wrapper, data, out, rank, merged=True):
    start = time.monotonic()
    cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    local = [row for i in range(0, len(data), 8) if (i//8) % WORLD == rank
             for row in data[i:i+8]]
    shard = out / f'rank-{rank}'
    if merged:
        old.merged_evaluation(wrapper, local, shard)
    else:
        old.decode(wrapper, local, shard)
    pieces = gather(read(shard / 'predictions.json'))
    if rank == 0:
        flat = [row for part in pieces for row in part]
        assert len(flat) == len(data) == len({x['utt_id'] for x in flat})
        by_id = {row['utt_id']: row for row in flat}
        predictions = [by_id[row['utt_id']] for row in data]
        save(out / 'predictions.json', predictions)
        save(out / 'summary.json', dict(**all_metrics(predictions), seconds=time.monotonic()-start,
             provenance='Four rank decoding receipts; original batch-of-8 membership preserved.',
             shard_predictions_sha256={f'rank-{r}': sha(out/f'rank-{r}'/'predictions.json') for r in range(WORLD)}))
    dist.barrier()
    torch.random.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    return read(out / 'summary.json')


def frozen_digest(model):
    return old.tensor_digest((n,p) for n,p in model.named_parameters()
                             if '.lora_' in n and not p.requires_grad)


def adapter_digest(model):
    return old.tensor_digest((n,p) for n,p in model.named_parameters() if '.lora_' in n)


def smoke(wrapper, rank, train, order, base_hash, initial_hash):
    params = [p for p in wrapper.model.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in params]
    frozen = frozen_digest(wrapper.model)
    checks = []
    wrapper.model.eval()  # Disable dropout only for the gradient equivalence check.
    for count in (32, 5):
        data = [train[i] for i in order[:count]]
        parallel_loss, parallel_grad, _ = backward(wrapper, data, rank)
        if rank == 0:
            serial_loss, serial_grad, _ = backward(wrapper, data, rank, distributed=False)
            relative = float((serial_grad-parallel_grad).norm()/serial_grad.norm())
            delta = float((serial_grad-parallel_grad).abs().max())
            assert relative < 1e-4 and abs(serial_loss-parallel_loss) < 1e-5
            torch.testing.assert_close(serial_grad, parallel_grad, rtol=2e-4, atol=2e-6)
            check = dict(audio=count, parallel_loss=parallel_loss, serial_loss=serial_loss,
                         gradient_relative_l2=relative, gradient_max_abs=delta, passed=True)
            checks.append(check)
            event(ROOT, 'gradient_equivalence', **check)
        dist.barrier()
    optimizer = torch.optim.AdamW(params, lr=CFG['lr'], betas=(.9,.999), eps=1e-8,
                                 weight_decay=0)
    wrapper.model.train()
    set_seed(42+rank)
    data = [train[i] for i in order[:31]] + [max(train,key=lambda r:float(r['duration']))]
    for step, batch in enumerate((data, data[:5]), 1):
        loss, gradient, _ = backward(wrapper, batch, rank)
        assert gradient.norm() > 0
        assert all(p.grad is None for n,p in wrapper.model.named_parameters() if not p.requires_grad)
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            event(ROOT, 'smoke_loss', step=step, loss=loss, gradient_norm=float(norm))
    assert any(not torch.equal(p, a) for p,a in zip(params,initial))
    assert frozen_digest(wrapper.model) == frozen and old.base_digest(wrapper.model) == base_hash
    updated = gather(adapter_digest(wrapper.model))
    assert len(set(updated)) == 1
    # Restore the exact fresh initialization; no smoke update enters formal training.
    with torch.no_grad():
        for p, a in zip(params,initial):
            p.copy_(a)
    assert adapter_digest(wrapper.model) == initial_hash
    save(ROOT / f'rank-{rank}' / 'smoke_passed.json', dict(passed=True, rank=rank,
         base_frozen=True, language_lora_frozen=True, audio_lora_updated=True,
         post_smoke_restored_exactly=True, actual_tail_audio=5,
         trainable_parameters=sum(p.numel() for p in params),
         peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
    dist.barrier()
    if rank == 0:
        save(ROOT / 'smoke_passed.json', dict(passed=True, gradient_checks=checks,
             updated_adapter_identical_across_ranks=True,
             rank_receipts=[read(ROOT/f'rank-{r}'/'smoke_passed.json') for r in range(WORLD)]))
        event(ROOT, 'smoke_complete', passed=True, world_size=WORLD)
    del optimizer, initial, parallel_grad, gradient
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()


def worker(rank, port):
    torch.set_num_threads(3)
    torch.cuda.set_device(0)  # Each worker sees exactly one physical GPU.
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=WORLD, timeout=timedelta(minutes=5))
    protocol = read(ROOT / 'protocol.json')
    assert all(sha(p)==h for p,h in protocol['source_sha256'].items())
    wrapper, base_hash = old.build(CFG)
    initial_hash = adapter_digest(wrapper.model)
    previous = read(PREVIOUS/'G_audio_lr5em5'/'smoke_passed.json')
    assert base_hash == previous['raw_base_tensor_sha256']
    assert initial_hash == previous['fresh_initial_adapter_sha256']
    assert len(set(gather((base_hash,initial_hash)))) == 1
    if rank == 0:
        save(OUT/'initialization.json', dict(raw_base_tensor_sha256=base_hash,
             initial_adapter_sha256=initial_hash, initial_adapter=None,
             matches_previous_G_exactly=True, world_size=WORLD,
             trainable_parameters=sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad),
             trainable_names=[n for n,p in wrapper.model.named_parameters() if p.requires_grad]))
        event(ROOT, 'initialization_verified', matches_previous_G=True, world_size=WORLD)
    train, dev, orders = read(ROOT/'train.json'), read(ROOT/'dev.json'), read(ROOT/'orders.json')
    smoke(wrapper, rank, train, orders[0], base_hash, initial_hash)
    evaluate(wrapper, dev, ROOT/'base_evaluation', rank)
    if rank == 0:
        assert read(ROOT/'base_evaluation'/'predictions.json') == read(PREVIOUS/'base_evaluation'/'predictions.json')
        event(ROOT, 'base_evaluation_verified', all_511_predictions_match=True)
    dist.barrier()
    assert adapter_digest(wrapper.model) == initial_hash
    params = [p for p in wrapper.model.parameters() if p.requires_grad]
    frozen = frozen_digest(wrapper.model)
    optimizer = torch.optim.AdamW(params, lr=CFG['lr'], betas=(.9,.999), eps=1e-8,
                                 weight_decay=0)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=TOTAL_STEPS)
    wrapper.model.train()
    wrapper.model.config.use_cache = False
    set_seed(42+rank)
    start = time.monotonic()
    if rank == 0:
        event(OUT, 'training_started', world_size=WORLD, epochs=EPOCHS, steps=TOTAL_STEPS, lr=CFG['lr'],
              initial_model='raw Qwen3-ASR + fresh audio-only LoRA')
    all_observed = []
    best_cer, best_step = float('inf'), None
    for epoch, order in enumerate(orders):
        epoch_observed = []
        for offset in range(0, len(order), 32):
            step = epoch*67 + offset//32 + 1
            batch = [train[i] for i in order[offset:offset+32]]
            tick = time.monotonic()
            loss, gradient, actual_ids = backward(wrapper, batch, rank)
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            lr = optimizer.param_groups[0]['lr']
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_observed.extend(actual_ids)
            all_observed.extend(actual_ids)
            if rank == 0:
                event(OUT, 'loss', step=step, epoch=epoch+len(epoch_observed)/2117,
                      loss=loss, grad_norm=float(norm), learning_rate=lr,
                      actual_batch=len(batch), actual_audio_exposures=len(all_observed),
                      seconds=time.monotonic()-tick)
            if step in EVAL_STEPS:
                checkpoint = OUT/f'checkpoint-{step}'
                if rank == 0:
                    if step % 67 == 0 and step//67 in MILESTONE_EPOCHS:
                        wrapper.model.save_pretrained(checkpoint)
                        save(checkpoint/'provenance.json', dict(step=step, config=CFG,
                             world_size=WORLD, initial_adapter=None, raw_base_tensor_sha256=base_hash,
                             actual_audio_exposures=len(all_observed), source_sha256=protocol['source_sha256']))
                    event(OUT, 'evaluation_started', step=step)
                metrics = evaluate(wrapper, dev, OUT/'evaluations'/f'step-{step}', rank)
                if rank == 0:
                    event(OUT, 'evaluation_complete', step=step, epoch=epoch+len(epoch_observed)/2117,
                          actual_audio_exposures=len(all_observed), metrics=metrics)
                    if metrics['CER_percent'] < best_cer:
                        best = OUT/'best-checkpoint'
                        wrapper.model.save_pretrained(best)
                        save(best/'provenance.json', dict(step=step, config=CFG,
                             world_size=WORLD, initial_adapter=None, raw_base_tensor_sha256=base_hash,
                             actual_audio_exposures=len(all_observed), source_sha256=protocol['source_sha256'],
                             CER_percent=metrics['CER_percent'], replaced_best_step=best_step))
                        event(OUT, 'best_checkpoint_updated', step=step, replaced_step=best_step,
                              CER_percent=metrics['CER_percent'], path=str(best))
                        best_cer, best_step = metrics['CER_percent'], step
        assert epoch_observed == [train[i]['utt_id'] for i in order]
        assert len(set(epoch_observed)) == 2117
        rng_states = gather(dict(rank=rank, cpu=torch.random.get_rng_state(), cuda=torch.cuda.get_rng_state().cpu()))
        if rank == 0:
            latest = OUT/'latest-checkpoint'
            wrapper.model.save_pretrained(latest)
            state_path = latest/'training_state.pt'
            state_pending = latest/'training_state.pt.pending'
            torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            step=step, epoch=epoch+1, rng_per_rank=rng_states,
                            actual_audio_exposures=len(all_observed)), state_pending)
            state_pending.replace(state_path)
            save(latest/'provenance.json', dict(step=step, epoch=epoch+1, config=CFG,
                 world_size=WORLD, initial_adapter=None, raw_base_tensor_sha256=base_hash,
                 actual_audio_exposures=len(all_observed), source_sha256=protocol['source_sha256'],
                 adapter_sha256=sha(latest/'adapter_model.safetensors'),
                 training_state_sha256=sha(state_path), retention='Rolling most recent completed epoch'))
            event(OUT, 'latest_checkpoint_updated', step=step, epoch=epoch+1,
                  path=str(latest), optimizer_scheduler_rng_saved=True)
            save(OUT/f'epoch-{epoch+1}-exposure.json', dict(epoch=epoch+1, audio=2117,
                 unique_audio=2117, utt_ids=epoch_observed, shared_order_exact=True,
                 provenance='Actual collator IDs gathered from all ranks after each optimizer batch; no sampler padding.'))
            event(OUT, 'epoch_coverage_verified', epoch=epoch+1, audio=2117, unique_audio=2117)
    assert len(all_observed) == 2117*EPOCHS
    assert old.base_digest(wrapper.model) == base_hash and frozen_digest(wrapper.model) == frozen
    assert len(set(gather(adapter_digest(wrapper.model)))) == 1
    if rank == 0:
        torch.save(optimizer.state_dict(), OUT/'final_optimizer.pt')
        torch.save(scheduler.state_dict(), OUT/'final_scheduler.pt')
    final = evaluate(wrapper, dev, OUT/'final_unmerged_evaluation', rank, merged=False)
    assert all(sha(p)==h for p,h in protocol['source_sha256'].items())
    if rank == 0:
        save(OUT/'complete.json', dict(passed=True, steps=TOTAL_STEPS, epochs=EPOCHS,
             actual_audio_exposures=2117*EPOCHS, all_epoch_orders_verified=True,
             base_unchanged=True, frozen_lora_unchanged=True, world_size=WORLD,
             final_adapters_equal_across_ranks=True, seconds=time.monotonic()-start,
             final_unmerged_metrics=final))
        event(OUT, 'complete', epochs=EPOCHS, steps=TOTAL_STEPS)
    dist.barrier()
    dist.destroy_process_group()


def supervise():
    assert ROOT.exists() and not (ROOT/'supervisor.json').exists()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    processes, handles = [], []
    save(ROOT/'supervisor.json', dict(pid=os.getpid(), port=port))
    try:
        for rank in range(WORLD):
            handle = (ROOT/f'rank-{rank}.log').open('w')
            handles.append(handle)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), PYTHONDONTWRITEBYTECODE='1')
            p = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker',
                                  '--rank', str(rank), '--port', str(port)],
                                 stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            processes.append(p)
            event(ROOT, 'process_started', rank=rank, gpu=rank, pid=p.pid)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('A four-GPU worker failed; stopping this experiment.')
            time.sleep(2)
        assert all(p.returncode == 0 for p in processes)
        assert read(OUT/'complete.json')['passed']
        save(ROOT/'complete.json', dict(passed=True, epochs=EPOCHS, world_size=WORLD))
        event(ROOT, 'experiment_complete')
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    torch.set_num_threads(3)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'supervise', 'worker'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    try:
        if args.mode == 'prepare':
            prepare()
        elif args.mode == 'supervise':
            supervise()
        else:
            worker(args.rank, args.port)
    except Exception as exc:
        event(ROOT if args.mode != 'worker' else ROOT/f'rank-{args.rank}', 'failure',
              mode=args.mode, error=repr(exc))
        raise
