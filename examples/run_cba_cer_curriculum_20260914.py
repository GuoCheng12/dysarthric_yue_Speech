"""CBA-inspired CER filtering on Qwen3-ASR; not the original SemScore system."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_recordings(train_ids, initial, current, threshold_percent=50.0):
    """Exclude only persistently high-CER items; complete Train is reconsidered."""
    allowed = set(train_ids)
    assert len(allowed) == len(train_ids) > 0
    scores = []
    for predictions in (initial, current):
        assert len(predictions) == len(allowed)
        assert {p['utt_id'] for p in predictions} == allowed
        assert all(p['chars'] > 0 and p['errors'] >= 0 for p in predictions)
        scores.append({p['utt_id']: 100 * p['errors'] / p['chars'] for p in predictions})
    before, after = scores
    return [uid for uid in train_ids
            if not (before[uid] > threshold_percent and after[uid] > threshold_percent)]


def draw_selected(selected, count, rng):
    assert selected and len(set(selected)) == len(selected)
    result = []
    while len(result) < count:
        sweep = list(selected)
        rng.shuffle(sweep)
        result.extend(sweep[:count - len(result)])
    return result


def load_splits(protocol):
    parts = [[json.loads(l) for l in Path(protocol[k]).read_text().splitlines() if l.strip()]
             for k in ('train_manifest', 'feedback_manifest', 'test_manifest')]
    ids = [{r['utt_id'] for r in part} for part in parts]
    assert all(len(part) == len(keys) > 0 for part, keys in zip(parts, ids))
    assert all(not ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    texts = [{r['clean_gt'] for r in part} for part in parts]
    patients = [{r['speaker_id'] for r in part} for part in parts]
    assert all(r['prompt'] == '' for part in parts for r in part)
    if protocol['setting'] == 'E':
        assert texts[0] == texts[1] == texts[2]
        assert all(not patients[i] & patients[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    else:
        assert protocol['setting'] == 'D'
        assert all(not texts[i] & texts[j] for i, j in ((0, 1), (0, 2), (1, 2)))
        assert patients[1] <= patients[0] and patients[2] <= patients[0]
    return parts


def launch(args):
    protocol = read(args.protocol)
    root = Path(protocol['root'])
    assert not (root / 'started.json').exists(), 'Use a fresh output directory.'
    load_splits(protocol)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'tmp').mkdir(exist_ok=True)
    save(root / 'started.json', {'unix': time.time(), 'gpus': args.gpus,
                                'protocol_sha256': sha(args.protocol)})
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    processes, logs = [], []
    try:
        for rank, device in enumerate(args.gpus.split(',')):
            log = (root / f'rank-{rank}.log').open('w'); logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, TMPDIR=str(root / 'tmp'), TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS='3', MKL_NUM_THREADS='3')
            command = [sys.executable, str(Path(__file__).resolve()), 'worker',
                       '--protocol', str(Path(args.protocol).resolve()), '--rank', str(rank),
                       '--world', str(len(args.gpus.split(','))), '--port', str(port)]
            processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            failed = [p.returncode for p in processes if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'Worker failed: {failed}')
            time.sleep(2)
        assert all(p.returncode == 0 for p in processes)
    finally:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            p.wait()
        for log in logs: log.close()


def worker(args):
    import torch
    import torch.distributed as dist
    from transformers import get_linear_schedule_with_warmup, set_seed
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    import run_fresh_setting_e_audio_lr2em4_4gpu_50epochs_20260911 as gpu
    from therapist_harness.sampling_curriculum_runtime import exposure_summary
    from therapist_harness.phonology import align_characters
    from therapist_harness.text import text_normalize

    protocol = read(args.protocol)
    out = Path(protocol['root'])
    assert all(sha(p) == value for p, value in protocol['source_sha256'].items())
    assert all(sha(p) == value for p, value in protocol['input_sha256'].items())
    os.environ['TMPDIR'] = str(out / 'tmp'); Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
    gpu.ROOT = gpu.OUT = out; gpu.WORLD = args.world
    gpu.CFG = dict(gpu.old.CONFIGS['G_audio_lr5em5'], lr=2e-4)
    torch.set_num_threads(3); torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{args.port}',
                           rank=args.rank, world_size=args.world, timeout=timedelta(minutes=30))
    train, dev, test = load_splits(protocol)
    if protocol['smoke']:
        train, dev, test = [part[:16] for part in (train, dev, test)]
    train_by_id = {r['utt_id']: r for r in train}
    rng = random.Random(protocol['seed'])
    started = time.monotonic()
    wrapper, base_hash = gpu.old.build(gpu.CFG)
    initial_adapter, frozen = gpu.adapter_digest(wrapper.model), gpu.frozen_digest(wrapper.model)
    assert len(set(gpu.gather((base_hash, initial_adapter)))) == 1
    params = [p for p in wrapper.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)
    total = protocol['total_optimizer_steps_limit']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=protocol['warmup_steps'],
                                               num_training_steps=total)
    set_seed(protocol['seed'] + args.rank)

    def log(stage, **fields):
        if args.rank == 0:
            gpu.event(out, stage, wall_seconds=time.monotonic()-started, **fields)

    def evaluate(data, step, split):
        directory = out / f'evaluations/step-{step}/{split}'
        metrics = gpu.evaluate(wrapper, data, directory, args.rank)
        metrics['provenance'] = f'{args.world} ranks; canonical batch-of-eight membership'
        if args.rank == 0: save(directory / 'summary.json', metrics)
        return metrics, read(directory / 'predictions.json')

    def checkpoint(name, step):
        if args.rank == 0:
            wrapper.model.save_pretrained(out / name)
            save(out / name / 'step.json', {'step': step})
        dist.barrier()

    if args.rank == 0:
        save(out / 'initialization.json', dict(base_tensor_sha256=base_hash,
             fresh_adapter_sha256=initial_adapter, seed=42, world_size=args.world,
             trainable_parameters=sum(p.numel() for p in params)))
    initial_metrics, initial = evaluate(train, 0, 'train')
    current = initial
    dev_metrics, _ = evaluate(dev, 0, 'dev')
    best, best_step = dev_metrics['patient_equal_CER_percent'], 0
    checkpoint('checkpoint-best', 0)
    log('initial_evaluation', train_CER=initial_metrics['CER_percent'], dev=dev_metrics)
    step, all_exposure = 0, []
    for stage, stage_start in enumerate(range(0, total, protocol['steps_per_stage']), 1):
        directory = out / f'stage-{stage:03d}'
        stage_steps = min(protocol['steps_per_stage'], total - stage_start)
        selected = list(train_by_id) if stage == 1 else select_recordings(
            list(train_by_id), initial, current, protocol['filter_cer_threshold_percent'])
        assert selected, 'CBA filtering selected no recordings.'
        ids = draw_selected(selected, stage_steps * 32, rng)
        assert len(set(gpu.gather(hashlib.sha256(json.dumps(ids).encode()).hexdigest()))) == 1
        if args.rank == 0:
            save(directory / 'selection.json', dict(step=step, selected_ids=selected,
                 excluded_ids=[uid for uid in train_by_id if uid not in set(selected)],
                 initial_scores=str(out / 'evaluations/step-0/train/predictions.json'),
                 current_scores=str(out / f'evaluations/step-{step}/train/predictions.json'),
                 rule='stage 1: full pool; thereafter exclude iff initial CER >50% AND current CER >50%',
                 threshold_percent=protocol['filter_cer_threshold_percent']))
            save(directory / 'sampling_plan.json', dict(utterance_ids=ids, stage_steps=stage_steps))
        log('stage_started', stage_index=stage, step=step, selected=len(selected), excluded=len(train)-len(selected))
        wrapper.model.train(); wrapper.model.config.use_cache = False
        observed = []
        for offset in range(0, len(ids), 32):
            loss, gradient, actual = gpu.backward(wrapper, [train_by_id[u] for u in ids[offset:offset+32]], args.rank)
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            lr = optimizer.param_groups[0]['lr']
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            step += 1; observed.extend(actual); all_exposure.extend(actual)
            log('loss', step=step, loss=loss, gradient_norm=float(norm), learning_rate=lr)
        assert observed == ids and scheduler.last_epoch == step and len(all_exposure) == step*32
        assert gpu.frozen_digest(wrapper.model) == frozen
        dev_metrics, _ = evaluate(dev, step, 'dev')
        if dev_metrics['patient_equal_CER_percent'] < best:
            best, best_step = dev_metrics['patient_equal_CER_percent'], step
            checkpoint('checkpoint-best', step)
        checkpoint('checkpoint-latest', step)
        if args.rank == 0:
            save(directory / 'actual_exposure.json', exposure_summary(observed, train_by_id))
            save(out / 'selected_checkpoint.json', dict(step=best_step, dev_patient_equal_CER_percent=best))
        log('stage_complete', step=step, dev=dev_metrics, best_step=best_step)
        if step < total:
            _, current = evaluate(train, step, 'train')

    assert gpu.old.base_digest(wrapper.model) == base_hash
    assert gpu.adapter_digest(wrapper.model) != initial_adapter
    assert len(set(gpu.gather(gpu.adapter_digest(wrapper.model)))) == 1
    endpoint, _ = evaluate(test, step, 'test-endpoint')
    weights = load_file(str(out / 'checkpoint-best/adapter_model.safetensors'))
    live = get_peft_model_state_dict(wrapper.model)
    assert live.keys() == weights.keys()
    with torch.no_grad():
        for name in live: live[name].copy_(weights[name])
    selected_metrics, predictions = evaluate(test, best_step, 'test-dev-selected')
    if args.rank == 0:
        def hard_breakdown(preds):
            canonical = {r['utt_id']: r for r in test}
            hard = [p for p in preds if p['zero_shot_bucket'] == 'hard']
            counts = dict(S=0, D=0, I=0); chars = 0
            for p in hard:
                reference = text_normalize(canonical[p['utt_id']]['clean_gt'])
                alignment = align_characters(reference, text_normalize(p['prediction']))
                for key, attr in [('S','substitution_count'),('D','deletion_count'),('I','insertion_count')]:
                    counts[key] += getattr(alignment, attr)
                chars += len(reference)
            return dict(counts=counts, reference_characters=chars,
                        rates={k:100*v/chars for k,v in counts.items()} if chars else None)
        endpoint_preds = read(out / f'evaluations/step-{step}/test-endpoint/predictions.json')
        save(out / 'complete.json', dict(smoke=protocol['smoke'], steps=step, exposures=len(all_exposure),
             best_step=best_step, endpoint=endpoint, dev_selected=selected_metrics,
             endpoint_hard_errors=hard_breakdown(endpoint_preds), selected_hard_errors=hard_breakdown(predictions),
             total_exposure=exposure_summary(all_exposure, train_by_id),
             wall_seconds=time.monotonic()-started, world_size=args.world,
             base_restored=True, frozen_adapters_unchanged=True, trainable_adapters_changed=True))
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['launch', 'worker'])
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=2)
    parser.add_argument('--port', type=int, default=29510)
    args = parser.parse_args()
    globals()[args.action](args)
