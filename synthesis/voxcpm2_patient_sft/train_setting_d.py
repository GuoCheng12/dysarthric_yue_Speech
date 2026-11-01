"""Setting D reference-conditioned LoRA; official VoxCPM2 model and packer."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.nn.utils.rnn import pad_sequence

SOURCE = Path('/data/qwen3-asr/third_party/VoxCPM2-probe')
BASE = Path('/data/qwen3-asr/models/tts/comparison24_v1/voxcpm2')
SG = Path('/data/qwen3-asr/records/cosyvoice3_reference_sft_setting_d_v1')
DATA = Path('/data/qwen3-asr/finetune/data_setting_d_v1')
ROOT = Path('/data/qwen3-asr/records/voxcpm2_reference_sft_setting_d_v1')
sys.path.insert(0, str(SOURCE / 'src'))
from voxcpm.model.voxcpm2 import VoxCPM2Model, LoRAConfig
from voxcpm.training.packers import AudioFeatureProcessingPacker

CONFIG = dict(seed=42, epochs=3, micro_batch=4, effective_batch=16,
              lr=1e-4, weight_decay=.01, warmup_steps=40, clip_norm=1.,
              eval_every_steps=66, eval_batch=2, eval_seed=20260906,
              lora=dict(enable_lm=True, enable_dit=True, enable_proj=False,
                        r=32, alpha=32, dropout=0.),
              input_sample_rate=16000, output_sample_rate=48000,
              minimum_free_bytes=20 * 1024**3)
KEYS = ('text_tokens', 'audio_feats', 'text_mask', 'audio_mask', 'loss_mask', 'labels')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def emit(event, **values):
    record = dict(event=event, time=time.time(), **values)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    with (ROOT / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    write(ROOT / 'status.json', record)


def check_space():
    free = shutil.disk_usage(ROOT).free
    assert free >= CONFIG['minimum_free_bytes'], f'Disk reserve reached: {free / 1024**3:.2f} GiB'
    return free


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_pairs(rows, epoch, seed=42):
    rng = random.Random(seed + epoch)
    pools = defaultdict(list)
    for row in rows:
        pools[row['speaker_id']].append(row['uid'])
    pairs = [{**r, 'ref_uid': rng.choice([u for u in pools[r['speaker_id']] if u != r['uid']])}
             for r in rows]
    rng.shuffle(pairs)
    return pairs


def prepare():
    assert not (ROOT / 'config.json').exists(), 'Preparation is already frozen'
    ROOT.mkdir(parents=True, exist_ok=True)
    check_space()
    sg_audio = dict(line.rstrip().split(' ', 1) for line in (SG / 'data/train/wav.scp').open(encoding='utf-8'))
    sg_text = dict(line.rstrip().split(' ', 1) for line in (SG / 'data/train/text').open(encoding='utf-8'))
    split_rows, sources = {}, {}
    for split in ('train', 'dev', 'test'):
        path = DATA / f'setting_d_{split}.jsonl'
        sources[str(path)] = digest(path)
        split_rows[split] = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    excluded = []
    rows = []
    for split, values in split_rows.items():
        for original in values:
            uid = original['utt_id'].replace('/', '__')
            if split == 'train' and uid not in sg_audio:
                excluded.append(original['utt_id'])
                continue
            if split == 'train':
                assert original['audio'] == sg_audio[uid]
                assert original['clean_gt'] == sg_text[uid]
            assert original['speaker_id'].split('_')[0] not in {'vlink201', 'vlink203', 'vlink225', 'vlink243'}
            assert original['utt_id'] != 'vlink211/7290411755587742893228'
            info = sf.info(original['audio'])
            rows.append({**original, 'uid': uid, 'original_split': split,
                         'partition': 'train' if split == 'train' else 'merged_test',
                         'seconds': info.duration, 'wav_sha256': digest(original['audio'])})
    train = [r for r in rows if r['partition'] == 'train']
    test = [r for r in rows if r['partition'] == 'merged_test']
    assert len(train) == len(sg_audio) == 2108 and len(test) == 511 and len(excluded) == 9
    assert len({r['speaker_id'] for r in train}) == 57
    assert len({r['uid'] for r in rows}) == len(rows)
    leakage = {}
    for key in ('prompt_id', 'prompt_normalized_sha16', 'prompt_jyutping_sha16', 'clean_gt', 'wav_sha256'):
        overlap = {r[key] for r in train} & {r[key] for r in test}
        assert not overlap, (key, overlap)
        leakage[key] = len(overlap)
    ref_path = SG / 'therapist_reference_manifest_v1.json'
    ref_info = {r['speaker_id']: r for r in read(ref_path)['references']}
    audio_to_uid = {r['audio']: r['uid'] for r in train}
    for r in rows:
        r['patient_group'] = ref_info[r['speaker_id']]['difficulty_group']
    for r in test:
        ref = ref_info[r['speaker_id']]
        r['ref_uid'] = audio_to_uid[ref['reference_wav']]
        assert ref['reference_wav_sha256'] == next(t['wav_sha256'] for t in train if t['uid'] == r['ref_uid'])
    # Fixed, distinct patients: three easy and three hard, selected before tuning.
    samples = []
    rng = random.Random(42)
    for group in ('easy', 'hard'):
        speakers = sorted({r['speaker_id'] for r in test if r['patient_group'] == group})
        rng.shuffle(speakers)
        for speaker in speakers[:3]:
            candidates = sorted([r for r in test if r['speaker_id'] == speaker], key=lambda r: r['seconds'])
            samples.append(candidates[len(candidates) // 2])
    assert len(samples) == 6
    stats = {'train': {'utterances': len(train), 'patients': 57, 'hours': sum(r['seconds'] for r in train) / 3600},
             'merged_test': {'utterances': len(test), 'patients': len({r['speaker_id'] for r in test}),
                             'hours': sum(r['seconds'] for r in test) / 3600,
                             'original_splits': dict(Counter(r['original_split'] for r in test))},
             'excluded_singletons_as_in_sg': excluded, 'train_test_overlap': leakage,
             'duration_range': [min(r['seconds'] for r in rows), max(r['seconds'] for r in rows)],
             'source_hashes': {**sources, str(ref_path): digest(ref_path),
                               str(SG / 'data/train/wav.scp'): digest(SG / 'data/train/wav.scp')},
             'reference_policy': '100% same-patient distinct TRAIN clip; train resampled each epoch; eval uses SG frozen reference',
             'evaluation_protocol': 'old dev + test merged; monitored, not an untouched final test',
             'conditioning': '[103, ref audio, 104, target text, 101, target audio, 102]; target-only loss',
             'wave_processing': 'mono + 16kHz resampling + right zero pad only; no denoising, VAD trimming or CER filtering'}
    write(ROOT / 'config.json', CONFIG)
    write(ROOT / 'data.json', rows)
    write(ROOT / 'samples.json', samples)
    write(ROOT / 'preparation.json', stats)
    emit('prepared', **stats)


def load_model():
    seed_all(CONFIG['seed'])
    model = VoxCPM2Model.from_local(str(BASE), optimize=False, training=True, device='cuda',
                                  lora_config=LoRAConfig(**CONFIG['lora']))
    with safe_open(BASE / 'model.safetensors', framework='pt') as f:
        base_keys = set(f.keys())
    vae_state = torch.load(BASE / 'audiovae.pth', map_location='cpu', weights_only=True, mmap=True)['state_dict']
    base_keys |= {'audio_vae.' + k for k in vae_state}
    expected = set(model.state_dict()) - {n for n, _ in model.named_parameters() if 'lora_' in n}
    assert base_keys == expected, (base_keys - expected, expected - base_keys)
    assert model.audio_vae.sample_rate == 16000 and model.sample_rate == 48000
    assert all('lora_' in n for n, p in model.named_parameters() if p.requires_grad)
    assert not any(p.requires_grad for p in model.audio_vae.parameters())
    return model.to('cuda')


class CachedPacker(AudioFeatureProcessingPacker):
    """Only replace frozen VAE extraction; retain official reference/target layout."""
    def extract_audio_feats(self, features):
        assert features.ndim == 3 and features.shape[1:] == (self.patch_size, self.feat_dim)
        return features.unsqueeze(0), features.shape[0] * self.patch_size / 25


def batch_for(rows, cache, tokens, packer):
    items = [packer.process_tts_data_with_ref(cache[r['ref_uid']], cache[r['uid']], tokens[r['uid']]) for r in rows]
    assert max(len(x[0]) for x in items) <= packer.max_len, 'No training audio may be truncated'
    batch = {key: pad_sequence([item[i] for item in items], batch_first=True).to('cuda')
             for i, key in enumerate(KEYS)}
    batch['position_ids'] = torch.arange(batch['text_tokens'].shape[1], device='cuda')[None].expand(len(rows), -1)
    return batch


def cache_features():
    import librosa
    assert not (ROOT / 'features.safetensors').exists()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    model = load_model().eval()
    rows = read(ROOT / 'data.json')
    packer = AudioFeatureProcessingPacker(1, model.config.max_length, model.patch_size, model.config.feat_dim, model.audio_vae)
    cache, tokens = {}, {}
    start = time.monotonic()
    with torch.inference_mode():
        for i, r in enumerate(rows):
            wave, sr = librosa.load(r['audio'], sr=16000, mono=True)
            assert sr == model.audio_vae.sample_rate
            raw = torch.from_numpy(wave).cuda()
            feats, _ = packer.extract_audio_feats(raw)
            cache[r['uid']] = feats[0].cpu().contiguous()
            tokens[r['uid']] = model.text_tokenizer(r['clean_gt'])
            assert tokens[r['uid']]
            if i == 0:
                direct = model._encode_wav(r['audio'], trim_silence_vad=False)
                encoder_max_abs_error = float((direct - cache[r['uid']]).abs().max())
                torch.testing.assert_close(direct, cache[r['uid']], rtol=1e-5, atol=1e-5)
            if (i + 1) % 100 == 0:
                emit('caching', completed=i + 1, total=len(rows), wall_seconds=time.monotonic() - start)
    save_file(cache, ROOT / 'features.safetensors')
    write(ROOT / 'tokens.json', tokens)
    write(ROOT / 'cache_receipt.json', {'features_sha256': digest(ROOT / 'features.safetensors'),
          'bytes': (ROOT / 'features.safetensors').stat().st_size, 'utterances': len(cache),
          'raw_packer_matches_inference_encoder': True, 'encoder_max_abs_error': encoder_max_abs_error,
          'encoder_test_rtol': 1e-5, 'encoder_test_atol': 1e-5, 'vae_cudnn_tf32': False,
          'wall_seconds': time.monotonic() - start})
    emit('cached', utterances=len(cache), free_GiB=check_space() / 1024**3)


def training_objects():
    model = load_model()
    cache = load_file(ROOT / 'features.safetensors')
    tokens = {k: torch.tensor(v, dtype=torch.int32) for k, v in read(ROOT / 'tokens.json').items()}
    packer = CachedPacker(1, model.config.max_length, model.patch_size, model.config.feat_dim, model.audio_vae)
    return model, cache, tokens, packer


def weighted_loss(result, frames, total_frames):
    return (result['loss/diff'] + result['loss/stop']) * (frames / total_frames)


def benchmark(micro_batch):
    model, cache, tokens, packer = training_objects()
    rows = [r for r in read(ROOT / 'data.json') if r['partition'] == 'train']
    pairs = train_pairs(rows, 1)
    # Upper bound over SAME-patient reference choices, not just a favourable short batch.
    for r in pairs:
        r['ref_uid'] = max((x['uid'] for x in rows if x['speaker_id'] == r['speaker_id'] and x['uid'] != r['uid']),
                           key=lambda uid: cache[uid].shape[0])
    pairs.sort(key=lambda r: len(cache[r['uid']]) + len(cache[r['ref_uid']]) + len(tokens[r['uid']]), reverse=True)
    model.train()
    model.audio_vae.eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CONFIG['lr'])
    durations = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(3):
        optimizer.zero_grad(set_to_none=True)
        batch = batch_for(pairs[:micro_batch], cache, tokens, packer)
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(**batch)
            loss = output['loss/diff'] + output['loss/stop']
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.monotonic() - start)
    record = dict(micro_batch=micro_batch, padded_tokens=batch['text_tokens'].shape[1],
                  target_seconds=[r['seconds'] for r in pairs[:micro_batch]], seconds=durations,
                  peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3,
                  peak_reserved_GiB=torch.cuda.max_memory_reserved() / 1024**3,
                  finite_loss=float(loss.detach()), grad_norm=float(grad), discarded_benchmark_updates=True)
    write(ROOT / f'benchmark_b{micro_batch}.json', record)
    emit('benchmarked', **record)


@torch.no_grad()
def evaluate(model, rows, cache, tokens, packer):
    model.eval()
    sums = {'diff': 0., 'stop': 0., 'frames': 0}
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(CONFIG['eval_seed'])
        torch.cuda.manual_seed_all(CONFIG['eval_seed'])
        for i in range(0, len(rows), CONFIG['eval_batch']):
            batch = batch_for(rows[i:i + CONFIG['eval_batch']], cache, tokens, packer)
            frames = int(batch['loss_mask'].sum())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(**batch, progress=1.)
            for name in ('diff', 'stop'):
                sums[name] += float(out['loss/' + name]) * frames
            sums['frames'] += frames
    result = {name: sums[name] / sums['frames'] for name in ('diff', 'stop')}
    result.update(loss=result['diff'] + result['stop'], utterances=len(rows), target_patches=sums['frames'])
    assert math.isfinite(result['loss'])
    return result


def save_adapter(model, step, epoch):
    check_space()
    folder = ROOT / 'checkpoints' / f'step_{step:06d}'
    folder.mkdir(parents=True, exist_ok=False)
    state = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}
    save_file(state, folder / 'lora_weights.safetensors')
    write(folder / 'lora_config.json', {'base_model': str(BASE), 'lora_config': model.lora_config.model_dump(),
          'step': step, 'epoch': epoch})
    return folder


def listen(step):
    torch.cuda.empty_cache()
    emit('generating_fixed_samples', step=step, patients=6)
    with (ROOT / f'listening_step_{step:06d}.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, str(Path(__file__).with_name('listen.py')), '--step', str(step)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    receipts = read(ROOT / 'listening' / f'step_{step:06d}' / 'receipts.json')
    emit('fixed_samples_generated', step=step, samples=len(receipts),
         length_cap_hits=sum(r['hit_length_limit'] for r in receipts))


def train():
    assert not (ROOT / 'runtime.json').exists(), 'Training already launched; do not restart implicitly'
    assert read(ROOT / 'config.json') == CONFIG
    assert (ROOT / f"benchmark_b{CONFIG['micro_batch']}.json").exists()
    assert CONFIG['effective_batch'] % CONFIG['micro_batch'] == 0
    model, cache, tokens, packer = training_objects()
    rows = read(ROOT / 'data.json')
    train_rows = [r for r in rows if r['partition'] == 'train']
    test_rows = [r for r in rows if r['partition'] == 'merged_test']
    params = [p for p in model.parameters() if p.requires_grad]
    steps_per_epoch = math.ceil(len(train_rows) / CONFIG['effective_batch'])
    total_steps = steps_per_epoch * CONFIG['epochs']
    optimizer = torch.optim.AdamW(params, lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    write(ROOT / 'runtime.json', dict(config=CONFIG, steps_per_epoch=steps_per_epoch, total_steps=total_steps,
          trainable_parameters=sum(p.numel() for p in params), base_model=str(BASE),
          source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
          trainer_sha256=digest(__file__), versions={m: importlib.metadata.version(m) for m in ('torch', 'torchaudio', 'transformers', 'safetensors')},
          normalization='sum target-patch losses / total target patches of effective batch; includes final partial batch',
          evaluation='all 511 rows, fixed ref/order/noise seed, fixed diffusion progress=1; repeated monitored test',
          precision='FP32 master parameters, BF16 autocast, no GradScaler',
          inference_mode='ref_audio only; no prompt transcript; implicit Cantonese',
          peak_benchmark=read(ROOT / f"benchmark_b{CONFIG['micro_batch']}.json")))
    emit('evaluating_initial', total_steps=total_steps)
    initial = evaluate(model, test_rows, cache, tokens, packer)
    write(ROOT / 'eval_step_000000.json', initial)
    emit('initial_evaluation', **initial)
    listen(0)
    best_loss, best_step = initial['loss'], 0
    evaluations = [initial['loss']]
    step, seen = 0, 0
    seed_all(CONFIG['seed'])
    start = time.monotonic()
    for epoch in range(1, CONFIG['epochs'] + 1):
        pairs = train_pairs(train_rows, epoch, CONFIG['seed'])
        write(ROOT / f'pairs_epoch_{epoch}.json', [{'target': r['uid'], 'reference': r['ref_uid']} for r in pairs])
        for offset in range(0, len(pairs), CONFIG['effective_batch']):
            group = pairs[offset:offset + CONFIG['effective_batch']]
            total_frames = sum(len(cache[r['uid']]) for r in group)
            model.train()
            model.audio_vae.eval()
            optimizer.zero_grad(set_to_none=True)
            next_step = step + 1
            if next_step <= CONFIG['warmup_steps']:
                scale = next_step / CONFIG['warmup_steps']
            else:
                scale = .5 * (1 + math.cos(math.pi * (next_step - CONFIG['warmup_steps']) / (total_steps - CONFIG['warmup_steps'])))
            for param_group in optimizer.param_groups:
                param_group['lr'] = CONFIG['lr'] * scale
            losses = {'diff': 0., 'stop': 0.}
            began = time.monotonic()
            for j in range(0, len(group), CONFIG['micro_batch']):
                chunk = group[j:j + CONFIG['micro_batch']]
                batch = batch_for(chunk, cache, tokens, packer)
                frames = int(batch['loss_mask'].sum())
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(**batch, progress=step / total_steps)
                    loss = weighted_loss(output, frames, total_frames)
                loss.backward()
                for name in losses:
                    losses[name] += float(output['loss/' + name].detach()) * frames / total_frames
            grad_norm = torch.nn.utils.clip_grad_norm_(params, CONFIG['clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            seen += len(group)
            emit('train_step', step=step, total_steps=total_steps, epoch=(epoch - 1) + (offset + len(group)) / len(pairs),
                 seen_targets=seen, targets_this_step=len(group), diff=losses['diff'], stop=losses['stop'],
                 loss=sum(losses.values()), grad_norm=float(grad_norm), lr=optimizer.param_groups[0]['lr'],
                 step_seconds=time.monotonic() - began, wall_seconds=time.monotonic() - start,
                 peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3)
            if step % CONFIG['eval_every_steps'] == 0 or step == total_steps:
                folder = save_adapter(model, step, epoch=(epoch - 1) + (offset + len(group)) / len(pairs))
                emit('evaluating', step=step, checkpoint=str(folder))
                result = evaluate(model, test_rows, cache, tokens, packer)
                if result['loss'] < best_loss:
                    best_loss, best_step = result['loss'], step
                result.update(step=step, checkpoint=str(folder), best_step=best_step, best_loss=best_loss)
                write(ROOT / f'eval_step_{step:06d}.json', result)
                write(ROOT / 'best_by_monitored_test_loss.json', {'step': best_step, 'loss': best_loss})
                # One replaceable resumable optimizer; never duplicate the frozen base model.
                state = {'optimizer': optimizer.state_dict(), 'step': step, 'epoch': epoch, 'offset': offset + len(group),
                         'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'config': CONFIG}
                tmp = ROOT / 'latest_optimizer.tmp.pt'
                torch.save(state, tmp)
                tmp.replace(ROOT / 'latest_optimizer.pt')
                evaluations.append(result['loss'])
                emit('evaluation', **result, free_GiB=check_space() / 1024**3)
                listen(step)
                # Stop only after two consecutive substantial rises, not one stochastic spike.
                if len(evaluations) >= 3 and all(x > best_loss * 1.15 for x in evaluations[-2:]):
                    emit('stopped_overfitting', step=step, best_step=best_step, best_loss=best_loss)
                    return
    emit('training_complete', step=step, epochs=CONFIG['epochs'], seen_targets=seen,
         best_step=best_step, best_loss=best_loss, free_GiB=check_space() / 1024**3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'cache', 'benchmark', 'train'])
    parser.add_argument('--micro-batch', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        if args.action == 'benchmark':
            benchmark(args.micro_batch)
        else:
            globals()[{'prepare': 'prepare', 'cache': 'cache_features', 'train': 'train'}[args.action]]()
    except Exception as error:
        emit('failed', action=args.action, error_type=type(error).__name__, message=str(error))
        raise


if __name__ == '__main__':
    main()
