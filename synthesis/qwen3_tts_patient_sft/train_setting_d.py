"""Qwen3-TTS Base: full-ICL, same-patient, target-only LoRA adaptation."""
import argparse
from collections import defaultdict
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
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

SOURCE = Path('/data/qwen3-asr/third_party/Qwen3-TTS-probe')
BASE = Path('/data/qwen3-asr/models/tts/comparison24_v1/qwen3tts')
VOX = Path('/data/qwen3-asr/records/voxcpm2_reference_sft_setting_d_v1')
SG = Path('/data/qwen3-asr/records/cosyvoice3_reference_sft_setting_d_v1')
ROOT = Path('/data/qwen3-asr/records/qwen3_tts_reference_sft_setting_d_v1')
CONFIG = dict(seed=42, epochs=3, micro_batch=4, effective_batch=16,
              lr=2e-5, weight_decay=.01, warmup_steps=20, clip_norm=1.,
              eval_every_steps=66, eval_batch=4, sub_loss_weight=.3,
              rank=16, alpha=32, dropout=0., language='Auto', non_streaming_mode=False,
              max_new_tokens=768, minimum_free_bytes=20 * 1024**3)
TARGET_MODULES = r'talker\.(model|code_predictor\.model)\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'
sys.path.insert(0, str(SOURCE))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def emit(event, **values):
    record = dict(event=event, time=time.time(), **values)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    with (ROOT / 'events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
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


def train_pairs(rows, epoch):
    rng = random.Random(CONFIG['seed'] + epoch)
    pools = defaultdict(list)
    for row in rows:
        pools[row['speaker_id']].append(row['uid'])
    paired = [{**r, 'ref_uid': rng.choice([uid for uid in pools[r['speaker_id']] if uid != r['uid']])}
              for r in rows]
    rng.shuffle(paired)
    return paired


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    check_space()
    rows = read(VOX / 'data.json')
    train = [r for r in rows if r['partition'] == 'train']
    test = [r for r in rows if r['partition'] == 'merged_test']
    sg_audio = dict(line.split(maxsplit=1) for line in (SG / 'data/train/wav.scp').read_text(encoding='utf-8').splitlines())
    assert {r['uid']: r['audio'] for r in train} == sg_audio
    assert len(train) == 2108 and len(test) == 511
    assert len({r['speaker_id'] for r in train}) == 57
    for key in ('prompt_id', 'prompt_normalized_sha16', 'prompt_jyutping_sha16', 'clean_gt', 'wav_sha256'):
        assert not {r[key] for r in train} & {r[key] for r in test}, key
    by_id = {r['uid']: r for r in train}
    for row in test:
        assert by_id[row['ref_uid']]['speaker_id'] == row['speaker_id']
    for row in rows:
        assert digest(row['audio']) == row['wav_sha256']
    download = read('/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_model_comparison24_v1/download_qwen3tts.json')
    for item in download['files']:
        assert digest(BASE / item['path']) == item['sha256'], item['path']
    write(ROOT / 'config.json', CONFIG)
    write(ROOT / 'data.json', rows)
    write(ROOT / 'samples.json', read(VOX / 'samples.json'))
    write(ROOT / 'preparation.json', dict(train_rows=2108, train_patients=57,
        train_hours=sum(r['seconds'] for r in train) / 3600, monitored_merged_test_rows=511,
        evaluation='old dev + test; repeated monitored set, not untouched final test',
        source_data_sha256=digest(VOX / 'data.json'), samples_sha256=digest(VOX / 'samples.json'),
        base_download=download, reference_policy='same-patient distinct train utterance; re-pair each epoch; fixed eval refs',
        conditioning='Native full ICL: reference transcript + reference codec tokens + speaker embedding + target text',
        loss='Target codebook0 + EOS CE; target codebooks1..15 CE weighted 0.3; no reference/padding loss',
        selection='Save/evaluate all six half-epoch points, stop on two consecutive joint acoustic regressions',
        acoustic_stop_rule='MCD > min(previous MCD)*1.05 AND CAMPPlus < max(previous CAMPPlus)-0.03 at two consecutive saved points',
        wave_processing='official loading/resampling only, no VAD/denoise/retry/CER filtering'))
    emit('prepared', train_rows=2108, patients=57, monitored_test_rows=511)


def load_pipeline(lora=True):
    from qwen_tts import Qwen3TTSModel
    from peft import LoraConfig, inject_adapter_in_model
    seed_all(CONFIG['seed'])
    pipeline = Qwen3TTSModel.from_pretrained(str(BASE), device_map='cuda:0',
        dtype=torch.bfloat16, attn_implementation='sdpa')
    model = pipeline.model
    model.requires_grad_(False)
    if lora:
        inject_adapter_in_model(LoraConfig(r=CONFIG['rank'], lora_alpha=CONFIG['alpha'],
            lora_dropout=CONFIG['dropout'], target_modules=TARGET_MODULES, bias='none'), model)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert 'lora_' in name
                parameter.data = parameter.data.float()
        assert sum('lora_B' in n for n, _ in model.named_parameters()) == (28 + 5) * 4
    assert model.tts_model_type == 'base'
    model.eval()
    return pipeline


def cache_features():
    assert not (ROOT / 'features.safetensors').exists()
    pipeline = load_pipeline(lora=False)
    features, tokens = {}, {}
    started = time.monotonic()
    rows = read(ROOT / 'data.json')
    for i, row in enumerate(rows):
        prompt = pipeline.create_voice_clone_prompt(ref_audio=row['audio'], ref_text=row['clean_gt'],
                                                    x_vector_only_mode=False)[0]
        codes = prompt.ref_code.detach().cpu().to(torch.int16).contiguous()
        assert codes.ndim == 2 and codes.shape[1] == 16 and 0 <= codes.min() <= codes.max() < 2048
        features[row['uid'] + '.codes'] = codes
        features[row['uid'] + '.speaker'] = prompt.ref_spk_embedding.detach().cpu().contiguous()
        tokens[row['uid']] = {
            'target': pipeline._tokenize_texts([pipeline._build_assistant_text(row['clean_gt'])])[0].cpu().tolist()[0],
            'ref': pipeline._tokenize_texts([pipeline._build_ref_text(row['clean_gt'])])[0].cpu().tolist()[0]}
        if (i + 1) % 200 == 0:
            emit('caching', completed=i + 1, total=len(rows), seconds=time.monotonic() - started)
    save_file(features, ROOT / 'features.safetensors')
    write(ROOT / 'tokens.json', tokens)
    emit('cache_complete', rows=len(rows), seconds=time.monotonic() - started,
         bytes=(ROOT / 'features.safetensors').stat().st_size)


def codec_embedding(talker, codes):
    embeddings = [talker.get_input_embeddings()(codes[:, 0])]
    embeddings += [talker.code_predictor.get_input_embeddings()[i - 1](codes[:, i]) for i in range(1, 16)]
    return torch.stack(embeddings, dim=1).sum(1)


def prefix_for(model, row, cache, tokens):
    """Mirror native Auto/ICL prefix, reuse official full-ICL packing function."""
    talker, config, device = model.talker, model.config, model.device
    target = torch.tensor([tokens[row['uid']]['target']], device=device)
    ref = torch.tensor([tokens[row['ref_uid']]['ref']], device=device)
    codes = cache[row['ref_uid'] + '.codes'].to(device=device, dtype=torch.long)
    speaker = cache[row['ref_uid'] + '.speaker'].to(device=device, dtype=model.dtype).view(1, 1, -1)
    bos, eos, pad = talker.text_projection(talker.get_text_embeddings()(torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]], device=device))).chunk(3, dim=1)
    tags = talker.get_input_embeddings()(torch.tensor([[config.talker_config.codec_nothink_id,
        config.talker_config.codec_think_bos_id, config.talker_config.codec_think_eos_id]], device=device))
    codec_pad = talker.get_input_embeddings()(torch.tensor([[config.talker_config.codec_pad_id]], device=device))
    header = torch.cat([tags, speaker, codec_pad], dim=1) + torch.cat([pad.expand(-1, 4, -1), bos], dim=1)
    role = talker.text_projection(talker.get_text_embeddings()(target[:, :3]))
    icl, trailing = model.generate_icl_prompt(text_id=target[:, 3:-5], ref_id=ref[:, 3:-2],
        ref_code=codes, tts_pad_embed=pad, tts_eos_embed=eos, non_streaming_mode=CONFIG['non_streaming_mode'])
    return torch.cat([role, header, icl], dim=1)[0], trailing[0], pad[0]


def supervision_positions(prefix_length, frames, device='cpu'):
    # Last reference frame predicts target frame 0; last target frame predicts EOS.
    return torch.arange(prefix_length - 1, prefix_length + frames, device=device)


def batch_for(model, rows, cache, tokens):
    packed, lengths, prefix_lengths, codes_list = [], [], [], []
    for row in rows:
        prefix, trailing, pad = prefix_for(model, row, cache, tokens)
        codes = cache[row['uid'] + '.codes'].to(device=model.device, dtype=torch.long)
        audio = codec_embedding(model.talker, codes)
        tail = torch.cat([trailing, pad.expand(max(0, len(codes) - len(trailing)), -1)])[:len(codes)]
        packed.append(torch.cat([prefix, audio + tail], dim=0))
        lengths.append(len(packed[-1]))
        prefix_lengths.append(len(prefix))
        codes_list.append(codes)
    inputs = pad_sequence(packed, batch_first=True)
    mask = torch.arange(inputs.shape[1], device=model.device)[None] < torch.tensor(lengths, device=model.device)[:, None]
    return inputs, mask.long(), prefix_lengths, codes_list


def loss_sums(model, rows, cache, tokens):
    inputs, mask, starts, codes_list = batch_for(model, rows, cache, tokens)
    positions, _ = model.talker.get_rope_index(mask)
    hidden = model.talker.model(inputs_embeds=inputs, attention_mask=mask,
        position_ids=positions, use_cache=False, return_dict=True).last_hidden_state
    first_hidden, first_labels, target_hidden = [], [], []
    for i, (start, codes) in enumerate(zip(starts, codes_list)):
        selected = hidden[i, supervision_positions(start, len(codes), hidden.device)]
        first_hidden.append(selected)
        first_labels.append(torch.cat([codes[:, 0], codes.new_tensor([model.config.talker_config.codec_eos_token_id])]))
        target_hidden.append(selected[:-1])
    logits = model.talker.codec_head(torch.cat(first_hidden))
    labels = torch.cat(first_labels)
    first = F.cross_entropy(logits.float(), labels, reduction='sum')
    codes = torch.cat(codes_list)
    target_hidden = torch.cat(target_hidden)
    # Each residual-code head predicts its own codebook. Do not apply causal label shifting across heads.
    sub_inputs = [target_hidden[:, None], model.talker.get_input_embeddings()(codes[:, :1])]
    sub_inputs += [model.talker.code_predictor.get_input_embeddings()[i - 1](codes[:, i:i+1]) for i in range(1, 15)]
    sub = model.talker.code_predictor.forward_finetune(inputs_embeds=torch.cat(sub_inputs, dim=1),
        labels=None, use_cache=False).logits
    residual = F.cross_entropy(sub.float().reshape(-1, sub.shape[-1]), codes[:, 1:].reshape(-1), reduction='sum')
    return {'first_sum': first, 'residual_sum': residual,
            'first_count': len(labels), 'residual_count': codes.shape[0] * 15}


def normalized_loss(sums, first_count, residual_count):
    return sums['first_sum'] / first_count + CONFIG['sub_loss_weight'] * sums['residual_sum'] / residual_count


def training_objects():
    assert read(ROOT / 'config.json') == CONFIG
    return load_pipeline(), load_file(ROOT / 'features.safetensors'), read(ROOT / 'tokens.json')


def verify_native(pipeline, row, cache, tokens):
    model = pipeline.model
    captured = {}
    class CapturedPrefix(Exception):
        pass
    def capture(**kwargs):
        captured.update(kwargs)
        raise CapturedPrefix()
    original = model.talker.generate
    model.talker.generate = capture
    try:
        with torch.no_grad():
            model.generate(input_ids=[torch.tensor([tokens[row['uid']]['target']], device=model.device)],
                ref_ids=[torch.tensor([tokens[row['ref_uid']]['ref']], device=model.device)], languages=['Auto'],
                voice_clone_prompt={'ref_code': [cache[row['ref_uid'] + '.codes'].to(device=model.device, dtype=torch.long)],
                    'ref_spk_embedding': [cache[row['ref_uid'] + '.speaker']], 'x_vector_only_mode': [False], 'icl_mode': [True]},
                non_streaming_mode=CONFIG['non_streaming_mode'])
    except CapturedPrefix:
        pass
    finally:
        model.talker.generate = original
    with torch.no_grad():
        prefix, tail, pad = prefix_for(model, row, cache, tokens)
        torch.testing.assert_close(prefix[None], captured['inputs_embeds'], rtol=0, atol=0)
        torch.testing.assert_close(tail[None], captured['trailing_text_hidden'], rtol=0, atol=0)
        torch.testing.assert_close(pad[None], captured['tts_pad_embed'], rtol=0, atol=0)
        packed, _, starts, codes = batch_for(model, [row], cache, tokens)
        torch.testing.assert_close(packed[0, :starts[0]], prefix, rtol=0, atol=0)
        assert starts[0] == len(prefix)
    return dict(target=row['uid'], reference=row['ref_uid'], prefix_frames=len(prefix),
                target_frames=len(codes[0]), prefix_max_abs_error=0., trailing_max_abs_error=0.)


def benchmark():
    pipeline, cache, tokens = training_objects()
    model = pipeline.model
    rows = [r for r in read(ROOT / 'data.json') if r['partition'] == 'train']
    pairs = train_pairs(rows, 1)
    pools = defaultdict(list)
    for row in rows:
        pools[row['speaker_id']].append(row['uid'])
    for row in pairs:
        row['ref_uid'] = max((u for u in pools[row['speaker_id']] if u != row['uid']), key=lambda u: len(cache[u + '.codes']))
    pairs.sort(key=lambda r: len(cache[r['uid'] + '.codes']) + len(cache[r['ref_uid'] + '.codes']), reverse=True)
    verified = [verify_native(pipeline, r, cache, tokens) for r in [pairs[0], *read(ROOT / 'samples.json')]]
    # Also exercise the uncommon case where reference codes end before reference+target text.
    short_cache = dict(cache)
    r = pairs[0]
    short_cache[r['ref_uid'] + '.codes'] = cache[r['ref_uid'] + '.codes'][:1]
    verified.append(verify_native(pipeline, r, short_cache, tokens))
    write(ROOT / 'native_alignment.json', verified)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    model.speaker_encoder.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=CONFIG['lr'])
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        start = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            sums = loss_sums(model, pairs[:CONFIG['micro_batch']], cache, tokens)
            loss = normalized_loss(sums, sums['first_count'], sums['residual_count'])
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        assert all(any(p.grad is not None and float(p.grad.abs().max()) > 0 for n, p in model.named_parameters()
                       if group in n and 'lora_B' in n) for group in ('talker.model.', 'talker.code_predictor.model.'))
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.monotonic() - start)
    result = dict(micro_batch=CONFIG['micro_batch'], seconds=times,
        loss=float(loss.detach()), grad_norm=float(grad), native_alignment_cases=len(verified),
        target_frames=[len(cache[r['uid'] + '.codes']) for r in pairs[:CONFIG['micro_batch']]],
        peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3,
        peak_reserved_GiB=torch.cuda.max_memory_reserved() / 1024**3,
        discarded_updates=True, trainable_parameters=sum(p.numel() for p in params))
    write(ROOT / 'benchmark.json', result)
    emit('benchmark_complete', **result)


@torch.no_grad()
def evaluate(model, rows, cache, tokens):
    model.eval()
    totals = dict(first_sum=0., residual_sum=0., first_count=0, residual_count=0)
    for i in range(0, len(rows), CONFIG['eval_batch']):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            sums = loss_sums(model, rows[i:i + CONFIG['eval_batch']], cache, tokens)
        for key, value in sums.items():
            totals[key] += float(value)
    return dict(first_loss=totals['first_sum'] / totals['first_count'],
        residual_loss=totals['residual_sum'] / totals['residual_count'],
        loss=normalized_loss(totals, totals['first_count'], totals['residual_count']), utterances=len(rows))


def monitor(step):
    emit('generating_monitor_samples', step=step, patients=6)
    torch.cuda.empty_cache()
    with (ROOT / f'monitor_step_{step:06d}.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, str(Path(__file__).with_name('monitor.py')), 'generate', '--step', str(step)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    result = read(ROOT / 'listening' / f'step_{step:06d}' / 'metrics.json')
    emit('acoustic_monitor_complete', step=step, **result)
    return result['overall']


def train():
    assert not (ROOT / 'runtime.json').exists(), 'Training already launched; no implicit restart'
    benchmark_result = read(ROOT / 'benchmark.json')
    pipeline, cache, tokens = training_objects()
    model = pipeline.model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    rows = read(ROOT / 'data.json')
    train_rows = [r for r in rows if r['partition'] == 'train']
    test_rows = [r for r in rows if r['partition'] == 'merged_test']
    params = [p for p in model.parameters() if p.requires_grad]
    per_epoch = math.ceil(len(train_rows) / CONFIG['effective_batch'])
    total_steps = per_epoch * CONFIG['epochs']
    optimizer = torch.optim.AdamW(params, lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    write(ROOT / 'runtime.json', dict(config=CONFIG, total_steps=total_steps, steps_per_epoch=per_epoch,
        trainable_parameters=sum(p.numel() for p in params), benchmark=benchmark_result,
        base=str(BASE), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
        trainer_sha256=digest(__file__), monitor_sha256=digest(Path(__file__).with_name('monitor.py')),
        lora_target_modules=TARGET_MODULES, versions={m: importlib.metadata.version(m) for m in ('torch', 'transformers', 'peft')},
        precision='Frozen BF16 base, FP32 LoRA/Adam states, BF16 autocast',
        normalization='Sum target CE / full effective-batch target count separately for codebook0+EOS and 15 residual codebooks',
        monitoring='all511 teacher-forced loss; same6 held-out real targets for generated MCD/CAMPPlus, no ASR CER selection'))
    emit('evaluating_initial', step=0, total_steps=total_steps)
    initial = evaluate(model, test_rows, cache, tokens)
    write(ROOT / 'eval_step_000000.json', initial)
    emit('initial_evaluation', step=0, **initial)
    initial_acoustic = monitor(0)
    best_mcd, best_sim = initial_acoustic['mcd_dtw_db'], initial_acoustic['campplus_to_real']
    best_mcd_step = best_sim_step = 0
    regressions = 0
    step = seen = 0
    seed_all(CONFIG['seed'])
    began = time.monotonic()
    for epoch in range(1, CONFIG['epochs'] + 1):
        pairs = train_pairs(train_rows, epoch)
        write(ROOT / f'pairs_epoch_{epoch}.json', [{'target': r['uid'], 'reference': r['ref_uid']} for r in pairs])
        for offset in range(0, len(pairs), CONFIG['effective_batch']):
            group = pairs[offset:offset + CONFIG['effective_batch']]
            nframes = sum(len(cache[r['uid'] + '.codes']) for r in group)
            denominators = (nframes + len(group), nframes * 15)
            model.train()
            model.speaker_encoder.eval()
            optimizer.zero_grad(set_to_none=True)
            next_step = step + 1
            scale = next_step / CONFIG['warmup_steps'] if next_step <= CONFIG['warmup_steps'] else .5 * (1 + math.cos(
                math.pi * (next_step - CONFIG['warmup_steps']) / (total_steps - CONFIG['warmup_steps'])))
            for pg in optimizer.param_groups:
                pg['lr'] = CONFIG['lr'] * scale
            totals = dict(first_sum=0., residual_sum=0., first_count=0, residual_count=0)
            started = time.monotonic()
            for j in range(0, len(group), CONFIG['micro_batch']):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    sums = loss_sums(model, group[j:j + CONFIG['micro_batch']], cache, tokens)
                    loss = normalized_loss(sums, *denominators)
                loss.backward()
                for key, value in sums.items():
                    totals[key] += float(value.detach()) if isinstance(value, torch.Tensor) else value
            assert (totals['first_count'], totals['residual_count']) == denominators
            grad = torch.nn.utils.clip_grad_norm_(params, CONFIG['clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            seen += len(group)
            emit('train_step', step=step, total_steps=total_steps, epoch=epoch - 1 + (offset + len(group)) / len(pairs),
                seen_targets=seen, targets_this_step=len(group), loss=normalized_loss(totals, *denominators),
                first_loss=totals['first_sum'] / denominators[0], residual_loss=totals['residual_sum'] / denominators[1],
                grad_norm=float(grad), lr=optimizer.param_groups[0]['lr'], step_seconds=time.monotonic() - started,
                wall_seconds=time.monotonic() - began, peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3)
            if step % CONFIG['eval_every_steps'] == 0 or step == total_steps:
                check_space()
                folder = ROOT / 'checkpoints' / f'step_{step:06d}'
                folder.mkdir(parents=True, exist_ok=False)
                state = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}
                save_file(state, folder / 'lora_weights.safetensors')
                write(folder / 'adapter.json', dict(step=step, config=CONFIG, base=str(BASE), target_modules=TARGET_MODULES))
                tmp = ROOT / 'latest_optimizer.tmp.pt'
                torch.save(dict(optimizer=optimizer.state_dict(), step=step, epoch=epoch,
                    offset=offset + len(group), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), tmp)
                tmp.replace(ROOT / 'latest_optimizer.pt')
                emit('evaluating_checkpoint', step=step, checkpoint=str(folder))
                result = evaluate(model, test_rows, cache, tokens)
                write(ROOT / f'eval_step_{step:06d}.json', dict(step=step, **result))
                emit('evaluation', step=step, **result)
                a = monitor(step)
                regressed = a['mcd_dtw_db'] > best_mcd * 1.05 and a['campplus_to_real'] < best_sim - .03
                regressions = regressions + 1 if regressed else 0
                if a['mcd_dtw_db'] < best_mcd:
                    best_mcd, best_mcd_step = a['mcd_dtw_db'], step
                if a['campplus_to_real'] > best_sim:
                    best_sim, best_sim_step = a['campplus_to_real'], step
                write(ROOT / 'best_by_acoustics.json', dict(best_mcd=best_mcd, best_mcd_step=best_mcd_step,
                    best_campplus=best_sim, best_campplus_step=best_sim_step, consecutive_joint_regressions=regressions))
                if regressions >= 2:
                    emit('stopped_acoustic_regression', step=step, best_mcd_step=best_mcd_step, best_campplus_step=best_sim_step)
                    return
    emit('training_complete', step=step, epochs=CONFIG['epochs'], seen_targets=seen,
        best_mcd_step=best_mcd_step, best_campplus_step=best_sim_step, free_GiB=check_space() / 1024**3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'cache', 'benchmark', 'train'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        globals()[{'cache': 'cache_features'}.get(args.action, args.action)]()
    except Exception as error:
        if ROOT.exists():
            emit('failed', action=args.action, error_type=type(error).__name__, message=str(error))
        raise
