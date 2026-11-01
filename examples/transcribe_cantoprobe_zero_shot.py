"""Unadapted ASR inference with per-audio outputs and resource receipts."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path

import librosa
import soundfile as sf
import torch


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['whisper', 'sensevoice', 'glm', 'funnano'], required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start-barrier', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    started = time.time()
    model_path = Path('/data/qwen3-asr/models') / {
        'whisper': 'whisper-large-v3', 'sensevoice': 'SenseVoiceSmall',
        'glm': 'GLM-ASR-Nano-2512',
        'funnano': 'Fun-ASR-Nano-2512',
    }[args.model]
    if args.model == 'whisper':
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, torch_dtype=torch.float16, local_files_only=True,
            attn_implementation='sdpa').to('cuda').eval()
        decoding = {'language': 'cantonese', 'task': 'transcribe', 'do_sample': False,
                    'num_beams': 1, 'max_new_tokens': 256, 'condition_on_prev_tokens': False}
    elif args.model == 'glm':
        from transformers import AutoProcessor, GlmAsrForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = GlmAsrForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, local_files_only=True,
            attn_implementation='sdpa').to('cuda').eval()
        decoding = {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 256}
    elif args.model == 'sensevoice':
        from funasr import AutoModel
        model = AutoModel(model=str(model_path), device='cuda:0', hub='hf',
                          disable_update=True, disable_pbar=True, ncpu=4)
        model.model.eval()
        decoding = {'language': 'yue', 'use_itn': False, 'batch_size': 1}
    else:
        from funasr import AutoModel
        model = AutoModel(model=str(model_path), device='cuda:0', hub='hf',
                          trust_remote_code=False, disable_update=True, disable_pbar=True, ncpu=4)
        model.model.eval()
        decoding = {'language': None, 'itn': False, 'hotwords': [], 'batch_size': 1,
                    'max_length': 256, 'llm_kwargs': {'do_sample': False, 'num_beams': 1}}
        token_trace = {}
        native_generate = model.model.llm.generate

        def traced_generate(*positional, **kwargs):
            sequences = native_generate(*positional, **kwargs)
            token_trace['ids'] = sequences[0].detach().cpu().tolist()
            return sequences

        model.model.llm.generate = traced_generate
    torch.cuda.synchronize()
    loaded = time.time()
    network = model.model if args.model in ('sensevoice', 'funnano') else model
    ready = {
        'model': args.model, 'path': str(model_path), 'pid': os.getpid(),
        'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'parameters': sum(x.numel() for x in network.parameters()),
        'dtype': str(next(network.parameters()).dtype),
        'loaded_unix': loaded, 'loading_seconds': loaded - started,
        'decoding': decoding, 'target_text_provided': False,
        'reference_text_provided': False, 'external_context': None,
        'transcription_instruction': (processor.default_transcription_prompt if args.model == 'glm'
                                     else model.model.get_prompt([], None, False) if args.model == 'funnano'
                                     else None),
        'versions': {x: importlib.metadata.version(x) for x in
                     ['torch', 'transformers', 'librosa', 'soundfile'] +
                     (['funasr'] if args.model in ('sensevoice', 'funnano') else [])},
    }
    save(args.output / 'ready.json', ready)
    if args.start_barrier:
        deadline = time.monotonic() + 300
        while not args.start_barrier.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Concurrent smoke did not release its start barrier')
            time.sleep(0.1)
    inference_start = time.time()
    predictions = []
    with (args.output / 'predictions.jsonl').open('x') as out:
        for row in map(json.loads, args.manifest.read_text().splitlines()):
            wave_path = Path(row['audio'])
            wave, sr = sf.read(wave_path, dtype='float32', always_2d=True)
            wave = wave.mean(axis=1)
            duration = len(wave) / sr
            if sr != 16000:
                wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
            t = time.perf_counter()
            with torch.inference_mode():
                if args.model == 'whisper':
                    inputs = processor(wave, sampling_rate=16000, return_tensors='pt',
                                       truncation=False, padding='max_length',
                                       return_attention_mask=True).to('cuda', dtype=torch.float16)
                    result = model.generate(**inputs, **decoding, return_timestamps=duration > 30,
                                            return_dict_in_generate=True)
                    text = processor.batch_decode(result['sequences'], skip_special_tokens=True)[0]
                    raw = text
                    if duration > 30:
                        sequences = [s['result']['sequences'].reshape(-1)
                                     for s in result['segments'][0]]
                    else:
                        sequences = [result['sequences'][0]]
                    eos = all(bool((s == model.generation_config.eos_token_id).any())
                              for s in sequences)
                    capped = not eos
                elif args.model == 'glm':
                    # Reject oversized input explicitly instead of allowing processor truncation.
                    limit = int(processor.max_audio_len // 30) * 30
                    if duration > limit:
                        raise ValueError(f'Audio exceeds GLM processor limit: {duration} > {limit}')
                    inputs = processor.apply_transcription_request(wave)
                    inputs = inputs.to('cuda', dtype=torch.bfloat16)
                    result = model.generate(**inputs, **decoding)
                    ids = result[:, inputs.input_ids.shape[1]:]
                    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
                    raw = text
                    eos_ids = model.generation_config.eos_token_id
                    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
                    eos = any(int(x) in eos_ids for x in ids[0])
                    capped = not eos and ids.shape[1] >= decoding['max_new_tokens']
                elif args.model == 'sensevoice':
                    result = model.generate(input=wave, cache={}, **decoding)
                    raw = result[0]['text']
                    text = re.sub(r'<\|[^>]*\|>', '', raw).strip()
                    eos, capped = None, False  # CTC decoding has no autoregressive EOS.
                else:
                    token_trace.clear()
                    # Its chat template accepts tensors; share the same 16 kHz waveform
                    # as the other models rather than resampling again from a file path.
                    result = model.generate(input=[torch.from_numpy(wave)], cache={}, **decoding)
                    raw = result[0]['text']
                    text = raw.strip()
                    ids = token_trace['ids']
                    eos_ids = model.model.llm.generation_config.eos_token_id
                    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
                    eos = any(x in eos_ids for x in ids)
                    capped = not eos and len(ids) >= decoding['max_length']
            torch.cuda.synchronize()
            pred = {'utt_id': row['utt_id'], 'audio': str(wave_path),
                    'audio_sha256': hashlib.sha256(wave_path.read_bytes()).hexdigest(),
                    'waveform_16k_sha256': hashlib.sha256(wave.tobytes()).hexdigest(),
                    'duration_seconds': duration, 'text': text, 'raw_text': raw,
                    'inference_seconds': time.perf_counter() - t,
                    'eos_observed': eos, 'generation_cap_reached': capped}
            out.write(json.dumps(pred, ensure_ascii=False) + '\n')
            out.flush()
            predictions.append(pred)
    total_audio = sum(x['duration_seconds'] for x in predictions)
    elapsed = sum(x['inference_seconds'] for x in predictions)
    save(args.output / 'complete.json', {
        **ready, 'completed_unix': time.time(), 'audio': len(predictions),
        'audio_seconds': total_audio, 'inference_seconds': elapsed,
        'inference_wall_seconds': time.time() - inference_start,
        'real_time_factor': elapsed / total_audio,
        'barrier_wait_seconds': inference_start - loaded,
        'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30,
        'generation_cap_count': sum(x['generation_cap_reached'] for x in predictions),
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })


if __name__ == '__main__':
    main()
