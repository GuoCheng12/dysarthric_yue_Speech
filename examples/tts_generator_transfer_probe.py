"""Run the fixed 96-pair generator acceptance probe, with auditable retries."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
B = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1')
ROOT = B / 'generator_transfer_probe_v2'
RECIPES = ['sg_instruct2', 'base_instruct2']
INSTRUCTION = 'You are a helpful assistant. 请用广东话表达。<|endofprompt|>'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def status(directory, state, **extra):
    value = {'status': state, 'time_unix': time.time(), **extra}
    temporary = directory / 'monitor.tmp'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(directory / 'monitor.json')


def prepare():
    cfg = read(ROOT / 'protocol.json')
    for path, expected in cfg['source_sha256'].items():
        assert digest(path) == expected, path
    cohort = read(ROOT / 'cohort.json')
    hashes = {}
    cases = []
    corrections = []
    for pair in cohort:
        source = B / 'tts_reference_policy_v1/selected' / f"{pair['sample_id']}.json"
        record = read(source)['records']['hard_multi']
        assert record['target_text'] == pair['target']['clean_gt']
        assert record['reference_wav'] == pair['reference']['reference_wav']
        assert record['reference_text'] == pair['reference']['reference_text']
        assert record['output_wav'] == pair['existing_sg_zero_shot']['audio']
        for key, sha in [('reference_wav', 'reference_wav_sha256'), ('output_wav', 'output_wav_sha256')]:
            assert digest(record[key]) == record[sha]
        if record['seed'] != pair['generation_seed']:
            corrections.append({'pair_id': pair['sample_id'], 'original_plan_seed': pair['generation_seed'], 'actual_old_audio_seed': record['seed']})
        cases.append({**pair, 'actual_baseline_seed': record['seed']})
        for path in [source, Path(record['reference_wav']), Path(record['output_wav']), Path(pair['target']['audio'])]:
            hashes[str(path)] = digest(path)
    for path in [Path(__file__), ROOT/'protocol.json', ROOT/'cohort.json',
                 REPO/'synthesis/cosyvoice3_patient_sft/synthesize_manifest.py',
                 REPO/'examples/resume_tts_reference_policy.py', REPO/'finetune/scripts/evaluate_asr_checkpoint.py']:
        hashes[str(path)] = digest(path)
    save(ROOT / 'execution.json', {
        'cases': cases, 'source_and_input_sha256': hashes,
        'baseline_seed_corrections': corrections,
        'seed_policy': 'Use the actual selected old audio seed in both new recipes; technical retries use seed+attempt, up to3 attempts, all recorded',
        'instruction': INSTRUCTION, 'selection': 'frozen96, no quality-based retries or replacements',
        'recipes': RECIPES, 'expected_new_audio': 192,
    })
    print({'prepared': len(cases), 'corrected_historical_retry_seeds': corrections}, flush=True)


class InstructProxy:
    def __init__(self, model):
        self.model = model
        self.sample_rate = model.sample_rate

    def inference_zero_shot(self, target, prompt, reference, **kwargs):
        return self.model.inference_instruct2(target, INSTRUCTION, reference, **kwargs)


def generate(recipe, directory, cfg):
    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts
    model, runtime = tts.load_fixed_runtime()
    from resume_tts_reference_policy import install_empty_token_guard
    install_empty_token_guard(model.model)
    if recipe == 'base_instruct2':
        from peft.tuners.tuners_utils import BaseTunerLayer
        layers = [layer for layer in model.model.llm.modules() if isinstance(layer, BaseTunerLayer)]
        assert layers
        for layer in layers:
            layer.enable_adapters(False)
        assert all(layer.disable_adapters for layer in layers)
        runtime['adapter_intervention'] = {'disabled_lora_layers': len(layers), 'base_weights': 'unchanged'}
    runtime['probe_inference'] = {'method': 'inference_instruct2', 'instruction': INSTRUCTION, 'technical_max_attempts': 3}
    save(directory / 'runtime.json', runtime)
    proxy = InstructProxy(model)
    outputs = []
    failures = []
    for i, pair in enumerate(cfg['cases']):
        status(directory, 'generating', complete=len(outputs), attempted_pairs=i, expected=96, current_pair=pair['sample_id'])
        ref = pair['reference']
        audio = directory / 'audio' / f"{pair['sample_id']}.wav"
        audio.parent.mkdir(exist_ok=True)
        assert not audio.exists()
        attempts = []
        for attempt in range(3):
            sample = tts.ManifestSample(sample_id=pair['sample_id'], target_text=pair['target']['clean_gt'],
                reference_text=ref['reference_text'], reference_wav=Path(ref['reference_wav']),
                reference_wav_sha256=ref['reference_wav_sha256'], output_wav=audio,
                seed=pair['actual_baseline_seed'] + attempt)
            try:
                record, temporary = tts._stage_audio_once(sample, proxy)
                os.link(temporary, audio)
                temporary.unlink()
                attempts.append({'attempt': attempt, 'seed': sample.seed, 'status': 'complete', 'record': record})
                old = pair['existing_sg_zero_shot']
                row = {**old, 'utt_id': f"{recipe}::{pair['sample_id']}", 'variant': recipe,
                    'audio': str(audio), 'audio_sha256': digest(audio), 'seconds': record['duration_seconds'],
                    'generation_attempt': attempt, 'generation_seed': sample.seed}
                outputs.append(row)
                break
            except Exception as error:
                attempts.append({'attempt': attempt, 'seed': sample.seed, 'status': 'failed', 'error': repr(error)})
        save(directory / 'receipts' / f"{pair['sample_id']}.json", attempts)
        if attempts[-1]['status'] != 'complete':
            failures.append(pair['sample_id'])
        print(f"[probe] {recipe} {i+1}/96 {pair['sample_id']} {attempts[-1]['status']}", flush=True)
    save(directory / 'generation_summary.json', {'expected': 96, 'complete': len(outputs), 'failed_pairs': failures})
    with (directory / 'input.jsonl').open('x') as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    assert len(outputs) == 96, failures


def run(recipe):
    cfg = read(ROOT / 'execution.json')
    for path, expected in cfg['source_and_input_sha256'].items():
        assert digest(path) == expected, path
    directory = ROOT / recipe
    directory.mkdir()
    status(directory, 'initializing', expected=96)
    try:
        generate(recipe, directory, cfg)
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        for judge, adapter in [('real_asr', B/'tts_increment_r_only1541_v1/r_only/seed_42/training/checkpoint-50'), ('base_asr', None)]:
            status(directory, 'evaluating', judge=judge, complete=96)
            command = [sys.executable, str(REPO/'finetune/scripts/evaluate_asr_checkpoint.py'),
                '--base-model-path', '/data/qwen3-asr/models/Qwen3-ASR-1.7B',
                '--split-jsonl', str(directory/'input.jsonl'), '--out-dir', str(directory/judge),
                '--batch-size', '8', '--critical-cer-threshold', '.5', '--max-new-tokens', '256']
            if adapter:
                command += ['--adapter-path', str(adapter)]
            save(directory / f'{judge}_execution.json', {'command': command})
            with (directory / f'{judge}.log').open('w') as log:
                subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        status(directory, 'complete', complete=96)
    except Exception as error:
        status(directory, 'failed', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--recipe', choices=RECIPES)
    args = parser.parse_args()
    assert args.prepare != bool(args.recipe)
    prepare() if args.prepare else run(args.recipe)
