"""Fixed reference-only inference, identical inputs/seeds before and after SFT."""
import argparse
import time

import numpy as np
import soundfile as sf
import torch

from train_setting_d import BASE, CONFIG, ROOT, LoRAConfig, VoxCPM2Model, digest, read, write
from voxcpm.model.utils import next_and_close


def run(step):
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    model = VoxCPM2Model.from_local(str(BASE), optimize=False, device='cuda',
                                  lora_config=LoRAConfig(**CONFIG['lora']))
    if step:
        loaded, skipped = model.load_lora_weights(str(ROOT / 'checkpoints' / f'step_{step:06d}'))
        assert not skipped
        assert set(loaded) == {n for n, _ in model.named_parameters() if 'lora_' in n}
    model.eval()
    by_id = {r['uid']: r for r in read(ROOT / 'data.json')}
    folder = ROOT / 'listening' / f'step_{step:06d}'
    folder.mkdir(parents=True, exist_ok=False)
    receipts = []
    with torch.inference_mode():
        for row in read(ROOT / 'samples.json'):
            reference = by_id[row['ref_uid']]
            start = time.monotonic()
            cache = model.build_prompt_cache(reference_wav_path=reference['audio'], trim_silence_vad=False)
            wave, _, features = next_and_close(model._generate_with_prompt_cache(
                target_text=row['clean_gt'], prompt_cache=cache, max_len=512,
                inference_timesteps=10, cfg_value=2., retry_badcase=False,
                retry_badcase_ratio_threshold=1000, streaming=False, seed=42))
            wave = wave.float().cpu().numpy().reshape(-1)
            assert wave.size and np.isfinite(wave).all()
            path = folder / (row['uid'] + '.wav')
            sf.write(path, wave, model.sample_rate, subtype='FLOAT')
            receipts.append({'step': step, 'target_uid': row['uid'], 'speaker_id': row['speaker_id'],
                'patient_group': row['patient_group'], 'text': row['clean_gt'], 'real_audio': row['audio'],
                'reference_audio': reference['audio'], 'reference_text': reference['clean_gt'],
                'reference_sha256': reference['wav_sha256'], 'seed': 42, 'audio': str(path),
                'sha256': digest(path), 'seconds': len(wave) / model.sample_rate,
                'sample_rate': model.sample_rate, 'samples': len(wave),
                'generated_patches': len(features), 'hit_length_limit': len(features) >= 512,
                'wall_seconds': time.monotonic() - start, 'conditioning': 'reference_only'})
            print(receipts[-1], flush=True)
    write(folder / 'receipts.json', receipts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, required=True)
    run(parser.parse_args().step)
