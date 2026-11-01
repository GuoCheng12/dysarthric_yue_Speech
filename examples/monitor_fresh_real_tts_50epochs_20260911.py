"""Compact live monitor for the fresh real2117 + Patient TTS1440 SFT run."""
from pathlib import Path
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/qwen3_asr_fresh_G_real2117_tts1440_lr2em4_4gpu_seed42_50epochs_20260911_v1')
OUT = ROOT/'G_audio_real2117_tts1440_lr2em4_4gpu_50epochs'


def events(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A live writer may not have completed the final line yet.
    return result


def render():
    all_events = events(ROOT/'events.jsonl') + events(OUT/'events.jsonl')
    all_events.sort(key=lambda r:r['utc'])
    losses = [r for r in all_events if r['stage']=='loss']
    evaluations = [r for r in all_events if r['stage']=='evaluation_complete']
    rows = [datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'Fresh Qwen3-ASR-1.7B + AUDIO LoRA | LR 2e-4 | seed42 | 4 GPUs',
            'Train: real 2117 + Patient TTS 1440 = 3557 / epoch | 50 epochs',
            'Global batch32 | 112 steps/epoch | 5600 steps | eval ~1/5 epoch',
            'Dev 511, CER% lower=better. Same reused development set.',
            'Historical R-only: 20.7933%; previous real-only G 50ep best: 20.9585%',
            'Fresh mixed run has more optimizer steps per epoch than real-only G.']
    if all_events:
        last=all_events[-1]
        rows.append(f"Stage: {last['stage']} @ {last['utc']}")
    if losses:
        r=losses[-1]
        rows.append(f"Step {r['step']}/5600 | epoch {r['epoch']:.3f}/50 | audio exposures {r['actual_audio_exposures']}/177850")
        rows.append(f"Loss {r['loss']:.6f} | grad norm {r['grad_norm']:.4f} | LR {r['learning_rate']:.3g} | step {r['seconds']:.2f}s")
    if evaluations:
        best=min(evaluations,key=lambda r:r['metrics']['CER_percent'])
        rows.append(f"Best CER {best['metrics']['CER_percent']:.4f}% at step {best['step']} / epoch {best['epoch']:.3f}")
        rows.append(' Recent eval: step epoch     all    easy  medium    hard vlink247   other')
        for r in evaluations[-5:]:
            m=r['metrics']
            vals=[m['CER_percent']]+[m[k]['CER_percent'] for k in ('easy','medium','hard','vlink247','other')]
            rows.append(f" {r['step']:>6} {r['epoch']:>5.2f} "+' '.join(f'{x:>7.2f}' for x in vals))
    else:
        rows.append('Waiting for first trained evaluation at step23 (after smoke + raw baseline).')
    completed=[r for r in all_events if r['stage']=='epoch_coverage_verified']
    if completed:
        r=completed[-1]
        rows.append(f"Coverage verified: epoch {r['epoch']}, real {r['real_audio']}, synthetic {r['synthetic_audio']}, unique {r['unique_audio']}")
    rows.append('GPU: index, used MiB, utilization %')
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    rows.extend(result.stdout.strip().splitlines())
    rows.append('Windows: 0 overview | 1 loss log | 2 supervisor | 3 GPU; Ctrl+b then number')
    rows.append('Scroll: mouse wheel or Ctrl+b then [ ; q exits scroll; Ctrl+b then d detaches')
    return rows


if __name__=='__main__':
    while True:
        rows=render()
        height=shutil.get_terminal_size((120,32)).lines
        print('\033[2J\033[H'+'\n'.join(rows[:max(height-1,1)]),end='',flush=True)
        time.sleep(3)
