"""Setting E fresh audio LoRA with merged test766."""
from pathlib import Path
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_e_v1/qwen3_asr_fresh_audio_lr2em4_4gpu_seed42_50epochs_20260911_v1')
OUT = ROOT/'E_audio_real1862_test766_lr2em4_4gpu_50epochs'


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
            'Setting E | Train: real1862, 46 patients | test766, 20 unseen patients',
            'Global batch32 | 59 steps/epoch | 2950 steps / 50 epochs | eval ~1/5 epoch',
            'Test = dev394 + test372 | All60 texts shared, patients disjoint | CER% lower=better.',
            'Best checkpoint selected by patient-equal CER; pooled CER also reported.',
            'Fresh raw Qwen + new audio LoRA. No previous trained adapter loaded.']
    if all_events:
        last=all_events[-1]
        rows.append(f"Stage: {last['stage']} @ {last['utc']}")
    baseline=ROOT/'base_evaluation'/'summary.json'
    if baseline.exists():
        m=json.loads(baseline.read_text())
        rows.append(f"Raw baseline: pooled {m['CER_percent']:.4f}%, patient-equal {m['patient_equal_CER_percent']:.4f}%")
    if losses:
        r=losses[-1]
        rows.append(f"Step {r['step']}/2950 | epoch {r['epoch']:.3f}/50 | audio exposures {r['actual_audio_exposures']}/93100")
        rows.append(f"Loss {r['loss']:.6f} | grad norm {r['grad_norm']:.4f} | LR {r['learning_rate']:.3g} | step {r['seconds']:.2f}s")
    if evaluations:
        best=min(evaluations,key=lambda r:r['metrics']['patient_equal_CER_percent'])
        rows.append(f"Best patient-equal CER {best['metrics']['patient_equal_CER_percent']:.4f}% at step {best['step']} / epoch {best['epoch']:.3f}")
        rows.append(' Recent eval: step epoch  pooled patient    easy  medium    hard')
        for r in evaluations[-5:]:
            m=r['metrics']
            vals=[m['CER_percent'],m['patient_equal_CER_percent']]+[m[k]['CER_percent'] for k in ('easy','medium','hard')]
            rows.append(f" {r['step']:>6} {r['epoch']:>5.2f} "+' '.join(f'{x:>7.2f}' for x in vals))
    else:
        rows.append('Waiting for first trained evaluation at step12 (after smoke + raw baseline).')
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
