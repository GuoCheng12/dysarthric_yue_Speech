"""Read-only monitor for the four fresh-base SFT recipes."""
import json
from pathlib import Path
from datetime import datetime,timezone

root=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/qwen3_asr_fresh_lora_sft_seed42_EFGH_20260910_v1')
def read(path):return json.loads(path.read_text())
def events(path):
    if not path.exists():return []
    return [json.loads(x) for x in path.read_text().splitlines(keepends=True) if x.endswith('\n') and x.strip()]

print('E/F/G/H | 原始 Qwen + 全新 LoRA | seed42 | 2117条 × 3 epoch | '+datetime.now(timezone.utc).strftime('%H:%M:%S UTC'))
print('E/F: 音频+语言; G/H: 仅音频; LR: 5e-5/2e-5/5e-5/2e-5 | CER 越低越好')
print('R-only 历史参照 CER 20.7933% | 单组日志: Ctrl+b 后按 2(E)/3(F)/4(G)/5(H)')
base=root/'base_evaluation/summary.json'
if base.exists():print(f"原始Qwen基线 CER：{read(base)['CER_percent']:.4f}%")
for gpu,name in enumerate(['E_full_lr5em5','F_full_lr2em5','G_audio_lr5em5','H_audio_lr2em5']):
    out=root/name
    smoke='smoke通过' if (out/'smoke_passed.json').exists() else 'smoke尚未通过'
    print(f'\nGPU {gpu} | {name} | {smoke}')
    logs=events(out/'events.jsonl');losses=[x for x in logs if x['stage']=='loss' and 'loss' in x]
    for x in losses[-1:]:
        print(f"  step {x['step']:3d}/201 | epoch {x.get('epoch',0):.3f} | loss {x['loss']:.5f} | lr {x['learning_rate']:.3g} | grad {x['grad_norm']:.4f}")
    evals=[x for x in logs if x['stage']=='evaluation_complete']
    best=min(evals,key=lambda x:x['metrics']['CER_percent']) if evals else None
    best_text=f"最佳 CER {best['metrics']['CER_percent']:.4f}% @step{best['step']}" if best else '尚无评估结果'
    print(f"  评估 {len(evals)}/15 | 完整数据覆盖 {len(list(out.glob('epoch-*-exposure.json')))}/3轮 | {best_text}")
    for x in evals[-1:]:
        m=x['metrics']
        print(f"  eval step{x['step']:3d} | CER {m['CER_percent']:.4f}% | easy {m['easy']['CER_percent']:.4f}% | hard {m['hard']['CER_percent']:.4f}% | exposure {x['actual_audio_exposures']}")
    for d in sorted((out/'evaluations').glob('step-*')):
        if (d/'summary.json').exists():continue
        progress=events(d/'events.jsonl')
        if progress:print(f"  正在评估 {d.name}: {progress[-1].get('completed',0)}/511")
    if (out/'complete.json').exists():print('  三轮训练及末端评估已全部完成')
failures=[x for x in events(root/'events.jsonl') if x['stage']=='failure']
if failures:print('\n失败记录：'+str(failures[-1]))
if (root/'complete.json').exists():print('\n四组均已完成。')
