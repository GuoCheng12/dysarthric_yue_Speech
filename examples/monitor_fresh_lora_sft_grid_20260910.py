"""Read-only monitor for the four fresh-base SFT recipes."""
import json
from pathlib import Path
from datetime import datetime,timezone

root=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/qwen3_asr_fresh_lora_sft_seed42_4configs_20260910_v1')
def read(path):return json.loads(path.read_text())
def events(path):
    if not path.exists():return []
    return [json.loads(x) for x in path.read_text().splitlines(keepends=True) if x.endswith('\n') and x.strip()]

print('原始 Qwen3-ASR-1.7B + 全新 LoRA | seed42 | 2117条真实语音 | 3 epoch')
print(datetime.now(timezone.utc).isoformat())
print('每卡一个独立配置；每组201步、6351次真实音频曝光；511条开发集只评估。')
print('R-only仅作历史成绩参照：CER 20.7933%；本轮模型不载入R-only。')
base=root/'base_evaluation/summary.json'
if base.exists():print(f"原始Qwen基线 CER：{read(base)['CER_percent']:.4f}%")
for gpu,name in enumerate(['A_legacy_full','B_alpha32_full','C_alpha32_language','D_alpha32_token_ce']):
    out=root/name
    print(f'\nGPU {gpu} | {name}')
    if (out/'smoke_passed.json').exists():print('  全新初始化 / 2步更新 / 保存 / 合并后精确恢复 smoke：通过')
    else:print('  smoke准备或执行中，尚未开始正式训练')
    logs=events(out/'events.jsonl');losses=[x for x in logs if x['stage']=='loss' and 'loss' in x]
    for x in losses[-2:]:
        print(f"  step {x['step']:3d}/201 | epoch {x.get('epoch',0):.3f} | loss {x['loss']:.5f} | lr {x['learning_rate']:.3g} | grad {x['grad_norm']:.4f}")
    evals=[x for x in logs if x['stage']=='evaluation_complete']
    print(f"  已完成 {len(evals)}/15 次正式评估；整轮覆盖核对 {len(list(out.glob('epoch-*-exposure.json')))}/3")
    for x in evals[-3:]:print(f"  eval step{x['step']:3d} | CER {x['metrics']['CER_percent']:.4f}% | exposure {x['actual_audio_exposures']}")
    for d in sorted((out/'evaluations').glob('step-*')):
        if (d/'summary.json').exists():continue
        progress=events(d/'events.jsonl')
        if progress:print(f"  正在评估 {d.name}: {progress[-1].get('completed',0)}/511")
    if (out/'complete.json').exists():print('  三轮训练及末端评估已全部完成')
failures=[x for x in events(root/'events.jsonl') if x['stage']=='failure']
if failures:print('\n失败记录：'+str(failures[-1]))
if (root/'complete.json').exists():print('\n四组均已完成。')
