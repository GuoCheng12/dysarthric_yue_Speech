#!/usr/bin/env python3
"""Read-only four-GPU monitor, with recovered draws distinguished from new draws."""
import json
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter

ROOT=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_design_exploration_20260908_v1/real2117_DPO_KL_seed42_3epochs_20260910_v1/resource_4gpu_v2')


def rows(path):
    if not path.exists():return []
    return [json.loads(x) for x in path.read_text().splitlines(keepends=True) if x.endswith('\n') and x.strip()]


def main():
    events=rows(ROOT/'events.jsonl');losses=rows(ROOT/'training/loss.jsonl')
    candidates={(r['audio_id'],r['sample_index']):r for r in rows(ROOT/'recovered_candidates.jsonl')}
    counts=[]
    for shard in range(4):
        records=rows(ROOT/f'sampling_shard{shard}/candidates.jsonl');counts.append(len(records))
        for row in records:candidates[row['audio_id'],row['sample_index']]=row
    by_audio=Counter(x[0] for x in candidates);complete=sum(v==4 for v in by_audio.values())
    phase='采样恢复已准备，三卡训练 smoke 检查中；正式 DPO 更新尚未开始'
    if (ROOT/'sampling_processes.json').exists():phase='四卡全量采样中；正式 DPO 更新尚未开始'
    if (ROOT/'training.log').exists():phase='采样完成，三卡准备冻结 reference 分数'
    if losses:phase='三卡正式 DPO 训练中；第四卡独立评估'
    if (ROOT/'training/result.json').exists():phase='训练完成，等待评估队列完成'
    if (ROOT/'complete.json').exists():phase='3 epoch 与 15 次评估已全部完成'
    if (ROOT/'failure.json').exists():phase='已停止：'+json.loads((ROOT/'failure.json').read_text())['error']
    if (ROOT/'user_stop_20260910.json').exists():phase='已按用户要求停止；保留已完成的训练与评估记录'
    print('真实 2117 DPO＋KL | seed42 | 3 epoch / 795 步 / 15 次评估')
    print(datetime.now(timezone.utc).strftime('%H:%M:%S UTC')+' | '+phase)
    print('资源：采样 GPU0–3；随后训练 GPU0–2，评估 GPU3。全局 batch=8。')
    print(f'\n随机采样：{complete}/2117 条完成 ({100*complete/2117:.1f}%)；{len(candidates)}/8468 次')
    print(f'已核对并接续旧记录 514 次；新增有效记录 {len(candidates)-514} 次。')
    for i,n in enumerate(counts):print(f'  GPU{i} 新执行已写入 {n} 次（其中包含接续记录）；负责 {530 if i==0 else 529} 条音频')
    print('\n正式训练：')
    if losses:
        latest=losses[-1];print(f"  step {latest['step']}/795 | epoch {latest['epoch']:.3f}/3")
        for x in losses[-4:]:print(f"  step{x['step']:3d} | loss {x['loss']:.6f} | DPO {x['DPO']:.6f} | KL×10 {x['KL_weighted']:.6f}")
    else:print('  0/795 次更新；尚无正式训练 loss')
    evals=[x for x in events if x['stage']=='evaluation_complete']
    print(f'\n正式评估：已完成 {len(evals)}/15 次；每 53 步提交一次，GPU3 异步评估')
    completed={x['step'] for x in evals}
    pending=[json.loads(p.read_text()) for p in sorted((ROOT/'eval_queue').glob('*.json'))]
    pending=[x for x in pending if x['step'] not in completed]
    running=[x for x in pending if (ROOT/f"evaluations/step-{x['step']}").is_dir()]
    if running:
        job=running[0];progress=None
        for line in (ROOT/'evaluation.log').read_text().splitlines(keepends=True):
            if not line.endswith('\n') or not line.startswith('{'):continue
            record=json.loads(line)
            if record.get('stage') in ('baseline_verified','evaluation_complete'):progress=None
            if 'decoded' in record:progress=record
        detail=(f"{progress['decoded']}/{progress['total']} 条 ({100*progress['decoded']/progress['total']:.1f}%)" if progress else '准备解码 / 汇总结果')
        print(f"  正在评估 step {job['step']} / epoch {job['epoch']:.1f}：{detail}")
    waiting=[x for x in pending if x not in running]
    if waiting:print('  排队 checkpoint：'+', '.join(str(x['step']) for x in waiting))
    print('  R-only 基线 CER：20.7933%；完整 511 条评估结束后显示 CER。')
    for x in evals[-4:]:print(f"  epoch{x['epoch']:.1f} | CER {x['metrics']['overall']['CER_percent']:.4f}% | ΔR-only {x['metrics']['overall']['delta_Ronly_pp']:+.4f}pp")
    for name,label in [('sampling_smoke','新旧采样一致性'),('smoke_3gpu_v3','三卡训练与串行梯度对照')]:
        path=ROOT/name/'result.json';print(f"\n{label} smoke："+('已通过（不计入正式训练）' if path.exists() and json.loads(path.read_text())['passed'] else '准备/检查中'))


if __name__=='__main__':main()
