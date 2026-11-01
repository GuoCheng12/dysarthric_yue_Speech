#!/usr/bin/env python3
"""Read-only tmux display; independent of the frozen sampling/training code."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_design_exploration_20260908_v1/real2117_DPO_KL_seed42_3epochs_20260910_v1')


def records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines(keepends=True):
        if line.endswith('\n') and line.strip():
            result.append(json.loads(line))
    return result


def main():
    plan = json.loads((ROOT/'protocol.json').read_text())
    events = records(ROOT/'events.jsonl')
    losses = records(ROOT/'training/loss.jsonl')
    evaluations = [e for e in events if e['stage'] == 'evaluation_complete']
    smoke_path = ROOT/'smoke/result.json'
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    draws = []
    for shard in range(2):
        path = ROOT/f'sampling_shard{shard}/candidates.jsonl'
        count = sum(line.endswith(b'\n') for line in path.open('rb')) if path.exists() else 0
        draws.append(count)
    complete_audio = sum(n//4 for n in draws)
    failure = ROOT/'failure.json'
    if failure.exists():
        phase = '已停止：'+json.loads(failure.read_text())['error']
    elif (ROOT/'complete.json').exists():
        phase = '正式训练与 15 次评估已全部完成'
    elif (ROOT/'training/result.json').exists():
        phase = '正式训练已完成，等待评估队列完成'
    elif losses:
        phase = '正式 DPO 训练中'
    elif (ROOT/'training.log').exists():
        phase = '全量采样已完成，正在计算冻结 reference 分数；正式更新尚未开始'
    else:
        phase = '全量 ASR 采样中；正式 DPO 更新尚未开始'
    print('真实 2117 DPO＋KL | seed42 | 正式计划：3 epoch / 795 步 / 15 次评估')
    print('刷新时间：'+datetime.now(timezone.utc).strftime('%H:%M:%S UTC'))
    print('\n当前阶段：'+phase)
    print(f'采样进度：{complete_audio}/2117 条语音 ({100*complete_audio/2117:.1f}%)；每条 4 次随机采样')
    for shard, count in enumerate(draws):
        print(f'  GPU {shard}：{count//4}/{1059 if shard==0 else 1058} 条完成，已写入 {count} 次采样')
    print('\n正式训练：')
    if losses:
        last = losses[-1]
        print(f"  已更新 {last['step']}/{plan['total_steps']} 步；epoch {last['epoch']:.3f}/3")
        for row in losses[-4:]:
            print(f"  step {row['step']:3d} | loss {row['loss']:.6f} | DPO {row['DPO']:.6f} | KL×10 {row['KL_weighted']:.6f}")
    else:
        print('  已更新 0/795 步；尚无正式训练 loss')
    print(f'\n正式评估：已完成 {len(evaluations)}/15 次；计划每 53 步评估一次')
    for event in evaluations[-4:]:
        metric = event['metrics']['overall']
        print(f"  epoch {event['epoch']:.1f} | step {event['step']} | CER {metric['CER_percent']:.4f}% | ΔR-only {metric['delta_Ronly_pp']:+.4f}pp")
    print('\n独立 smoke 检查（不计入上面的正式训练）：')
    if smoke:
        print(f"  {'已通过' if smoke['passed'] else '未通过'}：{smoke['real_KL_unique']} 条语音、{smoke['distinct_pairs']} 个偏好对、{smoke['steps']} 步检查性更新")
        print('  smoke 权重未用于正式起点；正式模型重新从 R-only 开始。')
    else:
        print('  尚未完成')
    print('\n窗口：monitor 整体进度 | loss 正式训练日志 | evaluation 正式评估 | smoke 检查日志')


if __name__ == '__main__':
    main()
