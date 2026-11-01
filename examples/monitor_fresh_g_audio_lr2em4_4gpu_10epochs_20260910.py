"""Ten-epoch G monitor, with both completed three-epoch G runs as references."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/qwen3_asr_fresh_G_lr2em4_4gpu_seed42_10epochs_20260910_v1')
OUT = ROOT/'G_audio_lr2em4_4gpu_10epochs'
RECORDS = ROOT.parent
LOW = RECORDS/'qwen3_asr_fresh_lora_sft_seed42_EFGH_20260910_v1/G_audio_lr5em5'
HIGH = RECORDS/'qwen3_asr_fresh_G_lr2em4_4gpu_seed42_20260910_v1/G_audio_lr2em4_4gpu'


def events(path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True)
            if s.endswith('\n') and s.strip()]


def evaluations(path):
    return {e['step']: e for e in events(path/'events.jsonl') if e['stage']=='evaluation_complete'}


def cer_at(records, step):
    return f"{records[step]['metrics']['CER_percent']:.2f}%" if step in records else '—'


logs = events(OUT/'events.jsonl')
stages = events(ROOT/'events.jsonl')
low, high = evaluations(LOW), evaluations(HIGH)
evals = [e for e in logs if e['stage']=='evaluation_complete']
by_step = {e['step']:e for e in evals}
losses = [e for e in logs if e['stage']=='loss' and 'loss' in e]
status = '全部完成' if (ROOT/'complete.json').exists() else '运行中'
failures = [e for e in stages if e['stage']=='failure']
if failures:
    status = '失败：'+failures[-1]['error']
print(f'G 音频 LoRA | LR 2e-4 | 四卡同步 | 10 epoch / 670 step | {status}')
print(datetime.now(timezone.utc).strftime('%H:%M:%S UTC')+' | 同一原始 Qwen + 全新 LoRA | seed42 | 2117条/轮 | 全局batch32')
print('本次线性衰减到670步，warmup仍5步；旧3轮衰减到201步，因此第3轮checkpoint并非同一训练轨迹。')
print('原始 Qwen CER 25.1352% | R-only 20.7933% | 旧G末端: LR5e-5 23.2272%, LR2e-4 23.4976%')
print('smoke: '+('通过' if (ROOT/'smoke_passed.json').exists() else '准备或执行中'))
for e in losses[-2:]:
    print(f"step {e['step']:3d}/670 | epoch {e['epoch']:.3f} | loss {e['loss']:.5f} | lr {e['learning_rate']:.3g} | grad {e['grad_norm']:.4f} | {e['seconds']:.1f}s/step")
pending = [e for e in logs if e['stage']=='evaluation_started' and e['step'] not in by_step]
if pending:
    print(f"正在四卡评估 step {pending[-1]['step']}（共511条）")
print('\n最近5次评估（CER越低越好；旧实验仅有前三轮）')
print(' step epoch | 新10轮   旧3轮2e-4  原G5e-5 | easy    hard    vlink247 other')
for e in evals[-5:]:
    m = e['metrics']
    print(f" {e['step']:3d}  {e['epoch']:.2f} | {m['CER_percent']:6.2f}% {cer_at(high,e['step']):>10} {cer_at(low,e['step']):>9} | {m['easy']['CER_percent']:6.2f}% {m['hard']['CER_percent']:6.2f}% {m['vlink247']['CER_percent']:6.2f}% {m['other']['CER_percent']:6.2f}%")
print('\n每轮末端 CER：')
for epoch in range(1,11):
    step = 67*epoch
    if step in by_step:
        m = by_step[step]['metrics']
        print(f"  epoch {epoch:2d} | 新 {m['CER_percent']:6.2f}% | 旧3轮2e-4 {cer_at(high,step):>7} | 原G5e-5 {cer_at(low,step):>7} | easy {m['easy']['CER_percent']:.2f}% hard {m['hard']['CER_percent']:.2f}%")
if evals:
    best = min(evals,key=lambda e:e['metrics']['CER_percent'])
    print(f"最佳 CER {best['metrics']['CER_percent']:.4f}% @step{best['step']} / epoch{best['epoch']:.3f}")
print(f'评估 {len(evals)}/50 | 实际整轮覆盖 {len(list(OUT.glob("epoch-*-exposure.json")))}/10 | 保留每轮末端+最佳checkpoint')
print('窗口: Ctrl+b 后按 0=监督 / 1=本页 / 2=loss日志 / 3=GPU')

report = ['# G 音频 LoRA：四卡、LR 2e-4、10 epoch', '', '状态：'+status, '',
          '本轮从相同原始Qwen+全新LoRA初始化开始，seed42，2117条真实音频/轮，全局batch32，rank16/alpha4/dropout0.05。',
          '线性学习率计划从原3轮201步延长到10轮670步，warmup保持5步；旧3轮结果保留，不能将两条轨迹拼接为连续训练。',
          '每约1/5轮评估完整511条，共50次；原有开发反馈集、BF16合并、greedy Cantonese、空上下文、cap256。',
          '前3轮训练顺序与之前一致，后7轮使用seed42+epoch继续生成顺序；每轮末批5条，无重复填充。',
          '所有评估记录保留；checkpoint仅保留10个整轮末端和1个滚动最佳，best_checkpoint_updated事件记录最佳替换。', '',
          '| step | epoch | 新10轮 CER | 旧3轮2e-4 | 原G5e-5 | easy | medium | hard | vlink247 | other |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for e in evals:
    m = e['metrics']
    report.append(f"| {e['step']} | {e['epoch']:.3f} | {m['CER_percent']:.4f}% | {cer_at(high,e['step'])} | {cer_at(low,e['step'])} | {m['easy']['CER_percent']:.4f}% | {m['medium']['CER_percent']:.4f}% | {m['hard']['CER_percent']:.4f}% | {m['vlink247']['CER_percent']:.4f}% | {m['other']['CER_percent']:.4f}% |")
report.extend(['', 'tmux：`tmux attach -t fresh-sft-g-10ep`'])
if ROOT.exists():
    p = ROOT/'comparison.zh.md.pending'
    p.write_text('\n'.join(report)+'\n')
    p.replace(ROOT/'comparison.zh.md')
