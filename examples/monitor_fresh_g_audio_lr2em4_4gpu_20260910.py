"""Compact live comparison against the completed G at identical update steps."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/qwen3_asr_fresh_G_lr2em4_4gpu_seed42_20260910_v1')
OUT = ROOT/'G_audio_lr2em4_4gpu'
PREVIOUS = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/qwen3_asr_fresh_lora_sft_seed42_EFGH_20260910_v1/G_audio_lr5em5')


def events(path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True)
            if s.endswith('\n') and s.strip()]


logs = events(OUT/'events.jsonl')
stages = events(ROOT/'events.jsonl')
old = {e['step']: e for e in events(PREVIOUS/'events.jsonl') if e['stage']=='evaluation_complete'}
evals = [e for e in logs if e['stage']=='evaluation_complete']
losses = [e for e in logs if e['stage']=='loss' and 'loss' in e]
status = '全部完成' if (ROOT/'complete.json').exists() else '运行中'
failures = [e for e in stages if e['stage']=='failure']
if failures:
    status = '失败：'+failures[-1]['error']
print(f'G 音频 LoRA | 新 LR 2e-4 vs 原 G 5e-5 | 四卡同步训练和评估 | {status}')
print(datetime.now(timezone.utc).strftime('%H:%M:%S UTC')+' | seed42 | 原始 Qwen + 全新 LoRA | 2117条 x 3轮 | 全局batch32')
print('原始 Qwen CER 25.1352% | 历史 R-only 20.7933% | CER越低越好；差值=新-原G')
print('smoke: '+('通过（梯度等价/末批5条/冻结/四卡权重一致）' if (ROOT/'smoke_passed.json').exists() else '准备或执行中'))
if stages:
    print('最近阶段: '+stages[-1]['stage'])
for e in losses[-2:]:
    print(f"step {e['step']:3d}/201 | epoch {e['epoch']:.3f} | loss {e['loss']:.5f} | lr {e['learning_rate']:.3g} | grad {e['grad_norm']:.4f} | {e['seconds']:.1f}s/step")
pending = [e for e in logs if e['stage']=='evaluation_started' and e['step'] not in {x['step'] for x in evals}]
if pending:
    step = pending[-1]['step']
    done = []
    for r in range(4):
        p = OUT/'evaluations'/f'step-{step}'/f'rank-{r}'
        if (p/'summary.json').exists():
            done.append(str(json.loads((p/'summary.json').read_text())['audio']))
        else:
            es = events(p/'events.jsonl')
            done.append(str(es[-1].get('completed',0)) if es else '0')
    print(f'正在评估 step {step} | 各卡已转写: '+ '/'.join(done))
print('\n step epoch | 新CER   原G CER   差值pp | easy    hard    vlink247 other')
report = ['# G：学习率 2e-4，四卡同步训练', '', '状态：'+status, '',
          '相同原始模型和全新LoRA初始化、rank16/alpha4、dropout0.05、seed42、2117条真实音频、3epoch、全局batch32。',
          '原G学习率5e-5；本轮2e-4。四卡保留原始每步32条及微批次4条，不重复末批5条；rank随机数流和归约舍入不保证与单卡逐位相同。',
          '评估保持原始每批8条，分到四卡。主结果为BF16合并、greedy Cantonese、空上下文、cap256；511条为既有开发反馈集。', '',
          '| step | epoch | 新 CER | 原 G CER | 差值百分点 | easy | hard | vlink247 | other |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for e in evals:
    m, prior = e['metrics'], old[e['step']]['metrics']['CER_percent']
    cer = m['CER_percent']
    print(f" {e['step']:3d}  {e['epoch']:.2f} | {cer:6.2f}% {prior:6.2f}% {cer-prior:+7.2f} | {m['easy']['CER_percent']:6.2f}% {m['hard']['CER_percent']:6.2f}% {m['vlink247']['CER_percent']:6.2f}% {m['other']['CER_percent']:6.2f}%")
    report.append(f"| {e['step']} | {e['epoch']:.3f} | {cer:.4f}% | {prior:.4f}% | {cer-prior:+.4f} | {m['easy']['CER_percent']:.4f}% | {m['hard']['CER_percent']:.4f}% | {m['vlink247']['CER_percent']:.4f}% | {m['other']['CER_percent']:.4f}% |")
if evals:
    best = min(evals, key=lambda e:e['metrics']['CER_percent'])
    print(f"\n最佳新CER {best['metrics']['CER_percent']:.4f}% @step{best['step']} | 原G最好 23.2272% @step201")
print(f'评估完成 {len(evals)}/15 | 整轮实际数据覆盖 {len(list(OUT.glob("epoch-*-exposure.json")))}/3')
print('窗口: Ctrl+b 后按 0=四卡监督 / 1=本页 / 2=rank0日志 / 3=GPU占用')
report.extend(['', 'tmux：`tmux attach -t fresh-sft-g-lr2e4`'])
if ROOT.exists():
    pending_file = ROOT/'comparison.zh.md.pending'
    pending_file.write_text('\n'.join(report)+'\n')
    pending_file.replace(ROOT/'comparison.zh.md')
