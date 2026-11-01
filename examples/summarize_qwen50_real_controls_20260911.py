"""Report saved Qwen trajectories alongside the completed Whisper controls."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1')
Q = ROOT / 'qwen3_asr_fresh_G_lr2em4_2gpu_seed42_50epochs_20260910_v1'
B = Q / 'G_audio_lr2em4_2gpu_50epochs'
W = ROOT / 'whisper_large_v3_real2117_lora_seed42_4configs_20260910_v2'
PROOF = ROOT / 'qwen50_whisper_real_controls_completion_20260910_v1/verification.json'


def read(path):
    return json.loads(path.read_text())


def events(path):
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True)
            if s.endswith('\n') and s.strip()]


def groups(m):
    return dict(all=m['CER_percent'], **{k: m[k]['CER_percent']
                for k in ['easy', 'medium', 'hard', 'vlink247', 'other']})


def main():
    es = events(B / 'events.jsonl')
    ev = [e for e in es if e['stage'] == 'evaluation_complete']
    losses = [e for e in es if e['stage'] == 'loss']
    best, latest = min(ev, key=lambda e: e['metrics']['CER_percent']), ev[-1]
    epochs = {e['step'] // 67: e for e in ev if e['step'] % 67 == 0}
    ce = {}
    for epoch in epochs:
        rows = [e for e in losses if (e['step'] - 1) // 67 + 1 == epoch]
        assert len(rows) == 67 and sum(e['actual_batch'] for e in rows) == 2117
        ce[epoch] = sum(e['loss'] * e['actual_batch'] for e in rows) / 2117
    whisper = read(W / 'final_findings.json')
    baselines = whisper['baselines']
    done = (Q / 'complete.json').exists()
    verified = done and PROOF.exists() and read(PROOF)['status'] == 'verified'
    status = '50轮完成且联合验收通过' if verified else ('训练结束，等待联合验收' if done else '训练运行中')
    now = datetime.now(timezone.utc).isoformat()
    lines = ['# Qwen 50轮与 Whisper 真实数据 SFT 对照', '', f'快照：{now}。状态：{status}。', '',
             f"Qwen 最佳 CER **{best['metrics']['CER_percent']:.4f}%**，位于第 {best['epoch']:.3f} 轮；"
             f"最新已评估第 {latest['epoch']:.3f} 轮，CER **{latest['metrics']['CER_percent']:.4f}%**。", '',
             '原始 Qwen3-ASR-1.7B＋全新 LoRA，seed42，仅更新 audio tower LoRA；LR2e-4、rank16、alpha4、dropout0.05、全局batch32，双卡同步。'
             '2117条真实音频每轮完整覆盖，计划50轮；每约1/5轮评估相同511条既有开发音频。'
             '训练48个目标句与开发12个目标句无重叠，患者有重叠。该开发集已被反复使用。', '',
             '|模型或阶段|总体 CER|easy|medium|hard|vlink247|other|', '|---|---:|---:|---:|---:|---:|---:|']
    table = [(n, baselines[n]) for n in ['原始 Qwen', '历史 Qwen R-only', '原始 Whisper']]
    table += [('Qwen 本轮最佳', groups(best['metrics'])), ('Qwen 本轮最新', groups(latest['metrics']))]
    if done:
        table.append(('Qwen 最终未合并 LoRA', groups(read(B / 'complete.json')['final_unmerged_metrics'])))
    for name, values in table:
        lines.append('|' + name + '|' + '|'.join(f'{values[k]:.4f}%' for k in ['all', 'easy', 'medium', 'hard', 'vlink247', 'other']) + '|')
    lines += ['', '逐阶段主指标沿用 BF16 合并 LoRA 评估；最后另做一次未合并评估，单列结果，不替换此前主曲线。', '',
              '|完整 epoch|平均在线训练 CE|开发 CER|', '|---:|---:|---:|']
    for epoch, e in epochs.items():
        lines.append(f"|{epoch}|{ce[epoch]:.6f}|{e['metrics']['CER_percent']:.4f}%|")
    lines += ['', '训练 CE 按每步实际音频数加权，是在线整轮均值，不是重新计算的 checkpoint 训练集 loss。', '',
              '一次已核查的短时尖峰：step3055→3069，CER由32.39%升至35.70%。'
              '其中一条 vlink247 音频出现273字的重复输出，增加255个字符错误；其余510条合计减少35个错误，净增220。'
              '该分解不剔除任何样本，也不等同于用单个循环解释整个长期退化趋势。'
              f'[逐条核查记录]({Q}/step3069_spike_decomposition.json)。'
              '同一条音频在step3270再次产生相同长循环，贡献255个新增错误，其余510条合计减少10个错误；'
              f'该次全量CER为36.36%。[第二次核查记录]({Q}/step3270_spike_decomposition.json)。', '',
              '本次50轮使用3350步线性衰减，旧10轮实验使用670步线性衰减，均只有5步 warmup。'
              '两者同为初始LR2e-4，但相同epoch的实际LR不同；GPU数量与dropout随机轨迹也不同，不能只归因于训练轮数。', '',
              f'![Qwen训练、分组评估与学习率]({Q}/training_trends.png)', '',
              f'[Whisper四组最终对照与三个研究问题的回答]({W}/final_findings.zh.md)；'
              f'[Qwen逐阶段语境替换追踪]({ROOT}/qwen_g50_contextual_substitution_audit_20260910_v1/report.zh.md)。', '',
              '语境替换规则是有限的转写分析，不能证明所有退化来自“更合理地猜词”。'
              '整句训练文本未命中也不能排除间接或片段记忆。跨模型比较还受到预训练、LoRA规模与训练设置差异影响。', '',
              f'[实际训练与评估事件]({B}/events.jsonl) · [训练协议]({Q}/protocol.json) · '
              f'[双卡正确性 smoke]({Q}/smoke_passed.json)']
    if verified:
        lines += ['', f'[联合完成校验]({PROOF})']
    (Q / 'final_findings.zh.md').write_text('\n'.join(lines) + '\n')
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    xs = [e['epoch'] for e in ev]
    ax[0, 0].plot(xs, [e['metrics']['CER_percent'] for e in ev], label='Qwen 50-epoch schedule')
    ax[0, 0].axhline(baselines['历史 Qwen R-only']['all'], ls='--', color='black', label='Historical R-only')
    ax[0, 0].scatter([best['epoch']], [best['metrics']['CER_percent']], color='red', zorder=4, label='Best observed')
    ax[0, 0].set_ylabel('Development CER (%)'); ax[0, 0].legend()
    for k in ['easy', 'medium', 'hard']:
        ax[0, 1].plot(xs, [e['metrics'][k]['CER_percent'] for e in ev], label=k)
    ax[0, 1].set_ylabel('Development CER (%)'); ax[0, 1].legend()
    ax[1, 0].plot(list(ce), list(ce.values()))
    ax[1, 0].set_yscale('log'); ax[1, 0].set_ylabel('Mean online training CE (log scale)')
    ax[1, 1].plot([e['epoch'] for e in losses], [e['learning_rate'] for e in losses])
    ax[1, 1].set_ylabel('Logged learning rate')
    for a in ax.flat:
        a.set_xlabel('Epoch'); a.grid(alpha=.2)
    for ext in ['png', 'svg']:
        fig.savefig(Q / f'training_trends.{ext}', dpi=160)
    plt.close(fig)
    summary = dict(utc=now, status=status, complete=done, jointly_verified=verified,
                   best=best, latest=latest, mean_online_train_CE_by_epoch=ce,
                   evaluations=len(ev), actual_audio_exposures=losses[-1]['actual_audio_exposures'],
                   source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [B / 'events.jsonl', Q / 'protocol.json', W / 'final_findings.json']})
    (Q / 'final_findings.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['status', 'evaluations', 'actual_audio_exposures']}))


if __name__ == '__main__':
    main()
