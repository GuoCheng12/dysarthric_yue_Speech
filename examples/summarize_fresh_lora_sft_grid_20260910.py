"""Live comparison of the four fresh-base recipes; no automatic model promotion."""
import argparse
import json
from pathlib import Path
import time

ROOT=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/qwen3_asr_fresh_lora_sft_seed42_4configs_20260910_v1')
NAMES=['A_legacy_full','B_alpha32_full','C_alpha32_language','D_alpha32_token_ce']


def events(path):
    if not path.exists():return []
    return [json.loads(x) for x in path.read_text().splitlines(keepends=True) if x.endswith('\n') and x.strip()]


def summarize():
    finished=(ROOT/'complete.json').exists()
    failed=any(x['stage']=='failure' for x in events(ROOT/'events.jsonl'))
    lines=['# 原始 Qwen3-ASR＋全新 LoRA：四配置 SFT','',
        '状态：'+('全部完成。' if finished else '出现失败，请查看 supervisor.log。' if failed else '训练与评估进行中。'),'',
        '四组均从原始 Qwen3-ASR-1.7B 载入，初始 adapter 为空；新 LoRA 的 B=0。四组原始基础张量及初始 A/B 哈希均相同。',
        '统一 seed42、2117条真实数据、3 epoch、每轮67步/全程201步。batch4×累积8；线性学习率衰减、warmup0.02、dropout0.05、weight decay0、clip1。',
        '511条开发反馈仅用于评估，与训练音频路径及内容哈希无交集。开发集经过反复使用，不是新的独立测试集。','',
        '| 配置 | alpha/rank | lr | 更新范围 | CE归一化 |',
        '|---|---:|---:|---|---|',
        '| A | 4/16 | 2e-4 | 音频＋语言LoRA | 每语音平均 |',
        '| B | 32/16 | 1e-4 | 音频＋语言LoRA | 每语音平均 |',
        '| C | 32/16 | 1e-4 | 仅语言LoRA | 每语音平均 |',
        '| D | 32/16 | 1e-4 | 音频＋语言LoRA | 每优化步全部有效token平均 |','',
        'A沿用原来的核心LoRA参数，完成全部真实数据训练；统一的3 epoch调度和batch与旧50步实验不同。B使用社区案例的alpha/lr组合，其他设置统一，不是复现其整套数据、增强、dropout或warmup配置。',
        'B/C仅改变音频侧是否可训练。C保留冻结、B恒为0的音频LoRA，以保持语言侧初始化和dropout随机序列一致。B/D仅改变CE归一化。A/B同时改变alpha与lr，只解释为两套参数组合的对照。','']
    base=ROOT/'base_evaluation/summary.json'
    if base.exists():lines.append(f"原始Qwen基线：{json.loads(base.read_text())['CER_percent']:.4f}% CER。")
    lines+=['历史R-only参照：20.7933% CER，旧checkpoint实际看过1600条真实语音；本轮模型未加载它。','',
        '主指标采用与历史一致的BF16合并评估、greedy Cantonese、空上下文、cap256。每次临时合并后从快照精确恢复基础权重，再继续训练；不依赖BF16减法恢复。','',
        '| 配置 | 最近完成步数 | 最近开发CER% | 看完整轮后的最低CER% | 对应步数 |',
        '|---|---:|---:|---:|---:|']
    records={};eligible=[]
    for name in NAMES:
        data=[x for x in events(ROOT/name/'events.jsonl') if x['stage']=='evaluation_complete']
        records[name]=data
        valid=[x for x in data if x['step']>=67]
        best=min(valid,key=lambda x:x['metrics']['CER_percent']) if valid else None
        if best:eligible.append(dict(config=name,step=best['step'],CER_percent=best['metrics']['CER_percent'],
            checkpoint=str(ROOT/name/f"checkpoint-{best['step']}")))
        recent=data[-1] if data else None
        lines.append('| '+name+' | '+(str(recent['step']) if recent else '—')+' | '+
            (f"{recent['metrics']['CER_percent']:.4f}" if recent else '—')+' | '+
            (f"{best['metrics']['CER_percent']:.4f}" if best else '尚未完整看过2117条')+' | '+(str(best['step']) if best else '—')+' |')
    lines+=['','## 各整轮末端','',
        '| 配置 | epoch1/step67 | epoch2/step134 | epoch3/step201 | 完整曝光回执 |',
        '|---|---:|---:|---:|---:|']
    for name,data in records.items():
        by_step={x['step']:x for x in data}
        vals=[f"{by_step[s]['metrics']['CER_percent']:.4f}%" if s in by_step else '—' for s in (67,134,201)]
        lines.append('| '+name+' | '+' | '.join(vals)+f" | {len(list((ROOT/name).glob('epoch-*-exposure.json')))}/3 |")
    if eligible:
        winner=min(eligible,key=lambda x:x['CER_percent'])
        lines+=['',('四组已完成结果中' if not finished else '本轮四组中')+f"，看过完整训练集之后最低的开发CER为 **{winner['CER_percent']:.4f}%**：{winner['config']}，step {winner['step']}。",
            f"权重：`{winner['checkpoint']}`。这只是开发集选出的候选，尚未替换现有ASR。"]
    else:winner=None
    lines+=['','## 全部评估点','',
        '| 配置 | step | epoch | 累计真实曝光 | CER% | vlink247 | 其他 | hard |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,data in records.items():
        for x in data:
            m=x['metrics']
            lines.append(f"| {name} | {x['step']} | {x['epoch']:.3f} | {x['actual_audio_exposures']} | {m['CER_percent']:.4f} | {m['vlink247']['CER_percent']:.4f} | {m['other']['CER_percent']:.4f} | {m['hard']['CER_percent']:.4f} |")
    lines+=['','监督窗口：`tmux attach -t fresh-sft-4cfg`。',
        '代码入口：`/data/qwen3-asr/repo/dysarthric_yue_Speech/examples/run_fresh_lora_sft_grid_20260910.py`。',
        '完整参数、初始化、音频哈希及每轮曝光回执保存在本目录。','']
    (ROOT/'comparison.zh.md').write_text('\n'.join(lines))
    (ROOT/'comparison.json').write_text(json.dumps(dict(complete=finished,failed=failed,
        checkpoints=records,best_complete_epoch_candidate=winner),ensure_ascii=False,indent=2))
    return finished or failed,tuple(len(records[n]) for n in NAMES)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    previous=None
    while True:
        done,status=summarize()
        if status!=previous:print('Completed eval points A/B/C/D:',status,flush=True);previous=status
        if done or not args.watch:break
        time.sleep(10)
