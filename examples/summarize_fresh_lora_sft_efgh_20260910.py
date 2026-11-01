"""Live report for the E/F/G/H scope-by-learning-rate SFT experiment."""
import argparse
import json
from pathlib import Path
import time

ROOT=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/qwen3_asr_fresh_lora_sft_seed42_EFGH_20260910_v1')
NAMES=['E_full_lr5em5','F_full_lr2em5','G_audio_lr5em5','H_audio_lr2em5']


def events(path):
    if not path.exists():return []
    return [json.loads(x) for x in path.read_text().splitlines(keepends=True) if x.endswith('\n') and x.strip()]


def write(path,value):
    temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(value)
    temp.replace(path)


def summarize():
    finished=(ROOT/'complete.json').exists()
    failed=any(x['stage']=='failure' for x in events(ROOT/'events.jsonl'))
    verified=(ROOT/'common_fresh_initialization_verified.json').exists()
    lines=['# E/F/G/H：更新范围 × 学习率','',
        '状态：'+('全部完成。' if finished else '出现失败，查看supervisor.log。' if failed else '运行中。'),'',
        '四组均从原始Qwen3-ASR-1.7B＋全新LoRA开始；不载入R-only或旧实验adapter。',
        '共同初始基础权重与LoRA A/B哈希：'+('已核对完全相同。' if verified else '待四组smoke完成后核对。'),
        '固定seed42、rank16、alpha4、dropout0.05、逐语音平均CE、batch4×累积8、linear schedule、warmup0.02、weight decay0、clip1。',
        '2117条真实音频每轮恰好一次，共3 epoch、201步。四组共用逐轮音频顺序；511条开发反馈音频与训练集路径及内容哈希隔离。','',
        '| GPU | 配置 | 更新范围 | 学习率 |','|---|---|---|---:|',
        '| 0 | E | 音频＋语言LoRA | 5e-5 |',
        '| 1 | F | 音频＋语言LoRA | 2e-5 |',
        '| 2 | G | 仅音频LoRA | 5e-5 |',
        '| 3 | H | 仅音频LoRA | 2e-5 |','',
        'G/H语言侧含LoRA参数均冻结，LoRA B保持零；仍允许loss穿过冻结语言网络反传至音频LoRA。音频/语言之外的原始权重全程冻结。',
        'E/F、G/H比较学习率；E/G、F/H比较是否更新语言LoRA。',
        '每轮step14/27/40/54/67的位置各评估一次（后续轮加67）；主评估BF16合并、greedy Cantonese、空上下文、cap256。每次评估后从快照精确恢复基础权重；最后补一次未合并评估。','',
        'easy/medium/hard是既有音频难度标签，不是互斥患者类别。511条是反复使用的开发反馈集，不是未接触的最终测试集。','']
    base=ROOT/'base_evaluation/summary.json'
    if base.exists():
        m=json.loads(base.read_text())
        lines.append(f"本轮原始Qwen复核：CER {m['CER_percent']:.4f}%，easy {m['easy']['CER_percent']:.4f}%，hard {m['hard']['CER_percent']:.4f}%。")
    lines+=['历史R-only参照：CER20.7933%，easy6.9761%，hard55.1282%。', '',
        '| 配置 | 最新完成评估step | 最新CER | 早期至今最低CER／step | 看满一轮后最低CER／step |','|---|---:|---:|---|---|']
    records={};eligible=[]
    for name in NAMES:
        data=[x for x in events(ROOT/name/'events.jsonl') if x['stage']=='evaluation_complete']
        records[name]=data
        best=min(data,key=lambda x:x['metrics']['CER_percent']) if data else None
        full=[x for x in data if x['step']>=67]
        fullbest=min(full,key=lambda x:x['metrics']['CER_percent']) if full else None
        last=data[-1] if data else None
        fmt=lambda x:f"{x['metrics']['CER_percent']:.4f}%／{x['step']}" if x else '—'
        lines.append(f"| {name} | {last['step'] if last else '—'} | {last['metrics']['CER_percent'] if last else '—'} | {fmt(best)} | {fmt(fullbest)} |")
        if fullbest:eligible.append(dict(config=name,step=fullbest['step'],CER_percent=fullbest['metrics']['CER_percent']))
    winner=min(eligible,key=lambda r:r['CER_percent']) if eligible else None
    lines+=['','## 整轮末端','', '| 配置 | epoch1 | epoch2 | epoch3 | 实际2117条覆盖回执 |','|---|---:|---:|---:|---:|']
    for name,rs in records.items():
        lookup={x['step']:x for x in rs}
        vals=[f"{lookup[s]['metrics']['CER_percent']:.4f}%" if s in lookup else '—' for s in [67,134,201]]
        lines.append('| '+name+' | '+' | '.join(vals)+f" | {len(list((ROOT/name).glob('epoch-*-exposure.json')))}/3 |")
    lines+=['','## 全部评估点','',
        '| 配置 | step | epoch | 真实曝光 | overall CER | easy | medium | hard | vlink247 | other |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,rs in records.items():
        for x in rs:
            m=x['metrics']
            scores=[m['CER_percent']]+[m[k]['CER_percent'] for k in ['easy','medium','hard','vlink247','other']]
            lines.append(f"| {name} | {x['step']} | {x['epoch']:.3f} | {x['actual_audio_exposures']} | "+' | '.join(f'{v:.4f}%' for v in scores)+' |')
    lines+=['','比较目标：区分仅延后退化与最低CER实际改善；同时检查看满一轮后hard收益、easy保持和新内容表现。',
        'tmux：`tmux attach -t fresh-sft-efgh`。代码：`examples/run_fresh_lora_sft_efgh_20260910.py`。','']
    write(ROOT/'comparison.zh.md','\n'.join(lines))
    write(ROOT/'comparison.json',json.dumps(dict(complete=finished,failed=failed,common_initialization_verified=verified,
        checkpoints=records,best_complete_epoch_candidate=winner),ensure_ascii=False,indent=2)+'\n')
    return finished or failed,tuple(len(records[n]) for n in NAMES)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    previous=None
    while True:
        done,status=summarize()
        if status!=previous:print('Completed E/F/G/H evaluations:',status,flush=True);previous=status
        if done or not args.watch:break
        time.sleep(10)
