"""Summarize fixed-pair audio dependence probes, including patient bootstrap."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np

from probe_fresh_sft_audio_dependence_20260910 import ROOT, CONFIGS, read, save


def values(row, field='text_nll'):
    correct = row['correct'][field]
    wrong = (row['wrong0'][field] + row['wrong1'][field]) / 2
    return correct, wrong, wrong - correct


def bootstrap_delta(early, late):
    assert [r['utt_id'] for r in early] == [r['utt_id'] for r in late]
    by_patient = defaultdict(list)
    for a, b in zip(early, late):
        by_patient[a['speaker_id']].append(values(b)[2] - values(a)[2])
    sums = np.array([sum(v) for v in by_patient.values()])
    counts = np.array([len(v) for v in by_patient.values()])
    ix = np.random.default_rng(42).integers(0, len(sums), (3000, len(sums)))
    means = sums[ix].sum(1) / counts[ix].sum(1)
    return dict(delta=float(sums.sum()/counts.sum()), patient_bootstrap_95=np.quantile(means,[.025,.975]).tolist(),
                patients=len(sums), resamples=3000,
                correct_delta=float(np.mean([values(b)[0]-values(a)[0] for a,b in zip(early,late)])),
                wrong_delta=float(np.mean([values(b)[1]-values(a)[1] for a,b in zip(early,late)])),
                permutation_delta=[float(np.mean([(b[f'wrong{k}']['text_nll']-b['correct']['text_nll']) -
                                                   (a[f'wrong{k}']['text_nll']-a['correct']['text_nll']) for a,b in zip(early,late)])) for k in range(2)])


def main():
    models = {}
    for cfg in CONFIGS:
        for path in (ROOT / cfg).glob('*_*.json'):
            if path.name.endswith(('_merged.json','_unmerged.json')):
                models[path.stem] = read(path)
    table = []
    for name, records in sorted(models.items()):
        for split in ['train','evaluation']:
            for difficulty in ['all','easy','medium','hard']:
                rs = [r for r in records if r['split']==split and (difficulty=='all' or r['difficulty']==difficulty)]
                for field in ['text_nll','first_nll','continuation_nll','supervised_nll']:
                    mean = np.array([values(r,field) for r in rs]).mean(0)
                    table.append(dict(model=name,split=split,difficulty=difficulty,field=field,audio=len(rs),
                                      correct=float(mean[0]),wrong=float(mean[1]),gap=float(mean[2])))
    comparisons=[]
    for cfg, early_step in CONFIGS.items():
        for mode in ['unmerged','merged']:
            early_name=f'{cfg[0]}_step{early_step}_{mode}'
            late_name=f'{cfg[0]}_step67_{mode}'
            if not all(n in models for n in [early_name,late_name]):
                continue
            for split in ['train','evaluation']:
                early=[r for r in models[early_name] if r['split']==split]
                late=[r for r in models[late_name] if r['split']==split]
                comparisons.append(dict(config=cfg,mode=mode,split=split,**bootstrap_delta(early,late)))
    completed=all((ROOT/cfg/'complete.json').exists() for cfg in CONFIGS)
    save(ROOT/'results.json',dict(complete=completed,summary=table,early_to_late=comparisons))
    with (ROOT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# 正确音频 vs 同患者错配音频：已有SFT checkpoint探针','',f'状态：{"完成" if completed else "进行中，仅含已完成前向结果"}。',
           '', '固定34位患者；训练面板272条、48句；评估面板265条、12句。每位患者最多8句，每句随机取一条；两边是同一批患者。它们是诊断面板，不能直接当作完整数据集指标。',
           '', '保持目标文本和teacher-forced正确前文不变；错配音频来自同患者的另一句，每条做两次固定排列。匹配优先接近时长、接近历史难度；难度并非强制一致。',
           '', '主要量：正文每token平均NLL，先在每条音频内平均再对面板平均。排除固定语言头和EOS。gap=两种错配平均NLL−正确音频NLL；gap正值代表正确音频有优势。报告首token与后续token作为补充。',
           '', '未合并采用BF16 backbone、FP32 adapter、BF16 autocast；合并为独立的同权重精度复核。无梯度、无优化器，不更改任何checkpoint。',
           '', '## 正文NLL', '', '| checkpoint | 面板 | 正确音频 | 错配音频 | gap |', '|---|---|---:|---:|---:|']
    for r in table:
        if r['difficulty']=='all' and r['field']=='text_nll':
            lines.append(f'| {r["model"]} | {r["split"]} | {r["correct"]:.4f} | {r["wrong"]:.4f} | {r["gap"]:.4f} |')
    lines+=['', '## 从早期最佳到step67的变化', '',
            '负的gap变化支持该指标下音频配对区分减弱；正值表示区分增强。必须同时看正确/错配绝对loss，不能把gap单独等同为听音能力或真实解码的因果解释。', '',
            '| 配置 | 方式 | 面板 | 正确NLL变化 | 错配NLL变化 | gap变化 | 患者bootstrap 95%区间 |', '|---|---|---|---:|---:|---:|---|']
    for r in comparisons:
        lo,hi=r['patient_bootstrap_95']
        lines.append(f'| {r["config"]} | {r["mode"]} | {r["split"]} | {r["correct_delta"]:+.4f} | {r["wrong_delta"]:+.4f} | {r["delta"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    lines+=['', '区间为固定面板内按患者聚类抽样3000次；它不覆盖重新选择训练/评估句子的变异。两次错配的分别变化保存在results.json；全部难度与首token等分项在summary.csv。',
            '', '本探针有正确目标前缀，不能独立验证自由生成时的音频依赖；下降也可能来自正确配对本身变差，上升也不意味着泛化改善。', '']
    (ROOT/'report.zh.md').write_text('\n'.join(lines))
    print(json.dumps(dict(complete=completed,variants=len(models),comparisons=comparisons),ensure_ascii=False))


if __name__=='__main__':
    main()
