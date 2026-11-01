"""Summarize completed Whisper controls without changing any training process."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from therapist_harness.text import text_normalize, edit_distance

ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1')
W = ROOT / 'whisper_large_v3_real2117_lora_seed42_4configs_20260910_v2'
Q = ROOT / 'qwen3_asr_fresh_G_lr2em4_2gpu_seed42_50epochs_20260910_v1'
RONLY = ROOT / 'qwen3_asr_lora_sft_audit_20260910_v2/evaluations/000_continue_original/dev_merged.predictions.json'
NAMES = ['W-A', 'W-B', 'W-C', 'W-D']


def read(path):
    return json.loads(path.read_text())


def metrics(rows):
    def cer(xs):
        return 100 * sum(x['errors'] for x in xs) / sum(x['chars'] for x in xs)
    return dict(all=cer(rows), **{k: cer([x for x in rows if x['zero_shot_bucket'] == k])
                                 for k in ['easy', 'medium', 'hard']},
                vlink247=cer([x for x in rows if x['speaker_id'] == 'vlink247']),
                other=cer([x for x in rows if x['speaker_id'] != 'vlink247']))


def main():
    assert read(W / 'complete.json')['passed'] and read(W / 'completion_verified.json')['passed']
    protocol = read(W / 'protocol.json')
    dev = {r['utt_id']: r for r in read(W / 'dev.json')}
    train = read(W / 'train.json')
    sources, baselines, results = {}, {}, {}
    for label, path in [('原始 Qwen', Q / 'base_evaluation/predictions.json'),
                        ('历史 Qwen R-only', RONLY), ('原始 Whisper', W / 'base_evaluation/predictions.json')]:
        rows = read(path)
        assert len(rows) == 511 and {r['utt_id'] for r in rows} == set(dev)
        for row in rows:
            d = dev[row['utt_id']]
            gt, hyp = text_normalize(d['clean_gt']), text_normalize(row['prediction'])
            assert row['errors'] == edit_distance(gt, hyp) and row['chars'] == len(gt)
            assert row['speaker_id'] == d['speaker_id'] and row['zero_shot_bucket'] == d['zero_shot_bucket']
        baselines[label] = metrics(rows)
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = read(W / 'base_evaluation/predictions.json')
    original_caps = {r['utt_id'] for r in raw if r['hit_token_cap']}
    original_noncap = [r for r in raw if r['utt_id'] not in original_caps]
    raw_noncap_cer = 100 * sum(r['errors'] for r in original_noncap) / sum(r['chars'] for r in original_noncap)
    for name in NAMES:
        branch = W / name
        es = [json.loads(s) for s in (branch / 'events.jsonl').read_text().splitlines() if s.strip()]
        ev = [e for e in es if e['stage'] == 'evaluation_complete']
        losses = [e for e in es if e['stage'] == 'loss']
        best = min(ev, key=lambda e: e['metrics']['CER_percent'])
        paths = [branch / f"evaluations/step-{s}/predictions.json" for s in [best['step'], 201]]
        before, after = [read(p) for p in paths]
        assert [r['utt_id'] for r in before] == [r['utt_id'] for r in after]
        union = {x['utt_id'] for x, y in zip(before, after) if x['hit_token_cap'] or y['hit_token_cap']}
        changes = {x['utt_id']: y['errors'] - x['errors'] for x, y in zip(before, after)}
        total_delta = sum(changes.values())
        cap_delta = sum(changes[k] for k in union)
        ce = {i: sum(e['loss'] * e['actual_batch'] for e in losses if (e['step'] - 1) // 67 == i - 1) / 2117
              for i in [1, 2, 3]}
        target_gains = {}
        for stage, rows in [('best', before), ('final', after)]:
            improved = 0
            for target in {d['clean_gt'] for d in dev.values()}:
                ids = {k for k, d in dev.items() if d['clean_gt'] == target}
                improved += sum(r['errors'] for r in rows if r['utt_id'] in ids) < sum(r['errors'] for r in raw if r['utt_id'] in ids)
            fixed = [r for r in rows if r['utt_id'] not in original_caps]
            target_gains[stage] = dict(improved_targets=improved,
                                      original_noncap_CER=100 * sum(r['errors'] for r in fixed) / sum(r['chars'] for r in fixed))
        results[name] = dict(config=protocol['configs'][name],
                            trainable_parameters=read(branch / 'smoke_passed.json')['trainable_parameters'],
                            best_step=best['step'], best_epoch=best['epoch'], best=metrics(before), final=metrics(after),
                            mean_online_train_loss_by_epoch=ce, target_gains=target_gains,
                            best_to_final_delta_errors=total_delta, cap_union_ids=sorted(union),
                            cap_union_delta_errors=cap_delta, other_delta_errors=total_delta-cap_delta,
                            best_to_final_delta_pp=100*total_delta/6656,
                            cap_union_delta_pp=100*cap_delta/6656, other_delta_pp=100*(total_delta-cap_delta)/6656)
        for p in paths:
            sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    trajectory = read(W / 'error_analysis/trajectories.json')
    assert all(len(trajectory[n]) == 16 for n in NAMES)
    counts_figure(trajectory)
    now = datetime.now(timezone.utc).isoformat()
    lines = ['# Whisper large-v3 真实数据 SFT：四组最终结果', '',
             f'生成时间：{now}。Whisper 四组均完成三轮；Qwen 50 轮实验单独记录，不以本报告宣称其完成。', '',
             '本轮最好的 Whisper 为 W-C 的第 1 轮 checkpoint，CER 27.04%；历史 Qwen R-only 为 20.79%。四组均明显改善了原始 Whisper，但都没有超过该基线。', '',
             f"官方模型 {protocol['model_id']}，固定 revision {protocol['revision']}。seed42；2117 条真实音频、48 个训练目标句；511 条既有开发音频、12 个不重叠目标句。开发集已多次用于反馈，不是未使用过的测试集。患者与训练集有重叠。", '',
             '统一 LoRA r16、alpha32、dropout0.05，global batch32、micro batch4，末批5条；线性学习率衰减、5步 warmup，每约1/5轮评估。每组从原始权重和全新 LoRA 开始。粤语 yue、转写、greedy、空提示，不跨独立音频继承上下文；全部音频均计入 CER。', '',
             '|配置|更新范围|初始 LR|可训练参数|最佳 epoch|最佳 CER|第三轮末 CER|末端比最佳增加|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        scope = 'encoder＋decoder LoRA' if r['config']['decoder'] else 'encoder LoRA'
        lines.append(f"|{name}|{scope}|{r['config']['lr']:.0e}|{r['trainable_parameters']:,}|{r['best_epoch']:.3f}|{r['best']['all']:.4f}%|{r['final']['all']:.4f}%|{r['best_to_final_delta_pp']:.4f} 点|")
    lines += ['', '**全量与分组 CER**', '', '|模型/阶段|总体|easy|medium|hard|vlink247|other|',
              '|---|---:|---:|---:|---:|---:|---:|']
    table = list(baselines.items()) + [(f'{n} {label}', r[stage]) for n, r in results.items()
                                     for stage, label in [('best', '最佳'), ('final', '第三轮末')]]
    for label, values in table:
        lines.append('|' + label + '|' + '|'.join(f'{values[k]:.4f}%' for k in ['all','easy','medium','hard','vlink247','other']) + '|')
    lines += ['', '**三个问题的回答**', '',
              '1. 原始 Whisper 在这批患者音频上的 CER 为 90.41%，hard 组为 177.08%。CER 超过100%来自插入和重复，并非计算上限错误。原始模型有15条生成达到长度上限；其余固定496条的 CER 为53.41%。原始表现较弱，不能把大幅相对提升直接等同于优于 Qwen。', '',
              '2. encoder-only 也有跨句适配收益：W-A、W-B 的最佳 checkpoint 均改善了12/12个未见训练目标句。收益不只来自原始15条长输出。该结果支持在这些未见句子上的识别改善，不能单独证明具体学到了哪些发音规律，也不能推出未见患者泛化。', '',
              '3. 在本次 seed42 和 LR5e-5 下，联合更新 decoder 的后期退化明显更大；W-D 的训练 CE 继续下降，开发 CER 却反弹。LR2e-5 的 W-C 早期最佳值更好，末端仍退化。冻结 decoder 的 W-A、W-B 也没有消除退化，因此不能把所有问题都归因于 decoder 参数更新。', '',
              '|配置|第1轮平均训练 CE|第3轮平均训练 CE|最佳点原始非触顶496条 CER|末端同一496条 CER|',
              '|---|---:|---:|---:|---:|']
    for n, r in results.items():
        lines.append(f"|{n}|{r['mean_online_train_loss_by_epoch'][1]:.4f}|{r['mean_online_train_loss_by_epoch'][3]:.4f}|{r['target_gains']['best']['original_noncap_CER']:.4f}%|{r['target_gains']['final']['original_noncap_CER']:.4f}%|")
    lines += ['', '训练 CE 是按实际音频数加权的在线整轮均值，不是重新计算的 checkpoint 训练集损失。496条队列由原始模型定义，后期新出现的长循环仍计入该队列；这不是剔除所有失败后的新主指标。', '',
              '**最佳点到末端：错误增加的组成**', '',
              '|配置|新增字符错误|两端任一触顶的音频数|这些音频的错误变化|其余音频的错误变化|',
              '|---|---:|---:|---:|---:|']
    for n, r in results.items():
        lines.append(f"|{n}|{r['best_to_final_delta_errors']}|{len(r['cap_union_ids'])}|{r['cap_union_delta_errors']}|{r['other_delta_errors']}|")
    lines += ['', '这是对全量 CER 变化的事后分解，所有511条仍保留。各组触顶集合不同，不能把“其余音频”当作统一的新测试集。未触顶输出也可能含长循环：W-D 末端已核查的三条非触顶重复输出合计增加448个字符错误，包含在上述1155中。因此该表不能解释为“循环错误与普通转写错误”的完整划分。', '',
              '**语境换词与训练文本吸附**', '',
              '原有三条规则只跟踪繳件→票券、鑲條裙褂→將條裙掛、書冊→書籍。它们不是全部语义错误的定义。四组所有阶段均未出现完整转写命中48个训练目标句；这不排除词片段或间接词汇效应。', '',
              '扩查全部早对后错候选后，W-C step148 存在同一音频从正确的“拎拐杖”变成“拿拐杖”或“攞拐杖”的实例；另有“人生→人身”“蠍子→竊子”等没有更合理的混淆。因此存在语义相近改写，但不支持把全部退化都描述为“越来越会用合理语境猜词”。这里比较转写和标注，没有重新听音核验这些个案。', '',
              f'[完整早对后错候选与实例]({W}/correctness_transition_analysis/report.zh.md)；[被检查词语在训练标签中的出现次数]({W}/correctness_transition_analysis/training_lexical_occurrences.json)；[逐句迁移]({W}/sentence_transfer_analysis/report.zh.md)。', '',
              '四组 smoke 均通过特殊 token、padding/EOS mask、梯度与冻结范围、保存重载 token 一致性检查。最终核验覆盖60次全量评估及原始基线，共31,171条预测、25,404次训练音频曝光；源码、输入、模型及 checkpoint 哈希均核对。', '',
              'Whisper 使用未合并 LoRA 评估，Qwen 历史指标沿用 BF16 合并评估。模型预训练和历史 R-only 训练配置不同；单个 seed、反复使用的开发集和最佳点选择也限制了结论。这里的差异不能全部归因于模型架构或语言 prior。', '',
              f'![训练和开发趋势]({W}/training_trends.png)', '',
              f'![三个固定词组规则和训练句命中]({W}/error_pattern_counts.png)', '',
              f'[核验记录]({W}/completion_verified.json) · [完整逐阶段指标]({W}/comparison.zh.md) · [原始15条与其余496条的固定队列分析]({W}/fixed_cohort_analysis/report.zh.md)', '',
              '模型与相关研究：[官方 large-v3](https://huggingface.co/openai/whisper-large-v3)；[英语 TORGO encoder-only 研究](https://aclanthology.org/2025.emnlp-main.1701/)。本实验不是该英语 small/medium 研究的精确复现。']
    (W / 'final_findings.zh.md').write_text('\n'.join(lines) + '\n')
    summary = dict(utc=now, baselines=baselines, results=results, raw_original_noncap_CER=raw_noncap_cer,
                   source_predictions_sha256=sources, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (W / 'final_findings.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({n: {k: v for k, v in r.items() if k in ['best_to_final_delta_errors','cap_union_delta_errors','other_delta_errors','mean_online_train_loss_by_epoch']} for n, r in results.items()}))


def counts_figure(trajectory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for name in NAMES:
        stages = trajectory[name]; x = [s.get('epoch', 0) for s in stages]
        for ax, key in zip(axes[:3], ['ticket','hang_skirt','book_synonym']):
            ax.plot(x, [s['pattern_counts'].get(key, 0) for s in stages], label=name)
        axes[3].plot(x, [s['exact_train'] for s in stages], label=name)
    for ax, title in zip(axes, ['Ticket substitution (n=46)','Hang-skirt substitution (n=43)','Book synonym (n=40)','Exact training-sentence matches']):
        highest = max(max(line.get_ydata()) for line in ax.lines)
        ax.set(title=title, xlabel='Epoch', ylabel='Number of audio samples', ylim=(0, max(1, highest+1)))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(W / 'error_pattern_counts.png', dpi=160); fig.savefig(W / 'error_pattern_counts.svg'); plt.close(fig)


if __name__ == '__main__':
    main()
