#!/usr/bin/env python3
"""Render the completed reference/dose experiment without changing its results."""

import argparse
from pathlib import Path

from mas_selection_dpo_probe import read, save
from mas_exploratory_round2_dpo_v1 import verify
from therapist_harness.store import sha256_file


def main(base):
    root = base / 'round2_reference_trajectory_v1'; cfg = verify(root)
    results = read(root / 'results.json'); assert results['status'] == 'complete'
    files = [Path(__file__), root / 'protocol.json', root / 'results.json']
    lines = ['同一次训练的 checkpoint：reference 选择与更新步数', '',
        '每个 reference 条件实际连续训练一次，共两次训练、六次完整511条开发评估。两边从同一1373错的parent开始；每条轨迹保存实际第4/8/12步。', '',
        '| 条件 | 步数 | CER % | 字符错误 | macro CER % | hard 错误 | 織删除 | 明确目标恢复 | 原验收 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    initial = read(base / 'parent_eval_check/summary.json')['metrics']
    current = read(base / 'ASR_target_format_dpo_probe_v1/edit_eos/evaluation_run/evaluation/summary.json')['metrics']
    for label, metrics in [('初始R-only', initial), ('本轮共同parent', current)]:
        m = metrics['overall']
        lines.append(f'| {label} | — | {100*m["pooled_cer"]:.6f} | {m["total_edits"]} | {100*m["patient_macro_cer"]:.6f} | {metrics["hard"]["total_edits"]} | — | — | 基线 |')
    timing = {}
    for arm in ['original_reference', 'MAS_reference']:
        record = results['arms'][arm]
        for step in ['4', '8', '12']:
            item = record['steps'][step]; m = item['metrics']; target = item['target']
            lines.append(f'| {arm} | {step} | {100*m["overall"]["pooled_cer"]:.6f} | {m["overall"]["total_edits"]} | {100*m["overall"]["patient_macro_cer"]:.6f} | {m["hard"]["total_edits"]} | {target["after"]["error_count"]} | {target["target_positions"]["recovered_count"]} | {"通过" if item["eligible"] else "未通过"} |')
            files.append(root / arm / f'step_{step}/results.json')
        snapshot = read(root / arm / 'training_receipts/trajectory_verified.json')
        timing[arm] = dict(training_seconds=record['training_wall_seconds'],
            evaluation_seconds=sum(x['evaluation_wall_seconds'] for x in record['steps'].values()),
            snapshot_seconds=sum(x['snapshot_wall_seconds'] for x in snapshot['snapshots']))
    lines += ['', '原验收同时比较初始R-only和本轮parent：pooled CER下降、macro最多退步0.1个百分点、hard最多退步0.5个百分点。hard相对初始最多694错，规则未改变。', '',
        '| 步数 | MAS选择相对原reference的字符错误变化 | CER变化（百分点） | 患者bootstrap 95%区间 |',
        '|---|---:|---:|---|']
    for step in ['4', '8', '12']:
        path = root / f'paired_reference_step_{step}.json'; pair = read(path)['groups']['overall']; files.append(path)
        a, b = [results['arms'][arm]['steps'][step]['metrics']['overall']['total_edits'] for arm in ['original_reference', 'MAS_reference']]
        lower, upper = pair['patient_cluster_bootstrap_delta_pp_95']
        lines.append(f'| {step} | {b-a:+d} | {pair["delta_pp"]:+.6f} | [{lower:+.6f}, {upper:+.6f}] |')
    lines += ['', '本次两边都使用新MAS内容，因此这里只估计额外reference选择的作用。独立短程运行得到过1364错/20.492788%，但其数值轨迹不同，不能替代本表中实际checkpoint结果。',
        '所有rejected仍来自生成这些偏好数据时共同parent的实际采样；没有新造负样本，没有重新生成或重新采样。真实96条完整GT前缀的KL保持与合成edit+EOS偏好目标均不变，DPO reference及KL teacher仍固定初始R-only。',
        '目标恢复统计基于配对真实目标字符位置；三文本粗审通过不证明生成音频逐字保真。511条是反复使用的开发反馈，本次seed42结果不等同独立泛化验证。旧账本未改写，候选参数尚未晋级。', '',
        '实际耗时（模型加载和teacher计算计入训练；三个checkpoint评估均计入评估）：']
    for arm, value in timing.items():
        lines.append(f'- {arm}：训练{value["training_seconds"]:.1f}秒，评估{value["evaluation_seconds"]:.1f}秒；快照保存开销{value["snapshot_seconds"]:.2f}秒（已包含在训练内）。')
    output = root / 'results.zh.md'
    with output.open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    save(root / 'report_receipt.json', dict(status='complete', report=str(output), report_sha256=sha256_file(output),
        source_sha256={str(p): sha256_file(p) for p in files}, timing=timing,
        no_new_evaluation=True, no_new_training=True, original_results_unchanged=True))
    print(str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
