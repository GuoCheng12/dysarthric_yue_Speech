#!/usr/bin/env python3
"""Persist development results and EOS diagnostics for the declared format probe."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from statistics import mean

from mas_selection_dpo_probe import read, rows, save
from mas_evaluation_report import compare
from therapist_harness.evidence import compare_prediction_snapshots
from therapist_harness.round_schema import ClusterDescriptor
from therapist_harness.store import sha256_file


def summarize(base, out):
    result = read(out / 'results.json')
    assert result['status'] == 'complete'
    assert read(out / 'verified.json')['status'] == 'passed'
    metrics = result['metrics']
    parent = read(base / 'parent_eval_check/summary.json')['metrics']
    groups = read(out / 'paired_parent.json')['groups']
    p30_delta = round(groups['vlink247']['delta_pp'] / 100 * 1384)
    evidence = read(base / 'lexical96_mas_v1/round_001/evidence.json')
    proposal = read(base / 'lexical96_mas_v1/round_001/proposal.json')
    ids = {x['subject_evidence_id'] for x in proposal['expected_outcomes']}
    clusters = [ClusterDescriptor(**{k: x[k] for k in ClusterDescriptor.model_fields})
                for x in evidence['error_clusters']
                if x['evidence_id'] in ids and x['operation'] == 'substitution']
    response = compare_prediction_snapshots(
        rows(base / 'parent_eval_check/predictions.jsonl'),
        rows(out / 'evaluation_run/evaluation/predictions.jsonl'),
        before_prediction_column='prediction', after_prediction_column='prediction',
        critical_cer_threshold=.5, monitored_clusters=clusters,
        off_target_max_per_level_per_direction=3)
    response = json.loads(json.dumps(asdict(response), default=lambda x: x.model_dump(mode='json')))
    response_path = out / 'target_response.json'
    if response_path.exists():
        assert read(response_path) == response
    else:
        save(response_path, response)
    destination = out / 'checkpoint_metadata_receipts'
    if not destination.exists():
        destination.mkdir()
        copied = []
        for name in ['residual_composition.json', 'trainer_state.json',
                     'optimizer_omission.json', 'adapter_config.json']:
            source = Path(result['checkpoint']) / name
            target = destination / name
            shutil.copyfile(source, target)
            digest = sha256_file(source)
            assert sha256_file(target) == digest
            copied.append(dict(source=str(source), copy=str(target), sha256=digest))
        save(destination / 'receipt.json', dict(copies=copied, model_weights_copied=False))
    return dict(CER_percent=100 * metrics['overall']['pooled_cer'],
                edits=metrics['overall']['total_edits'],
                macro_CER_percent=100 * metrics['overall']['patient_macro_cer'],
                hard_edits=metrics['hard']['total_edits'],
                vlink247_edit_delta=p30_delta,
                other_edit_delta=metrics['overall']['total_edits'] - parent['overall']['total_edits'] - p30_delta,
                vlink247_CER_percent=100 * groups['vlink247']['after_pooled'],
                other_CER_percent=100 * groups['other_patients']['after_pooled'],
                gate_checks=result['gate_checks'], eligible=result['eligible_for_next_round_parent'])


def main(base, stage):
    first = base / 'ASR_target_format_dpo_probe_v1'
    root = first if stage == 'format' else base / 'ASR_target_format_training_seeds_v1'
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == h for f, h in cfg['source_sha256'].items())
    paths = ({arm: first / arm for arm in ['full', 'edit_eos']} if stage == 'format' else
             {'42': first / 'edit_eos', **{str(s): root / f'seed_{s}' for s in [43, 44]}})
    arms = {name: summarize(base, path) for name, path in paths.items()}
    result = dict(status='complete', arms=arms, parent_promoted=False,
                  actual_new_DPO_training=True, actual_new_MAS_decision=False,
                  actual_new_generation=False, actual_new_sampling=False, postprocessing_used=False,
                  evaluation_scope='Repeatedly used511 development recordings, not an independent held-out final test.',
                  checkpoint_storage='Temporary local scratch; persistent small receipts and evaluation outputs.')
    parent = read(base / 'parent_eval_check/summary.json')['metrics']['overall']
    lines = ['本轮使用实际 DPO 更新后的模型直接转写；没有推理后处理。', '',
             '| 条件 | CER % | 字符错误 | macro CER % | hard 错误 | vlink247 错误变化 | 其他错误变化 | 原门槛 |',
             '|---|---:|---:|---:|---:|---:|---:|---|',
             f'| R-only | {100 * parent["pooled_cer"]:.6f} | 1384 | {100 * parent["patient_macro_cer"]:.6f} | 688 | 0 | 0 | 起点 |']
    for name, arm in arms.items():
        lines.append(f'| {name} | {arm["CER_percent"]:.6f} | {arm["edits"]} | {arm["macro_CER_percent"]:.6f} | {arm["hard_edits"]} | {arm["vlink247_edit_delta"]:+d} | {arm["other_edit_delta"]:+d} | {"通过" if arm["eligible"] else "未通过"} |')
    if stage == 'format':
        raw = read(first / 'KL_raw_full.json')['patient_weighted']
        normalized = read(first / 'KL_normalized_full.json')['patient_weighted']
        drop = raw['all_token_mean_KL'] - normalized['all_token_mean_KL']
        eos_drop_fraction = (raw['EOS_contribution'] - normalized['EOS_contribution']) / drop
        result['KL_full_scoring'] = dict(raw=raw, normalized=normalized,
                                          EOS_fraction_of_KL_reduction=eos_drop_fraction)
        result['paired_mask_vs_full'] = compare(
            first / 'full/evaluation_run/evaluation/predictions.jsonl',
            first / 'edit_eos/evaluation_run/evaluation/predictions.jsonl')
        lines += ['', '相同15条 MAS 音频、真实采样负例及训练曝光，只将 chosen 的句末标点移到原始目标元数据保留。'
                  '整句/差异+EOS 分别训练12步，lr=1e-4、beta=0.1；96条真实训练语音提供 KL 保持，未加真实 CE。', '',
                  f'完整96条真实训练语音的训练后加权 KL：{raw["all_token_mean_KL"]:.8f} → {normalized["all_token_mean_KL"]:.8f}。'
                  f'降幅的 {100 * eos_drop_fraction:.2f}% 来自 EOS 位置，正文贡献也从 {raw["text_contribution"]:.8f} 降到 {normalized["text_contribution"]:.8f}。',
                  '这支持标签格式引起句尾训练冲突；不能把它当成全部 CER 退步的原因。此诊断与训练日志末步使用的样本/更新时间不同。', '',
                  '去标点后的整句 CER 比对应原始格式少8个错误；差异+EOS 总错误数仍为1373，vlink247退步从+18缩到+9，其他患者收益也减少。'
                  'seed42 的总 CER 未因去标点额外提高。',
                  'chosen 长度变化会改变配对 padding，部分 rejected 冻结参考分数也有 BF16 数值差异；差异计分还会改变对齐。完整诊断保留在 actual_format_score_diagnostic.json。']
    else:
        result['mean_CER_percent'] = mean(x['CER_percent'] for x in arms.values())
        result['mean_delta_from_R_only_pp'] = result['mean_CER_percent'] - 100 * parent['pooled_cer']
        result['gate_pass_seeds'] = [s for s, x in arms.items() if x['eligible']]
        raw = read(base / 'edit_mask_training_seeds_v1/results.json')
        result['raw_seed_result_source'] = str(base / 'edit_mask_training_seeds_v1/results.json')
        result['raw_seed_result_sha256'] = sha256_file(Path(result['raw_seed_result_source']))
        lines += ['', f'三训练 seed 平均 CER 为 {result["mean_CER_percent"]:.6f}%，相对 R-only 为 {result["mean_delta_from_R_only_pp"]:+.6f} 个百分点。',
                  '三个分支都从原 seed42 R-only 开始，固定同一15条 MAS 偏好和合成曝光顺序，只改变 residual 初始化与真实训练 shuffle 的 seed。'
                  '不代表三个不同 R-only 起点或三次独立生成。',
                  '逐 seed 的原始格式配对对照保存在 paired_same_seed_raw_target.json；seed42 的对应文件为 paired_raw_target_control.json。']
    lines += ['', '原始粗审与实际负例来源不变；粗审通过不等于逐字音频保真。'
              '511条为反复使用的开发反馈，当前结果不是独立最终测试。未自动提升 parent。',
              'checkpoint 权重目前位于 /tmp；训练回执、预测、评测和元数据已持久保存在此记录目录。']
    sources = [Path(__file__), root / 'protocol.json', base / 'parent_eval_check/summary.json']
    for out in paths.values():
        sources += [out / n for n in ['results.json', 'verified.json', 'paired_parent.json',
                                      'target_response.json', 'checkpoint_metadata_receipts/receipt.json']]
    if stage == 'format':
        sources += [first / n for n in ['KL_raw_full.json', 'KL_normalized_full.json', 'actual_format_score_diagnostic.json']]
    result['source_sha256'] = {str(f): sha256_file(f) for f in sources}
    save(root / 'results.json', result)
    with (root / 'results.zh.md').open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['source_sha256', 'paired_mask_vs_full']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['format', 'seeds'])
    parser.add_argument('base', type=Path)
    args = parser.parse_args()
    main(args.base.resolve(), args.stage)
