#!/usr/bin/env python3
"""Compare actual matched real preference training and incremental MAS DPO."""

import json
from pathlib import Path
import sys

from mas_real_preference_bridge import ARMS, verify
from mas_selection_dpo_probe import read, rows, save
from mas_evaluation_report import compare
from report_ASR_target_format import summarize
from therapist_harness.store import sha256_file


def main(base, root=None):
    root = root or base / 'real_preference_bridge_v1'
    cfg = verify(root)
    assert read(root / 'smoke/verified.json')['status'] == 'passed'
    summaries = {arm: summarize(base, root / arm) for arm in ARMS}
    training = [root / arm / 'training_receipts' for arm in ARMS]
    records = [rows(path / 'training_exposures.jsonl') for path in training]
    assert records[0] == [x for x in records[1] if x['source'] == 'real']
    results = [read(path / 'training_result.json') for path in training]
    assert results[0]['policy_trainable_sha256_before'] == results[1]['policy_trainable_sha256_before']
    refs = [read(path / 'reference_log_probs.json')['values'] for path in training]
    assert refs[0] == {k: v for k, v in refs[1].items() if k in refs[0]}
    protocols = [read(path / 'run_protocol.json') for path in training]
    for protocol in protocols:
        assert protocol['reference_asr_sha256'] == protocol['policy_parent_sha256'] == cfg['source_protocol']['initial_adapter']['sha256']
    for field in ['model_artifacts', 'reference_artifacts', 'policy_parent_artifacts']:
        assert protocols[0][field] == protocols[1][field]
    comparison = compare(root / 'R_DPO/evaluation_run/evaluation/predictions.jsonl',
                         root / 'R_MAS_DPO/evaluation_run/evaluation/predictions.jsonl')
    save(root / 'paired_incremental_MAS.json', comparison)
    save(root / 'cross_arm_verified.json', dict(status='passed', same_initial_residual_parameters=True,
        exact_real_pair_exposures_weights_and_reference_scores=True, common_parent_frozen_reference=True,
        preserved_independent_sampling_receipts=True))
    source_files = [Path(__file__), root / 'protocol.json', root / 'data_receipt.json', root / 'cpu_smoke.json',
                    root / 'smoke/verified.json', root / 'paired_incremental_MAS.json', root / 'cross_arm_verified.json']
    for arm in ARMS:
        source_files += [root / arm / n for n in ['results.json', 'verified.json', 'target_response.json',
            'training_receipts/training_result.json', 'training_receipts/training_exposures.jsonl',
            'training_receipts/reference_log_probs.json', 'evaluation_run/evaluation/predictions.jsonl']]
    output = dict(status='complete', arms=summaries,
        MAS_incremental_CER_pp=comparison['groups']['overall']['delta_pp'],
        MAS_incremental_bootstrap95_pp=comparison['groups']['overall']['patient_cluster_bootstrap_delta_pp_95'],
        parent_promoted=False, actual_new_DPO_training=True, actual_new_generation=False,
        actual_new_sampling=False, actual_new_MAS_decision=False, postprocessing_used=False,
        source_sha256={str(f): sha256_file(f) for f in source_files})
    save(root / 'results.json', output)
    lines = ['本次直接比较真实偏好 DPO 和附加 MAS 合成偏好的增量，不使用真实 CE、真实 KL 或推理后处理。', '',
             '| 条件 | CER % | 字符错误 | macro CER % | hard 错误 | vlink247 错误变化 | 其他错误变化 | 原门槛 |',
             '|---|---:|---:|---:|---:|---:|---:|---|',
             '| R-only | 20.793269 | 1384 | 19.107915 | 688 | 0 | 0 | 起点 |']
    for arm, x in summaries.items():
        lines.append(f'| {arm} | {x["CER_percent"]:.6f} | {x["edits"]} | {x["macro_CER_percent"]:.6f} | {x["hard_edits"]} | {x["vlink247_edit_delta"]:+d} | {x["other_edit_delta"]:+d} | {"通过" if x["eligible"] else "未通过"} |')
    lines += ['', f'MAS 对 R/DPO 的增量 CER 为 {output["MAS_incremental_CER_pp"]:+.6f} 个百分点。',
              '实际132个真实偏好对来自96条已采样真实语音中的50条，其余46条全对、不构造负样本。'
              '37条此前曾在 R-only 起点训练中获得正权重曝光，13条未曾获得；两分支的这部分真实数据与曝光完全相同。',
              f'两侧同一初始 R-only parent/reference、同一 residual 初始化，12步、lr={cfg["arguments"]["R_DPO"]["lr"]}、beta=0.1、192次真实音频曝光。'
              '混合侧另加15条 MAS 偏好、96次合成曝光，合成 DPO loss 系数0.25。每条音频的独特错误平均，不按错误个数加权。',
              '来源回执分别为真实4次采样/cap256、合成16次采样/cap128；其他解码参数和采样 parent 一致，未伪造合并采样回执。'
              '原三文本粗审保留，合成 chosen 仅统一句末标点，未证明音频逐字保真。',
              '这轮改变了真实训练信号和目标配置，不能与之前合成 DPO＋真实 KL 结果作单因素因果比较。'
              '511条仍为反复使用的开发反馈，bootstrap 未校正多次实验选择，不是独立最终测试。',
              'checkpoint 权重位于 /tmp，训练与评测回执持久保存；未自动提升 parent。']
    with (root / 'results.zh.md').open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None)
