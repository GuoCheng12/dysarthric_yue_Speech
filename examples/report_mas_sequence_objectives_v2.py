#!/usr/bin/env python3
"""Compare complete sequence-objective pilots without changing their ledgers."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mas_sequence_objective_pilot_v1 as common
from mas_sequence_objective_pilot_v1 import read, rows, gzip_rows, report, sha256_file, write_immutable_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=['ipo', 'risk'])
    args = parser.parse_args()
    root = Path('/tmp/mas_IPO_DPO_mean_v1' if args.experiment == 'ipo' else '/tmp/mas_character_risk_v2')
    arms = ['dpo', 'ipo'] if args.experiment == 'ipo' else ['ce', 'mcer']
    protocols = {a: read(root / a / 'protocol.json') for a in arms}
    results = {a: read(root / a / 'result.json') for a in arms}
    trained = {a: read(root / a / 'training_result.json') for a in arms}
    assert all(x['status'] == 'training_and_evaluation_complete' for x in results.values())
    assert protocols[arms[0]]['parent_sha256'] == protocols[arms[1]]['parent_sha256']
    assert protocols[arms[0]]['source_sha256'] == protocols[arms[1]]['source_sha256']
    report.rows = gzip_rows
    paths = {a: root / a / 'evaluation/predictions.jsonl.gz' for a in arms}
    if args.experiment == 'ipo':
        exposure = {a: rows(root / a / 'exposures.jsonl') for a in arms}
        assert exposure['dpo'] == exposure['ipo']
        assert read(root / 'dpo/reference_scores.json') == read(root / 'ipo/reference_scores.json')
        assert trained['dpo']['policy_trainable_sha256_before'] == trained['ipo']['policy_trainable_sha256_before']
        slot_note = 'Legacy exposure objective names dpo identify replayed preference slots. Actual objective is protocol.mode and the real/synthetic_dpo or real/synthetic_ipo training loss keys; no DPO loss was substituted for IPO.'
    else:
        exposure = {a: rows(root / a / 'training_audio.jsonl') for a in arms}
        keys = ['step', 'audio_id', 'audio_sha256', 'loss_weight', 'CE_coefficient']
        assert [{k: e[k] for k in keys} for e in exposure['ce']] == [{k: e[k] for k in keys} for e in exposure['mcer']]
        assert trained['ce']['initial_parameter_sha256'] == trained['mcer']['initial_parameter_sha256']
        assert trained['ce']['nbest_receipt'] == trained['mcer']['nbest_receipt']
        slot_note = 'Risk uses character edit counts; CE coefficient and real exposure order are matched. N-best lists fixed at common initial parent.'
    assert trained[arms[0]]['frozen_base_sha256'] == trained[arms[1]]['frozen_base_sha256']
    conditions = {}
    for arm in arms:
        pred = gzip_rows(paths[arm])
        assert len(pred) == 511
        metrics = read(paths[arm].parent / 'summary.json')['metrics']
        focus = {label: sum(r['edit_count'] for r in pred if r['speaker_id'].startswith('vlink247') == want)
                 for label, want in [('vlink247', True), ('others', False)]}
        conditions[arm] = dict(metrics=metrics, focus_errors=focus,
            comparison=read(root / arm / 'comparison.json'), training=trained[arm],
            result=results[arm])
    delta = report.compare(paths[arms[0]], paths[arms[1]])
    result = dict(status='complete_matched_objective_pilot', conditions=conditions,
        paired_objective_comparison=delta, common_parent_initial_state_and_exposure_verified=True,
        exposure_semantics=slot_note, no_promotion=True,
        source_sha256={str(Path(__file__)): sha256_file(Path(__file__))})
    write_immutable_json(root / 'results.json', result)
    lines = ['序列目标配对实验', '', '| 条件 | 字错 | CER % | hard 字错 | vlink247 字错 | 其他字错 | 原验收 |',
             '|---|---:|---:|---:|---:|---:|---|', '| 初始 R-only | 1384 | 20.793269 | 688 | 385 | 999 | 起点 |']
    for arm, r in conditions.items():
        m, f = r['metrics'], r['focus_errors']
        lines.append(f'| {arm} | {m["overall"]["total_edits"]} | {100*m["overall"]["pooled_cer"]:.6f} | {m["hard"]["total_edits"]} | {f["vlink247"]} | {f["others"]} | {r["result"]["passes_original_gate"]} |')
    g = delta['groups']['overall']
    lines += ['', f'第二组相对第一组 CER 变化：{g["delta_pp"]:+.6f} 个百分点，患者 bootstrap 95% 区间：{g["patient_cluster_bootstrap_delta_pp_95"]}。',
              '', '完整511条开发反馈，尚非独立测试。共同起点和实际曝光已核对。没有晋级模型。', '', slot_note]
    with (root / 'results.zh.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps({a: dict(errors=r['metrics']['overall']['total_edits'], gate=r['result']['passes_original_gate']) for a, r in conditions.items()}), flush=True)


if __name__ == '__main__':
    main()
