#!/usr/bin/env python3
"""Frozen evaluation of matched SG allocation; original gate and transfer kept distinct."""
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mas_synthetic_utility_allocation_dpo_v1 as run


def main():
    root = run.ROOT
    plan = run.read(run.PLAN / 'plan.json')
    cases, exposures = {}, {}
    report = run.live.report
    report.rows = run.live.gzip_rows
    for mode in ['uniform', 'MAS']:
        out = root / mode
        result = run.read(out / 'result.json')
        assert result['status'] == 'training_and_evaluation_complete'
        protocol = run.read(out / 'protocol.json')
        assert all(run.sha256_file(Path(p)) == h for p, h in protocol['source_sha256'].items())
        trained = run.read(out / 'training_result.json')
        exposures[mode] = run.rows(out / 'training_exposures.jsonl')
        sg = [r for r in exposures[mode] if r['source'] == 'synthetic']
        occurrence = Counter(aid for aid, visit in {(e['audio_id'], e['occurrence']) for e in sg})
        assert len(occurrence) == 8 and set(occurrence.values()) == {4}
        assert len({(r['audio_id'], r['occurrence']) for r in sg}) == 32
        predictions = run.live.gzip_rows(out / 'evaluation/predictions.jsonl.gz')
        assert len(predictions) == 511
        focus = {label: sum(r['edit_count'] for r in predictions if r['speaker_id'].startswith('vlink247') == want)
                 for label, want in [('vlink247', True), ('others', False)]}
        cases[mode] = dict(metrics=run.read(out / 'evaluation/summary.json')['metrics'],
            focus_errors=focus, versus_initial=run.read(out / 'comparison.json'), completion=result,
            training=trained, synthetic_unique_pairs=len({r['preference_key'] for r in sg}),
            actual_audio_visits=occurrence)
    assert [r for r in exposures['uniform'] if r['source'] == 'real'] == [r for r in exposures['MAS'] if r['source'] == 'real']
    assert cases['uniform']['training']['initial_residual_sha256'] == cases['MAS']['training']['initial_residual_sha256']
    assert cases['uniform']['training']['unchanged_base_sha256'] == cases['MAS']['training']['unchanged_base_sha256']
    uniform = root / 'uniform/evaluation/predictions.jsonl.gz'
    selected = root / 'MAS/evaluation/predictions.jsonl.gz'
    paired = report.compare(uniform, selected)
    overall, hard = paired['groups']['overall'], paired['groups']['difficulty:hard']
    additional = bool(overall['after_pooled'] < overall['before_pooled'] and
        overall['after_patient_macro'] - overall['before_patient_macro'] <= .001 and
        hard['after_pooled'] - hard['before_pooled'] <= .005)
    supported = bool(cases['MAS']['completion']['passes_original_gate'] and additional)
    result = dict(status='complete_MAS_synthetic_utility_allocation_DPO', conditions=cases,
        MAS_vs_uniform=paired, passes_incremental_gate=additional, supports_preliminary_MAS_gain=supported,
        exact_common_real_and_KL_exposure=True, common_initial_parameters_verified=True,
        selections=plan['selections'], aggregate_gradient_dots=plan['aggregate_gradient_dots'],
        held_out_C_used_for_selection=False, promoted=False,
        scope='Same existing SG pool and generation/review, eight unique audios x4visits each. Tests actual MAS audio allocation, not a reference-only causal effect or proof of word-perfect SG.',
        source_sha256={str(p): run.sha256_file(p) for p in [Path(__file__), run.PLAN / 'plan.json', uniform, selected]})
    run.write_immutable_json(root / 'results.json', result)
    lines = ['MAS按实际合成音频训练效用分配：配对DPO', '',
        '| 条件 | 字错 | CER % | hard字错 | vlink247 | 其他 |',
        '|---|---:|---:|---:|---:|---:|', '| 初始R-only | 1384 | 20.793269 | 688 | 385 | 999 |']
    for mode, r in cases.items():
        m, f = r['metrics'], r['focus_errors']
        lines.append(f'| {mode} 8条SG ×4次 | {m["overall"]["total_edits"]} | {100*m["overall"]["pooled_cer"]:.6f} | {m["hard"]["total_edits"]} | {f["vlink247"]} | {f["others"]} |')
    ci = [float(x) for x in overall['patient_cluster_bootstrap_delta_pp_95']]
    lines += ['', f'MAS相对随机对照：{overall["delta_pp"]:+.6f}个百分点，患者bootstrap95%区间{ci}。',
        f'同时通过原始门槛及MAS增量门槛：{supported}。无自动晋级。', '',
        '两臂共同initial R-only、固定DPO reference、32真实DPO音频/91对、256KL visits，edit+EOS beta0.1、lr1e-4、KL10、4steps。每段SG内部平均实际错误对；32条合成曝光为8段音频各4次，音频身份变化是干预。', '',
        'MAS仅看A/B两组固定真实训练GT anchor的梯度和实际内容/reference卡片；C16条未提供给MAS，选择冻结后才汇总为迁移诊断。梯度代理不等于CER，SG粗审与chosen约定不变。', '',
        '这仍是反复使用的511条开发反馈，不能视为独立封存测试集。完整逐条预测、训练曝光、MAS原始响应和来源hash均保留。']
    with (root / 'results.zh.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(errors={k: v['metrics']['overall']['total_edits'] for k, v in cases.items()},
        supported=supported, aggregate_gradient_dots=plan['aggregate_gradient_dots'])), flush=True)


if __name__ == '__main__':
    main()
