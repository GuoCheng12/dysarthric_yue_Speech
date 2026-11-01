#!/usr/bin/env python3
"""Report all fixed-data optimization seeds and their matched real controls."""
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mas_synthetic_fixed_subset_replication_v1 as run


def original_gate(comparison):
    overall, hard = comparison['groups']['overall'], comparison['groups']['difficulty:hard']
    return bool(overall['after_pooled'] < overall['before_pooled'] and
        overall['after_patient_macro'] - overall['before_patient_macro'] <= .001 and
        hard['after_pooled'] - hard['before_pooled'] <= .005)


def main():
    report = run.live.report
    report.rows = run.live.gzip_rows
    sources = [Path(__file__), run.PLAN / 'plan.json']
    seeds = {}
    for seed in [42, 43, 44]:
        if seed == 42:
            roots = dict(R=run.OLD / 'R_DPO_KL', R_SG=run.CANDIDATE / 'uniform')
            evaluation = dict(R=roots['R'] / 'evaluation_run/evaluation', R_SG=roots['R_SG'] / 'evaluation')
        else:
            roots = {mode: run.REPLICATION / f'seed_{seed}' / mode for mode in ['R', 'R_SG']}
            evaluation = {mode: path / 'evaluation' for mode, path in roots.items()}
        cases, exposures, initial, base = {}, {}, {}, {}
        for mode, out in roots.items():
            trained = run.read(out / 'training_result.json')
            initial[mode] = trained.get('initial_residual_sha256', trained.get('policy_trainable_sha256_before'))
            base[mode] = trained.get('unchanged_base_sha256', trained.get('frozen_base_sha256'))
            assert initial[mode] and base[mode]
            if seed != 42:
                result = run.read(out / 'result.json')
                assert result['status'] == 'training_and_evaluation_complete'
                proto = run.read(out / 'protocol.json')
                assert proto['settings']['seed'] == seed
                assert all(run.sha256_file(Path(p)) == h for p, h in proto['source_sha256'].items())
            predictions = evaluation[mode] / 'predictions.jsonl.gz'
            rows = run.live.gzip_rows(predictions)
            assert len(rows) == 511
            versus_initial = report.compare(run.live.BASELINE, predictions)
            cases[mode] = dict(metrics=run.read(evaluation[mode] / 'summary.json')['metrics'],
                versus_initial=versus_initial, passes_original_gate=original_gate(versus_initial),
                focus_errors={label: sum(r['edit_count'] for r in rows if r['speaker_id'].startswith('vlink247') == want)
                              for label, want in [('vlink247', True), ('others', False)]})
            exposures[mode] = run.rows(out / 'training_exposures.jsonl')
            sources.extend([predictions, out / 'training_result.json', out / 'training_exposures.jsonl'])
        assert initial['R'] == initial['R_SG'] and base['R'] == base['R_SG']
        assert [r for r in exposures['R_SG'] if r['source'] == 'real'] == exposures['R']
        sg_occurrences = {(r['audio_id'], r['occurrence']) for r in exposures['R_SG'] if r['source'] == 'synthetic'}
        assert len(sg_occurrences) == 32 and len({r[0] for r in sg_occurrences}) == 8
        paired = report.compare(evaluation['R'] / 'predictions.jsonl.gz', evaluation['R_SG'] / 'predictions.jsonl.gz')
        seeds[str(seed)] = dict(conditions=cases, R_SG_vs_R=paired, common_initial_residual_sha256=initial['R'],
            common_base_sha256=base['R'], common_real_KL_exposures_verified=True)
    means = {}
    for mode in ['R', 'R_SG']:
        cases = [x['conditions'][mode] for x in seeds.values()]
        edits = [x['metrics']['overall']['total_edits'] for x in cases]
        means[mode] = dict(mean_errors=statistics.mean(edits), min_errors=min(edits), max_errors=max(edits),
            mean_CER=statistics.mean(x['metrics']['overall']['pooled_cer'] for x in cases),
            mean_hard_errors=statistics.mean(x['metrics']['hard']['total_edits'] for x in cases),
            mean_patient_macro_CER=statistics.mean(x['metrics']['overall']['patient_macro_cer'] for x in cases),
            original_gate_passes=sum(x['passes_original_gate'] for x in cases))
    result = dict(status='complete_fixed_SG_subset_optimization_seed_replication', seeds=seeds, means=means,
        mean_R_SG_minus_R_pp=100 * (means['R_SG']['mean_CER'] - means['R']['mean_CER']),
        mean_R_SG_minus_initial_pp=100 * (means['R_SG']['mean_CER'] - 1384 / 6656),
        original_initial_errors=1384, optimized_initialization_seeds=[42, 43, 44],
        data_selection_seed=42, candidate_sampling_seed=42, parent_R_only_unchanged=True, promoted=False,
        interpretation='Fixed data, same8SG and same real/KL schedule; only residual optimization initialization changes. Not independent full-pipeline seeds. No evidence of MAS utility-selection benefit is inferred from this random-subset replication.',
        source_sha256={str(p): run.sha256_file(p) for p in sources})
    run.write_immutable_json(run.REPLICATION / 'results.json', result)
    lines = ['固定8条SG：优化初始化种子复现', '',
        '| 优化seed | R DPO字错 | R+固定SG DPO字错 | R hard | R+SG hard | R+SG原门槛 |',
        '|---|---:|---:|---:|---:|---|']
    for seed, x in seeds.items():
        a, b = x['conditions']['R'], x['conditions']['R_SG']
        lines.append(f'| {seed} | {a["metrics"]["overall"]["total_edits"]} | {b["metrics"]["overall"]["total_edits"]} | {a["metrics"]["hard"]["total_edits"]} | {b["metrics"]["hard"]["total_edits"]} | {b["passes_original_gate"]} |')
    lines += ['', '初始R-only固定为1384字错（20.793269%）、hard688。',
        f'R三seed平均字错{means["R"]["mean_errors"]:.3f}；R+SG为{means["R_SG"]["mean_errors"]:.3f}，范围{means["R_SG"]["min_errors"]}—{means["R_SG"]["max_errors"]}。',
        f'R+SG相对R平均CER差{result["mean_R_SG_minus_R_pp"]:+.6f}个百分点；相对initial为{result["mean_R_SG_minus_initial_pp"]:+.6f}个百分点。', '',
        '没有重新选样、生成或采样候选；数据来源保持seed42。每个优化seed内，两臂初始参数与真实/KL曝光逐项一致。初始R-only与固定DPO reference均保持不变。',
        'seed42 R复用已有严格确定性对照的完整预测/回执，旧失败权重不可用；没有声称重新加载它。seed42随机SG候选已持久保存并独立加载核对。',
        '这不支持已经失败的MAS梯度选样策略，不是独立封存测试验证；不自动晋级，不用均值替代每个模型的原始验收。']
    with (run.REPLICATION / 'results.zh.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(means=means, mean_R_SG_minus_initial_pp=result['mean_R_SG_minus_initial_pp'])), flush=True)


if __name__ == '__main__':
    main()
