#!/usr/bin/env python3
"""Report the final fixed-subset real-preference removal ablation."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mas_fixed_subset_s_dpo_real_kl_v1 as run


def main():
    out = run.ROOT / 'S_KL'
    completed = run.read(out / 'result.json')
    assert completed['status'] == 'training_and_evaluation_complete'
    protocol = run.read(out / 'protocol.json')
    assert all(run.sha256_file(Path(p)) == h for p, h in protocol['source_sha256'].items())
    old_out = run.PREVIOUS / 'uniform'
    trained = run.read(out / 'training_result.json')
    old = run.read(old_out / 'training_result.json')
    assert trained['initial_residual_sha256'] == old['initial_residual_sha256']
    assert trained['unchanged_base_sha256'] == old['unchanged_base_sha256']
    actual = run.rows(out / 'training_exposures.jsonl')
    previous = run.rows(old_out / 'training_exposures.jsonl')
    assert actual == [r for r in previous if r['objective'] == 'real_symmetric_kl' or r['source'] == 'synthetic']
    real = [r for r in actual if r['source'] == 'real']
    sg = [r for r in actual if r['source'] == 'synthetic']
    assert len(real) == 256 and all(r['objective'] == 'real_symmetric_kl' for r in real)
    assert len({r['preference_key'] for r in sg}) == 23
    assert len({(r['audio_id'], r['occurrence']) for r in sg}) == 32
    report = run.live.report
    report.rows = run.live.gzip_rows
    current = out / 'evaluation/predictions.jsonl.gz'
    previous_predictions = old_out / 'evaluation/predictions.jsonl.gz'
    comparison = report.compare(previous_predictions, current)
    initial = run.read(out / 'comparison.json')
    metrics = run.read(out / 'evaluation/summary.json')['metrics']
    predicted = run.live.gzip_rows(current)
    assert len(predicted) == 511
    focus = {name: sum(r['edit_count'] for r in predicted if r['speaker_id'].startswith('vlink247') == want)
             for name, want in [('vlink247', True), ('others', False)]}
    support = bool(completed['passes_original_gate'] and metrics['overall']['total_edits'] < 1382)
    assert support == completed['supports_real_DPO_masking']
    result = dict(status='complete_fixed_subset_S_DPO_real_KL_ablation', metrics=metrics, focus_errors=focus,
        versus_initial=initial, versus_full_R_SG=comparison, supports_real_DPO_masking=support,
        training=trained, completion=completed, exact_prior_SG_and_KL_exposures_verified=True,
        actual_MAS_decision=str(run.DECISION), promoted=False,
        source_sha256={str(p): run.sha256_file(p) for p in [Path(__file__), current, previous_predictions]})
    run.write_immutable_json(run.ROOT / 'results.json', result)
    lines = ['固定SG子集：删除新增真实DPO项', '',
        '| 条件 | 字错 | CER % | hard字错 |', '|---|---:|---:|---:|',
        '| initial R-only | 1384 | 20.793269 | 688 |',
        '| 完整R+固定SG DPO，seed42 | 1382 | 20.763221 | 694 |',
        f'| 仅固定SG DPO+realKL，seed42 | {metrics["overall"]["total_edits"]} | {100*metrics["overall"]["pooled_cer"]:.6f} | {metrics["hard"]["total_edits"]} |', '',
        f'原验收通过：{completed["passes_original_gate"]}；real-DPO masking假设支持：{support}。无晋级。', '',
        '从同一R-only parent开始，DPO reference仍固定initial R-only ASR。只删除91次真实DPO pair训练；保留23个实际SG pairs、32次SG audio visits及256次真实KL。所有SG音频、候选、mask、顺序和超参数保持一致。', '',
        '该分支名称是SG-DPO+realKL，不能说真实语音在参与偏好训练。判据在训练前由MAS固定：须过original gate且总字错<1382；否则关闭该固定子集，转向新的matched SG probe。', '',
        '这是一次seed42消融，使用同一511开发反馈。通过粗审仍不代表逐字音文保真；文本转写的phonology signature不能单独证明声学纠错。']
    with (run.ROOT / 'results.zh.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(errors=metrics['overall']['total_edits'], hard=metrics['hard']['total_edits'],
        original_gate=completed['passes_original_gate'], supports_masking=support, focus=focus)), flush=True)


if __name__ == '__main__':
    main()
