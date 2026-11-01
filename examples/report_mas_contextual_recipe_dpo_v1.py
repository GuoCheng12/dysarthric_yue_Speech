#!/usr/bin/env python3
"""Real511 checkpoint gate and paired coda recovery for the matched SG recipe trial."""
import gzip
import json
from pathlib import Path

import mas_contextual_recipe_dpo_data_v1 as data
import mas_evaluation_report as cer_report
from therapist_harness.evidence import compare_prediction_snapshots, _key_descriptor

base, ROOT = data.base, data.ROOT


def rows(path):
    path = Path(path)
    text = gzip.decompress(path.read_bytes()).decode() if path.suffix == '.gz' else path.read_text()
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def main():
    paths = dict(initial=base.B / 'parent_eval_check/predictions.jsonl',
        R_DPO=Path('/tmp/mas_deterministic_matched_DPO_v1/R_DPO_KL/evaluation_run/evaluation/predictions.jsonl.gz'),
        MAS=ROOT / 'training/MAS/evaluation/predictions.jsonl.gz',
        fixed=ROOT / 'training/fixed/evaluation/predictions.jsonl.gz')
    observed = {a: rows(p) for a, p in paths.items()}
    initial_ids = [x['utt_id'] for x in observed['initial']]
    assert len(initial_ids) == 511 and len(set(initial_ids)) == 511
    for a in observed:
        assert [x['utt_id'] for x in observed[a]] == initial_ids
    descriptors = [_key_descriptor(('substitution', 'coda', a, b, None, None)) for a, b in [('ng','n'), ('n','ng')]]
    target_comparisons = {}
    for before, after in [('initial','MAS'), ('initial','fixed'), ('fixed','MAS')]:
        comparison = compare_prediction_snapshots(observed[before], observed[after],
            before_prediction_column='prediction', after_prediction_column='prediction',
            critical_cer_threshold=.5, monitored_clusters=descriptors, off_target_max_per_level_per_direction=2)
        target_comparisons[before + '_to_' + after] = dict(
            monitored_clusters=[x.model_dump(mode='json') for x in comparison.monitored_cluster_changes],
            off_target_changes=[x.model_dump(mode='json') for x in comparison.off_target_changes])
    summary = {}
    for arm, records in observed.items():
        groups = {'all': records, 'hard': [x for x in records if x['zero_shot_bucket'] == 'hard'],
            'vlink247': [x for x in records if x['speaker_id'] == 'vlink247'],
            'others': [x for x in records if x['speaker_id'] != 'vlink247']}
        summary[arm] = {k: dict(errors=sum(x['edit_count'] for x in rs),
            characters=sum(x['normalized_reference_characters'] for x in rs)) for k, rs in groups.items()}
        for value in summary[arm].values(): value['CER_percent'] = value['errors'] / value['characters'] * 100
        if arm in ['MAS','fixed']:
            result = base.read(ROOT / 'training' / arm / 'result.json')
            summary[arm]['passes_original_gate'] = result['passes_original_gate']
    assert summary['initial']['all']['errors'] == 1384
    assert summary['R_DPO']['all']['errors'] == 1399
    exposures = {a: rows(ROOT / 'training' / a / 'training_exposures.jsonl') for a in ['MAS','fixed']}
    assert [x for x in exposures['MAS'] if x['source']=='real'] == [x for x in exposures['fixed'] if x['source']=='real']
    cer_report.rows = rows
    paired_CER = {a + '_to_' + b: cer_report.compare(paths[a], paths[b])
        for a, b in [('initial','MAS'), ('initial','fixed'), ('R_DPO','MAS'), ('R_DPO','fixed'), ('fixed','MAS')]}
    result = dict(summary=summary, target_comparisons=target_comparisons, paired_CER=paired_CER,
        interpretation='Same new MAS texts and patients; compare chosen recipe policy and resulting matched-budget actual-pair allocation against fixed recipe. Real replay, initial parent and frozen reference identical. Repeated511 development data; one exploratoryseed, no automatic promotion or independent-test claim.',
        sources={str(f): base.sha256_file(f) for f in [Path(__file__), Path(cer_report.__file__), *paths.values(), ROOT / 'training_plan/plan.json']})
    base.save(ROOT / 'results.json', result)
    lines = ['新内容与患者条件下：MAS配方与固定配方 DPO', '', '|条件|真实511 CER|错误数|hard错误|原验收|', '|---|---:|---:|---:|---|']
    for arm, r in summary.items():
        lines.append(f'|{arm}|{r["all"]["CER_percent"]:.4f}%|{r["all"]["errors"]}/6656|{r["hard"]["errors"]}/1248|{r.get("passes_original_gate", "基准/已有对照")}|')
    lines += ['', result['interpretation']]
    base.write_immutable_bytes(ROOT / 'results.zh.md', ('\n'.join(lines)+'\n').encode())
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
