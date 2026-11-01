"""Compare actual coda recovery with the matched real-only DPO control."""
from pathlib import Path
import json
import report_mas_contextual_recipe_dpo_v1 as report
from therapist_harness.evidence import compare_prediction_snapshots, _key_descriptor

base, ROOT = report.base, report.ROOT


def main():
    paths = dict(initial=base.B / 'parent_eval_check/predictions.jsonl',
        R_DPO=Path('/tmp/mas_deterministic_matched_DPO_v1/R_DPO_KL/evaluation_run/evaluation/predictions.jsonl.gz'),
        MAS=ROOT / 'training/MAS/evaluation/predictions.jsonl.gz',
        fixed=ROOT / 'training/fixed/evaluation/predictions.jsonl.gz')
    observed = {a: report.rows(p) for a, p in paths.items()}
    descriptors = [_key_descriptor(('substitution', 'coda', a, b, None, None)) for a, b in [('ng','n'), ('n','ng')]]
    comparisons = {}
    for a, b in [('initial','R_DPO'), ('R_DPO','MAS'), ('R_DPO','fixed')]:
        r = compare_prediction_snapshots(observed[a], observed[b], before_prediction_column='prediction',
            after_prediction_column='prediction', critical_cer_threshold=.5, monitored_clusters=descriptors,
            off_target_max_per_level_per_direction=2)
        comparisons[a + '_to_' + b] = [x.model_dump(mode='json') for x in r.monitored_cluster_changes]
    base.save(ROOT / 'real_DPO_target_control.json', dict(comparisons=comparisons,
        sources={str(f): base.sha256_file(f) for f in [Path(__file__), *paths.values()]},
        scope='Matched R-DPO control is required for attributing any target-count decrease to additional SG. Conservative same-carrier transitions and unresolved states retained.'))
    print(json.dumps({k: [dict(id=x['descriptor']['evidence_id'], before=x['before']['error_count'], after=x['after']['error_count'],
        positions={f:x['target_positions'][f] for f in ['before_error_count','after_error_count','recovered_count','regressed_count','before_unresolved_count','after_unresolved_count']}) for x in v]
        for k, v in comparisons.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
