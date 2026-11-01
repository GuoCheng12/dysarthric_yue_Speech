#!/usr/bin/env python3
"""Audit completed coverage arms and report raw development and target responses."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

from mas_real_coverage_probe import ARMS, verify
from mas_selection_dpo_probe import read, rows, save
from mas_evaluation_report import compare
from therapist_harness.evidence import compare_prediction_snapshots
from therapist_harness.round_schema import ClusterDescriptor
from therapist_harness.store import sha256_file


def main(base):
    root=base/'real_coverage_dpo_probe_v1';cfg=verify(root)
    parent=read(base/'parent_eval_check/summary.json')['metrics']
    before=rows(base/'parent_eval_check/predictions.jsonl')
    evidence=read(base/'lexical96_mas_v1/round_001/evidence.json')
    proposal=read(base/'lexical96_mas_v1/round_001/proposal.json')
    ids={x['subject_evidence_id'] for x in proposal['expected_outcomes']}
    clusters=[ClusterDescriptor(**{k:x[k] for k in ClusterDescriptor.model_fields})
        for x in evidence['error_clusters'] if x['evidence_id'] in ids and x['operation']=='substitution']
    results={};protocols={};training={};sources=[Path(__file__),base/'parent_eval_check/summary.json',
        base/'parent_eval_check/predictions.jsonl',root/'protocol.json',root/'smoke/verified.json']
    for arm in ARMS:
        out=root/arm;result=read(out/'results.json');assert result['status']=='complete'
        assert read(out/'verified.json')['status']=='passed'
        protocols[arm]=read(out/'training/run_protocol.json');training[arm]=read(out/'training/training_result.json')
        response=compare_prediction_snapshots(before,rows(out/'evaluation_run/evaluation/predictions.jsonl'),
            before_prediction_column='prediction',after_prediction_column='prediction',critical_cer_threshold=.5,
            monitored_clusters=clusters,off_target_max_per_level_per_direction=3)
        response=json.loads(json.dumps(asdict(response),default=lambda x:x.model_dump(mode='json')))
        save(out/'target_response.json',response)
        groups=read(out/'paired_parent.json')['groups']
        metrics=result['metrics'];all_edits=metrics['overall']['total_edits']
        patient_delta=round(groups['vlink247']['delta_pp']/100*1384)
        results[arm]=dict(pooled_CER_percent=100*metrics['overall']['pooled_cer'],edits=all_edits,
            macro_CER_percent=100*metrics['overall']['patient_macro_cer'],
            hard_CER_percent=100*metrics['hard']['pooled_cer'],hard_edits=metrics['hard']['total_edits'],
            vlink247_edit_delta=patient_delta,other_edit_delta=all_edits-parent['overall']['total_edits']-patient_delta,
            original_gate=result['gate_checks'],eligible_for_next_round_parent=result['eligible_for_next_round_parent'],
            peak_gradient_norm=max(x['gradient_norm'] for x in training[arm]['history']),
            clipped_steps=sum(x['gradient_norm']>1 for x in training[arm]['history']))
        sources += [out/n for n in ['results.json','verified.json','paired_parent.json','target_response.json',
            'training/training_result.json','training/training_exposures.jsonl','training/run_protocol.json',
            'training/real_teacher_cache.json','training/reference_log_probs.json','evaluation_run/evaluation/predictions.jsonl']]
    a,b=[protocols[k] for k in ARMS]
    differences=[k for k in set(a['arguments'])|set(b['arguments']) if a['arguments'].get(k)!=b['arguments'].get(k)]
    assert set(differences)=={'real_train_file','output_dir'}
    assert all(a[k]==b[k] for k in ['reference_asr_sha256','policy_parent_sha256','reference_artifacts','model_artifacts'])
    assert training[ARMS[0]]['policy_trainable_sha256_before']==training[ARMS[1]]['policy_trainable_sha256_before']
    old_index_path=base.parent/'tts_utility_diagnosis_v1/additive/real_w0/seed_42/index.json'
    seen={x['audio'] for x in read(old_index_path) if x['domain']=='real'}
    control_real=rows(cfg['arguments'][ARMS[0]]['real_train_file']);diverse_real=rows(cfg['arguments'][ARMS[1]]['real_train_file'])
    control_ids={x['audio_id'] for x in control_real};extra=[x for x in diverse_real if x['audio_id'] not in control_ids]
    assert len(extra)==46 and all(x['audio'] in seen for x in extra)
    actual=[rows(root/arm/'training/training_exposures.jsonl') for arm in ARMS]
    signature=lambda data:[(x['step'],x['source'],x['occurrence'],x['loss_weight']) for x in data]
    assert signature(actual[0])==signature(actual[1])
    assert [x for x in actual[0] if x['source']=='synthetic']==[x for x in actual[1] if x['source']=='synthetic']
    cross=compare(root/ARMS[0]/'evaluation_run/evaluation/predictions.jsonl',root/ARMS[1]/'evaluation_run/evaluation/predictions.jsonl')
    save(root/'paired_coverage_comparison.json',cross)
    sources += [old_index_path,root/'paired_coverage_comparison.json']
    save(root/'cross_arm_verified.json',dict(status='passed',common_initial_parent_and_frozen_reference=True,
        initial_residual_parameters_identical=True,matched_actual_exposure_timing_and_weights=True,
        identical_synthetic96_exposure_records=True,additional46_recordings_previously_trained=True,
        training_argument_differences=differences,actual_real_counts=[dict(Counter(x['source'] for x in data)) for data in actual]))
    save(root/'results.json',dict(status='complete',arms=results,
        diverse_minus_control_CER_pp=cross['groups']['overall']['delta_pp'],
        actual_new_DPO_training=True,actual_new_generation=False,actual_new_sampling=False,postprocessing=False,
        parent_promoted=False,interpretation='Exploratory comparison on repeatedly used511 development samples; not an independent final test.',
        source_sha256={str(f):sha256_file(f) for f in sources}))
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
