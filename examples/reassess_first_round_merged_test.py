"""Re-report the existing M0 and pilot checkpoints on the user's merged test.

Reuse verified row-level predictions; decode only the missing original-test half.
This is a new evaluation artifact, not a rewrite of the old dev-only episode.
"""

import csv
import io
import json
from pathlib import Path
import subprocess
import sys

from pydantic import TypeAdapter
from therapist_harness.evidence import evaluate_snapshot, compare_prediction_snapshots, PairedSnapshotEvaluation
from therapist_harness.round_schema import EvaluationPlan
from therapist_harness.store import sha256_file, sha256_directory, write_immutable_json, write_immutable_bytes

REPO = Path(__file__).resolve().parents[1]
RECORDS = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1')
PILOT = RECORDS / 'therapist_m0_b8g4_pilot_v1'
M0 = RECORDS / 'real_only_m0_b8g4_utterance_mean_v1'
DATA = Path('/data/qwen3-asr/finetune/data_setting_d_v1')
MERGED = Path('/data/qwen3-asr/finetune/data_setting_d_merged_test_v1/setting_d_merged_test.jsonl')
ZERO = Path('/data/qwen3-asr/inference/outputs/setting_d_seen_patient_unseen_content_v1_qc_frozen')
OUT = PILOT / 'merged_test_reassessment_v1'


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def combine(scope, sources):
    by_id = {}
    for path, column in sources:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                assert row['utt_id'] not in by_id
                by_id[row['utt_id']] = {**row, 'prediction': row[column]}
    assert set(by_id) == {r['utt_id'] for r in scope}
    rows = []
    for row in scope:
        predicted = by_id[row['utt_id']]
        for field in ('speaker_id', 'prompt_id', 'clean_gt', 'zero_shot_bucket'):
            assert predicted[field] == row[field], (row['utt_id'], field)
        rows.append({**row, 'prediction': predicted['prediction']})
    return rows


def record(name, scope, sources, checkpoint):
    rows = combine(scope, sources)
    metrics = evaluate_snapshot(rows, prediction_column='prediction', critical_cer_threshold=0.5).model_dump(mode='json')
    target = OUT / name
    write_immutable_bytes(target / 'predictions.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode())
    summary = {'model': name, 'evaluation_scope': 'train_2117_test_511_no_dev',
               'split': str(MERGED), 'split_sha256': sha256_file(MERGED),
               'checkpoint': str(checkpoint) if checkpoint else None,
               'checkpoint_sha256': sha256_directory(checkpoint) if checkpoint else None,
               'prediction_sources': [{'path': str(p), 'column': c, 'sha256': sha256_file(p)} for p,c in sources],
               'metrics': metrics}
    write_immutable_json(target / 'summary.json', summary)
    print(json.dumps({'model': name, 'CER_percent': {k:100*v['mean_utterance_cer'] for k,v in metrics.items()}}, ensure_ascii=False), flush=True)
    return summary, rows


def paired_endpoint(before, after):
    plan = EvaluationPlan.model_validate_json((PILOT/'evaluation_plan.json').read_text())
    paired = compare_prediction_snapshots(before,after,before_prediction_column='prediction',
        after_prediction_column='prediction',critical_cer_threshold=0.5,
        monitored_clusters=[m.descriptor for m in plan.monitored_clusters],
        off_target_max_per_level_per_direction=3)
    write_immutable_json(OUT/'paired_m0_vs_step67.json',TypeAdapter(PairedSnapshotEvaluation).dump_python(paired,mode='json'))


def main():
    OUT.mkdir(exist_ok=False)
    scope = jsonl(MERGED)
    train = jsonl(DATA / 'setting_d_train.jsonl')
    original = jsonl(DATA / 'setting_d_dev.jsonl') + jsonl(DATA / 'setting_d_test.jsonl')
    assert len(scope) == 511 and len(train) == 2117
    assert len({r['utt_id'] for r in scope}) == 511
    assert {r['utt_id'] for r in scope} == {r['utt_id'] for r in original}
    original_by_id = {r['utt_id']:r for r in original}
    for row in scope:
        for field in ('audio','clean_gt','speaker_id','prompt_id','zero_shot_bucket'):
            assert row[field] == original_by_id[row['utt_id']][field]
    for field in ('utt_id','prompt_id','prompt_normalized_sha16','prompt_jyutping_sha16'):
        assert not {r[field] for r in train} & {r[field] for r in scope}
    write_immutable_json(OUT / 'scope_audit.json', {
        'train_rows':2117, 'test_rows':511, 'test_speakers':len({r['speaker_id'] for r in scope}),
        'test_contents':len({r['prompt_id'] for r in scope}), 'original_dev_rows':258, 'original_test_rows':253,
        'merged_manifest_sha256':sha256_file(MERGED), 'training_rerun':False,
        'historical_episode_overwritten':False, 'source_sha256':sha256_file(Path(__file__))})
    results = []
    result, _ = record('zero_shot', scope, [(ZERO / f'setting_d_{s}.csv','raw_pred') for s in ('dev','test')], None)
    results.append(result)
    m0_checkpoint = M0 / 'training/checkpoint-50'
    for sub in ('dev_step_50','test_selected_m0'):
        assert json.loads((M0 / f'evaluation/{sub}/summary.json').read_text())['adapter_path'] == str(m0_checkpoint)
    result, before = record('m0_step50', scope, [(M0 / f'evaluation/{s}/predictions.csv','prediction') for s in ('dev_step_50','test_selected_m0')], m0_checkpoint)
    results.append(result)
    evaluate_steps(results, before, scope, (67,10,20,30,40,50,60))


def evaluate_steps(results, before, scope, steps):
    for step in steps:
        checkpoint = PILOT / f'training/checkpoint-{step}'
        old_dev = PILOT / f'evaluation/dev_step_{step}'
        assert json.loads((old_dev/'summary.json').read_text())['adapter_path'] == str(checkpoint)
        extra = OUT / f'original_test_step_{step}'
        with (OUT/f'decode_step_{step}.log').open('x') as log:
            subprocess.run([sys.executable,str(REPO/'finetune/scripts/evaluate_asr_checkpoint.py'),
                '--base-model-path','/data/qwen3-asr/models/Qwen3-ASR-1.7B',
                '--adapter-path',str(checkpoint),'--split-jsonl',str(DATA/'setting_d_test.jsonl'),
                '--out-dir',str(extra),'--batch-size','8','--critical-cer-threshold','0.5',
                '--max-new-tokens','256'],stdout=log,stderr=subprocess.STDOUT,check=True)
        result, after = record(f'round_step{step}', scope,
            [(old_dev/'predictions.csv','prediction'),(extra/'predictions.csv','prediction')],checkpoint)
        results.append(result)
        if step == 67:
            paired_endpoint(before, after)
    order = ['zero_shot','m0_step50']+[f'round_step{s}' for s in (10,20,30,40,50,60,67)]
    results.sort(key=lambda r:order.index(r['model']))
    write_immutable_json(OUT/'results.json',results)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['model','overall_mean_CER_pct','easy_mean_CER_pct','medium_mean_CER_pct','hard_mean_CER_pct','patient_macro_CER_pct','overall_pooled_CER_pct'])
    for result in results:
        m = result['metrics']
        writer.writerow([result['model']]+[100*m[k]['mean_utterance_cer'] for k in ('overall','easy','medium','hard')]+[100*m['overall'][k] for k in ('patient_macro_cer','pooled_cer')])
    write_immutable_bytes(OUT/'comparison.csv',buffer.getvalue().encode())
    print('Complete: merged-test results and paired endpoint analysis saved.',flush=True)


def finish_from_saved_endpoint():
    """Explicit continuation after the endpoint serialization failure; no re-decode."""
    scope = jsonl(MERGED)
    results = [json.loads((OUT/name/'summary.json').read_text()) for name in ('zero_shot','m0_step50','round_step67')]
    for result in results:
        assert result['split_sha256'] == sha256_file(MERGED)
    before = jsonl(OUT/'m0_step50/predictions.jsonl')
    after = jsonl(OUT/'round_step67/predictions.jsonl')
    paired_endpoint(before, after)
    write_immutable_json(OUT/'serialization_fix.json', {
        'error':'PairedSnapshotEvaluation is a dataclass, not a Pydantic BaseModel',
        'fix':'TypeAdapter(PairedSnapshotEvaluation).dump_python(mode=json)',
        'predictions_changed':False, 'source_sha256':sha256_file(Path(__file__))})
    evaluate_steps(results,before,scope,(10,20,30,40,50,60))


if __name__ == '__main__':
    if sys.argv[1:] == ['--finish-from-saved-endpoint']:
        finish_from_saved_endpoint()
    else:
        assert len(sys.argv) == 1
        main()
