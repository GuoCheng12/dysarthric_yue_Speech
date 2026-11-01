#!/usr/bin/env python3
"""Resume the observed review token-limit interruption without fabricating review."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

from mas_selection_dpo_probe import REPO, execute, jsonl, read, rows, save
from therapist_harness.review_worker import ReviewASRWorker
from therapist_harness.store import sha256_file
from therapist_harness.transcript_review import validate_transcript_review


def prepare(base, root):
    old = base / 'adaptive96_random_reference_data_v1'
    failure = read(old / 'generation/generation_failure.json')
    assert failure['stage'] == 'review_transcription'
    broken = old / 'generation/attempts/0078_23aa1be41e45/attempt_1.transcription.json'
    assert read(broken)['technical_error'] == 'asr_token_limit'
    completed = sorted((old / 'generation/attempts').glob('*/assignment_result.json'))
    assert len(completed) == 78
    done = {read(p)['sample_id'] for p in completed}
    original = rows(old / 'manifest.jsonl')
    remaining = [dict(x, output_wav=str(root / 'audio' / Path(x['output_wav']).name))
                 for x in original if x['sample_id'] not in done | {failure['failed_sample_id']}]
    assert len(remaining) == 17
    root.mkdir(exist_ok=False); (root / 'audio').mkdir()
    for name in ['protocol.json', 'generation_plan.json', 'materialization_sidecar.json']:
        save(root / name, read(old / name))
    jsonl(root / 'manifest.jsonl', original)
    jsonl(root / 'remaining_manifest.jsonl', remaining)
    plan = read(old / 'generation_plan.json')
    ids = {x['sample_id'] for x in remaining}
    plan['choices'] = [x for x in plan['choices'] if x['sample_id'] in ids]
    save(root / 'remaining_generation_plan.json', plan)
    sources = [Path(__file__), broken, old / 'generation/generation_failure.json',
               old / 'manifest.jsonl', old / 'generation_execution/supervisor_launch.json'] + completed
    for p in completed:
        x = read(p); sources += [Path(x['output_wav']), Path(x['review']['transcription_receipt_path'])]
    sources += [root / n for n in ['protocol.json', 'manifest.jsonl', 'generation_plan.json',
                                  'materialization_sidecar.json', 'remaining_manifest.jsonl', 'remaining_generation_plan.json']]
    provenance = read(old / 'provenance.json')
    provenance.update(recovery_source=str(old), source_sha256={**provenance['source_sha256'],
                      **{str(p): sha256_file(p) for p in sources}})
    save(root / 'provenance.json', provenance)
    save(root / 'data_stage_execution.json', dict(
        source_sha256=provenance['source_sha256'], old_completed_assignments=[str(p) for p in completed],
        failed_sample_id=failure['failed_sample_id'], failed_transcription=str(broken),
        migration='Re-transcribe the 78 unchanged completed waveforms with a fresh fixed-R-only worker; reuse each actual Agent response only if its three input texts are byte-identical. Preserve old interrupted session and receipts. Generate only the 17 unattempted assignments. The token-limited case is technically unresolved, unadmitted, with no invented Agent judgment.',
        admission_and_training_objective_changed=False))


def run(base, root):
    started = time.time(); binding = read(root / 'data_stage_execution.json')
    assert all(sha256_file(Path(p)) == h for p, h in binding['source_sha256'].items())
    p = read(root / 'protocol.json'); c = p['preference_loop']
    migration = root / 'migration'; migration.mkdir()
    worker = ReviewASRWorker(base_model=p['base_model_dir'], adapter=c['review_asr_adapter'],
                             session_path=migration / 'review_worker.json')
    outputs = []; receipts = []
    try:
        for i, source in enumerate(binding['old_completed_assignments']):
            previous = read(Path(source)); review = previous['review']
            validate_transcript_review(review, c['review_asr_adapter_sha256'],
                                       c['review_asr_base_model_sha256'], require_worker_closed=False)
            path = worker.transcribe(previous['output_wav'], migration / f'{i:03d}.transcription.json')
            transcribed = read(path)
            assert transcribed['prediction'] == review['asr_transcription']
            assert transcribed['raw_output'] == review['asr_raw_output']
            migrated = dict(review, transcription_receipt_path=str(path), transcription_receipt_sha256=sha256_file(path))
            current = deepcopy(previous); current['review'] = migrated
            current['attempt_records'][-1]['review'] = migrated
            current['reuse_provenance'] = dict(assignment_result=source, assignment_result_sha256=sha256_file(Path(source)),
                new_ASR_transcription=str(path), actual_new_Agent_call=False,
                original_Agent_response_id=review['response_id'], exact_three_text_input_equality=True)
            save(migration / f'{i:03d}.assignment_result.json', current)
            outputs.append(current); receipts.append(current['reuse_provenance'])
        worker.close()
    finally:
        if not worker.closed: worker.abort()
    for x in outputs:
        validate_transcript_review(x['review'], c['review_asr_adapter_sha256'], c['review_asr_base_model_sha256'])
    migration_seconds = time.time() - started
    save(migration / 'receipt.json', dict(status='complete', migrated=len(outputs), actual_new_ASR_transcriptions=len(outputs),
        actual_new_Agent_calls=0, old_worker_graceful_close_claimed=False, new_worker_closed=True,
        elapsed_seconds=migration_seconds, assignments=receipts))
    command = [sys.executable, str(REPO / 'examples/generate_reviewed_sg_recovery_v2.py'),
        '--manifest', str(root / 'remaining_manifest.jsonl'), '--out-dir', str(root / 'remaining_generation'),
        '--generation-plan', str(root / 'remaining_generation_plan.json'), '--review-mode', 'asr_transcript',
        '--review-asr-worker', '--audio-review-model', c['audio_review_model'],
        '--audio-review-base-url', c['audio_review_base_url'], '--review-asr-model', p['base_model_dir'],
        '--review-asr-adapter', c['review_asr_adapter'], '--review-asr-adapter-sha256', c['review_asr_adapter_sha256'],
        '--review-asr-base-model-sha256', c['review_asr_base_model_sha256'], '--review-allow-pending']
    execute(root, command, 'remaining_generation')
    fresh = read(root / 'remaining_generation/generation_summary.json'); outputs += fresh['outputs']
    old = base / 'adaptive96_random_reference_data_v1'
    failed = read(old / 'generation/attempts/0078_23aa1be41e45/attempt_1.generation.json')
    original = {x['sample_id']: x for x in rows(root / 'manifest.jsonl')}
    failed.update(admitted=False, action='technical_unresolved', review=None,
        assignment_seed=original[failed['sample_id']]['seed'],
        unresolved_transcription=binding['failed_transcription'], actual_Agent_judgment_for_final_attempt=False)
    outputs.append(failed); outputs.sort(key=lambda x:list(original).index(x['sample_id']))
    accepted = []
    for x in outputs:
        if not x['admitted']: continue
        accepted.append(dict(original[x['sample_id']], source='synthetic', audio_id=x['sample_id'], utt_id=x['sample_id'],
            audio=x['output_wav'], audio_sha256=x['output_wav_sha256'], target_text=x['target_text'], clean_gt=x['target_text'],
            prompt='', language='Cantonese', reference_audio=x['reference_wav'], reference_text=x['reference_text'],
            generation_mode=x['generation_mode'], generation_seed=x['seed'], review=x['review'],
            chosen_status='intended_target_after_coarse_review_not_word_verified'))
    generation = root / 'generation'; generation.mkdir()
    jsonl(generation / 'accepted.jsonl', accepted)
    jsonl(generation / 'reviews.jsonl', [dict(x['review'],sample_id=x['sample_id'],audio_id=x['sample_id'])
                                       for x in outputs if x['review'] is not None])
    save(generation / 'generation_summary.json', dict(status='complete_with_technical_unresolved_case', outputs=outputs,
        output_count=96, admitted_count=len(accepted), pending_count=fresh['pending_count'],
        excluded_count=fresh['excluded_count'], technical_unresolved_count=1,
        generation_attempt_count=sum(len(x['attempt_records']) for x in outputs if 'attempt_records' in x) + 2,
        elapsed_seconds=time.time() - read(old / 'generation_execution/supervisor_launch.json')['started_at'],
        timing=dict(migration_seconds=migration_seconds, remaining_generation=fresh['timing'],
                    timing_scope='Wall time since original launch includes interruption and recovery; original per-attempt timing remains in source records.'),
        original_failure=str(old / 'generation/generation_failure.json'), migration_receipt=str(migration / 'receipt.json'),
        independent_generation_fragment=str(root / 'remaining_generation/generation_summary.json')))
    sample = [sys.executable, str(REPO / 'examples/sample_asr_preferences.py'), 'sample',
        '--inputs', str(generation / 'accepted.jsonl'), '--real-train', p['real_train']['path'],
        '--feedback', p['feedback_split']['path'], '--out-dir', str(root / 'sampling'), '--base-model', p['base_model_dir'],
        '--current-adapter', p['initial_adapter']['path'], '--expected-parent-sha256', p['initial_adapter']['sha256'],
        '--samples', str(c['samples_per_audio']), '--seed', str(p['lora']['seed']), '--temperature', str(c['temperature']),
        '--top-p', str(c['top_p']), '--max-new-tokens', str(c['max_new_tokens']), '--include-greedy']
    execute(root, sample, 'sampling')
    assert all(sha256_file(Path(q)) == h for q, h in binding['source_sha256'].items())
    save(root / 'data_stage_result.json', dict(status='complete', actual_DPO_training_executed=False,
        recovered_completed_audio=78, attempted_remaining_audio=17, technically_unresolved_audio=1,
        independent_sampling_receipt=str(root / 'sampling/sampling_receipt.json'),
        original_failure_preserved=True, actual_new_Agent_calls_for_migration=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run']); parser.add_argument('base', type=Path)
    args = parser.parse_args(); base = args.base.resolve(); root = base / 'adaptive96_random_reference_data_recovery_v2'
    (prepare if args.stage == 'prepare' else run)(base, root)
