#!/usr/bin/env python3
"""Matched full-set LLM correction control without task-specific error feedback."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from mas_selection_dpo_probe import read, rows, save, jsonl
from mas_baseline_text_ablation import Correction
from mas_evaluation_report import compare
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file
from therapist_harness.text import edit_distance, text_normalize


def request_digest(request):
    return hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def parse_response(response):
    assert response['status'] == 'completed'
    messages = [x for x in response['output'] if x.get('type') == 'message' and x.get('phase') == 'final_answer']
    assert len(messages) == 1
    return Correction.model_validate_json(''.join(x['text'] for x in messages[0]['content'] if x['type'] == 'output_text'))


def main(base):
    root = base / 'llm_full_no_feedback_correction_v1'
    root.mkdir(exist_ok=False)
    old = base / 'llm_baseline_text_ablation_v1'
    parent_path = base / 'parent_eval_check/predictions.jsonl'
    protocol = RunProtocol.model_validate(read(base / 'lexical96_mas_v1/protocol.json'))
    original = rows(parent_path)
    template = read(old / 'requests/u000/request.json')
    user = json.loads(template['input'][1]['content'])
    user['aggregate_feedback'] = dict(coupling='No task-specific confusion feedback provided.',
        observed_surface_confusions=[], scope='Generic Cantonese ASR text-correction control.')
    template['input'][1]['content'] = json.dumps(user, ensure_ascii=False)
    save(root / 'protocol.json', dict(source_prediction=str(parent_path), source_prediction_sha256=sha256_file(parent_path),
        source_execution_sha256=sha256_file(Path(__file__)), source_policy=template,
        full_audio_count=len(original), one_utterance_per_request=True, max_concurrent_requests=3,
        cache_policy='Exact full request identity. For duplicate prior successful responses choose the first original request path in sorted order, without consulting labels or CER. Retain duplicate provenance.',
        input_policy='R-only transcript only, with task-specific error feedback explicitly empty. Every utterance is processed; no TTS model, specialist hypothesis or disagreement gate is required.',
        actual_new_asr_decoding=False, actual_training=False,
        evaluation_scope='Repeated 511-row development feedback, not a sealed test.'))
    cache = {}
    duplicates = []
    save(root / 'prior_duplicate_responses.json', duplicates)
    mapping = {}
    jobs = {}
    for row in original:
        request = json.loads(json.dumps(template))
        user = json.loads(request['input'][1]['content'])
        assert set(user) == {'aggregate_feedback', 'baseline_transcript'}
        user['baseline_transcript'] = row['prediction']
        request['input'][1]['content'] = json.dumps(user, ensure_ascii=False)
        digest = request_digest(request)
        mapping[row['utt_id']] = digest
        if digest not in cache:
            jobs[digest] = request
    save(root / 'utterance_requests.json', mapping)
    save(root / 'work_plan.json', dict(audio_count=len(original), distinct_requests=len(set(mapping.values())),
        prior_exact_request_cache_hits=len(set(mapping.values()) & cache.keys()), new_distinct_requests=len(jobs),
        cache_selection_uses_performance=False))
    print(read(root / 'work_plan.json'), flush=True)

    def dispatch(item):
        digest, request = item
        directory = root / 'requests' / digest
        save(directory / 'request.json', request)
        response = serialize_response(execute_therapist_request(protocol, request))
        save(directory / 'response.json', response)
        correction = parse_response(response)
        entry = dict(request_sha256=digest, request_path=str(directory / 'request.json'),
            request_file_sha256=sha256_file(directory / 'request.json'), response_path=str(directory / 'response.json'),
            response_sha256=sha256_file(directory / 'response.json'), prediction=correction.prediction,
            reason=correction.reason, reused_response=False)
        save(directory / 'correction.json', entry)
        return digest, entry

    completed = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(dispatch, item) for item in jobs.items()]):
            digest, entry = future.result()
            cache[digest] = entry
            completed += 1
            print(dict(completed_new_requests=completed, total_new_requests=len(jobs)), flush=True)
    used = {digest: cache[digest] for digest in set(mapping.values())}
    save(root / 'response_provenance.json', used)
    predictions = []
    for row in original:
        entry = used[mapping[row['utt_id']]]
        prediction = entry['prediction']
        reference = text_normalize(row['clean_gt'])
        hypothesis = text_normalize(prediction)
        edits = edit_distance(reference, hypothesis)
        cer = edits / len(reference)
        predictions.append(dict(row, prediction=prediction, edit_count=edits, cer=cer,
            normalized_reference_characters=len(reference), critical=not hypothesis or cer >= .5, exact=edits == 0))
    path = root / 'predictions.jsonl'
    jsonl(path, predictions)
    metrics = evaluate_snapshot(predictions, prediction_column='prediction', critical_cer_threshold=.5)
    paired = compare(parent_path, path)
    save(root / 'summary.json', dict(metrics=metrics.model_dump(mode='json'), actual_new_asr_decoding=False,
         original_asr_prediction_sha256=sha256_file(parent_path), actual_new_llm_requests=len(jobs)))
    save(root / 'paired_parent.json', paired)
    save(root / 'results.json', dict(status='complete', metrics=metrics.model_dump(mode='json'),
        requires_tts_or_specialist=False, actual_new_llm_requests=len(jobs), exact_request_cache=True,
        method='Frozen R-only ASR output followed by the same LLM text corrector without task-specific error feedback.'))
    print(metrics.overall.model_dump(mode='json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
