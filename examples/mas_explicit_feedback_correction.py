#!/usr/bin/env python3
"""Test named confusion directions with the unchanged single-transcript policy."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

from mas_selection_dpo_probe import read, rows, save, jsonl
from mas_full_text_correction_probe import request_digest, parse_response
from mas_evaluation_report import compare
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file
from therapist_harness.text import edit_distance, text_normalize


def explicit_feedback(feedback):
    """Preserve observations and counts; make the intended/error direction explicit."""
    observations = []
    for item in feedback['observed_surface_confusions']:
        component, intended_phone, observed_phone, intended, observed = item['key']
        observations.append(dict(component=component, intended_phone=intended_phone,
            observed_asr_phone=observed_phone, intended_character=intended,
            observed_asr_character=observed, count=item['count']))
    return dict(coupling=feedback['coupling'], observed_surface_confusions=observations,
        scope='Previously observed aggregate development error feedback. It does not identify the true transcript for the presented item.')


def main(base):
    started = time.monotonic()
    root = base / 'llm_explicit_feedback_correction_v1'
    root.mkdir(exist_ok=False)
    parent_path = base / 'parent_eval_check/predictions.jsonl'
    feedback_path = base / 'lexical_feedback_v1.json'
    template_path = base / 'llm_baseline_text_ablation_v1/requests/u000/request.json'
    protocol_path = base / 'lexical96_mas_v1/protocol.json'
    protocol = RunProtocol.model_validate(read(protocol_path))
    original = rows(parent_path)
    template = read(template_path)
    user = json.loads(template['input'][1]['content'])
    old_feedback = user['aggregate_feedback']
    user['aggregate_feedback'] = explicit_feedback(read(feedback_path))
    # This experiment changes only the serialization of the same feedback facts.
    assert old_feedback['coupling'] == user['aggregate_feedback']['coupling']
    assert old_feedback['scope'] == user['aggregate_feedback']['scope']
    template['input'][1]['content'] = json.dumps(user, ensure_ascii=False)
    files = [Path(__file__), Path(__file__).with_name('mas_full_text_correction_probe.py'),
             parent_path, feedback_path, template_path, protocol_path]
    hashes = {str(p): sha256_file(p) for p in files}
    save(root / 'protocol.json', dict(source_sha256=hashes, source_policy=template,
        intervention='Replace positional confusion keys with named intended/observed fields. Preserve all observations and counts, the system prompt, ASR outputs and LLM settings.',
        evaluation_scope='Repeated 511-row development feedback, not a sealed test.',
        input_policy='Single R-only hypothesis plus aggregate feedback; no item label, audio, patient identity or alternative hypothesis.',
        actual_training=False, actual_new_asr_decoding=False, max_concurrent_requests=3,
        smoke='First three distinct requests sequentially; validate response completion/schema before dispatching the rest. No performance gate.',
        cache_policy='One response per exact full request identity, with no performance-based selection.'))
    jobs, mapping = {}, {}
    for row in original:
        request = json.loads(json.dumps(template))
        user = json.loads(request['input'][1]['content'])
        assert set(user) == {'aggregate_feedback', 'baseline_transcript'}
        user['baseline_transcript'] = row['prediction']
        request['input'][1]['content'] = json.dumps(user, ensure_ascii=False)
        digest = request_digest(request)
        jobs[digest] = request
        mapping[row['utt_id']] = digest
    save(root / 'utterance_requests.json', mapping)
    save(root / 'work_plan.json', dict(audio_count=len(original), distinct_requests=len(jobs),
        actual_new_requests=len(jobs), prior_response_reuse=False))

    def dispatch(item):
        digest, request = item
        directory = root / 'requests' / digest
        save(directory / 'request.json', request)
        t0 = time.monotonic()
        response = serialize_response(execute_therapist_request(protocol, request))
        save(directory / 'response.json', response)
        correction = parse_response(response)
        entry = dict(request_path=str(directory / 'request.json'), request_sha256=digest,
            request_file_sha256=sha256_file(directory / 'request.json'),
            response_path=str(directory / 'response.json'), response_sha256=sha256_file(directory / 'response.json'),
            prediction=correction.prediction, reason=correction.reason,
            elapsed_seconds=time.monotonic() - t0)
        save(directory / 'correction.json', entry)
        return digest, entry

    entries = {}
    items = list(jobs.items())
    for item in items[:3]:
        digest, entry = dispatch(item)
        entries[digest] = entry
    save(root / 'smoke.json', dict(status='passed', actual_calls=len(entries),
        request_keys=list(entries), validates='Completed structured responses, unchanged single-hypothesis input contract; no claims about correction accuracy.'))
    print(dict(smoke='passed', completed=3, total=len(jobs)), flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(dispatch, item) for item in items[3:]]):
            digest, entry = future.result()
            entries[digest] = entry
            print(dict(completed=len(entries), total=len(jobs)), flush=True)
    assert all(sha256_file(Path(p)) == value for p, value in hashes.items())
    save(root / 'response_provenance.json', entries)
    predictions = []
    for row in original:
        prediction = entries[mapping[row['utt_id']]]['prediction']
        reference, hypothesis = text_normalize(row['clean_gt']), text_normalize(prediction)
        edits = edit_distance(reference, hypothesis)
        cer = edits / len(reference)
        predictions.append(dict(row, prediction=prediction, edit_count=edits, cer=cer,
            normalized_reference_characters=len(reference), critical=not hypothesis or cer >= .5, exact=edits == 0))
    path = root / 'predictions.jsonl'
    jsonl(path, predictions)
    metrics = evaluate_snapshot(predictions, prediction_column='prediction', critical_cer_threshold=.5)
    save(root / 'paired_parent.json', compare(parent_path, path))
    save(root / 'summary.json', dict(metrics=metrics.model_dump(mode='json'), actual_new_asr_decoding=False,
        actual_new_llm_requests=len(jobs), elapsed_seconds=time.monotonic() - started))
    save(root / 'results.json', dict(status='complete', method='LLM post-correction with explicit confusion direction',
        requires_tts_or_specialist=False, metrics=metrics.model_dump(mode='json')))
    print(metrics.overall.model_dump(mode='json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
