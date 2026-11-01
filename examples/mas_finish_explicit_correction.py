#!/usr/bin/env python3
"""Finalize the observed interrupted text probe with explicit response reuse."""

import argparse
from pathlib import Path
import json

from mas_selection_dpo_probe import read, rows, save, jsonl
from mas_full_text_correction_probe import request_digest, parse_response
from mas_evaluation_report import compare
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file
from therapist_harness.text import edit_distance, text_normalize


def main(base):
    old = base / 'llm_explicit_feedback_correction_v1'
    log = base / 'launch_llm_explicit_feedback_correction_v1/supervisor.log'
    assert 'Traceback' in log.read_text(), 'Wait for the original executor to drain and report its failure.'
    root = base / 'llm_explicit_feedback_correction_v2'
    root.mkdir(exist_ok=False)
    mapping = read(old / 'utterance_requests.json')
    keys = set(mapping.values())
    assert keys == {p.parent.name for p in (old / 'requests').glob('*/request.json')}
    original_protocol = read(old / 'protocol.json')
    assert all(sha256_file(Path(p)) == digest for p, digest in original_protocol['source_sha256'].items())
    protocol = RunProtocol.model_validate(read(base / 'lexical96_mas_v1/protocol.json'))
    save(root / 'protocol.json', dict(source_policy=original_protocol['source_policy'],
        original_execution=str(old), original_protocol_sha256=sha256_file(old / 'protocol.json'),
        original_failure_log=str(log), original_failure_log_sha256=sha256_file(log),
        recovery_source_sha256=sha256_file(Path(__file__)),
        recovery='Retain every completed response. Retry only requests with no saved response after the observed transport failure. Preserve the old run and source hashes.',
        evaluation_scope=original_protocol['evaluation_scope']))
    entries, retries = {}, []
    for key in sorted(keys):
        request_path = old / 'requests' / key / 'request.json'
        request = read(request_path)
        assert request_digest(request) == key
        response_path = request_path.with_name('response.json')
        reused = response_path.exists()
        if not reused:
            request_path = root / 'requests' / key / 'request.json'
            save(request_path, request)
            response_path = request_path.with_name('response.json')
            response = serialize_response(execute_therapist_request(protocol, request))
            save(response_path, response)
            retries.append(dict(request_sha256=key, original_request=str(old / 'requests' / key / 'request.json'),
                original_response_absent=True, new_request=str(request_path), new_response=str(response_path)))
        response = read(response_path)
        correction = parse_response(response)
        entries[key] = dict(request_path=str(request_path), request_sha256=key,
            request_file_sha256=sha256_file(request_path), response_path=str(response_path),
            response_sha256=sha256_file(response_path), prediction=correction.prediction,
            reason=correction.reason, reused_response=reused)
    save(root / 'response_provenance.json', entries)
    save(root / 'recovery_receipt.json', dict(actual_new_calls=len(retries), retries=retries,
        reused_completed_responses=len(entries) - len(retries), original_files_overwritten=False))
    save(root / 'utterance_requests.json', mapping)
    parent = base / 'parent_eval_check/predictions.jsonl'
    predictions = []
    for row in rows(parent):
        prediction = entries[mapping[row['utt_id']]]['prediction']
        reference, hypothesis = text_normalize(row['clean_gt']), text_normalize(prediction)
        edits = edit_distance(reference, hypothesis)
        cer = edits / len(reference)
        predictions.append(dict(row, prediction=prediction, edit_count=edits, cer=cer,
            normalized_reference_characters=len(reference), critical=not hypothesis or cer >= .5, exact=edits == 0))
    jsonl(root / 'predictions.jsonl', predictions)
    metrics = evaluate_snapshot(predictions, prediction_column='prediction', critical_cer_threshold=.5)
    save(root / 'summary.json', dict(metrics=metrics.model_dump(mode='json'), actual_new_asr_decoding=False))
    save(root / 'paired_parent.json', compare(parent, root / 'predictions.jsonl'))
    save(root / 'results.json', dict(status='complete', metrics=metrics.model_dump(mode='json'),
        actual_new_recovery_calls=len(retries), reused_completed_responses=len(entries) - len(retries),
        actual_training=False, actual_new_asr_decoding=False))
    print(metrics.overall.model_dump(mode='json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
