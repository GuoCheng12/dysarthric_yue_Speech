#!/usr/bin/env python3
"""Measure residual DPO value under an identical frozen LLM correction policy."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

from mas_selection_dpo_probe import read, rows, save, jsonl
from mas_full_text_correction_probe import request_digest, parse_response
from mas_evaluation_report import compare
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.schema import RunProtocol
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.store import sha256_file
from therapist_harness.text import edit_distance, text_normalize


def main(base, corrector_root):
    started = time.monotonic()
    root = base / 'llm_corrected_dpo_comparison_v1'
    root.mkdir(exist_ok=False)
    prior = corrector_root
    assert read(prior / 'results.json')['status'] == 'complete'
    cache = read(prior / 'response_provenance.json')
    for digest, entry in cache.items():
        assert request_digest(read(entry['request_path'])) == digest
        assert sha256_file(Path(entry['response_path'])) == entry['response_sha256']
        assert parse_response(read(entry['response_path'])).prediction == entry['prediction']
    template = read(prior / 'protocol.json')['source_policy']
    protocol_path = base / 'lexical96_mas_v1/protocol.json'
    protocol = RunProtocol.model_validate(read(protocol_path))
    paths = dict(parent=base / 'parent_eval_check/predictions.jsonl',
        mas=base / 'lexical96_mas_v1/round_001/mas_dpo/evaluation/predictions.jsonl',
        random=base / 'lexical96_random_control_v1/evaluation/predictions.jsonl')
    originals = {arm: rows(path) for arm, path in paths.items()}
    files = [Path(__file__), Path(__file__).with_name('mas_full_text_correction_probe.py'),
        protocol_path, prior / 'protocol.json', prior / 'response_provenance.json',
        prior / 'predictions.jsonl', *paths.values()]
    hashes = {str(p): sha256_file(p) for p in files}
    save(root / 'protocol.json', dict(source_sha256=hashes, correction_policy=template,
        intervention='Apply the exact same frozen single-hypothesis corrector to R-only, MAS-reference DPO, and random-reference DPO outputs.',
        hypothesis='Test whether TTS DPO contributes information beyond the R-only plus LLM correction baseline. Improvement over raw R-only alone cannot establish DPO value.',
        actual_new_asr_decoding=False, actual_new_training=False, max_concurrent_requests=3,
        exact_full_request_cache=True, no_per_item_label_in_llm_input=True,
        evaluation_scope='Repeated development feedback; not an independent test.'))
    jobs, mappings = {}, {}
    for arm, original in originals.items():
        mappings[arm] = {}
        for row in original:
            request = json.loads(json.dumps(template))
            user = json.loads(request['input'][1]['content'])
            assert set(user) == {'aggregate_feedback', 'baseline_transcript'}
            user['baseline_transcript'] = row['prediction']
            request['input'][1]['content'] = json.dumps(user, ensure_ascii=False)
            digest = request_digest(request)
            mappings[arm][row['utt_id']] = digest
            if digest not in cache:
                jobs[digest] = request
    assert set(mappings['parent'].values()) <= cache.keys()
    save(root / 'utterance_requests.json', mappings)
    save(root / 'work_plan.json', dict(audio_count_per_arm={a: len(x) for a, x in originals.items()},
        new_distinct_requests=len(jobs), reused_distinct_requests=len(set.union(*(set(x.values()) for x in mappings.values())) & cache.keys())))
    print(read(root / 'work_plan.json'), flush=True)

    def dispatch(item):
        digest, request = item
        directory = root / 'requests' / digest
        save(directory / 'request.json', request)
        response = serialize_response(execute_therapist_request(protocol, request))
        save(directory / 'response.json', response)
        correction = parse_response(response)
        entry = dict(request_path=str(directory / 'request.json'), request_sha256=digest,
            request_file_sha256=sha256_file(directory / 'request.json'), response_path=str(directory / 'response.json'),
            response_sha256=sha256_file(directory / 'response.json'), prediction=correction.prediction,
            reason=correction.reason)
        save(directory / 'correction.json', entry)
        return digest, entry

    completed = 0
    failures = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(dispatch, item): item[0] for item in jobs.items()}
        for future in as_completed(futures):
            try:
                digest, entry = future.result()
            except Exception as error:
                failures.append(dict(request_sha256=futures[future], error_type=type(error).__name__, error=str(error)))
                save(root / 'failures.json', failures)
                print(dict(failed_request=futures[future], completed=completed, total=len(jobs)), flush=True)
                continue
            cache[digest] = entry
            completed += 1
            print(dict(completed=completed, total=len(jobs)), flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} requests failed; completed responses preserved, see failures.json')
    used = set.union(*(set(x.values()) for x in mappings.values()))
    save(root / 'response_provenance.json', {key: cache[key] for key in sorted(used)})
    results = {}
    for arm, original in originals.items():
        predictions = []
        for row in original:
            prediction = cache[mappings[arm][row['utt_id']]]['prediction']
            reference, hypothesis = text_normalize(row['clean_gt']), text_normalize(prediction)
            edits = edit_distance(reference, hypothesis)
            cer = edits / len(reference)
            predictions.append(dict(row, prediction=prediction, edit_count=edits, cer=cer,
                normalized_reference_characters=len(reference), critical=not hypothesis or cer >= .5, exact=edits == 0))
        path = root / arm / 'predictions.jsonl'
        jsonl(path, predictions)
        metrics = evaluate_snapshot(predictions, prediction_column='prediction', critical_cer_threshold=.5)
        save(root / arm / 'summary.json', dict(metrics=metrics.model_dump(mode='json'), actual_new_asr_decoding=False))
        results[arm] = metrics.model_dump(mode='json')
    assert sha256_file(root / 'parent/predictions.jsonl') == sha256_file(prior / 'predictions.jsonl')
    for arm in ['mas', 'random']:
        save(root / arm / 'paired_corrected_parent.json', compare(root / 'parent/predictions.jsonl', root / arm / 'predictions.jsonl'))
    assert all(sha256_file(Path(p)) == digest for p, digest in hashes.items())
    save(root / 'results.json', dict(status='complete', arms=results, actual_new_llm_calls=len(jobs),
        elapsed_seconds=time.monotonic() - started, actual_new_training=False))
    print({arm: m['overall']['pooled_cer'] for arm, m in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('--corrector-root', required=True, type=Path)
    args = parser.parse_args()
    main(args.base.resolve(), args.corrector_root.resolve())
