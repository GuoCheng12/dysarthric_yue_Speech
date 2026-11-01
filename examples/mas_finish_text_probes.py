#!/usr/bin/env python3
"""Recover observed gateway rejections and canonically finish the text probes."""

import argparse
import json
from pathlib import Path

from mas_selection_dpo_probe import read, rows, save, jsonl
from mas_transcript_routing_single import Choices
from mas_baseline_text_ablation import Correction
from mas_evaluation_report import compare
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.schema import RunProtocol
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.store import sha256_file
from therapist_harness.text import text_normalize, edit_distance


def main(base):
    root = base / 'text_probe_recovery_v1'
    root.mkdir(exist_ok=False)
    p = RunProtocol.model_validate(read(base / 'lexical96_mas_v1/protocol.json'))
    single = base / 'mas_transcript_routing_single_v1'
    ablation = base / 'llm_baseline_text_ablation_v1'
    for source in [single, ablation]:
        for path, digest in read(source / 'protocol.json')['source_sha256'].items():
            assert sha256_file(Path(path)) == digest
    save(root / 'protocol.json', dict(source_sha256={str(path): sha256_file(path)
         for path in [Path(__file__), single / 'protocol.json', ablation / 'protocol.json']},
         recovery='Only retry actual gateway_concurrency_limit responses, sequentially with identical saved requests; reuse every completed response explicitly. No new ASR decoding or training.'))
    receipts = []

    def payload(response_path):
        response = read(response_path)
        request_path = response_path.with_name('request.json')
        receipt = dict(original_request=str(request_path), original_request_sha256=sha256_file(request_path),
            original_response=str(response_path), original_response_sha256=sha256_file(response_path),
            reused_completed_response=response['status'] == 'completed')
        if response['status'] != 'completed':
            assert response.get('error', {}).get('code') == 'gateway_concurrency_limit'
            directory = root / 'retries' / f'{len(receipts):03d}'
            request = read(request_path)
            save(directory / 'request.json', request)
            response = serialize_response(execute_therapist_request(p, request))
            save(directory / 'response.json', response)
            receipt.update(new_response=str(directory / 'response.json'),
                           new_response_sha256=sha256_file(directory / 'response.json'))
            save(directory / 'provenance.json', receipt)
            if response['status'] != 'completed':
                raise RuntimeError('Sequential replay did not complete; preserved for inspection')
            print(dict(retried=str(response_path.relative_to(base)), status='completed'), flush=True)
        receipts.append(receipt)
        messages = [x for x in response['output'] if x.get('type') == 'message' and x.get('phase') == 'final_answer']
        assert len(messages) == 1
        return ''.join(x['text'] for x in messages[0]['content'] if x['type'] == 'output_text')

    parent_path = base / 'parent_eval_check/predictions.jsonl'
    original = rows(parent_path)
    by_id = {row['utt_id']: row for row in original}

    def export(directory, replacements):
        predictions = []
        for row in original:
            prediction = replacements.get(row['utt_id'], row['prediction'])
            reference = text_normalize(row['clean_gt'])
            hypothesis = text_normalize(prediction)
            edits = edit_distance(reference, hypothesis)
            cer = edits / len(reference)
            predictions.append(dict(row, prediction=prediction, cer=cer, edit_count=edits,
                normalized_reference_characters=len(reference), critical=not hypothesis or cer >= .5, exact=edits == 0))
        path = directory / 'predictions.jsonl'
        jsonl(path, predictions)
        metrics = evaluate_snapshot(predictions, prediction_column='prediction', critical_cer_threshold=.5)
        paired = compare(parent_path, path)
        save(directory / 'summary.json', dict(metrics=metrics.model_dump(mode='json'),
             actual_new_asr_decoding=False, evaluation='Canonical scoring of actual saved/recovered LLM outputs.'))
        save(directory / 'paired_parent.json', paired)
        return dict(metrics=metrics.model_dump(mode='json'), groups=paired['groups'])

    routed_root = base / 'mas_transcript_routing_single_v2'
    routed_root.mkdir(exist_ok=False)
    routed = {}
    for arm, source_path in [('mas', base / 'lexical96_mas_v1/round_001/mas_dpo/evaluation/predictions.jsonl'),
                             ('random', base / 'lexical96_random_control_v1/evaluation/predictions.jsonl')]:
        alternatives = {row['utt_id']: row for row in rows(source_path)}
        mapping = read(single / arm / 'key_mapping.json')
        replacements = {}
        decisions = []
        for index, key in enumerate(sorted(mapping)):
            directory = single / arm / f'batch_{index:03d}'
            user = json.loads(read(directory / 'request.json')['input'][1]['content'])
            assert set(user) == {'feedback', 'items'} and len(user['items']) == 1
            item = user['items'][0]
            assert set(item) == {'key', 'baseline', 'specialist'} and item['key'] == key
            utt_id = mapping[key]
            assert item['baseline'] == by_id[utt_id]['prediction'] and item['specialist'] == alternatives[utt_id]['prediction']
            choice = Choices.model_validate_json(payload(directory / 'response.json')).choices
            assert len(choice) == 1 and choice[0].key == key
            decisions.append(choice[0].model_dump(mode='json'))
            if choice[0].selected == 'specialist':
                replacements[utt_id] = alternatives[utt_id]['prediction']
        save(routed_root / arm / 'decisions.json', decisions)
        routed[arm] = dict(export(routed_root / arm, replacements), specialist_selected=len(replacements),
                           reviewed=len(mapping), one_utterance_per_request=True)
    save(routed_root / 'results.json', dict(status='complete', arms=routed, single_model_dpo_gain_claimed=False))

    corrected = {}
    for key, utt_id in read(ablation / 'key_mapping.json').items():
        directory = ablation / 'requests' / key
        user = json.loads(read(directory / 'request.json')['input'][1]['content'])
        assert set(user) == {'aggregate_feedback', 'baseline_transcript'}
        assert user['baseline_transcript'] == by_id[utt_id]['prediction']
        corrected[utt_id] = Correction.model_validate_json(payload(directory / 'response.json')).prediction
    ablation_root = base / 'llm_baseline_text_ablation_v2'
    ablation_root.mkdir(exist_ok=False)
    ablated = {}
    for arm, eligible in read(ablation / 'triggers.json').items():
        assert set(eligible) <= corrected.keys()
        ablated[arm] = dict(export(ablation_root / arm, {key: corrected[key] for key in eligible}),
                           trigger_count=len(eligible), specialist_hypothesis_provided=False)
    save(ablation_root / 'results.json', dict(status='complete', arms=ablated,
        scope='Baseline text only on the identical specialist-disagreement triggers; not a standalone no-TTS system.'))
    save(root / 'response_provenance.json', receipts)
    receipt_path = root / 'response_provenance.json'
    for directory, source in [(routed_root, single), (ablation_root, ablation)]:
        save(directory / 'provenance.json', dict(original_execution=str(source),
            original_protocol_sha256=sha256_file(source / 'protocol.json'),
            recovery_receipts=str(receipt_path), recovery_receipts_sha256=sha256_file(receipt_path),
            source_execution_sha256=sha256_file(Path(__file__)), old_outputs_overwritten=False))
    save(root / 'results.json', dict(status='complete', reused_completed_responses=sum(x['reused_completed_response'] for x in receipts),
        new_api_requests=sum(not x['reused_completed_response'] for x in receipts),
        routing={k:v['metrics']['overall'] for k,v in routed.items()},
        baseline_text_ablation={k:v['metrics']['overall'] for k,v in ablated.items()}))
    print(read(root / 'results.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
