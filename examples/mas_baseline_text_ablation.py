#!/usr/bin/env python3
"""LLM correction using only R-only text on the same disagreement trigger sets."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, read, rows, save, jsonl
from mas_evaluation_report import compare
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.schema import RunProtocol
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.store import sha256_file
from therapist_harness.text import edit_distance, text_normalize


class Correction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prediction: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=240)


def main(base):
    from openai.lib._pydantic import to_strict_json_schema

    root = base / 'llm_baseline_text_ablation_v1'
    root.mkdir(exist_ok=False)
    protocol = RunProtocol.model_validate(read(base / 'lexical96_mas_v1/protocol.json'))
    parent_path = base / 'parent_eval_check/predictions.jsonl'
    original = rows(parent_path)
    by_id = {x['utt_id']: x for x in original}
    paths = dict(mas=base / 'lexical96_mas_v1/round_001/mas_dpo/evaluation/predictions.jsonl',
                 random=base / 'lexical96_random_control_v1/evaluation/predictions.jsonl')
    triggers = {}
    for arm, path in paths.items():
        alternatives = {x['utt_id']: x for x in rows(path)}
        assert alternatives.keys() == by_id.keys()
        triggers[arm] = {key for key in by_id if text_normalize(by_id[key]['prediction']) != text_normalize(alternatives[key]['prediction'])}
    union = sorted(set.union(*triggers.values()))
    files = [Path(__file__), parent_path, *paths.values(), base / 'lexical_feedback_v1.json',
             REPO / 'src/therapist_harness/agent.py', REPO / 'src/therapist_harness/evidence.py']
    hashes = {str(p): sha256_file(p) for p in files}
    feedback = read(base / 'lexical_feedback_v1.json')
    feedback_text = dict(coupling=feedback['coupling'], observed_surface_confusions=feedback['observed_surface_confusions'],
        scope='Previously observed aggregate development error feedback. It does not identify the true transcript for the presented item.')
    save(root / 'protocol.json', dict(source_sha256=hashes, model=protocol.agent.model,
        reasoning_effort='medium', one_utterance_per_request=True,
        scope='Same disagreement triggers as the two specialist selectors, but the LLM sees only the baseline transcript. Tests the information contributed by the specialist hypothesis. Triggers still require specialist outputs; this is not a standalone no-TTS system.',
        input_fields=['aggregate_feedback', 'baseline_transcript'],
        actual_new_asr_decoding=False, actual_new_training=False))
    mapping = {f'u{i:03d}': key for i, key in enumerate(union)}
    save(root / 'key_mapping.json', mapping)
    save(root / 'triggers.json', {arm: sorted(keys) for arm, keys in triggers.items()})

    def dispatch(item):
        key, utt_id = item
        directory = root / 'requests' / key
        request = dict(model=protocol.agent.model, reasoning={'effort': 'medium'}, store=False,
            stream=True, max_output_tokens=8192,
            input=[dict(role='system', content='''Review one Cantonese ASR transcript using aggregate observed lexical-confusion feedback.
You have no audio, no reference transcript, and no other ASR hypothesis.
Make only concrete, context-supported corrections to likely ASR errors. A known
confusion does not justify changing every matching word. Ordinary fluency alone
is not evidence: patients may read unusual sentences. Preserve Cantonese wording,
particles, repetitions and content; do not paraphrase or make stylistic improvements.
Keep the original transcript wherever the evidence is insufficient. Return the
complete proposed transcript and one brief reason. Do not claim to know the audio truth.'''),
                dict(role='user', content=json.dumps(dict(aggregate_feedback=feedback_text,
                     baseline_transcript=by_id[utt_id]['prediction']), ensure_ascii=False))],
            text={'format': {'type': 'json_schema', 'name': 'baseline_text_correction', 'strict': True,
                           'schema': to_strict_json_schema(Correction)}})
        save(directory / 'request.json', request)
        response = serialize_response(execute_therapist_request(protocol, request))
        save(directory / 'response.json', response)
        if response['status'] != 'completed':
            raise RuntimeError('baseline correction incomplete')
        messages = [(i, x) for i, x in enumerate(response['output']) if x.get('type') == 'message' and x.get('phase') == 'final_answer']
        if len(messages) != 1:
            raise ValueError('expected one explicit final answer')
        index, message = messages[0]
        correction = Correction.model_validate_json(''.join(x['text'] for x in message['content'] if x['type'] == 'output_text'))
        save(directory / 'correction.json', dict(correction.model_dump(mode='json'), output_index=index,
             response_sha256=sha256_file(directory / 'response.json')))
        return utt_id, correction.prediction

    corrected = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(dispatch, item) for item in mapping.items()]):
            key, prediction = future.result()
            corrected[key] = prediction
            print(dict(completed=len(corrected), total=len(mapping)), flush=True)
    results = {}
    for arm, eligible in triggers.items():
        predictions = []
        for row in original:
            prediction = corrected[row['utt_id']] if row['utt_id'] in eligible else row['prediction']
            reference = text_normalize(row['clean_gt'])
            normalized = text_normalize(prediction)
            edits = edit_distance(reference, normalized)
            cer = edits / len(reference)
            predictions.append(dict(row, prediction=prediction, edit_count=edits, cer=cer,
                normalized_reference_characters=len(reference), critical=not normalized or cer >= .5, exact=edits == 0))
        path = root / arm / 'predictions.jsonl'
        jsonl(path, predictions)
        metrics = evaluate_snapshot(predictions, prediction_column='prediction', critical_cer_threshold=.5)
        save(root / arm / 'summary.json', dict(metrics=metrics.model_dump(mode='json'), actual_new_asr_decoding=False))
        comparison = compare(parent_path, path)
        save(root / arm / 'paired_parent.json', comparison)
        results[arm] = dict(trigger_count=len(eligible), metrics=metrics.model_dump(mode='json'), groups=comparison['groups'])
    assert all(sha256_file(Path(p)) == digest for p, digest in hashes.items())
    save(root / 'results.json', dict(status='complete', arms=results,
        actual_independent_llm_requests=len(mapping), specialist_transcripts_provided_to_llm=False))
    print({arm: value['metrics']['overall']['pooled_cer'] for arm, value in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
