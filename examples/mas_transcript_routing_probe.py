#!/usr/bin/env python3
"""Actual text-only MAS selection between cached same-audio ASR transcripts."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, read, rows, save, jsonl
from mas_evaluation_report import compare
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file
from therapist_harness.text import edit_distance, text_normalize


class Choice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    key: str
    selected: Literal['baseline', 'specialist']
    reason: str = Field(min_length=1, max_length=240)


class Choices(BaseModel):
    model_config = ConfigDict(extra='forbid')
    choices: list[Choice]


def main(base):
    from openai.lib._pydantic import to_strict_json_schema

    root = base / 'mas_transcript_routing_probe_v1'
    root.mkdir(exist_ok=False)
    protocol = RunProtocol.model_validate(read(base / 'lexical96_mas_v1/protocol.json'))
    baseline_path = base / 'parent_eval_check/predictions.jsonl'
    source_paths = dict(mas=base / 'lexical96_mas_v1/round_001/mas_dpo/evaluation/predictions.jsonl',
                        random=base / 'lexical96_random_control_v1/evaluation/predictions.jsonl')
    baseline = rows(baseline_path)
    by_id = {x['utt_id']: x for x in baseline}
    sources = {name: {x['utt_id']: x for x in rows(path)} for name, path in source_paths.items()}
    files = [Path(__file__), baseline_path, *source_paths.values(), base / 'lexical_feedback_v1.json',
             REPO / 'src/therapist_harness/agent.py', REPO / 'examples/mas_evaluation_report.py']
    hashes = {str(p): sha256_file(p) for p in files}
    feedback = read(base / 'lexical_feedback_v1.json')
    feedback_text = dict(coupling=feedback['coupling'], observed_surface_confusions=feedback['observed_surface_confusions'],
        scope='Previously observed aggregate development error feedback. It does not identify the true transcript for any presented item.')
    save(root / 'protocol.json', dict(source_sha256=hashes, actual_new_asr_sampling=False,
        actual_new_training=False, method='Online text-only MAS transcript selection on disagreements; identical normalized predictions keep the R-only output.',
        model=protocol.agent.model, reasoning_effort='medium', batch_size=16,
        input_policy='Only opaque keys, two ASR hypotheses and shared aggregate word-confusion feedback; no audio, per-item reference transcript, CER, patient label or model outcome.',
        comparison='Same selector for MAS-reference and random-reference specialists; report separately from single-model DPO. Repeated development feedback, not an independent test.'))
    jobs = []
    mappings = {}
    for arm, alternatives in sources.items():
        assert alternatives.keys() == by_id.keys()
        keys = {}
        cards = []
        for row in baseline:
            other = alternatives[row['utt_id']]
            assert other['audio'] == row['audio'] and other['clean_gt'] == row['clean_gt']
            if text_normalize(other['prediction']) == text_normalize(row['prediction']):
                continue
            key = f'u{len(keys):03d}'
            keys[key] = row['utt_id']
            cards.append(dict(key=key, baseline=row['prediction'], specialist=other['prediction']))
        mappings[arm] = keys
        save(root / arm / 'key_mapping.json', keys)
        for start in range(0, len(cards), 16):
            batch = cards[start:start + 16]
            request = dict(model=protocol.agent.model, reasoning={'effort': 'medium'}, store=False,
                stream=True, max_output_tokens=8192,
                input=[dict(role='system', content='''Select one of two actual Cantonese ASR hypotheses for each same-audio item.
The baseline is the retained real-data model. The specialist is a candidate update.
You have text evidence only and cannot listen or prove the audio's true words.
Use the shared observed lexical-confusion feedback and sentence context to identify
specific, supported corrections. Ordinary fluency alone is not evidence: patients may
read unusual sentences, and Cantonese particles, repetitions and wording matter.
Keep the baseline where the specialist offers no concrete correction or introduces
unsupported changes. A known confusion does not justify automatically replacing
every matching word. Select a COMPLETE supplied hypothesis; do not rewrite or merge.
Return exactly one choice for every provided key and a brief reason. No scores.'''),
                    dict(role='user', content=json.dumps(dict(feedback=feedback_text, items=batch), ensure_ascii=False))],
                text={'format': {'type': 'json_schema', 'name': 'transcript_selection', 'strict': True,
                               'schema': to_strict_json_schema(Choices)}})
            directory = root / arm / f'batch_{start // 16:03d}'
            save(directory / 'request.json', request)
            jobs.append((arm, directory, batch, request))

    def dispatch(job):
        arm, directory, batch, request = job
        response = serialize_response(execute_therapist_request(protocol, request))
        save(directory / 'response.json', response)
        if response['status'] != 'completed':
            raise RuntimeError('transcript selector response incomplete')
        messages = [(i, x) for i, x in enumerate(response['output']) if x.get('type') == 'message' and x.get('phase') == 'final_answer']
        if len(messages) != 1:
            raise ValueError('expected one explicit final answer')
        index, message = messages[0]
        parsed = Choices.model_validate_json(''.join(x['text'] for x in message['content'] if x['type'] == 'output_text'))
        selected = {x.key: x.model_dump(mode='json') for x in parsed.choices}
        assert len(selected) == len(parsed.choices) and selected.keys() == {x['key'] for x in batch}
        save(directory / 'selection.json', dict(output_index=index, choices=list(selected.values()),
             response_sha256=sha256_file(directory / 'response.json')))
        return arm, selected

    choices = {name: {} for name in sources}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(dispatch, job) for job in jobs]):
            arm, selected = future.result()
            choices[arm].update(selected)
            print(dict(arm=arm, reviewed=len(choices[arm]), total=len(mappings[arm])), flush=True)
    results = {}
    for arm, decisions in choices.items():
        assert decisions.keys() == mappings[arm].keys()
        selected_ids = {mappings[arm][key] for key, value in decisions.items() if value['selected'] == 'specialist'}
        predictions = []
        for row in baseline:
            prediction = sources[arm][row['utt_id']]['prediction'] if row['utt_id'] in selected_ids else row['prediction']
            reference = text_normalize(row['clean_gt'])
            edits = edit_distance(reference, text_normalize(prediction))
            predictions.append(dict(row, prediction=prediction, edit_count=edits, cer=edits / len(reference),
                                    normalized_reference_characters=len(reference), critical=edits / len(reference) > .5, exact=edits == 0))
        path = root / arm / 'predictions.jsonl'
        jsonl(path, predictions)
        paired = compare(baseline_path, path)
        save(root / arm / 'paired_parent.json', paired)
        results[arm] = dict(specialist_selected=len(selected_ids), llm_reviewed=len(decisions),
                           groups=paired['groups'], requires_llm_at_inference=True)
    assert all(sha256_file(Path(p)) == digest for p, digest in hashes.items())
    save(root / 'results.json', dict(status='complete', arms=results, single_model_dpo_gain_claimed=False))
    print({name: value['groups']['overall'] for name, value in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
