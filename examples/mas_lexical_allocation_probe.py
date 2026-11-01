#!/usr/bin/env python3
"""Allocate a fixed synthetic DPO exposure budget using realized lexical errors."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from mas_audio_allocation_probe import Allocation, run
from mas_selection_dpo_probe import REPO, read, rows, save
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file


def prepare(main, root):
    from openai.lib._pydantic import to_strict_json_schema

    root.mkdir(exist_ok=False)
    p = RunProtocol.model_validate(read(main / 'protocol.json'))
    assert p.preference_loop.comparison == 'synthetic_only'
    d = main / 'round_001'
    baseline = read(d / 'training_allocation.json')
    pairs = rows(d / 'mas_dpo/train.jsonl')
    by_audio = defaultdict(list)
    for pair in pairs:
        assert pair['source'] == 'synthetic' and pair['current_asr_sha256'] == p.initial_adapter.sha256
        by_audio[pair['audio_id']].append(pair)
    roles = {x['sample_id']: x for x in read(d / 'materialization_sidecar.json')['samples']}
    audit = {x['audio_id']: x for x in read(d / 'sampling/preference_audit.json')['audio_results']}
    greedy = {x['audio_id']: x['prediction'] for x in rows(d / 'sampling/greedy.jsonl')}
    outputs = {x['sample_id']: x for x in read(d / 'generation/generation_summary.json')['outputs']}
    keys = {f'a{i:03d}': audio_id for i, audio_id in enumerate(sorted(by_audio))}
    cards = []
    for key, audio_id in keys.items():
        output = outputs[audio_id]
        assert output['admitted'] and output['review']['verdict'] == 'pass'
        generation = output['attempt_records'][-1]['generation']
        cards.append(dict(audio_key=key, role=roles[audio_id]['role'],
            evidence_ids=roles[audio_id]['evidence_ids'], target=by_audio[audio_id][0]['chosen'],
            reference_id=generation['generation_choice']['reference_id'],
            reference_text=output['review']['reference_text'],
            mode=generation['generation_mode'], instruction=generation['effective_instruction_id'],
            coarse_review_reason=output['review']['reason'], greedy=greedy[audio_id],
            correct_random_draws=audit[audio_id]['correct_samples'], error_origin=audit[audio_id]['error_origin'],
            actual_distinct_rejected=[x['rejected'] for x in by_audio[audio_id]]))
    context = read(Path(p.preference_loop.generation_design_path))['proposal_context']
    request = dict(model=p.agent.model, reasoning={'effort': p.agent.reasoning_effort},
        store=False, stream=True, max_output_tokens=32768,
        input=[dict(role='system', content='''Allocate the fixed actual audio exposure budget for a Cantonese patient-ASR DPO probe.
Use the current real lexical-feedback hypothesis and actual synthetic ASR mistakes.
Choose supplied audio keys and positive integer counts that sum to the total budget.
Unlisted keys receive zero exposure; all original audio, coarse reviews and pairs remain intact.
You may redistribute exposure ACROSS roles as well as among audio within a role.
The prior role proportions are a control condition, not mandatory quotas here.
Explain how selected actual mistakes support the real correction goal, reverse contrasts,
protection and retention. Do not force large maintenance exposure merely because the text
was named maintenance; assess whether it supplies useful preference information.
More errors, harder speech, and more distinct negatives do not imply greater value.
Use diverse contexts and avoid concentrating the budget on one uncertain realization.
You have text evidence only. Do not reopen coarse review, assert word fidelity, introduce
CER/phoneme admission thresholds, or classify unclear patient speech as failed generation.
All chosen/rejected texts and per-audio pair weights stay unchanged. Never invent negatives.
Parent, frozen initial R-only DPO reference, loss, learning rate and total steps stay fixed.
The experiment tests whether allocating by realized lexical signal improves on a fixed role
budget. State a concrete expected real-development response; do not claim a benefit yet.'''),
            dict(role='user', content=json.dumps(dict(proposal=read(d / 'proposal.json'),
                real_feedback_and_completed_pilot=context, baseline_role_exposures=baseline['role_exposures'],
                total_exposures=baseline['total_audio_exposures'], audio_cards=cards), ensure_ascii=False))],
        text={'format': {'type': 'json_schema', 'name': 'lexical_audio_allocation',
                       'strict': True, 'schema': to_strict_json_schema(Allocation)}})
    save(root / 'request.json', request)
    save(root / 'audio_keys.json', keys)
    response = serialize_response(execute_therapist_request(p, request))
    save(root / 'response.json', response)
    if response['status'] != 'completed':
        raise RuntimeError('allocation response did not complete')
    messages = [(i, x) for i, x in enumerate(response['output'])
                if x.get('type') == 'message' and x.get('phase') == 'final_answer']
    if len(messages) != 1:
        raise ValueError('expected one explicit final-answer message')
    index, message = messages[0]
    decision = Allocation.model_validate_json(''.join(x['text'] for x in message['content'] if x['type'] == 'output_text'))
    save(root / 'response_extraction.json', dict(output_index=index, message_id=message['id'],
         response_sha256=sha256_file(root / 'response.json')))
    selected = {x.audio_key: x for x in decision.allocations}
    assert len(selected) == len(decision.allocations) and selected.keys() <= keys.keys()
    sequence = [keys[key] for key, item in selected.items() for _ in range(item.exposures)]
    assert len(sequence) == baseline['total_audio_exposures']
    random.Random(p.lora.seed + 3017).shuffle(sequence)
    per_step = p.preference_loop.synthetic_per_step
    allocation = dict(baseline, steps=[sequence[i:i + per_step] for i in range(0, len(sequence), per_step)],
        role_exposures=dict(Counter(roles[x]['role'] for x in sequence)),
        rule='Actual MAS allocation across roles and audio; same total budget, unchanged complete preference manifest.')
    save(root / 'decision.json', decision.model_dump(mode='json'))
    save(root / 'allocation.json', allocation)
    files = [Path(__file__), REPO / 'examples/mas_audio_allocation_probe.py',
        main / 'protocol.json', Path(p.preference_loop.generation_design_path),
        d / 'proposal.json', d / 'mas_dpo/train.jsonl', d / 'sampling/sampling_receipt.json',
        d / 'sampling/preference_audit.json', d / 'sampling/greedy.jsonl',
        d / 'generation/generation_summary.json', d / 'materialization_sidecar.json',
        d / 'training_allocation.json', root / 'allocation.json', root / 'request.json', root / 'response.json',
        REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py', REPO / 'src/therapist_harness/dpo_training.py']
    save(root / 'protocol.json', dict(source_protocol=p.model_dump(mode='json'), source_root=str(main),
        source_preferences=str(d / 'mas_dpo/train.jsonl'), source_sha256={str(f): sha256_file(f) for f in files},
        selected_audio=len(selected), actual_audio_exposures=len(sequence),
        actual_new_sampling=False, actual_new_generation=False,
        scope='Independent across-role allocation probe. Same complete pairs, parent/reference, DPO and total exposure; role composition and order intentionally differ. No new baseline evaluation is passed to MAS.'))
    print(dict(selected_audio=len(selected), role_exposures=allocation['role_exposures']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--main-root', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.main_root.resolve(), args.root.resolve())
    run(args.root.resolve())
