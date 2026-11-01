#!/usr/bin/env python3
"""Select one actual rejected transcript per audio, with a matched rule control."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_explicit_feedback_correction import explicit_feedback
from mas_evaluation_report import compare
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file
from therapist_harness.text import edit_distance, text_normalize
from therapist_harness.dpo_training import group_training_rows


class Choice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    audio_key: str
    negative_key: str
    reason: str = Field(min_length=1, max_length=180)


class Decision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    hypothesis: str
    selections: list[Choice]
    expected_real_response: str


def prepare(base, root):
    from openai.lib._pydantic import to_strict_json_schema

    root.mkdir(exist_ok=False)
    source = base / 'lexical96_mas_v1/round_001'
    protocol_path = base / 'lexical96_mas_v1/protocol.json'
    p = RunProtocol.model_validate(read(protocol_path))
    original_path = source / 'mas_dpo/train.jsonl'
    arguments = read(source / 'mas_dpo/training/run_protocol.json')['arguments']
    by_audio = defaultdict(list)
    for row in rows(original_path):
        assert row['current_asr_sha256'] == p.initial_adapter.sha256 and row['source'] == 'synthetic'
        by_audio[row['audio_id']].append(row)
    sidecar = {x['sample_id']: x for x in read(source / 'materialization_sidecar.json')['samples']}
    audit = {x['audio_id']: x for x in read(source / 'sampling/preference_audit.json')['audio_results']}
    greedy = {x['audio_id']: x['prediction'] for x in rows(source / 'sampling/greedy.jsonl')}
    cards, candidates, audio_keys = [], {}, {}
    for i, (audio_id, group) in enumerate(by_audio.items()):
        key = f'a{i:03d}'
        audio_keys[key] = audio_id
        options = []
        for j, row in enumerate(group):
            negative_key = f'{key}.n{j:02d}'
            candidates[negative_key] = row
            options.append(dict(negative_key=negative_key, rejected=row['rejected'], actual_sample_indices=row['sample_indices']))
        cards.append(dict(audio_key=key, target=group[0]['chosen'], role=sidecar[audio_id]['role'],
            greedy=greedy[audio_id], correct_random_draws=audit[audio_id]['correct_samples'],
            error_origin=audit[audio_id]['error_origin'], actual_rejected_options=options))
    save(root / 'design.json', dict(hypothesis='Average gradients over several unrelated sampled errors may dilute a localized correction. Test one selected actual negative per audio.',
        paired_audio=len(by_audio), original_pairs=sum(map(len, by_audio.values())),
        preserved='Audio, target, coarse review, original samples, R-only parent/reference, full audio exposure order, standard DPO loss and optimizer.',
        changed='Select one existing negative and renormalize that audio pair weight to 1. Compare MAS selection with minimum normalized character edit distance, ties by original pair order.',
        review_or_admission_changed=False, actual_new_generation=False, actual_new_sampling=False,
        loss='Unchanged standard DPO on the selected full chosen/rejected strings.',
        raw_asr_acceptance_only=True))
    request = dict(model=p.agent.model, reasoning={'effort': p.agent.reasoning_effort}, store=False,
        stream=True, max_output_tokens=32768,
        input=[dict(role='system', content='''Select exactly ONE supplied actual ASR error transcript for EACH audio card.
This is a training-negative selection experiment in the Cantonese patient-ASR MAS.
Keep the complete original intended target and selected rejected text unchanged.
Prefer a useful localized correction connected to the observed real errors, with
few unrelated differences; a tiny punctuation/particle difference is not automatically
more useful than a relevant lexical mistake. Use the actual greedy and random
outputs to reason about the information in a pair. Do not select solely because
an error is longer, more numerous, or the audio seems harder. Do not invent a
negative, skip an audio, change its exposure count, or replace its target.
You see text evidence only. Coarse review remains unchanged and does not prove
word-level audio fidelity. This is not a new quality filter or a CER admission
threshold; unclear patient speech is not a failure. All choices must use the
provided negative keys for that audio. State a brief learning reason per choice
and a concrete overall expected real-development response. The objective remains
standard full-transcript audio DPO from R-only with its frozen R-only reference.'''),
            dict(role='user', content=json.dumps(dict(real_feedback=explicit_feedback(read(base / 'lexical_feedback_v1.json')),
                previous_result='The complete-pair MAS DPO and the new decoder-only DPO both failed to improve raw real-development CER. Training loss decreased, but real errors increased. No inference postprocessing benefit is counted.',
                audio_cards=cards), ensure_ascii=False))],
        text={'format': {'type': 'json_schema', 'name': 'negative_selection', 'strict': True,
                         'schema': to_strict_json_schema(Decision)}})
    save(root / 'request.json', request)
    response = serialize_response(execute_therapist_request(p, request))
    save(root / 'response.json', response)
    assert response['status'] == 'completed'
    messages = [x for x in response['output'] if x.get('type') == 'message' and x.get('phase') == 'final_answer']
    assert len(messages) == 1
    decision = Decision.model_validate_json(''.join(x['text'] for x in messages[0]['content'] if x['type'] == 'output_text'))
    choices = {x.audio_key: x for x in decision.selections}
    assert len(choices) == len(decision.selections) and choices.keys() == audio_keys.keys()
    save(root / 'mas_decision.json', decision.model_dump(mode='json'))
    selection_ids = {}
    for arm in ['nearest', 'mas']:
        selected, receipt = [], []
        for card in cards:
            key = card['audio_key']
            allowed = [x['negative_key'] for x in card['actual_rejected_options']]
            if arm == 'nearest':
                selected_key = min(allowed, key=lambda k: edit_distance(text_normalize(candidates[k]['chosen']), text_normalize(candidates[k]['rejected'])))
                reason = 'Minimum normalized character edit distance among actual errors; first original pair breaks ties.'
            else:
                selected_key, reason = choices[key].negative_key, choices[key].reason
            assert selected_key in allowed
            original = candidates[selected_key]
            updated = dict(original, pair_weight=1.)
            assert {k: v for k, v in updated.items() if k != 'pair_weight'} == {k: v for k, v in original.items() if k != 'pair_weight'}
            selected.append(updated)
            receipt.append(dict(audio_id=original['audio_id'], original_pair_id=original['pair_id'],
                negative_key=selected_key, original_pair_weight=original['pair_weight'], training_pair_weight=1., reason=reason))
        assert len(group_training_rows(selected, 'dpo')['synthetic']) == len(by_audio)
        jsonl(root / arm / 'train.jsonl', selected)
        save(root / arm / 'selection_receipt.json', dict(original_preferences=str(original_path),
            original_preferences_sha256=sha256_file(original_path), original_sampling_receipt=str(source / 'sampling/sampling_receipt.json'),
            original_sampling_receipt_sha256=sha256_file(source / 'sampling/sampling_receipt.json'),
            actual_new_sampling=False, changed_fields=['pair_weight'], selections=receipt))
        selection_ids[arm] = [x['pair_id'] for x in selected]
    files = [Path(__file__), Path(__file__).with_name('mas_explicit_feedback_correction.py'),
        REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py', REPO / 'src/therapist_harness/dpo_training.py',
        REPO / 'finetune/scripts/evaluate_asr_checkpoint.py', protocol_path, original_path,
        source / 'sampling/sampling_receipt.json', source / 'sampling/preference_audit.json',
        source / 'sampling/greedy.jsonl', source / 'materialization_sidecar.json',
        source / 'mas_dpo/training/run_protocol.json', source / 'mas_dpo/training/training_exposures.jsonl',
        Path(arguments['synthetic_allocation_file']), base / 'lexical_feedback_v1.json', root / 'request.json', root / 'response.json']
    files += [root / arm / name for arm in ['nearest', 'mas'] for name in ['train.jsonl', 'selection_receipt.json']]
    save(root / 'protocol.json', dict(source_sha256={str(f): sha256_file(f) for f in files}, source_protocol=p.model_dump(mode='json'),
        source_arguments=arguments, original_exposures=str(source / 'mas_dpo/training/training_exposures.jsonl'),
        paired_audio=len(by_audio), selected_pairs_per_arm=len(by_audio),
        different_negative_choices=sum(a != z for a, z in zip(selection_ids['nearest'], selection_ids['mas'], strict=True))))
    first = read(arguments['synthetic_allocation_file'])['steps'][0][:2]
    jsonl(root / 'smoke_preferences.jsonl', [x for x in rows(root / 'nearest/train.jsonl') if x['audio_id'] in first])
    save(root / 'smoke_allocation.json', dict(steps=[first, list(reversed(first))]))
    print(dict(prepared=True, selected_pairs_per_arm=len(by_audio), different_choices=read(root / 'protocol.json')['different_negative_choices']), flush=True)


def exposure_order(records):
    seen, sequence = set(), []
    for row in records:
        key = (row['step'], row['source'], row['audio_id'], row['occurrence'])
        if key not in seen:
            seen.add(key)
            sequence.append(key)
    return sequence


def run(base, root, arm, smoke):
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == digest for f, digest in cfg['source_sha256'].items())
    if not smoke:
        assert read(root / 'smoke/verified.json')['status'] == 'passed'
    directory = root / ('smoke' if smoke else arm)
    if smoke:
        directory.mkdir(exist_ok=False)
    arguments = dict(cfg['source_arguments'], output_dir=str(directory / 'training'), train_file=str(root / arm / 'train.jsonl'))
    if smoke:
        arguments.update(train_file=str(root / 'smoke_preferences.jsonl'), synthetic_allocation_file=str(root / 'smoke_allocation.json'), max_steps=2, synthetic_per_step=2)
    command = [sys.executable, str(REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py')]
    for key, value in arguments.items():
        if value is not None:
            command.extend(['--' + key, str(value)])
    execute(directory, command, 'training')
    result = read(directory / 'training/training_result.json')
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert math.isclose(result['history'][0]['loss'], math.log(2), abs_tol=1e-4)
    actual = rows(directory / 'training/training_exposures.jsonl')
    if not smoke:
        assert exposure_order(actual) == exposure_order(rows(cfg['original_exposures']))
        assert len(actual) == 96
    save(directory / 'verified.json', dict(status='passed', reference_unchanged=True,
        initial_dpo_log2=True, exact_original_audio_exposure_order=None if smoke else True,
        selected_pairs_are_original_actual_samples=True))
    if smoke:
        print('Two-step selected-negative DPO smoke passed', flush=True)
        return
    p = cfg['source_protocol']
    execute(directory, [sys.executable, str(REPO / 'finetune/scripts/evaluate_asr_checkpoint.py'),
        '--base-model-path', p['base_model_dir'], '--adapter-path', str(directory / 'training/checkpoint-12'),
        '--split-jsonl', p['feedback_split']['path'], '--out-dir', str(directory / 'evaluation'),
        '--batch-size', '8', '--critical-cer-threshold', '.5', '--max-new-tokens', '256'], 'evaluation')
    m = read(directory / 'evaluation/summary.json')['metrics']
    parent = read(base / 'parent_eval_check/summary.json')['metrics']
    gate = dict(pooled_improves=m['overall']['pooled_cer'] < parent['overall']['pooled_cer'],
        macro_regression_within_limit=m['overall']['patient_macro_cer'] - parent['overall']['patient_macro_cer'] <= .001,
        hard_regression_within_limit=m['hard']['pooled_cer'] - parent['hard']['pooled_cer'] <= .005)
    save(directory / 'paired_parent.json', compare(Path(p['initial_adapter_dev_predictions']['path']), directory / 'evaluation/predictions.jsonl'))
    assert all(sha256_file(Path(f)) == digest for f, digest in cfg['source_sha256'].items())
    save(directory / 'results.json', dict(status='complete', metrics=m, gate_checks=gate,
        eligible_for_next_round_parent=all(gate.values()), postprocessing_used=False))
    print(dict(arm=arm, metrics=m['overall'], gate=gate), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'run'])
    parser.add_argument('base', type=Path)
    parser.add_argument('--arm', choices=['nearest', 'mas'], default='nearest')
    args = parser.parse_args()
    base = args.base.resolve()
    root = base / 'negative_selection_probe_v1'
    if args.stage == 'prepare':
        prepare(base, root)
    else:
        run(base, root, args.arm, args.stage == 'smoke')
