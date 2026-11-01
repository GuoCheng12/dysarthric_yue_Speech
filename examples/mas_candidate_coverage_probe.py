#!/usr/bin/env python3
"""Compare MAS negative selection from four versus sixteen actual ASR draws."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_residual_dpo_probe import evaluation_command
from mas_preference_target_feedback import actual_events
from mas_evaluation_report import compare
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.preference_data import build_preference_pairs
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file


class Choice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    audio_key: str
    negative_key: str
    reason: str = Field(min_length=1, max_length=240)


class Decision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    choices: list[Choice] = Field(min_length=15, max_length=15)
    hypothesis: str = Field(min_length=1)
    limitation: str = Field(min_length=1)


def prepare(base, root):
    root.mkdir(exist_ok=False)
    source = base / 'lexical96_mas_v1/round_001/sampling'
    old = base / 'actual_pair_allocation_probe_v1'
    p = read(base / 'lexical96_mas_v1/protocol.json')
    selections = read(old / 'selection_receipt.json')['selections']
    ids = {x['audio_id'] for x in selections}
    inputs = [x for x in rows(source / 'validated_inputs.jsonl') if x['audio_id'] in ids]
    assert len(inputs) == len(ids) == 15
    assert all(x['review']['verdict'] == 'pass' for x in inputs)
    for x in inputs:
        assert sha256_file(Path(x['audio'])) == x['audio_sha256']
    jsonl(root / 'inputs.jsonl', inputs)
    save(root / 'allocation.json', read(old / 'allocation.json'))
    save(root / 'purposes.json', selections)
    files = [Path(__file__), REPO / 'examples/sample_asr_preferences.py',
        REPO / 'examples/mas_preference_target_feedback.py', REPO / 'examples/evaluate_residual_asr.py',
        REPO / 'examples/mas_residual_dpo_probe.py', REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py',
        REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py', REPO / 'src/therapist_harness/preference_data.py',
        REPO / 'src/therapist_harness/dpo_training.py',
        source / 'validated_inputs.jsonl', source / 'candidates.jsonl', source / 'sampling_receipt.json',
        old / 'selection_receipt.json', old / 'allocation.json', base / 'lexical96_mas_v1/protocol.json',
        base / 'actual_pair_allocation_recovery_v1/results.json',
        base / 'actual_pair_allocation_recovery_v1/selected_pure_target_response.json',
        base / 'residual_dpo_probe_v1/initial/identity_verified.json',
        root / 'inputs.jsonl', root / 'allocation.json', root / 'purposes.json']
    save(root / 'protocol.json', dict(source_protocol=p,
        source_sha256={str(f): sha256_file(f) for f in files},
        original_sampling=str(source), samples_per_audio=16, audio_count=15,
        arms=['first4', 'all16'], steps=12, audio_exposures=96, seed=42,
        actual_new_generation=False, actual_new_sampling=True, actual_new_dpo_training=True,
        hypothesis='Four stochastic draws may miss useful localized negatives. Hold the selected audio, target/reference, review and actual exposure order fixed; let the same MAS selector choose from either four or sixteen freshly observed draws.',
        control='Two actual MAS calls with identical instructions and exposure purposes; each sees only its available candidate view. Selection stochasticity is not eliminated. Both train one real negative per same audio.',
        sampling='One honest independent sixteen-draw receipt; first4 is an explicitly identified subset, not a fabricated four-draw execution or silent merge with old draws.',
        objective='Standard same-audio synthetic-only residual DPO, lr1e-5 beta0.1, merged frozen initial R-only base and fixed initial R-only reference.',
        acceptance='Existing raw ASR gate: pooled improves, macro regression<=0.1pp, hard regression<=0.5pp. No inference postprocessing.'))


def verify(root):
    c = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == h for f, h in c['source_sha256'].items())
    return c


def sample(root):
    cfg = verify(root)
    p = cfg['source_protocol']
    execute(root, [sys.executable, str(REPO / 'examples/sample_asr_preferences.py'), 'sample',
        '--inputs', str(root / 'inputs.jsonl'), '--real-train', p['real_train']['path'],
        '--feedback', p['feedback_split']['path'], '--out-dir', str(root / 'sampling'),
        '--base-model', p['base_model_dir'], '--current-adapter', p['initial_adapter']['path'],
        '--expected-parent-sha256', p['initial_adapter']['sha256'], '--samples', '16', '--seed', '42',
        '--temperature', '.8', '--top-p', '.95', '--max-new-tokens', '128', '--include-greedy'], 'sampling')
    d = root / 'sampling'
    receipt = read(d / 'sampling_receipt.json')
    candidates = rows(d / 'candidates.jsonl')
    inputs = rows(d / 'validated_inputs.jsonl')
    assert receipt['candidate_count'] == len(candidates) == 240
    assert receipt['candidates_sha256'] == sha256_file(d / 'candidates.jsonl')
    previous = read(Path(cfg['original_sampling']) / 'sampling_receipt.json')
    for field in ['current_asr_sha256', 'base_model_sha256', 'seed', 'temperature', 'top_p',
                  'top_k', 'max_new_tokens', 'language', 'prompt', 'real_train_sha256', 'feedback_sha256']:
        assert receipt[field] == previous[field], field
    original = {(x['audio_id'], x['sample_index']): x for x in rows(Path(cfg['original_sampling']) / 'candidates.jsonl')}
    replay = [dict(audio_id=x['audio_id'], sample_index=x['sample_index'],
        exact_tokens=x['generated_token_ids'] == original[(x['audio_id'], x['sample_index'])]['generated_token_ids'],
        same_seed=x['seed'] == original[(x['audio_id'], x['sample_index'])]['seed'])
        for x in candidates if x['sample_index'] < 4]
    save(root / 'original_draw_comparison.json', dict(comparisons=replay,
        exact_token_replays=sum(x['exact_tokens'] for x in replay), total=60,
        reuse_claim='All 240 draws were actually decoded again; comparison does not import old predictions.'))
    for arm, n in [('first4', 4), ('all16', 16)]:
        out = root / arm
        out.mkdir(exist_ok=False)
        view = [x for x in candidates if x['sample_index'] < n]
        pairs, audit = build_preference_pairs(inputs, view, p['initial_adapter']['sha256'],
            expected_samples=n, greedy_candidates=rows(d / 'greedy.jsonl'))
        jsonl(out / 'candidate_view.jsonl', view)
        jsonl(out / 'available_preferences.jsonl', pairs)
        save(out / 'preference_audit.json', audit)
        save(out / 'view_receipt.json', dict(actual_sampling_receipt=str(d / 'sampling_receipt.json'),
            actual_sampling_receipt_sha256=sha256_file(d / 'sampling_receipt.json'),
            actual_candidates_sha256=sha256_file(d / 'candidates.jsonl'),
            view_sha256=sha256_file(out / 'candidate_view.jsonl'),
            available_preferences_sha256=sha256_file(out / 'available_preferences.jsonl'),
            selection=f'sample_index < {n}', samples_per_audio_in_view=n,
            actual_execution_samples_per_audio=16, independent_execution_for_this_view=False))
        assert len({x['audio_id'] for x in pairs}) == 15, 'All-correct audio cannot be forced into matched training'
        print(dict(arm=arm, actual_pairs=len(pairs)), flush=True)
    verify(root)


def choose(root, arm):
    from openai.lib._pydantic import to_strict_json_schema

    cfg = verify(root)
    p = RunProtocol.model_validate(cfg['source_protocol'])
    d = root / arm
    pairs = rows(d / 'available_preferences.jsonl')
    assert sha256_file(d / 'available_preferences.jsonl') == read(d / 'view_receipt.json')['available_preferences_sha256']
    by_audio = defaultdict(list)
    for row in pairs:
        by_audio[row['audio_id']].append(row)
    purposes = {x['audio_id']: x for x in read(root / 'purposes.json')}
    greedy = {x['audio_id']: x['prediction'] for x in rows(root / 'sampling/greedy.jsonl')}
    keys, audio_keys, cards = {}, {}, []
    for i, (audio_id, group) in enumerate(by_audio.items()):
        audio_key = f'a{i:02d}'
        audio_keys[audio_key] = audio_id
        options = []
        for j, row in enumerate(group):
            key = f'{audio_key}.n{j:02d}'
            keys[key] = row
            events = [dict(level=e.level, intended_unit=e.reference_unit, observed_unit=e.hypothesis_unit,
                intended_character=e.reference_character, observed_character=e.hypothesis_character)
                for e in actual_events(row['chosen'], row['rejected']) if e.context_sensitive is False]
            options.append(dict(negative_key=key, rejected=row['rejected'],
                sample_indices=row['sample_indices'], observed_stable_phone_events=events))
        cards.append(dict(audio_key=audio_key, target=group[0]['chosen'],
            reference_text=group[0]['metadata']['review']['reference_text'], greedy=greedy[audio_id],
            purpose=purposes[audio_id]['purpose'], exposures=purposes[audio_id]['exposures'], negatives=options))
    request = dict(model=p.agent.model, reasoning={'effort': p.agent.reasoning_effort}, store=False, stream=True,
        max_output_tokens=12000, input=[dict(role='system', content='''Select one actual negative per supplied audio for a MAS Cantonese patient-ASR DPO probe.
All 15 audio have already passed the unchanged simple three-text coarse review.
The target remains the intended chosen text; word fidelity has not been proved.
Do not change targets, invent errors, rerate admission, infer listening or add a
CER threshold. The observed family events describe textual substitutions only.
Use actual negative differences and supplied normal decoding to identify useful
correction signals. Prefer a targeted correction over a large bundle of unrelated
changes when such an actual candidate exists, while considering the fixed role.
Do not assume that more errors or a more fluent rejected sentence is better.
The primary real feedback is i→yu, protected m→n, counterbalance i→e. Real errors
include 鉗→鉛 (both i→yu and m→n, not independent mechanisms), 蠍→橛/決, 件→券,
繳→竄, 尖→專, 展→轉. Maintenance concerns existing z/easy and j/medium capability.
Previous use of these same 15 audio and fixed exposures did not beat R-only:
20.868389% versus20.793269%. Increasing primary-family exposure to40 recovered
only3 original real target positions and increased insertions in their sentences.
Now select among the actual candidates provided. The audio and96 exposures in
12steps stay fixed. Each chosen negative receives that audio's full weight; no
real KL/SFT term is added. If no candidate addresses a role, say so briefly and
choose the best available actual one without claiming the role is fulfilled.
Return exactly one supplied negative_key for each audio_key, a short reason and
a prospective hypothesis plus limitation. Benefit will be judged on real audio.'''),
            dict(role='user', content=json.dumps(dict(audio_cards=cards), ensure_ascii=False))],
        text={'format': {'type': 'json_schema', 'name': 'actual_negative_choice', 'strict': True,
                         'schema': to_strict_json_schema(Decision)}})
    save(d / 'request.json', request)
    response = serialize_response(execute_therapist_request(p, request))
    save(d / 'response.json', response)
    assert response['status'] == 'completed'
    final = [x for x in response['output'] if x.get('type') == 'message' and x.get('phase') == 'final_answer']
    assert len(final) == 1
    decision = Decision.model_validate_json(''.join(x['text'] for x in final[0]['content'] if x['type'] == 'output_text'))
    assert len({x.audio_key for x in decision.choices}) == 15
    assert {x.audio_key for x in decision.choices} == set(audio_keys)
    selected, mapping = [], []
    for x in decision.choices:
        row = keys[x.negative_key]
        assert row['audio_id'] == audio_keys[x.audio_key]
        selected.append(dict(row, pair_weight=1.0))
        mapping.append(dict(audio_id=row['audio_id'], source_pair_id=row['pair_id'],
            original_pair_weight=row['pair_weight'], source_sample_indices=row['sample_indices'], reason=x.reason))
    jsonl(d / 'preferences.jsonl', selected)
    save(d / 'mas_decision.json', decision.model_dump(mode='json'))
    save(d / 'selection_receipt.json', dict(selections=mapping, changed_pair_fields=['pair_weight'],
        available_preferences_sha256=sha256_file(d / 'available_preferences.jsonl'),
        preferences_sha256=sha256_file(d / 'preferences.jsonl'),
        actual_mas_response_sha256=sha256_file(d / 'response.json'), view_receipt_sha256=sha256_file(d / 'view_receipt.json')))
    verify(root)


def train(base, root, arm, smoke=False):
    cfg = verify(root)
    p = cfg['source_protocol']
    source = root / arm
    assert sha256_file(source / 'preferences.jsonl') == read(source / 'selection_receipt.json')['preferences_sha256']
    assert read(base / 'residual_dpo_probe_v1/initial/identity_verified.json')['status'] == 'passed'
    d = root / 'smoke' if smoke else source
    if smoke:
        d.mkdir(exist_ok=False)
        keys = list(dict.fromkeys(x for step in read(root / 'allocation.json')['steps'] for x in step))[:2]
        jsonl(d / 'preferences.jsonl', [x for x in rows(source / 'preferences.jsonl') if x['audio_id'] in keys])
        save(d / 'allocation.json', dict(steps=[keys, list(reversed(keys))]))
    else:
        assert read(root / 'smoke/verified.json')['status'] == 'passed'
    steps = 2 if smoke else 12
    execute(d, [sys.executable, str(REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py'),
        '--model_path', p['base_model_dir'], '--policy_adapter_path', p['initial_adapter']['path'],
        '--reference_adapter_path', p['initial_adapter']['path'], '--train_file', str(d / 'preferences.jsonl'),
        '--synthetic_allocation_file', str(d / 'allocation.json' if smoke else root / 'allocation.json'),
        '--output_dir', str(d / 'training'), '--max_steps', str(steps), '--real_per_step', '0',
        '--synthetic_per_step', '2' if smoke else '8', '--synthetic_weight', '1',
        '--lr', '1e-5', '--beta', '.1', '--seed', '42'], 'training')
    r = read(d / 'training/training_result.json')
    assert r['policy_changed'] and r['reference_files_unchanged']
    assert math.isclose(r['history'][0]['loss'], math.log(2), abs_tol=1e-4)
    assert read(d / 'training/residual_verified.json')['status'] == 'passed'
    save(d / 'verified.json', dict(status='passed', initial_log2=True, policy_changed=True, frozen_base_reference_unchanged=True))
    if smoke:
        return
    execute(d, evaluation_command(p, d / 'evaluation_run', d / 'training/checkpoint-12'), 'evaluation')
    metrics = read(d / 'evaluation_run/evaluation/summary.json')['metrics']
    parent = read(base / 'parent_eval_check/summary.json')['metrics']
    gate = dict(pooled_improves=metrics['overall']['pooled_cer'] < parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer']-parent['overall']['patient_macro_cer'] <= .001,
        hard_regression_within_limit=metrics['hard']['pooled_cer']-parent['hard']['pooled_cer'] <= .005)
    save(d / 'paired_parent.json', compare(Path(p['initial_adapter_dev_predictions']['path']), d / 'evaluation_run/evaluation/predictions.jsonl'))
    save(d / 'results.json', dict(status='complete', metrics=metrics, gate_checks=gate,
        eligible_for_next_round_parent=all(gate.values()), actual_new_dpo_training=True, postprocessing_used=False))
    verify(root)


def supervise(base, root):
    sample(root)
    def parallel(stage):
        jobs = []
        for gpu, arm in enumerate(['first4', 'all16']):
            command = [sys.executable, str(Path(__file__).resolve()), stage, str(base), '--arm', arm]
            with (root / arm / (stage + '.supervisor.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
            save(root / arm / (stage + '.launch.json'), dict(pid=child.pid, gpu=gpu, command=command))
            jobs.append(child)
        codes = [x.wait() for x in jobs]
        assert codes == [0, 0], codes
    parallel('choose')
    train(base, root, 'first4', smoke=True)
    parallel('train')
    actual = [rows(root / a / 'training/training_exposures.jsonl') for a in ['first4', 'all16']]
    identity = lambda data: [(x['step'], x['source'], x['audio_id'], x['occurrence'], x['loss_weight']) for x in data]
    assert identity(actual[0]) == identity(actual[1])
    assert Counter(x['audio_id'] for x in actual[0]) == Counter(x for step in read(root / 'allocation.json')['steps'] for x in step)
    save(root / 'cross_arm_verification.json', dict(status='passed', same_actual_audio_exposure_order_weights=True,
        same_initial_parent=True, same_frozen_reference=True, original_96_exposures_retained=True,
        candidate_views='Explicit subsets of independently completed sixteen-draw execution; no forged sampling receipt'))
    save(root / 'results.json', dict(status='complete', arms={a: read(root / a / 'results.json') for a in ['first4', 'all16']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'supervise', 'choose', 'train'])
    parser.add_argument('base', type=Path)
    parser.add_argument('--arm', choices=['first4', 'all16'], default='first4')
    args = parser.parse_args()
    base = args.base.resolve()
    root = base / 'candidate_coverage_probe_v1'
    if args.stage == 'prepare':
        prepare(base, root)
    elif args.stage == 'supervise':
        supervise(base, root)
    elif args.stage == 'choose':
        choose(root, args.arm)
    else:
        train(base, root, args.arm)
