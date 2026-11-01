#!/usr/bin/env python3
"""Within-patient alternate references for the six actual round2 target texts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, read, rows, save, jsonl, execute
from mas_patient_retention_probe import request_mas
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file, sha256_directory


class Choice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    audio_id: str
    option: Literal['original', 'alternate']
    reason: str = Field(min_length=1, max_length=500)


class Selection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reflection: str = Field(min_length=1, max_length=2000)
    choices: list[Choice] = Field(min_length=6, max_length=6)
    hypothesis: str = Field(min_length=1, max_length=1000)
    expected_observation: str = Field(min_length=1, max_length=1000)
    observation_against_hypothesis: str = Field(min_length=1, max_length=1000)


def verify(root):
    plan = read(root / 'plan.json')
    assert all(sha256_file(Path(f)) == h for f, h in plan['source_sha256'].items())
    assert sha256_directory(Path(plan['current_parent'])) == plan['current_parent_sha256']
    return plan


def prepare(base, root):
    source = base / 'exploratory_round2_v1/adaptive_data'
    previous = base / 'exploratory_round2_v1/DPO'; cfg = read(previous / 'protocol.json')
    assert all(sha256_file(Path(f)) == h for f, h in cfg['source_sha256'].items())
    result = read(previous / 'results.json'); assert result['status'] == 'complete'
    plan = cfg['plan']; cards = read(base / 'reference_catalog.json')['cards']
    by_hash = {x['reference_wav_sha256']: x for x in cards}
    roles = {x['sample_id']: x['role'] for x in read(source / 'slots.json')}
    originals = {x['sample_id']: x for x in read(source / 'generation/generation_summary.json')['outputs']}
    targets = [x for x in rows(source / 'manifest.jsonl') if roles[x['sample_id']] == 'target']; assert len(targets) == 6
    manifest, choices, cases = [], [], []
    root.mkdir(exist_ok=False); (root / 'audio').mkdir()
    for x in targets:
        old = originals[x['sample_id']]; ref = by_hash[x['reference_wav_sha256']]
        alternatives = [c for c in cards if c['voice_group'] == ref['voice_group'] and c['reference_id'] != ref['reference_id']]
        assert len(alternatives) == 1 and old['admitted'] and old['attempt'] == 0
        alt = alternatives[0]
        manifest.append(dict(x, reference_text=alt['reference_text'], reference_wav=alt['reference_wav'],
            reference_wav_sha256=alt['reference_wav_sha256'], output_wav=str(root / 'audio' / (x['sample_id'] + '.wav'))))
        choices.append(dict(sample_id=x['sample_id'], reference_id=alt['reference_id'], mode=old['generation_mode'],
            instruction_id=old['effective_instruction_id'], reference_wav_sha256=alt['reference_wav_sha256'],
            target_text_sha256=hashlib.sha256(x['target_text'].encode()).hexdigest(),
            reason='Within-patient alternate reference pilot. Same target, initial mode/instruction and TTS seed; unchanged coarse review/rulebook.'))
        cases.append(dict(audio_id=x['sample_id'], voice_group=ref['voice_group'], target_text=x['target_text'],
            original_reference=ref['reference_id'], alternate_reference=alt['reference_id']))
    jsonl(root / 'manifest.jsonl', manifest); save(root / 'cases.json', cases)
    save(root / 'generation_plan.json', dict(schema_version='generation_plan_v1', choices=choices,
        hypothesis='Actual within-patient trial output and reference experience may improve reference selection for the already useful target curriculum.',
        expected_observation='Evaluate selected audio DPO against original references from the same current1373parent, preserving initialR-only teacher and matched exposures.'))
    files = [Path(__file__), previous / 'protocol.json', previous / 'results.json', previous / 'declared_target_observations.json',
        source / 'generation/generation_summary.json', source / 'manifest.jsonl', source / 'sampling/sampling_receipt.json',
        base / 'reference_experience_v2/MAS_context.json', base / 'reference_experience_v2/receipt.json',
        root / 'manifest.jsonl', root / 'cases.json', root / 'generation_plan.json']
    save(root / 'plan.json', dict(current_parent=plan['current_parent'], current_parent_sha256=plan['current_parent_sha256'],
        source_protocol=plan['source_protocol'], original_data_root=str(source), previous_DPO_root=str(previous),
        source_sha256={**cfg['source_sha256'], **{str(f): sha256_file(f) for f in files}},
        parent_choice='Retain current1373 checkpoint; the round2 MAS1375 endpoint failed pooled continuation and original hard gate. Inherit its observations, not its parameters.',
        intervention='Generate only six alternate-reference target audio. Same patient/text/mode/instruction/seed; nine non-target audio remain original. Actual MAS selection occurs after both options have real coarse-review and current-ASR sampling observations.',
        prospective_DPO='Original versus MAS-selected references, same currentparent, edit+EOS KL10,12steps,96synthetic/192real exposures, role32/16/16/32 on common actual paired slots. Reuse a completed control only if its actual pair inputs and exposure plan are identical; otherwise retrain the matched subset.',
        all_correct='Skip actual all-correct audio; no synthetic negative invention.', original_ledgers_unchanged=True))
    verify(root); print(dict(status='prepared', alternate_audio=6), flush=True)


def run_data(root):
    plan = verify(root); p = RunProtocol.model_validate(plan['source_protocol']); c = p.preference_loop
    execute(root, [sys.executable, str(REPO / 'examples/generate_reviewed_sg_recovery_v2.py'),
        '--manifest', str(root / 'manifest.jsonl'), '--out-dir', str(root / 'generation'),
        '--generation-plan', str(root / 'generation_plan.json'), '--review-mode', 'asr_transcript', '--review-asr-worker',
        '--audio-review-model', c.audio_review_model, '--audio-review-base-url', c.audio_review_base_url,
        '--review-asr-model', p.base_model_dir, '--review-asr-adapter', c.review_asr_adapter,
        '--review-asr-adapter-sha256', c.review_asr_adapter_sha256, '--review-asr-base-model-sha256', c.review_asr_base_model_sha256,
        '--review-allow-pending'], 'generation')
    execute(root, [sys.executable, str(REPO / 'examples/sample_residual_asr_preferences_v1.py'), 'sample',
        '--inputs', str(root / 'generation/accepted.jsonl'), '--real-train', p.real_train.path, '--feedback', p.feedback_split.path,
        '--out-dir', str(root / 'sampling'), '--base-model', p.base_model_dir, '--current-adapter', plan['current_parent'],
        '--initial-r-only-adapter', p.initial_adapter.path, '--expected-parent-sha256', plan['current_parent_sha256'],
        '--samples', '4', '--include-greedy', '--seed', '42', '--temperature', '.8', '--top-p', '.95', '--max-new-tokens', '128'], 'sampling')
    verify(root); save(root / 'data_complete.json', dict(status='complete', audit=read(root / 'sampling/preference_audit.json')))


def select(base, root):
    from openai.lib._pydantic import to_strict_json_schema

    plan = verify(root); p = RunProtocol.model_validate(plan['source_protocol'])
    source = Path(plan['original_data_root']); previous = Path(plan['previous_DPO_root'])
    assert read(root / 'data_complete.json')['status'] == 'complete'
    options = {}; receipts = {}; files = []
    for name, directory in [('original', source), ('alternate', root)]:
        sample = directory / 'sampling'; receipt = read(sample / 'sampling_receipt.json'); receipts[name] = receipt
        assert receipt['status'] == 'complete' and receipt['current_asr_sha256'] == plan['current_parent_sha256']
        assert receipt['initial_R_only_sha256'] == p.initial_adapter.sha256
        manifest = rows(sample / 'validated_inputs.jsonl'); audit = {x['audio_id']: x for x in read(sample / 'preference_audit.json')['audio_results']}
        draws = rows(sample / 'candidates.jsonl'); greedy = {x['audio_id']: x for x in rows(sample / 'greedy.jsonl')}
        options[name] = {x['audio_id']: dict(reference_text=x['reference_text'], audio_sha256=x['audio_sha256'],
            target=x['clean_gt'], review={k: x['review'][k] for k in ['verdict', 'reason', 'asr_transcription']},
            greedy=greedy[x['audio_id']]['prediction'], actual_samples=[y['prediction'] for y in draws if y['audio_id'] == x['audio_id']],
            pair_audit={k: v for k, v in audit[x['audio_id']].items() if k != 'audio_id'}, generation_mode=x['generation_mode']) for x in manifest}
        files += [sample / n for n in ['sampling_receipt.json', 'validated_inputs.jsonl', 'candidates.jsonl', 'greedy.jsonl', 'preference_audit.json']]
    assert all(receipts['original'][k] == receipts['alternate'][k] for k in ['current_asr_sha256', 'initial_R_only_sha256', 'temperature', 'top_p', 'seed', 'samples_per_audio', 'max_new_tokens', 'greedy_config'])
    cases = [dict(x, options={name: data[x['audio_id']] for name, data in options.items() if x['audio_id'] in data}) for x in read(root / 'cases.json')]
    refs = {x[k] for x in cases for k in ['original_reference', 'alternate_reference']}
    experience = read(base / 'reference_experience_v2/MAS_context.json')
    context = dict(cases=cases, reference_experience=[x for x in experience['cards'] if x['reference_id'] in refs],
        historical_sampling_protocols=experience['sampling_protocols'], historical_card_limits=experience['semantics'],
        previous_results={a: {k: v for k, v in x.items() if k in ['metrics', 'initial_R_only_gate', 'current_parent_gate']} for a, x in read(previous / 'results.json')['arms'].items()},
        previous_target_observations=read(previous / 'declared_target_observations.json'),
        parent_choice=plan['parent_choice'], prospective_DPO=plan['prospective_DPO'])
    body = json.dumps(context, ensure_ascii=False); assert '/data/' not in body and 'vlink' not in body
    out = root / 'MAS_selection'; out.mkdir(); save(out / 'context.json', context)
    request = dict(model=p.agent.model, reasoning={'effort': p.agent.reasoning_effort}, store=False, stream=True,
        max_output_tokens=10000, input=[dict(role='system', content='''You are MAS reflecting on a completed Cantonese patient-ASR DPO round and selecting concrete TTS reference outputs.
The previous adaptive curriculum had1375errors versus fixed1388, but currentparent1373 and initialR-only1384. It did NOT pass the original hard gate (696versus688 errors). It DID show named-target evidence: 織 deletion8->5 and4 resolved target positions recovered versus0 in the fixed control. Do not conflate target transfer with overall success or word-verified generated audio. The previous selected-parent three-seed mean tied R-only.
Retain the current1373parameter endpoint; use the completed experiment as experience. InitialR-only remains the frozen DPO AND real-KL teacher. Six target sentences now each have original and, if admitted, one newly generated alternate SAME-PATIENT reference option. Initial TTS mode/instruction/seed and target were held fixed; the coarse review/rulebook is unchanged. The other nine original curriculum audio stay fixed.
Reflect on actual outcomes, then choose original or alternate for EACH exact supplied audio_id. Only choose an available option. Reference experience cards contain actual past three-text review and sampling observations, separated by sampling parent/config. They are not reference-quality scores or causal downstream effects. More wrong samples/pairs is not automatically more useful. All-correct is allowed and contributes no fabricated negative. Keep original if the alternate lacks a useful basis; do not force changes just to make an experiment.
Use target text, reference text, observed ASR transcriptions and actual candidate errors to reason briefly about plausible preference usefulness and obvious reference-specific output issues. No CER admission threshold, phoneme audit, strict word-verification, patient-severity inference or listening claim. Simple coarse pass is not proof the audio reads every target word. Do not relabel chosen to an ASR prediction.
The next DPO comparison will use the common actual paired role slots, identical96synthetic/192real exposures,12steps edit+EOS,KL10,lr1e-4,beta.1,seed42, samecurrent1373parent. No promotion of the1375endpoint. If selections leave the actual data unchanged, report that honestly; do not demand redundant training. Return a Chinese reflection, exactly6choices and a testable hypothesis with prospective observations for and against it.'''),
            dict(role='user', content=body)], text={'format': dict(type='json_schema', name='round2_reference_selection', strict=True, schema=to_strict_json_schema(Selection))})
    save(out / 'request.json', request); response = request_mas(p, request, out); assert response['status'] == 'completed'
    messages = [x for x in response['output'] if x.get('type') == 'message' and x.get('phase') == 'final_answer']; assert len(messages) == 1
    selection = Selection.model_validate_json(''.join(x['text'] for x in messages[0]['content'] if x['type'] == 'output_text'))
    chosen = {x.audio_id: x for x in selection.choices}; assert len(chosen) == 6 and chosen.keys() == {x['audio_id'] for x in cases}
    assert all(chosen[x['audio_id']].option in x['options'] for x in cases)
    save(out / 'selection.json', selection.model_dump(mode='json'))
    verify(root); save(root / 'selection_receipt.json', dict(status='complete', actual_MAS_call=True,
        changed_reference_audio=sum(x.option == 'alternate' for x in chosen.values()), response_id=response['id'],
        source_sha256={str(f): sha256_file(f) for f in files + [out / n for n in ['context.json', 'request.json', 'response.json', 'selection.json']]},
        source_receipts=dict(original=str(source / 'sampling/sampling_receipt.json'), alternate=str(root / 'sampling/sampling_receipt.json')),
        actual_DPO_training_executed=False, original_ledgers_unchanged=True))
    print(dict(status='selected', changed=sum(x.option == 'alternate' for x in chosen.values())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['prepare', 'data', 'select', 'all']); parser.add_argument('base', type=Path)
    args = parser.parse_args(); base = args.base.resolve(); root = base / 'round2_reference_selection_probe_v1'
    if args.stage in ['prepare', 'all']: prepare(base, root)
    if args.stage in ['data', 'all']: run_data(root)
    if args.stage in ['select', 'all']: select(base, root)
