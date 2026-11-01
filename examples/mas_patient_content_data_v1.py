#!/usr/bin/env python3
"""Hold patient/reference/recipe fixed and ask MAS for error-conditioned text."""

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, read, rows, save, jsonl
from mas_patient_retention_probe import request_mas
from mas_patient_assignment_data_v1 import run
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file
from therapist_harness.text import text_normalize

NAME = 'patient_content_probe_v1'


class Practice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sample_id: str
    target_text: str = Field(min_length=4, max_length=50)
    feedback_ids: list[str] = Field(min_length=1, max_length=3)
    reason: str = Field(min_length=1, max_length=350)


class ContentProposal(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reflection: str = Field(min_length=1, max_length=1500)
    hypothesis: str = Field(min_length=1, max_length=1200)
    practices: list[Practice] = Field(min_length=24, max_length=24)
    observation_against_hypothesis: str = Field(min_length=1, max_length=800)


def prescribe(base, root):
    from openai.lib._pydantic import to_strict_json_schema

    source = base / 'patient_assignment_probe_v1'
    old = read(source / 'plan.json'); p = RunProtocol.model_validate(old['source_protocol'])
    assert all(sha256_file(Path(f)) == h for f,h in old['source_sha256'].items())
    assignment = read(source / 'patient_assignment_plan.json'); voice = assignment['voice_group']
    patient = read(source / 'patient_key_map.json')[voice]
    before_path = base / 'parent_eval_check/predictions.jsonl'
    after_path = base / 'patient_assignment_DPO_v1/MAS_patient/evaluation_run/evaluation/predictions.jsonl'
    after = {x['utt_id']:x for x in rows(after_path)}
    feedback = []; crosswalk = {}
    for index, x in enumerate(x for x in rows(before_path) if x['speaker_id'] == patient):
        key = f'E{index+1:02d}'; y = after[x['utt_id']]; crosswalk[key] = x['utt_id']
        assert all(x[k] == y[k] for k in ['clean_gt','speaker_id','audio'])
        feedback.append(dict(feedback_id=key, intended=x['clean_gt'], parent_prediction=x['prediction'],
            last_DPO_prediction=y['prediction'], parent_errors=x['edit_count'], last_DPO_errors=y['edit_count']))
    cases = read(source / 'cases.json')
    generation = {x['sample_id']:x for x in read(source / 'generation/generation_summary.json')['outputs']}
    sampling = read(source / 'sampling/preference_audit.json')
    feedback_export = base / 'patient_assignment_feedback_export_v1/proposal_context.json'
    summary = read(base / 'patient_assignment_DPO_v1/results.json')
    clean = {a:{k:v for k,v in x.items() if k in ['CER_percent','edits','macro_CER_percent','hard_edits','eligible']} for a,x in summary['arms'].items()}
    task_cards = [dict(sample_id=x['sample_id'], role=x['role'], previous_target=x['target_text'],
        reference_id=generation[x['sample_id']]['generation_choice']['reference_id'],
        reference_text=generation[x['sample_id']]['reference_text'],
        fixed_mode=generation[x['sample_id']]['generation_mode'],
        fixed_instruction_id=generation[x['sample_id']]['effective_instruction_id'],
        previous_SG_greedy=next(g['prediction'] for g in rows(source / 'sampling/greedy.jsonl') if g['audio_id'] == x['sample_id'])) for x in cases]
    payload = dict(selected_voice_group=voice, real_error_feedback=feedback, fixed_case_roles=task_cards,
        completed_experiment=dict(arms=clean, selected_patient_and_others=summary['selected_patient_and_others'],
            MAS_minus_original_pp=summary['MAS_minus_original_CER_pp'], bootstrap95_pp=summary['bootstrap95_pp']),
        surface_diagnostic={k:v for k,v in read(source / 'surface_error_alignment_diagnostic.json').items() if k != 'source_sha256'},
        previous_sampling={k:v for k,v in sampling.items() if k not in ['audio_results','current_asr_sha256']})
    body = json.dumps(payload, ensure_ascii=False); assert '/data/' not in body and 'vlink' not in body
    root.mkdir(exist_ok=False); out = root / 'MAS_content'; out.mkdir()
    save(root / 'feedback_key_map.json', crosswalk)
    files = [Path(__file__), Path(__file__).with_name('mas_patient_assignment_data_v1.py'),
        Path(__file__).with_name('mas_patient_retention_probe.py'), before_path, after_path, feedback_export,
        base / 'patient_assignment_DPO_v1/results.json', source / 'surface_error_alignment_diagnostic.json',
        source / 'patient_assignment_plan.json', source / 'cases.json', source / 'generation/generation_summary.json',
        source / 'sampling/preference_audit.json', source / 'sampling/greedy.jsonl', root / 'feedback_key_map.json']
    hashes = {**old['source_sha256'], **{str(f):sha256_file(f) for f in files}}
    save(root / 'input_sources.json', hashes)
    request = dict(model=p.agent.model, reasoning={'effort':p.agent.reasoning_effort}, store=False, stream=True,
        max_output_tokens=14000, input=[dict(role='system', content='''You are the MAS practice-design agent for Cantonese dysarthric ASR.
Reflect on the completed patient-allocation DPO probe and propose24 NEW natural
Cantonese practice texts using this patient's actual real errors. Return Chinese
reflection, one hypothesis, one practice for each exact sample_id and a concrete
observation against the hypothesis. Each practice cites 1-3 supplied feedback IDs.
Keep each case's declared role: target8, contrast4, protection4, maintenance8.
Target should address specific attested confusion or deletion patterns; contrast
should make the competing word/meaning legitimately correct in another natural
sentence, not label every competing form wrong. Protection and maintenance should
retain useful everyday recognition, including patterns still correct in supplied
transcripts. Explain links briefly; a role is not a separate loss or proof of value.
Prior patient-only reassignment changed selected-patient errors124->121, still
worse than R-only120, and worsened overall1424->1440. No checkpoint is promoted.
Do not present this as a successful overall improvement or assume all TTS is noise.
The prior24tasks had19distinct texts designed from aggregate errors. This probe
holds patient, exact reference per case, effective TTS mode/instruction and seeds
fixed to the already-generated patient arm and changes ONLY the target text policy.
Write24distinct short, ordinary, grammatical sentences, approximately8-18Chinese
characters each, with plausible meaning. Do not repeat any supplied complete real
feedback sentence or previous SG target. Reuse an attested word in a NEW sentence
when useful; do not simply paraphrase the entire development utterance or drill
one rare word in every item. Prefer up to3 concrete error families while retaining
different everyday contexts. Never infer acoustic/clinical properties from text.
The zero shared surface-substitution signatures and11of66 target-character
positions covered are descriptive only, not phonological coverage or fidelity.
An ASR substitution can be a homophone/spelling/semantic error; do not fabricate
phoneme diagnoses. Actual synthesis and sampling may fail to create the intended
training contrast; the ledger will report that honestly.
Review stays the same three-text coarse Agent/rulebook, no CER rejection threshold,
strict word/phoneme audit or listening requirement. Same frozen initialR-only
samples4 actual candidates plus greedy per audio. All-correct contributes no DPO;
one repeated actual error may form a pair. Chosen remains the intended text after
coarse review, not certified word fidelity. Both training arms start at initial
R-only, with frozen initialR-only DPO reference, same12step edit+EOS DPO,96synthetic
and192realKL occurrences and unchanged weighting/optimizer. Common actual paired
case IDs are matched; previous audio/review/candidates are reused with provenance.
All511 real-development recordings and original pooled/macro/hard acceptance
remain. This is a bounded exploratory content-policy probe, not a request to
change reference, dose, loss, evaluation or to claim core ledger promotion.'''),
            dict(role='user', content=body)], text={'format':dict(type='json_schema', name='patient_content_proposal',
                strict=True, schema=to_strict_json_schema(ContentProposal))})
    save(out / 'request.json', request); response = request_mas(p, request, out)
    assert response['status'] == 'completed'
    messages = [x for x in response['output'] if x.get('type') == 'message' and x.get('phase') == 'final_answer']; assert len(messages) == 1
    proposal = ContentProposal.model_validate_json(''.join(x['text'] for x in messages[0]['content'] if x['type'] == 'output_text'))
    choices = {x.sample_id:x for x in proposal.practices}
    assert len(choices) == 24 and set(choices) == {x['sample_id'] for x in cases}
    forbidden = {text_normalize(x['clean_gt']) for path in [p.real_train.path,p.feedback_split.path] for x in rows(path)}
    forbidden |= {text_normalize(x['target_text']) for x in cases}
    texts = [text_normalize(x.target_text) for x in proposal.practices]
    assert len(set(texts)) == 24 and not (set(texts) & forbidden)
    assert all(set(x.feedback_ids) <= set(crosswalk) for x in proposal.practices)
    assignments = []; plan_choices = []; new_cases = []; (root / 'audio').mkdir()
    for case in cases:
        key = case['sample_id']; action = choices[key]; old_audio = generation[key]
        assignments.append(dict(sample_id=key, target_text=action.target_text, seed=old_audio['seed'],
            reference_text=old_audio['reference_text'], reference_wav=old_audio['reference_wav'],
            reference_wav_sha256=old_audio['reference_wav_sha256'], output_wav=str(root / 'audio' / (key + '.wav'))))
        plan_choices.append(dict(old_audio['generation_choice'], mode=old_audio['generation_mode'],
            instruction_id=old_audio['effective_instruction_id'], reason='Fixed patient/reference/recipe content-policy control.',
            target_text_sha256=hashlib.sha256(action.target_text.encode()).hexdigest()))
        new_cases.append(dict(case, target_text=action.target_text, previous_target_text=case['target_text'],
            seed=old_audio['seed'], fixed_mode=old_audio['generation_mode'], fixed_instruction_id=old_audio['effective_instruction_id']))
    save(out / 'proposal.json', proposal.model_dump(mode='json'))
    jsonl(root / 'manifest.jsonl', assignments); save(root / 'cases.json', new_cases)
    save(root / 'generation_plan.json', dict(schema_version='generation_plan_v1', choices=plan_choices,
        hypothesis=proposal.hypothesis, expected_observation=proposal.observation_against_hypothesis))
    save(root / 'role_sidecar.json', read(source / 'role_sidecar.json'))
    save(root / 'patient_key_map.json', read(source / 'patient_key_map.json'))
    save(root / 'patient_assignment_plan.json', dict(schema_version='external_fixed_patient_content_v1',
        voice_group=voice, patient_reference_and_effective_recipe_unchanged=True, hypothesis=proposal.hypothesis))
    assert all(sha256_file(Path(f)) == h for f,h in hashes.items())
    save(root / 'plan.json', dict(source_protocol=p.model_dump(mode='json'), source_root=str(source), source_sha256=hashes,
        selected_cases=[x['sample_id'] for x in cases], intervention='Only content policy changes: previous aggregate-error texts versus MAS selected-patient real-error practices. Fixed patient/reference/effective initial recipe/TTS seed; same actual sampling parent/decode and training recipe.',
        control_preferences=str(base / 'patient_assignment_DPO_v1/MAS_patient/preferences.jsonl'),
        acceptance='Original pooled/macro/hard real-development gate plus selected-patient/other-patient reporting. No automatic promotion.',
        scope='External prospective content probe; prior sources/ledgers untouched; no claim of core Proposal/ReflectionPack commit.'))
    bound = [out / 'request.json', out / 'response.json', out / 'proposal.json'] + [root / n for n in ['manifest.jsonl','cases.json','generation_plan.json','role_sidecar.json','patient_key_map.json','patient_assignment_plan.json','plan.json']]
    save(root / 'decision_receipt.json', dict(actual_MAS_call=True, response_id=response['id'],
        bound_sha256={str(f):sha256_file(f) for f in bound}, no_complete_train_or_feedback_text_reused=True))
    print(dict(status='prescribed', voice_group=voice, new_distinct_texts=24), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['prescribe','run'])
    parser.add_argument('base', type=Path); args = parser.parse_args(); base = args.base.resolve(); root = base / NAME
    if args.stage == 'prescribe': prescribe(base, root)
    else: run(root)
