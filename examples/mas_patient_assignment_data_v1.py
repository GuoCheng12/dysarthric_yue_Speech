#!/usr/bin/env python3
"""Prospective same-content probe allowing MAS to choose a target patient."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_patient_retention_probe import request_mas
from therapist_harness.generation_controller import load_catalog, public_cards
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file
from therapist_harness.transcript_review import validate_transcript_review

NAME = 'patient_assignment_probe_v1'
QUOTAS = dict(target=8, contrast=4, protection=4, maintenance=8)


class ReferenceChoice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sample_id: str
    reference_id: str
    reason: str = Field(min_length=1, max_length=240)


class PatientAssignment(BaseModel):
    model_config = ConfigDict(extra='forbid')
    voice_group: str
    patient_reason: str = Field(min_length=1, max_length=1000)
    hypothesis: str = Field(min_length=1, max_length=1000)
    choices: list[ReferenceChoice] = Field(min_length=24, max_length=24)
    observation_against_hypothesis: str = Field(min_length=1, max_length=800)


def verify(root):
    cfg = read(root / 'plan.json')
    assert all(sha256_file(Path(f)) == h for f, h in cfg['source_sha256'].items())
    return cfg


def prepare(base, root):
    root.mkdir(exist_ok=False)
    source = base / 'adaptive_mas96_data_v1/round_001'
    old = read(base / 'adaptive96_matched_DPO_recovery_v2/protocol.json')
    assert all(sha256_file(Path(f)) == h for f, h in old['source_sha256'].items())
    p = RunProtocol.model_validate(old['source_protocol'])
    catalog_path = base / 'reference_catalog.json'
    real = rows(p.real_train.path)
    cards = load_catalog(dict(catalog_path=str(catalog_path), catalog_sha256=sha256_file(catalog_path)), real)
    by_wave = {x['reference_wav']:x for x in cards.values()}
    voices = {x['speaker_id']:x['voice_group'] for x in cards.values()}
    before = rows(base / 'parent_eval_check/predictions.jsonl')
    after_path = base / 'adaptive96_matched_DPO_recovery_v2/MAS_reference/evaluation_run/evaluation/predictions.jsonl'
    after = {x['utt_id']:x for x in rows(after_path)}
    outputs = read(source / 'generation/generation_summary.json')['outputs']
    generated = {x['sample_id']:x for x in outputs}
    sidecar = read(source / 'materialization_sidecar.json')
    roles = {x['sample_id']:x for x in sidecar['samples']}
    control_path = base / 'adaptive96_matched_DPO_recovery_v2/MAS_reference/preferences.jsonl'
    paired = {x['audio_id'] for x in rows(control_path)}
    audit = {x['audio_id']:x for x in read(source / 'sampling/preference_audit.json')['audio_results']}
    coverage = []
    for patient, voice in sorted(voices.items()):
        dev = [x for x in before if x['speaker_id'] == patient]
        sg = [x for x in outputs if by_wave[x['reference_wav']]['speaker_id'] == patient]
        for x in dev:
            assert all(x[k] == after[x['utt_id']][k] for k in ['audio', 'clean_gt', 'speaker_id'])
        coverage.append(dict(voice_group=voice, real_train_audio=sum(x['speaker_id'] == patient for x in real),
            real_dev_audio=len(dev), dev_characters=sum(x['normalized_reference_characters'] for x in dev),
            parent_dev_errors=sum(x['edit_count'] for x in dev),
            MAS_dev_errors=sum(after[x['utt_id']]['edit_count'] for x in dev),
            generated_audio=len(sg), actual_matched_DPO_audio=sum(x['sample_id'] in paired for x in sg),
            all_correct_generated_audio=sum(audit[x['sample_id']]['correct_samples'] == 4 for x in sg),
            eligible=bool(dev)))
    save(root / 'patient_key_map.json', {v:k for k,v in voices.items()})
    save(root / 'patient_coverage.json', dict(patients=coverage, total_real_train=len(real), total_real_dev=len(before),
        total_generated=len(outputs), total_matched_DPO_audio=len(paired),
        interpretation='Observed training/development coverage, not clinical severity or causal proof. Prior materialization fixed patients before the MAS within-patient reference planner.'))
    rng = random.Random(42); selected = []
    for role, quota in QUOTAS.items():
        groups = defaultdict(list)
        for key in sorted(paired):
            if roles[key]['role'] == role: groups[roles[key]['stimulus_id']].append(key)
        keys = sorted(groups); rng.shuffle(keys)
        for group in groups.values(): rng.shuffle(group)
        chosen = []
        while len(chosen) < quota:
            for stimulus in keys:
                if groups[stimulus] and len(chosen) < quota: chosen.append(groups[stimulus].pop())
            assert any(groups.values()) or len(chosen) == quota
        selected.extend(chosen)
    assert len(selected) == len(set(selected)) == 24
    cases = []
    for key in selected:
        x = generated[key]; card = by_wave[x['reference_wav']]
        assert x['admitted'] and x['attempt'] == 0 and x['seed'] == x['assignment_seed']
        validate_transcript_review(x['review'], p.preference_loop.review_asr_adapter_sha256,
                                  p.preference_loop.review_asr_base_model_sha256)
        assert sha256_file(Path(x['output_wav'])) == x['output_wav_sha256']
        cases.append(dict(sample_id=key, target_text=x['target_text'], role=roles[key]['role'],
            stimulus_id=roles[key]['stimulus_id'], seed=x['assignment_seed'],
            fixed_mode=x['generation_mode'], fixed_instruction_id=x['effective_instruction_id'],
            original_voice_group=card['voice_group'], original_reference_id=card['reference_id']))
    save(root / 'cases.json', cases)
    save(root / 'role_sidecar.json', dict(schema_version='external_patient_assignment_roles_v1',
        samples=[dict(sample_id=x['sample_id'], role=x['role']) for x in cases],
        scope='Role-only allocation input; not a core all_patients_balanced materialization sidecar.'))
    reference_history = []
    for rid, card in cards.items():
        sg = [x for x in outputs if x['reference_wav'] == card['reference_wav']]
        reference_history.append(dict(reference_id=rid, generated=len(sg), coarse_pass=sum(x['admitted'] for x in sg),
            all_correct=sum(audit[x['sample_id']]['correct_samples'] == 4 for x in sg),
            paired_in_previous_matched_training=sum(x['sample_id'] in paired for x in sg)))
    save(root / 'reference_history.json', reference_history)
    files = [Path(__file__), Path(__file__).with_name('mas_patient_retention_probe.py'), catalog_path,
        source / 'synthesis_manifest.jsonl', source / 'materialization_sidecar.json', source / 'generation_plan.json',
        source / 'generation/generation_summary.json', after_path, control_path,
        base / 'parent_eval_check/predictions.jsonl', base / 'real_KL_prefix_probe_v1/results.json']
    files += [source / 'sampling' / n for n in ['validated_inputs.jsonl', 'candidates.jsonl', 'greedy.jsonl',
        'sampling_receipt.json', 'preferences.jsonl', 'preference_audit.json']]
    files += [root / n for n in ['patient_key_map.json', 'patient_coverage.json', 'cases.json', 'role_sidecar.json', 'reference_history.json']]
    files += [Path(generated[key]['output_wav']) for key in selected]
    files += [REPO / 'examples/generate_reviewed_sg_recovery_v2.py', REPO / 'examples/generate_reviewed_sg_resumable.py',
        REPO / 'examples/sample_asr_preferences.py', REPO / 'src/therapist_harness/generation_controller.py',
        REPO / 'src/therapist_harness/materialize.py']
    save(root / 'plan.json', dict(source_protocol=old['source_protocol'], source_root=str(source),
        source_sha256={**old['source_sha256'], **{str(f):sha256_file(f) for f in files}},
        control_preferences=str(control_path), selected_cases=selected, arms=['original_patient', 'MAS_patient'],
        intervention='Same24 cases/texts/roles/seeds/initial recipes; MAS chooses one patient and an existing train-only reference for each case. Original audio/review/sampling reused with separate provenance.',
        inherited_protocol_scope='Historical model, data, review and training settings only. New explicit patient_assignment_plan governs voice allocation; no claim of unchanged all_patients_balanced materialization.',
        sampling='4actual stochastic draws plus greedy from initial R-only, distinct new receipt; same logical case IDs give matched RNG seeds, actual waveform hashes differ.',
        training='Common actual paired cases only, requested role8/4/4/8 renormalized only if a role lacks pairs. Identical96synthetic and192real audio occurrences,12steps lr1e-4 beta.1 edit+EOS DPO plus GT-prefix realKL10 with existing focus weighting; frozen initialR-only.',
        admission='Existing three-text Agent/rulebook unchanged; all-correct skipped, repeated same actual error gives one pair; no CER threshold or word-fidelity claim.',
        acceptance='Original pooled/macro/hard real-development gate; report chosen-patient and other-patient metrics. No automatic promotion from a single exploratory seed.',
        storage='Audio and execution receipts in workspace; training weights in /tmp. Prior sources and ledgers untouched.'))
    verify(root)
    print(dict(status='prepared', cases=len(cases), distinct_texts=len({x['target_text'] for x in cases})), flush=True)


def prescribe(base, root):
    from openai.lib._pydantic import to_strict_json_schema

    cfg = verify(root); p = RunProtocol.model_validate(cfg['source_protocol']); out = root / 'MAS_assignment'; out.mkdir()
    cases = read(root / 'cases.json')
    payload = dict(patient_coverage=read(root / 'patient_coverage.json'),
        reference_cards=public_cards({x['reference_id']:x for x in read(base / 'reference_catalog.json')['cards']}),
        reference_history=read(root / 'reference_history.json'), cases=cases,
        previous_result=dict(parent_CER_percent=20.79326923076923, MAS_CER_percent=21.06370192307692,
            random_reference_CER_percent=21.349158653846153,
            prefix_probe=read(base / 'real_KL_prefix_probe_v1/results.json')['arms']))
    # Remove local patient identifiers from the aggregate prefix summary.
    payload['previous_result']['prefix_probe'] = {k:{f:v for f,v in x.items() if f in ['CER_percent','edits','hard_edits','macro_CER_percent']} for k,x in payload['previous_result']['prefix_probe'].items()}
    body = json.dumps(payload, ensure_ascii=False); assert '/data/' not in body and 'vlink' not in body
    request = dict(model=p.agent.model, reasoning={'effort':p.agent.reasoning_effort}, store=False, stream=True,
        max_output_tokens=10000, input=[dict(role='system', content='''You are the MAS patient/reference assignment agent for Cantonese dysarthric ASR.
Choose ONE eligible voice_group using the measured real-error burden, existing
generated/trained coverage and available training-only references. Assign one
reference from that chosen patient's two available recordings to each of24cases.
This prospective intervention lets MAS choose who receives practice; prior MAS
could only change references within patients fixed by a balanced allocator.
Keep the supplied target texts, roles, seeds and per-case mode/instruction fixed.
These24control cases were selected before your decision from previously paired
audio, fixedseed42 and role8/4/4/8 with stimulus diversity. Their old actual
audio/reviews/4draws are the control. Generate24new audio; train only common cases
with actual wrong candidates. All-correct means no DPO pair, never invent errors.
The same12step edit+EOS DPO, realKL retention, initialR-only parent/reference,
actual exposure schedule and real-development evaluation apply to both arms.
Pick the patient to address a concrete coverage hypothesis, not because high
error automatically means high training value. Do not claim a patient ID or
reference transcript proves severity, timbre, slurring or acoustic fidelity.
Reference cards give text, duration and prior use only; acoustic traits unknown.
You may balance the two references when neither has demonstrated transfer.
More candidate errors or a coarse pass is not a word-fidelity certificate.
Keep the existing simple three-text review and regeneration rulebook, no extra
scores, CER cutoff, phoneme checks or listening requirement. Do not change loss,
generate new text, remove development patients or request permission. Respond in
Chinese with a brief patient reason, case reference reasons, hypothesis and a
specific observation that would count against it. No new outcome is known.'''),
            dict(role='user', content=body)], text={'format':dict(type='json_schema', name='patient_assignment',
                strict=True, schema=to_strict_json_schema(PatientAssignment))})
    save(out / 'request.json', request); response = request_mas(p, request, out)
    assert response['status'] == 'completed'
    final = [x for x in response['output'] if x.get('type') == 'message' and x.get('phase') == 'final_answer']; assert len(final) == 1
    decision = PatientAssignment.model_validate_json(''.join(x['text'] for x in final[0]['content'] if x['type'] == 'output_text'))
    assert next(x for x in payload['patient_coverage']['patients'] if x['voice_group'] == decision.voice_group)['eligible']
    cards = {x['reference_id']:x for x in read(base / 'reference_catalog.json')['cards']}
    choices = {x.sample_id:x for x in decision.choices}
    assert len(choices) == 24 and set(choices) == {x['sample_id'] for x in cases}
    assignments = []; compiled = []; (root / 'audio').mkdir()
    for case in cases:
        key = case['sample_id']; action = choices[key]; ref = cards[action.reference_id]
        assert ref['voice_group'] == decision.voice_group
        assignments.append(dict(sample_id=key, target_text=case['target_text'], seed=case['seed'],
            reference_text=ref['reference_text'], reference_wav=ref['reference_wav'],
            reference_wav_sha256=ref['reference_wav_sha256'], output_wav=str(root / 'audio' / (key + '.wav'))))
        compiled.append(dict(sample_id=key, reference_id=action.reference_id, reason=action.reason,
            mode=case['fixed_mode'], instruction_id=case['fixed_instruction_id'],
            reference_wav_sha256=ref['reference_wav_sha256'], target_text_sha256=hashlib.sha256(case['target_text'].encode()).hexdigest()))
    save(out / 'decision.json', decision.model_dump(mode='json'))
    save(root / 'patient_assignment_plan.json', dict(schema_version='external_patient_assignment_v1',
        voice_group=decision.voice_group, hypothesis=decision.hypothesis, cases=cases,
        actual_decision_sha256=sha256_file(out / 'decision.json'), original_protocol_voice_allocator_overridden=True))
    jsonl(root / 'manifest.jsonl', assignments)
    save(root / 'generation_plan.json', dict(schema_version='generation_plan_v1', choices=compiled,
        hypothesis=decision.hypothesis, expected_observation=decision.observation_against_hypothesis))
    verify(root)
    save(root / 'decision_receipt.json', dict(actual_MAS_call=True, response_id=response['id'],
        source_hashes_unchanged=True, bound_sha256={str(f):sha256_file(f) for f in [out / 'request.json', out / 'response.json', out / 'decision.json', root / 'patient_assignment_plan.json', root / 'manifest.jsonl', root / 'generation_plan.json']}))
    print(dict(status='prescribed', voice_group=decision.voice_group, cases=24), flush=True)


def run(root):
    cfg = verify(root); p = RunProtocol.model_validate(cfg['source_protocol']); c = p.preference_loop
    assert all(sha256_file(Path(f)) == h for f,h in read(root / 'decision_receipt.json')['bound_sha256'].items())
    command = [sys.executable, str(REPO / 'examples/generate_reviewed_sg_recovery_v2.py'),
        '--manifest', str(root / 'manifest.jsonl'), '--out-dir', str(root / 'generation'),
        '--generation-plan', str(root / 'generation_plan.json'), '--review-mode', 'asr_transcript',
        '--review-asr-worker', '--audio-review-model', c.audio_review_model,
        '--audio-review-base-url', c.audio_review_base_url, '--review-asr-model', p.base_model_dir,
        '--review-asr-adapter', c.review_asr_adapter, '--review-asr-adapter-sha256', c.review_asr_adapter_sha256,
        '--review-asr-base-model-sha256', c.review_asr_base_model_sha256]
    if c.review_allow_pending: command += ['--review-allow-pending']
    execute(root, command, 'generation')
    verify(root)
    command = [sys.executable, str(REPO / 'examples/sample_asr_preferences.py'), 'sample',
        '--inputs', str(root / 'generation/accepted.jsonl'), '--real-train', p.real_train.path,
        '--feedback', p.feedback_split.path, '--out-dir', str(root / 'sampling'), '--base-model', p.base_model_dir,
        '--current-adapter', p.initial_adapter.path, '--expected-parent-sha256', p.initial_adapter.sha256,
        '--samples', '4', '--seed', '42', '--temperature', str(c.temperature), '--top-p', str(c.top_p),
        '--max-new-tokens', str(c.max_new_tokens), '--include-greedy']
    execute(root, command, 'sampling'); verify(root)
    save(root / 'data_complete.json', dict(status='complete', generation=read(root / 'generation/generation_summary.json')['admitted_count'],
        sampling=read(root / 'sampling/preference_audit.json'), parent_promoted=False))
    print(dict(status='data_complete', paired=read(root / 'sampling/preference_audit.json')['paired_audio_count']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['prepare','prescribe','run','all'])
    parser.add_argument('base', type=Path); args = parser.parse_args(); base = args.base.resolve(); root = base / NAME
    if args.stage in ['prepare','all']: prepare(base, root)
    if args.stage in ['prescribe','all']: prescribe(base, root)
    if args.stage in ['run','all']: run(root)
