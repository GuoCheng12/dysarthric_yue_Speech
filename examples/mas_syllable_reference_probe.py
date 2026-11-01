#!/usr/bin/env python3
"""Fresh MAS content and same-patient reference retrieval, followed by matched DPO."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_evaluation_report import compare
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.grounding import validate_stimulus_grounding
from therapist_harness.materialize import materialize_proposal, build_materialization_sidecar
from therapist_harness.schema import RunProtocol, EvidencePack, Proposal, canonical_sha256, validate_proposal
from therapist_harness.store import sha256_file
from therapist_harness.training_allocation import compile_role_allocation


class Exercise(BaseModel):
    model_config = ConfigDict(extra='forbid')
    parent_stimulus_id: str
    text: str
    rationale: str


class ContentDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    hypothesis: str
    expected_response: str
    exercises: list[Exercise] = Field(min_length=6, max_length=6)


def prepare(base, root):
    import pycantonese
    import soundfile as sf
    from openai.lib._pydantic import to_strict_json_schema

    root.mkdir(exist_ok=False)
    donor = base / 'lexical24_loop_v1'
    pdata = read(donor / 'protocol.json')
    real = rows(pdata['real_train']['path'])
    frozen = read(pdata['frozen_reference_manifest']['path'])
    by_speaker = defaultdict(list)
    for row in real:
        if 0 < float(row['duration']) <= 30:
            by_speaker[row['speaker_id']].append(row)
    texts = {x['clean_gt'] for x in real}
    carrier_texts = {t for t in texts if any(jp and 'zim1' in jp.split() for _, jp in pycantonese.characters_to_jyutping(t))}
    assert carrier_texts
    groups = {x['speaker_id']: x['difficulty_group'] for x in frozen['references']}
    reference_pairs, controls, cards = {}, [], []
    for speaker in sorted(groups):
        pool = by_speaker[speaker]
        carriers = sorted((x for x in pool if x['clean_gt'] in carrier_texts), key=lambda x: x['utt_id'])
        others = [x for x in pool if x['clean_gt'] not in carrier_texts]
        if not carriers or not others:
            continue
        carrier = carriers[0]
        # Selection uses duration and stable identity, never reference ASR CER.
        control = min(others, key=lambda x: (abs(float(x['duration']) - float(carrier['duration'])),
            abs(len(x['clean_gt']) - len(carrier['clean_gt'])), x['utt_id']))
        voice = f'voice{len(reference_pairs):03d}'
        pair = {}
        for arm, row in [('carrier', carrier), ('control', control)]:
            record = dict(speaker_id=speaker, difficulty_group=groups[speaker], reference_text=row['clean_gt'],
                reference_wav=row['audio'], reference_wav_sha256=sha256_file(Path(row['audio'])), utt_id=row['utt_id'])
            pair[arm] = record
            cards.append(dict(record, reference_id=f'{voice}.{arm}', voice_group=voice,
                seconds=sf.info(row['audio']).duration, acoustic_traits='unknown_without_listening',
                origin='original' if arm == 'control' else 'transcript_zim1_carrier'))
        reference_pairs[speaker] = dict(voice_group=voice, **pair)
        controls.append(pair['control'])
    assert {x['difficulty_group'] for x in controls} == {'easy', 'medium', 'hard'}
    save(root / 'reference_pairs.json', reference_pairs)
    save(root / 'reference_catalog.json', dict(schema_version='reference_candidates_v1', cards=cards))
    save(root / 'reference_manifest.json', dict(schema_version='1', profiles=frozen['profiles'], references=controls,
        source_manifest_path=pdata['real_train']['path'], source_manifest_sha256=pdata['real_train']['sha256']))
    design = dict(strategy='paired_syllable_reference', catalog_path=str(root / 'reference_catalog.json'),
        catalog_sha256=sha256_file(root / 'reference_catalog.json'),
        recipe_options=[dict(mode='zero_shot', instruction_id='cantonese_en')],
        hypothesis='Test same-patient reference transcripts containing zim1 versus duration-matched transcripts without it. Textual phonological coverage is not verified acoustic realization.')
    save(root / 'generation_design.json', design)
    pdata['run_id'] = root.name
    pdata['frozen_reference_manifest'].update(path=str(root / 'reference_manifest.json'), sha256=sha256_file(root / 'reference_manifest.json'))
    pdata['preference_loop'].update(generation_design_path=str(root / 'generation_design.json'),
        generation_design_sha256=sha256_file(root / 'generation_design.json'))
    p = RunProtocol.model_validate(pdata)
    save(root / 'protocol.json', p.model_dump(mode='json'))
    edata = read(donor / 'round_001/evidence.json')
    edata['run_id'] = p.run_id
    evidence = EvidencePack.model_validate(edata)
    save(root / 'evidence.json', evidence.model_dump(mode='json'))
    old = read(donor / 'round_001/proposal.json')
    ids = {eid for x in old['stimuli'] for eid in x['evidence_ids']}
    selected_evidence = [x for x in edata['error_clusters'] + edata['capability_anchors'] if x['evidence_id'] in ids]
    save(root / 'design.json', dict(arms=['carrier', 'control'], seed=42, texts=6, audio_per_arm=24,
        dpo_steps=4, matched_audio_exposures=32, reference_patients=len(reference_pairs),
        intervention='Same patient, target text, TTS seed and initial recipe; change only reference retrieval. Carrier is first eligible zim1 recording by utt_id. Control minimizes duration difference then text-length difference and utt_id.',
        content_intervention='Fresh MAS proposal centered on 尖 rather than the previously problematic 鉗/蠍. Both reference arms share the same proposal.',
        comparison_scope='Reference strategy comparison on this new content; no isolated causal comparison with earlier content.',
        training_match='Use the intersection of actual paired audio IDs, retain all their actual errors, and share one role-balanced exposure schedule. No new admission criterion or invented negative.',
        review='Same simple three-text coarse review, frozen R-only greedy worker and regeneration rulebook. No listening/word-fidelity claim or CER cutoff.',
        checkpoint_gate='Raw real-development pooled CER improves; macro regression<=0.1pp, hard<=0.5pp.'))
    request = dict(model=p.agent.model, reasoning={'effort': p.agent.reasoning_effort}, store=False, stream=True,
        max_output_tokens=16384, input=[dict(role='system', content='''Create six fresh Cantonese practice sentences for a controlled MAS-to-TTS-to-DPO probe.
Replace each of the six supplied exercises once, inheriting its role, evidence IDs,
speaker profile and four-copy budget. The first four (target/contrast/protection)
must contain 尖, preferably in ordinary 尖/尖銳 contexts. Do not use 鉗 or 蠍.
The real feedback includes 尖→專; this is a narrow feasible-target test, not a claim
that it accounts for the majority of errors. The reference pool contains 詹 zim1,
matching the written pronunciation of 尖, whereas no kim4/kit3 carrier was found.
Keep the original phonological grounding: the i/e contrast needs both i and e
carriers (for example a natural use of 車); m/n protection needs m and n carriers.
Maintenance exercises retain their z-onset/easy and j-onset/medium anchors. Include
one natural TRUE use of 專 in a maintenance sentence to discourage indiscriminate
專→尖 correction while still satisfying that anchor. No speech-style directions,
tongue twisters or packed word lists. Aim for 12–20 characters, maximum 24.
Use Traditional Chinese and natural Cantonese. Every whole sentence must be new;
do not copy a supplied original or evaluation sentence. Return only the six
replacement texts, their learning reasons, and a prospective overall hypothesis.
Both reference arms will receive these exact same texts. Same-audio DPO negatives
will come only from actual R-only samples. Written readings do not prove how the
generated audio is pronounced; keep the coarse review unchanged.'''),
            dict(role='user', content=json.dumps(dict(original_proposal=old, current_real_evidence=selected_evidence,
                prior_results='Full-pair, decoder-only and one-negative MAS DPO did not beat R-only; this probe changes realizable content and reference retrieval rather than counting postprocessing gains.'), ensure_ascii=False))],
        text={'format': {'type': 'json_schema', 'name': 'syllable_reference_content', 'strict': True,
                         'schema': to_strict_json_schema(ContentDecision)}})
    save(root / 'content_request.json', request)
    response = serialize_response(execute_therapist_request(p, request))
    save(root / 'content_response.json', response)
    assert response['status'] == 'completed'
    final = [x for x in response['output'] if x.get('type') == 'message' and x.get('phase') == 'final_answer']
    assert len(final) == 1
    decision = ContentDecision.model_validate_json(''.join(x['text'] for x in final[0]['content'] if x['type'] == 'output_text'))
    replacements = {x.parent_stimulus_id: x for x in decision.exercises}
    assert len(replacements) == 6 and replacements.keys() == {x['stimulus_id'] for x in old['stimuli']}
    stimuli = []
    for i, item in enumerate(old['stimuli']):
        new = replacements[item['stimulus_id']]
        if i < 4:
            assert '尖' in new.text and '鉗' not in new.text and '蠍' not in new.text
        stimuli.append(dict(item, stimulus_id=f'stim.r1.{item["role"]}.zim{i:02d}', text=new.text, rationale=new.rationale))
    assert any('專' in x['text'] for x in stimuli if x['role'] == 'maintenance')
    proposal = Proposal.model_validate(dict(old, evidence_sha256=canonical_sha256(evidence),
        hypothesis=decision.hypothesis, expected_response=decision.expected_response, stimuli=stimuli))
    validate_proposal(proposal, evidence, p)
    validate_stimulus_grounding(proposal, evidence)
    save(root / 'proposal.json', proposal.model_dump(mode='json'))
    speaker_by_hash = {x['reference_wav_sha256']: x['speaker_id'] for x in controls}
    for arm in ['carrier', 'control']:
        directory = root / arm
        (directory / 'audio').mkdir(parents=True)
        assignments = materialize_proposal(proposal, evidence, p, output_root=directory / 'audio')
        choices = []
        for row in assignments:
            speaker = speaker_by_hash[row['reference_wav_sha256']]
            pair = reference_pairs[speaker]
            selected = pair[arm]
            row.update({k: selected[k] for k in ['reference_wav', 'reference_wav_sha256', 'reference_text']})
            choices.append(dict(sample_id=row['sample_id'], reference_id=f'{pair["voice_group"]}.{arm}',
                mode='zero_shot', instruction_id='cantonese_en',
                reference_wav_sha256=selected['reference_wav_sha256'],
                target_text_sha256=hashlib.sha256(row['target_text'].encode()).hexdigest()))
        save(directory / 'generation_plan.json', dict(schema_version='generation_plan_v1', choices=choices))
        jsonl(directory / 'manifest.jsonl', assignments)
        sidecar = build_materialization_sidecar(proposal, evidence, p, assignments, generation_plan_path=directory / 'generation_plan.json')
        save(directory / 'materialization_sidecar.json', sidecar.model_dump(mode='json'))
    source_files = [Path(__file__), donor / 'protocol.json', donor / 'round_001/evidence.json', donor / 'round_001/proposal.json',
        Path(p.real_train.path), Path(p.feedback_split.path), Path(p.initial_adapter_dev_predictions.path),
        REPO / 'examples/generate_reviewed_sg_recovery_v2.py', REPO / 'examples/generate_reviewed_sg_resumable.py',
        REPO / 'examples/generate_reviewed_sg.py', REPO / 'examples/sample_asr_preferences.py',
        REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py', REPO / 'finetune/scripts/evaluate_asr_checkpoint.py',
        REPO / 'src/therapist_harness/dpo_training.py', REPO / 'src/therapist_harness/review_worker.py']
    source_files += list(root.glob('*.json'))
    source_files += [root / arm / name for arm in ['carrier', 'control'] for name in ['manifest.jsonl', 'generation_plan.json', 'materialization_sidecar.json']]
    save(root / 'sources.json', {str(f): sha256_file(f) for f in source_files})
    print(dict(prepared=True, reference_patients=len(reference_pairs), audio_per_arm=24, texts=[x.text for x in proposal.stimuli]), flush=True)


def verify(root):
    assert all(sha256_file(Path(f)) == digest for f, digest in read(root / 'sources.json').items())
    return RunProtocol.model_validate(read(root / 'protocol.json'))


def generate_sample(root, arm):
    p = verify(root)
    c = p.preference_loop
    d = root / arm
    execute(d, [sys.executable, str(REPO / 'examples/generate_reviewed_sg_recovery_v2.py'),
        '--manifest', str(d / 'manifest.jsonl'), '--out-dir', str(d / 'generation'),
        '--generation-plan', str(d / 'generation_plan.json'), '--review-mode', 'asr_transcript',
        '--review-asr-worker', '--review-allow-pending', '--audio-review-model', c.audio_review_model,
        '--audio-review-base-url', c.audio_review_base_url, '--review-asr-model', p.base_model_dir,
        '--review-asr-adapter', c.review_asr_adapter, '--review-asr-adapter-sha256', c.review_asr_adapter_sha256,
        '--review-asr-base-model-sha256', c.review_asr_base_model_sha256], 'generation')
    execute(d, [sys.executable, str(REPO / 'examples/sample_asr_preferences.py'), 'sample',
        '--inputs', str(d / 'generation/accepted.jsonl'), '--real-train', p.real_train.path,
        '--feedback', p.feedback_split.path, '--out-dir', str(d / 'sampling'),
        '--base-model', p.base_model_dir, '--current-adapter', p.initial_adapter.path,
        '--expected-parent-sha256', p.initial_adapter.sha256, '--samples', '4', '--seed', '42',
        '--temperature', str(c.temperature), '--top-p', str(c.top_p), '--max-new-tokens', str(c.max_new_tokens), '--include-greedy'], 'sampling')
    verify(root)


def match(root):
    p = verify(root)
    pairs = {arm: rows(root / arm / 'sampling/preferences.jsonl') for arm in ['carrier', 'control']}
    common = set.intersection(*({x['audio_id'] for x in value} for value in pairs.values()))
    save(root / 'matching_receipt.json', dict(common_paired_audio=sorted(common), actual_new_sampling=False,
        scope='Training uses shared eligible case IDs only. Both complete sampling receipts and all review decisions remain unchanged.',
        source_sha256={str(root / arm / 'sampling/sampling_receipt.json'): sha256_file(root / arm / 'sampling/sampling_receipt.json') for arm in pairs}))
    if len(common) < 2:
        save(root / 'results.json', dict(status='insufficient_common_pairs_no_update', common_paired_audio=len(common)))
        return False
    for arm, value in pairs.items():
        jsonl(root / arm / 'matched_preferences.jsonl', [x for x in value if x['audio_id'] in common])
    allocation = compile_role_allocation(read(root / 'carrier/materialization_sidecar.json'),
        rows(root / 'carrier/matched_preferences.jsonl'), steps=4, per_step=8, seed=42)
    save(root / 'matched_allocation.json', allocation)
    keys = list(dict.fromkeys(x for step in allocation['steps'] for x in step))[:2]
    assert len(keys) == 2
    jsonl(root / 'smoke_preferences.jsonl', [x for x in rows(root / 'carrier/matched_preferences.jsonl') if x['audio_id'] in keys])
    save(root / 'smoke_allocation.json', dict(steps=[keys, list(reversed(keys))]))
    files = [root / 'matched_allocation.json', root / 'matching_receipt.json', root / 'smoke_preferences.jsonl', root / 'smoke_allocation.json']
    files += [root / arm / 'matched_preferences.jsonl' for arm in pairs]
    save(root / 'training_sources.json', {str(f): sha256_file(f) for f in files})
    print(dict(common_paired_audio=len(common), matched_exposures=32), flush=True)
    return True


def train(root, arm, smoke=False):
    p = verify(root)
    assert all(sha256_file(Path(f)) == digest for f, digest in read(root / 'training_sources.json').items())
    if not smoke:
        assert read(root / 'smoke/verified.json')['status'] == 'passed'
    d = root / ('smoke' if smoke else arm)
    steps = 2 if smoke else 4
    execute(d, [sys.executable, str(REPO / 'finetune/scripts/qwen3_asr_audio_dpo.py'),
        '--model_path', p.base_model_dir, '--policy_adapter_path', p.initial_adapter.path,
        '--reference_adapter_path', p.initial_adapter.path,
        '--train_file', str(root / 'smoke_preferences.jsonl' if smoke else d / 'matched_preferences.jsonl'),
        '--synthetic_allocation_file', str(root / ('smoke_allocation.json' if smoke else 'matched_allocation.json')),
        '--output_dir', str(d / 'training'), '--max_steps', str(steps), '--real_per_step', '0',
        '--synthetic_per_step', '2' if smoke else '8', '--synthetic_weight', '1',
        '--lr', str(p.lora.learning_rate), '--beta', str(p.preference_loop.beta), '--seed', '42'], 'training')
    result = read(d / 'training/training_result.json')
    assert result['policy_changed'] and result['reference_files_unchanged']
    assert math.isclose(result['history'][0]['loss'], math.log(2), abs_tol=1e-4)
    save(d / 'verified.json', dict(status='passed', reference_unchanged=True, initial_dpo_log2=True))
    if smoke:
        return
    execute(d, [sys.executable, str(REPO / 'finetune/scripts/evaluate_asr_checkpoint.py'),
        '--base-model-path', p.base_model_dir, '--adapter-path', str(d / f'training/checkpoint-{steps}'),
        '--split-jsonl', p.feedback_split.path, '--out-dir', str(d / 'evaluation'),
        '--batch-size', '8', '--critical-cer-threshold', '.5', '--max-new-tokens', '256'], 'evaluation')
    save(d / 'paired_parent.json', compare(Path(p.initial_adapter_dev_predictions.path), d / 'evaluation/predictions.jsonl'))
    save(d / 'results.json', dict(status='complete', metrics=read(d / 'evaluation/summary.json')['metrics'], postprocessing_used=False))
    verify(root)


def supervise(base, root):
    def parallel(stage):
        jobs = []
        for gpu, arm in enumerate(['carrier', 'control']):
            command = [sys.executable, str(Path(__file__).resolve()), stage, str(base), '--arm', arm]
            log = (root / arm / f'{stage}.supervisor.log').open('x')
            process = subprocess.Popen(command, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT)
            save(root / arm / f'{stage}.launch.json', dict(pid=process.pid, gpu=gpu, command=command))
            jobs.append((process, log))
        codes = []
        for process, log in jobs:
            codes.append(process.wait())
            log.close()
        if any(codes):
            raise RuntimeError(f'{stage} failed: {codes}; completed arm preserved')
    parallel('generate_sample')
    if not match(root):
        return
    train(root, 'carrier', smoke=True)
    parallel('train')
    results = {arm: read(root / arm / 'results.json') for arm in ['carrier', 'control']}
    save(root / 'results.json', dict(status='complete', arms=results, actual_new_generation=True, actual_new_dpo_training=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'generate_sample', 'match', 'smoke', 'train', 'supervise'])
    parser.add_argument('base', type=Path)
    parser.add_argument('--arm', choices=['carrier', 'control'], default='carrier')
    args = parser.parse_args()
    base = args.base.resolve()
    root = base / 'syllable_reference_probe_v1'
    if args.stage == 'prepare':
        prepare(base, root)
    elif args.stage == 'generate_sample':
        generate_sample(root, args.arm)
    elif args.stage == 'match':
        match(root)
    elif args.stage in ['smoke', 'train']:
        train(root, args.arm, args.stage == 'smoke')
    else:
        supervise(base, root)
