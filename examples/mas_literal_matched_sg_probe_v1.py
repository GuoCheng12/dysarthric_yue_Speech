#!/usr/bin/env python3
"""New literal-character MAS proposal and a 24-audio matched-reference probe."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from openai.lib._pydantic import to_strict_json_schema

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from therapist_harness.agent import execute_therapist_request, serialize_response, terminal_output_text
from therapist_harness.generation_controller import load_catalog, public_cards
from therapist_harness.generation_design import load_generation_choices
from therapist_harness.schema import EvidencePack, RunProtocol
from therapist_harness.store import sha256_file, write_immutable_json, write_immutable_bytes
from therapist_harness.text import identity_keys, text_normalize

B = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_design_exploration_20260908_v1')
DONOR = Path('/tmp/mas_kl_dpo_loop_v3_formal96')
ROOT = B / 'literal_matched_SG24_v1'
read = lambda p: json.loads(Path(p).read_text())
rows = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
save = write_immutable_json


class SentencePair(BaseModel):
    model_config = ConfigDict(extra='forbid')
    evidence_id: str
    target_text: str = Field(min_length=6, max_length=25)
    counter_text: str = Field(min_length=6, max_length=25)
    reason_zh: str


class VoiceChoice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    voice_group: str
    preferred_reference_id: str
    comparator_reference_id: str
    observed_basis_zh: str
    expected_reference_effect_zh: str


class MatchedProposal(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reflection_zh: str
    hypothesis_zh: str
    pairs: list[SentencePair] = Field(min_length=3, max_length=3)
    voices: list[VoiceChoice] = Field(min_length=2, max_length=2)
    observation_against_hypothesis_zh: str


def jsonl(path, values):
    write_immutable_bytes(path, ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in values).encode())


def verify():
    hashes = read(ROOT / 'sources.json')
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    return RunProtocol.model_validate(read(ROOT / 'source_protocol.json'))


def propose():
    ROOT.mkdir(exist_ok=False)
    p = RunProtocol.model_validate(read(DONOR / 'protocol.json'))
    evidence = EvidencePack.model_validate(read(DONOR / 'round_001/evidence.json'))
    assert evidence.snapshots.current.checkpoint_sha256 == p.initial_adapter.sha256
    closure = read(B / 'fixed_subset_S_DPO_real_KL_v1/S_KL/result.json')
    assert not closure['supports_real_DPO_masking'] and not closure['passes_original_gate']
    catalog = B / 'reference_catalog.json'
    real, dev = rows(p.real_train.path), rows(p.feedback_split.path)
    cards = load_catalog(dict(catalog_path=str(catalog), catalog_sha256=sha256_file(catalog)), real)
    assert not {str(Path(x['reference_wav']).resolve()) for x in cards.values()} & {str(Path(x['audio']).resolve()) for x in dev}
    before = rows(B / 'parent_eval_check/predictions.jsonl')
    glyphs = [x.model_dump(mode='json') for x in evidence.error_clusters
              if x.level == 'character' and x.operation == 'substitution' and x.targetable and x.tts_actionable]
    history_path = B / 'reference_experience_v2/MAS_context.json'
    history = read(history_path)
    patient_cards = []
    for voice in sorted({x['voice_group'] for x in cards.values()}):
        card = next(x for x in cards.values() if x['voice_group'] == voice)
        subset = [x for x in before if x['speaker_id'] == card['speaker_id']]
        patient_cards.append(dict(voice_group=voice, difficulty_group=card['difficulty_group'],
            dev_audio=len(subset), dev_errors=sum(x['edit_count'] for x in subset),
            dev_characters=sum(x['normalized_reference_characters'] for x in subset),
            train_audio=sum(x['speaker_id'] == card['speaker_id'] for x in real)))
    compact_history = [{k: x[k] for k in ['reference_id', 'observed_unique_audio', 'admitted_audio',
        'required_retry_audio', 'review_verdicts', 'actual_recipes', 'sampling_by_protocol']} for x in history['cards']]
    context = dict(current_parent='initial R-only:1384/6656 errors,hard688/1248; no endpoint promoted',
        fixed_subset_closed=closure, previous_summary='Fixed random8 R+SG DPO optimization42/43/44 errors1382/1390/1410; initial1384. MAS utility8 selection1409. Removing realDPO gives1382/hard695, no gate. Existing i->yu text signatures did not establish intended correction. Do not reuse that subset or tune adjacent loss/lr/freeze/ranking knobs.',
        literal_error_options=glyphs, reference_cards=public_cards(cards), patient_cards=patient_cards,
        historical_reference_observations=compact_history, history_semantics=history['semantics'],
        forbidden_whole_feedback_sentences=sorted({x['clean_gt'] for x in dev}),
        forbidden_whole_training_sentences=sorted({x['clean_gt'] for x in real}),
        design='3 new natural target/counter sentence pairs x2patients x2within-patient references =24audio. References are preferredA vs comparatorB chosen prospectively. All cells fixed instruct2+cantonese_zh, same per-case TTS seed and actual ASR sampling seeds;4 stochastic draws+separate greedy from unchangedinitial. No training until actual delivery report. New matched content tests literal target delivery, not phoneme fidelity.',
        constraints='Choose only supplied substitution IDs, one or several IDs may be used across3pairs. Target sentence must contain reference character; counter sentence must naturally use hypothesis character as correct, not call it a mistake. Prefer8-18 Chinese characters and everyday Cantonese; maximum24 surface characters and terminal punctuation. Each pair should be semantically coherent examples, not packed drills. Exactly2different patients with dev feedback, one hard and one easy/medium; use both catalog references for each. Give evidence-based prospective reference preference; no clinical/acoustic inference from text, duration, patient label or ASR difficulty. More wrong candidates is not utility. Keep existing three-text obvious-failure review and rulebook, no CER admission threshold. All-correct skips; identical wrong can form1pair. Every reference is train-only. This is a probe, not511CER efficacy; final training retains initial reference/original checkpoint gate.')
    save(ROOT / 'source_protocol.json', p.model_dump(mode='json'))
    save(ROOT / 'evidence.json', evidence.model_dump(mode='json'))
    save(ROOT / 'request_context.json', context)
    files = [Path(__file__), DONOR / 'protocol.json', DONOR / 'round_001/evidence.json', catalog, history_path,
        B / 'fixed_subset_S_DPO_real_KL_v1/S_KL/result.json', Path(p.real_train.path), Path(p.feedback_split.path),
        B / 'parent_eval_check/predictions.jsonl']
    files += [REPO / x for x in ['examples/generate_reviewed_sg_recovery_v2.py', 'examples/generate_reviewed_sg_resumable.py',
        'examples/sample_asr_preferences.py', 'examples/transcribe_sg_review.py', 'src/therapist_harness/review_worker.py',
        'src/therapist_harness/transcript_review.py', 'src/therapist_harness/prompts/transcript_review_v2.md',
        'src/therapist_harness/generation_design.py', 'src/therapist_harness/preference_data.py',
        'synthesis/cosyvoice3_patient_sft/synthesize_manifest.py']]
    save(ROOT / 'sources.json', {str(f): sha256_file(f) for f in files})
    request = dict(model=p.agent.model, reasoning={'effort': 'high'}, store=False, stream=True, max_output_tokens=6000,
        input=[dict(role='system', content='你是实际 MAS 的内容与 reference 设计者。固定旧子集已关闭。依据当前真实字错、训练录音事实和历史生成记录，设计一个新24音频配对探针。选择必须可解释、可推翻；不能宣称已有性能提升或已听过音频。返回简洁中文理由和自然繁体粤语句子。'),
               dict(role='user', content=json.dumps(context, ensure_ascii=False))],
        text={'format': dict(type='json_schema', name='literal_matched_SG24_proposal', strict=True,
                             schema=to_strict_json_schema(MatchedProposal))})
    save(ROOT / 'request.json', request)
    response = execute_therapist_request(p, request)
    save(ROOT / 'response.json', serialize_response(response))
    assert response.status == 'completed'
    decision = MatchedProposal.model_validate_json(terminal_output_text(response))
    save(ROOT / 'proposal.json', decision.model_dump())
    compile_proposal(p, decision, cards, glyphs, real, dev)
    verify()
    print(decision.model_dump_json(indent=2), flush=True)


def compile_proposal(p, decision, cards, glyphs, real, dev):
    known = {x['evidence_id']: x for x in glyphs}
    voices = decision.voices
    assert len({x.voice_group for x in voices}) == 2
    selected = []
    for v in voices:
        a, b = cards[v.preferred_reference_id], cards[v.comparator_reference_id]
        assert a['voice_group'] == b['voice_group'] == v.voice_group and a['speaker_id'] == b['speaker_id']
        assert a['reference_wav_sha256'] != b['reference_wav_sha256']
        assert any(x['speaker_id'] == a['speaker_id'] for x in dev)
        selected.append((a, b))
    assert sum(a['difficulty_group'] == 'hard' for a, b in selected) == 1
    forbidden = [identity_keys(t) for t in {x['clean_gt'] for x in real + dev}]
    all_texts = [t for pair in decision.pairs for t in [pair.target_text, pair.counter_text]]
    assert len({text_normalize(t) for t in all_texts}) == 6
    for pair in decision.pairs:
        event = known[pair.evidence_id]
        assert event['reference_unit'] in text_normalize(pair.target_text)
        assert event['hypothesis_unit'] in text_normalize(pair.counter_text)
        for t in [pair.target_text, pair.counter_text]:
            assert len(text_normalize(t)) <= 24 and t[-1] in '。？！!?'
            key = identity_keys(t)
            assert not any(any(key[k] == f[k] for k in key) for f in forbidden), t
    for arm, ref_index in [('A', 0), ('B', 1)]:
        out = ROOT / arm; out.mkdir(); (out / 'audio').mkdir()
        manifest, choices, sidecar = [], [], []
        for i, pair in enumerate(decision.pairs):
            event = known[pair.evidence_id]
            for role, text in [('target', pair.target_text), ('counter', pair.counter_text)]:
                for v, refs in enumerate(selected):
                    card = refs[ref_index]
                    sid = f'literal24.p{i}.{role}.voice{v}'
                    seed = 42 + i * 4 + (role == 'counter') * 2 + v
                    manifest.append(dict(sample_id=sid, target_text=text, reference_text=card['reference_text'],
                        reference_wav=card['reference_wav'], reference_wav_sha256=card['reference_wav_sha256'],
                        output_wav=str(out / 'audio' / f'{sid}.wav'), seed=seed))
                    choices.append(dict(sample_id=sid, reference_id=card['reference_id'], mode='instruct2',
                        instruction_id='cantonese_zh', reference_wav_sha256=card['reference_wav_sha256'],
                        target_text_sha256=hashlib.sha256(text.encode()).hexdigest()))
                    sidecar.append(dict(sample_id=sid, pair_index=i, role=role, reference_id=card['reference_id'],
                        speaker_id=card['speaker_id'], difficulty_group=card['difficulty_group'], evidence_id=pair.evidence_id,
                        intended_reference=event['reference_unit'], intended_hypothesis=event['hypothesis_unit']))
        jsonl(out / 'manifest.jsonl', manifest)
        save(out / 'generation_plan.json', dict(schema_version='generation_plan_v1', choices=choices))
        save(out / 'role_sidecar.json', dict(schema_version='matched_literal_probe_roles_v1', samples=sidecar))
        load_generation_choices(out / 'generation_plan.json', {x['sample_id']: x for x in manifest})
        smoke = ROOT / f'smoke_{arm}'; smoke.mkdir(); (smoke / 'audio').mkdir()
        jsonl(smoke / 'manifest.jsonl', [dict(manifest[0], output_wav=str(smoke / 'audio' / f'{manifest[0]["sample_id"]}.wav'))])
        save(smoke / 'generation_plan.json', dict(schema_version='generation_plan_v1', choices=choices[:1]))
    save(ROOT / 'plan.json', dict(schema_version='matched_literal_SG_probe_v1', planned_audio=24,
        parent_sha256=p.initial_adapter.sha256, DPO_reference_sha256=p.initial_adapter.sha256,
        paired_factors='Same6texts,2patients,logical IDs,TTS seeds,ASR seeds and initial instruct2+cantonese_zh; change only within-patient reference betweenA/B.',
        scope='Bounded external proposal adapter reuses unchanged core EvidencePack,SG,review,sampling,pair builder. Counter means competing character is correct, not the core contrast role requiring same-reference comparator. No fabricated core ledger commit.',
        review='Unchanged three-text coarse Agent and finite rulebook; no CER threshold or word/phoneme fidelity assertion.',
        analysis='Compare first attempts separately from rulebook finals. Report literal intended substitutions and correct counter-carrier retention using actual candidates, paired-audio availability and review outcomes. No training-value claim from error counts.',
        next_training='After delivery: actual MAS chooses bounded training allocation; separateA/B source receipts retained; commoninitialparent and frozeninitialreference, matched real exposure order and original511gate. No automatic checkpoint promotion.',
        smoke='One actual generated/reviewed waveform perarm, then verified immutable reuse inside24plannedaudio; no extra training sample.',
        manifest_hashes={str(x): sha256_file(x) for arm in ['A','B','smoke_A','smoke_B'] for x in (ROOT / arm).glob('*.json*')}))


def generate(arm, smoke):
    p = verify(); c = p.preference_loop
    assert all(sha256_file(Path(f)) == h for f, h in read(ROOT / 'plan.json')['manifest_hashes'].items())
    out = ROOT / (f'smoke_{arm}' if smoke else arm)
    command = [sys.executable, str(REPO / 'examples/generate_reviewed_sg_recovery_v2.py'),
        '--manifest', str(out / 'manifest.jsonl'), '--out-dir', str(out / 'generation'),
        '--generation-plan', str(out / 'generation_plan.json'), '--review-mode', 'asr_transcript',
        '--review-asr-worker', '--review-allow-pending', '--audio-review-model', c.audio_review_model,
        '--audio-review-base-url', c.audio_review_base_url, '--review-asr-model', p.base_model_dir,
        '--review-asr-adapter', c.review_asr_adapter, '--review-asr-adapter-sha256', c.review_asr_adapter_sha256,
        '--review-asr-base-model-sha256', c.review_asr_base_model_sha256]
    if not smoke:
        previous = ROOT / f'smoke_{arm}' / 'generation'
        summary = read(previous / 'generation_summary.json')
        assert len(summary['outputs']) == 1
        source = next(previous.rglob('assignment_result.json'))
        old = read(source)
        assert old['generation_mode'] == 'instruct2' and old['effective_instruction_id'] == 'cantonese_zh'
        save(out / 'reuse.json', dict(assignments=[dict(sample_id=old['sample_id'], assignment_result=str(source),
            assignment_result_sha256=sha256_file(source), provenance='Same execution smoke, completed reviewed assignment; identity/audio checked by generator.')]))
        command += ['--reuse-manifest', str(out / 'reuse.json')]
    save(out / 'generation_launch.json', dict(command=command, start=time.time()))
    with (out / 'generation.log').open('x') as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify()
    print(json.dumps(dict(stage='generation_complete', arm=arm, smoke=smoke,
        summary={k:v for k,v in read(out / 'generation/generation_summary.json').items() if k in ['output_count','admitted_count','timing','elapsed_seconds']})), flush=True)


def sample(arm):
    p = verify(); c = p.preference_loop; out = ROOT / arm
    command = [sys.executable, str(REPO / 'examples/sample_asr_preferences.py'), 'sample',
        '--inputs', str(out / 'generation/accepted.jsonl'), '--real-train', p.real_train.path,
        '--feedback', p.feedback_split.path, '--out-dir', str(out / 'sampling'), '--base-model', p.base_model_dir,
        '--current-adapter', p.initial_adapter.path, '--expected-parent-sha256', p.initial_adapter.sha256,
        '--samples', '4', '--seed', '42', '--temperature', str(c.temperature), '--top-p', str(c.top_p),
        '--max-new-tokens', str(c.max_new_tokens), '--include-greedy']
    save(out / 'sampling_launch.json', dict(command=command, start=time.time()))
    with (out / 'sampling.log').open('x') as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify()
    print(json.dumps(dict(stage='sampling_complete', arm=arm)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['propose','generate','sample'])
    parser.add_argument('--arm', choices=['A','B'], default='A')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage == 'propose': propose()
    elif args.stage == 'generate': generate(args.arm, args.smoke)
    else: sample(args.arm)
