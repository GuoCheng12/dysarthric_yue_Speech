#!/usr/bin/env python3
"""Patient-grounded literal exercises and separate-waveform recipe selection."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from pydantic import BaseModel, ConfigDict, Field
from openai.lib._pydantic import to_strict_json_schema
import mas_literal_matched_sg_probe_v1 as base
from therapist_harness.evidence import _read_snapshot
from therapist_harness.phonology import map_text_to_syllables
from therapist_harness.schema import Stimulus

PREVIOUS = base.B / 'contextual_recipe_DPO24_v1'
ROOT = base.B / 'literal_opportunity_selector_DPO_v1'
base.ROOT = ROOT
VOICES = ['voice004', 'voice005', 'voice009', 'voice023']


class Exercise(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str
    rationale_zh: str


class PatientPractice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    voice_group: str
    exercises: list[Exercise] = Field(min_length=2, max_length=2)


class Proposal(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reflection_zh: str
    practices: list[PatientPractice] = Field(min_length=4, max_length=4)
    what_this_tests_zh: str


def prepare():
    ROOT.mkdir(exist_ok=False)
    p = base.RunProtocol.model_validate(base.read(PREVIOUS / 'source_protocol.json'))
    previous = base.read(PREVIOUS / 'results.json')
    assert previous['summary']['MAS']['all']['errors'] == 1401
    assert not previous['summary']['MAS']['passes_original_gate']
    assert not previous['summary']['fixed']['passes_original_gate']
    baseline = base.B / 'parent_eval_check/predictions.jsonl'
    before = _read_snapshot(base.rows(baseline), role='current', prediction_column='prediction', critical_cer_threshold=.5)
    catalog = base.read(base.B / 'reference_catalog.json')['cards']
    speaker_to_voice = {x['speaker_id']: x['voice_group'] for x in catalog}
    actual = defaultdict(list); count = 0
    for row in before:
        for e in row.phone_events:
            if e.context_sensitive is not False or e.operation != 'substitution' or e.level != 'coda': continue
            if (e.reference_unit, e.hypothesis_unit) != ('ng', 'n'): continue
            count += 1; voice = speaker_to_voice[row.speaker_id]
            if voice in VOICES:
                actual[voice].append(dict(target=row.clean_gt, prediction=row.hypothesis,
                    content_id=row.content_id, utterance_id=row.utt_id, event=asdict(e)))
    assert count == 24 and set(actual) == set(VOICES)
    patients = [dict(voice_group=v, actual_events=actual[v],
        actual_carrier_characters=sorted({e['event']['reference_character'] for e in actual[v]})) for v in VOICES]
    real, dev = base.rows(p.real_train.path), base.rows(p.feedback_split.path)
    forbidden = sorted({x['clean_gt'] for x in real + dev} |
        {x['text'] for x in base.read(PREVIOUS / 'stimuli.json')})
    context = dict(patients=patients, previous_real_results=previous['summary'],
        actual_MAS_reflection=base.read(PREVIOUS / 'MAS_reflection/reflection.json'),
        bounded_execution=base.read(PREVIOUS / 'next_selector_minimal_plan.json'),
        forbidden_whole_sentences=forbidden,
        task='For eachofthese4patients write2newnaturalTraditionalCantonese sentences using their own actualng->n referencecharacters and meaningfulwordcontexts. Eachsentence containsatleast2occurrences ofthatpatient\'s listedcharacters withresolvedngcoda; ifonlyonecharacterexists, repeatit naturallytwice, do notinventanotherpatientcarrier. Includeatleastone naturaln-coda word forobservingreverseerrors. Aim12-20characters,max26normalizedcharacters. No articulatory/clinical claims or mouthinstructions; theseare ordinarycontenttexts. Do notreusewholetrain/dev/previoussentences. Recipes are alreadyfixed tothispatient\'s previousMASversusfixedrecipes; do notchooseadditionalmodes/voices or inventacousticfeatures.4patients2texts2recipes4waveforms=64newaudio; wave0selectsrecipe usingactualASRliteral/codaerrors, waves1-3heldout fromselectionandcantrain. Standard4ASRdraws/audio. Reviewunchangedsimplethree-text. ThisisthenextboundedDPOexperiment,notanotherhyperparameterscan. OriginalcheckpointgateandcommonrealDPO32/KL256 remainunchanged; correctreflection\'s tighterhardlimit andoldRcontrolcount asnotedinboundedexecution.')
    base.save(ROOT / 'source_protocol.json', p.model_dump(mode='json'))
    base.save(ROOT / 'request_context.json', context)
    files = [Path(__file__), Path(base.__file__), PREVIOUS / 'source_protocol.json', PREVIOUS / 'sources.json',
        PREVIOUS / 'results.json', PREVIOUS / 'MAS_reflection/reflection.json', PREVIOUS / 'next_selector_minimal_plan.json',
        baseline, base.B / 'reference_catalog.json', Path(p.real_train.path), Path(p.feedback_split.path)]
    hashes = {**base.read(PREVIOUS / 'sources.json'), **{str(f): base.sha256_file(f) for f in files}}
    assert all(base.sha256_file(Path(f)) == h for f, h in hashes.items())
    base.save(ROOT / 'sources.json', hashes)
    print(json.dumps({x['voice_group']: x['actual_carrier_characters'] for x in patients}, ensure_ascii=False), flush=True)


def propose():
    p = base.verify(); context = base.read(ROOT / 'request_context.json')
    request = dict(model=p.agent.model, reasoning={'effort':'high'}, store=False, stream=True, max_output_tokens=6000,
        input=[dict(role='system', content='你是实际MAS内容Agent。根据每个患者真实发生过的字级错误写具体自然新粤语句子，不能用其他患者的错误或一般ng词代替该患者的literal carriers。保持简单SG和既定有限实验。'),
            dict(role='user', content=json.dumps(context, ensure_ascii=False))],
        text={'format':dict(type='json_schema', name='patient_literal_opportunity_proposal', strict=True, schema=to_strict_json_schema(Proposal))})
    base.save(ROOT / 'request.json', request)
    response = base.execute_therapist_request(p, request); base.save(ROOT / 'response.json', base.serialize_response(response))
    assert response.status == 'completed'
    proposal = Proposal.model_validate_json(base.terminal_output_text(response))
    base.save(ROOT / 'proposal.json', proposal.model_dump())
    compile_proposal(proposal, context, p)
    print(proposal.model_dump_json(indent=2), flush=True)


def compile_proposal(proposal, context, p):
    assert Counter(x.voice_group for x in proposal.practices) == Counter(VOICES)
    practices = {x.voice_group: x for x in proposal.practices}
    patients = {x['voice_group']: x for x in context['patients']}
    recipes = {x['voice_group']: x for x in base.read(PREVIOUS / 'proposal.json')['voices']}
    refs = {x['reference_id']: x for x in base.read(base.B / 'reference_catalog.json')['cards']}
    forbidden = [base.identity_keys(x) for x in context['forbidden_whole_sentences']]
    stimuli = {}
    for v in VOICES:
        carriers = set(patients[v]['actual_carrier_characters'])
        stimuli[v] = []
        for i, x in enumerate(practices[v].exercises):
            s = Stimulus(stimulus_id=f'litopp.{v}.t{i}', text=x.text, role='target',
                evidence_ids=['coda-sub-f19ef0ec14f4'], speaker_profile_id='all_patients_balanced',
                copies=8, rationale=x.rationale_zh)
            assert len(base.text_normalize(s.text)) <= 26
            keys = base.identity_keys(s.text)
            assert not any(any(keys[k] == f[k] for k in keys) for f in forbidden)
            mapping = map_text_to_syllables(s.text)
            assert sum(pos.resolved and pos.coda == 'ng' and pos.character in carriers for pos in mapping.positions) >= 2, (v, s.text)
            assert any(pos.resolved and pos.coda == 'n' for pos in mapping.positions), s.text
            stimuli[v].append(s.model_dump(mode='json'))
    assert len({base.text_normalize(x['text']) for ss in stimuli.values() for x in ss}) == 8
    for arm in ['history', 'alternate']:
        out = ROOT / arm; out.mkdir(); (out / 'audio').mkdir()
        manifest, choices, sidecar = [], [], []
        for wave in range(4):
            for i in range(2):
                for j, v in enumerate(VOICES):
                    recipe = recipes[v]; rid = recipe['reference_id'] if arm == 'history' else v + '.ref0'
                    mode = recipe['mode'] if arm == 'history' else 'instruct2'; ref = refs[rid]
                    text = stimuli[v][i]['text']; sid = f'litopp.{v}.t{i}.w{wave}'
                    manifest.append(dict(sample_id=sid, target_text=text, reference_text=ref['reference_text'],
                        reference_wav=ref['reference_wav'], reference_wav_sha256=ref['reference_wav_sha256'],
                        output_wav=str(out / 'audio' / f'{sid}.wav'), seed=42+wave*8+i*4+j))
                    choices.append(dict(sample_id=sid, reference_id=rid, mode=mode, instruction_id='cantonese_zh',
                        reference_wav_sha256=ref['reference_wav_sha256'], target_text_sha256=hashlib.sha256(text.encode()).hexdigest()))
                    sidecar.append(dict(sample_id=sid, case_id=f'litopp.{v}.t{i}', stimulus_id=stimuli[v][i]['stimulus_id'],
                        wave=wave, split='selector' if wave == 0 else 'training', voice_group=v, speaker_id=ref['speaker_id'],
                        role='target', literal_carriers=patients[v]['actual_carrier_characters'], reference_id=rid))
        base.jsonl(out / 'manifest.jsonl', manifest)
        base.save(out / 'generation_plan.json', dict(schema_version='generation_plan_v1', choices=choices))
        base.save(out / 'role_sidecar.json', dict(samples=sidecar))
        base.load_generation_choices(out / 'generation_plan.json', {x['sample_id']:x for x in manifest})
    base.save(ROOT / 'stimuli.json', stimuli)
    base.save(ROOT / 'plan.json', dict(version='literal_opportunity_selector_v1', audio_per_recipe=32, total_new_audio=64,
        initial_parent_sha256=p.initial_adapter.sha256,
        source_protocol_scope='64audio/4patient external plan replaces source96materialization budget; fixed model/data/review/decoding retained.',
        selection='Perpatient/text: countdistinctactualwrongpairs onwave0 containingconservative ng->n eventatoneofthatpatient\'s realliteralreferencecharacters. Morequalifyingpairs wins; tiehistory. Noheldoutwaves1-3 readbyselector. AllASRdraws areactual4/currentinitialR-only.',
        training='Historyuseshistoryrecipes;opportunityusesselectedrecipes. Onlywaves1-3cantrain; commonpreviousMASprotection/maintenance8audio,SG24target+4protection+4maintenancevisits. SameR32audio91pairs+KL256;4steps andoriginalsettings/gate. Ownactualeligiblepairs, nofake negatives ornewSGthreshold.',
        boundedness='One64audiotest. Identicalrecipechoices or noactualliteraltrainingopportunities mustnotbe representedas informative different-arm DPO. No extra grid/closedpool revival.',
        manifest_hashes={str(f):base.sha256_file(f) for a in ['history','alternate'] for f in (ROOT/a).glob('*.json*')}))
    base.verify()


def generate(arm, smoke):
    p = base.verify(); c = p.preference_loop
    assert all(base.sha256_file(Path(f)) == h for f,h in base.read(ROOT / 'plan.json')['manifest_hashes'].items())
    parent = ROOT / arm; out = ROOT / (f'smoke_{arm}' if smoke else arm)
    if smoke:
        out.mkdir(exist_ok=False); (out / 'audio').mkdir()
        manifest = base.rows(parent / 'manifest.jsonl')[:4]
        assert len(manifest) == 4 and all(x['sample_id'].endswith('.t0.w0') for x in manifest)
        for x in manifest: x['output_wav'] = str(out / 'audio' / Path(x['output_wav']).name)
        base.jsonl(out / 'manifest.jsonl', manifest)
        ids = {x['sample_id'] for x in manifest}; plan = base.read(parent / 'generation_plan.json')
        plan['choices'] = [x for x in plan['choices'] if x['sample_id'] in ids]
        base.save(out / 'generation_plan.json', plan)
    command = [sys.executable, str(base.REPO / 'examples/generate_reviewed_sg_recovery_v2.py'),
        '--manifest',str(out/'manifest.jsonl'),'--out-dir',str(out/'generation'),'--generation-plan',str(out/'generation_plan.json'),
        '--review-mode','asr_transcript','--review-asr-worker','--review-allow-pending',
        '--audio-review-model',c.audio_review_model,'--audio-review-base-url',c.audio_review_base_url,
        '--review-asr-model',p.base_model_dir,'--review-asr-adapter',c.review_asr_adapter,
        '--review-asr-adapter-sha256',c.review_asr_adapter_sha256,'--review-asr-base-model-sha256',c.review_asr_base_model_sha256]
    if not smoke:
        previous = ROOT / f'smoke_{arm}' / 'generation'
        assert len(base.read(previous/'generation_summary.json')['outputs']) == 4
        files = sorted(previous.rglob('assignment_result.json')); assert len(files) == 4
        base.save(out/'reuse.json',dict(assignments=[dict(sample_id=base.read(f)['sample_id'],assignment_result=str(f),
            assignment_result_sha256=base.sha256_file(f),provenance='Same execution fourpatient smoke; identity/audio/review hashes verified by generator.') for f in files]))
        command += ['--reuse-manifest',str(out/'reuse.json')]
    base.save(out/'generation_launch.json',dict(command=command,start=time.time(),entry_sha256=base.sha256_file(Path(__file__))))
    with (out/'generation.log').open('x') as log:
        subprocess.run(command,cwd=base.REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    base.verify(); print(json.dumps(dict(stage='generated',arm=arm,smoke=smoke)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['prepare','propose','generate','sample'])
    parser.add_argument('--arm',choices=['history','alternate']); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.stage in ['prepare','propose']: globals()[args.stage]()
    elif args.stage == 'generate': assert args.arm; generate(args.arm,args.smoke)
    else: assert args.arm and not args.smoke; base.sample(args.arm)
