#!/usr/bin/env python3
"""MAS paired natural/odd sentences with matched character bags and recipes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from pydantic import BaseModel, ConfigDict, Field
from openai.lib._pydantic import to_strict_json_schema
import mas_literal_opportunity_selector_v1 as previous
from therapist_harness.phonology import map_text_to_syllables

base = previous.base
PREVIOUS = previous.ROOT
ROOT = base.B / 'semantic_context_matched_DPO_v1'
base.ROOT = ROOT
VOICES = previous.VOICES
PROMPT = base.REPO / 'src/therapist_harness/prompts/transcript_review_semantic_probe_v1.md'


class Contrast(BaseModel):
    model_config = ConfigDict(extra='forbid')
    case_id: str
    natural_text: str
    odd_text: str
    changed_semantic_relation_zh: str


class Proposal(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reflection_on_completed_DPO_zh: str
    hypothesis_zh: str
    contrasts: list[Contrast] = Field(min_length=8, max_length=8)
    observation_against_hypothesis_zh: str


def prepare():
    ROOT.mkdir(exist_ok=False)
    p = base.RunProtocol.model_validate(base.read(PREVIOUS / 'source_protocol.json'))
    old = base.read(PREVIOUS / 'request_context.json')
    stimuli = base.read(PREVIOUS / 'stimuli.json')
    cases = [dict(case_id=f'semctx.{v}.t{i}', voice_group=v, natural_text=s['text'])
             for v in VOICES for i,s in enumerate(stimuli[v])]
    train, dev = base.rows(p.real_train.path), base.rows(p.feedback_split.path)
    context = dict(patients=old['patients'], cases=cases,
        completed_previous_DPO=base.read(PREVIOUS / 'results.json')['summary'],
        previous_actual_opportunities={a:x['aggregate'] for a,x in base.read(PREVIOUS/'delivery_report.json')['arms'].items()},
        real_training_style_examples=sorted({x['clean_gt'] for x in train})[:16],
        forbidden_whole_real_sentences=sorted({x['clean_gt'] for x in train+dev}),
        user_hypothesis='Natural coherent proposals may let ASR fill words from semantic context. Test deliberately odd, weak-context elicitation sentences closer to unusual combinations in real data. This is a hypothesis, not an established cause; real data also contain ordinary sentences.',
        design='Use the supplied 8 natural sentences UNCHANGED. For each, make ONE semantically odd yet pronounceable Cantonese sentence by reordering existing words/constituents. EXACT SAME normalized character multiset and length in each pair, different order. Keep target words intact where possible and preserve resolved ng-coda carrier counts. Swap actors/objects/actions to reduce familiar semantic expectations. Avoid gibberish character scrambling, list-like articulation drills, extra repetitions, phonetic annotations or instructions. Do not add/remove characters. Four patients, two text pairs each, three independently seeded waveforms per text and arm =48 freshly generated audio. Same frozen historical MAS reference/mode/instruction, TTS seeds, sampling4+greedy from initialR-only. No new reference selector. Review accepts intentional oddness equally in both arms. Compare all generated examples; actual training uses matched audio slots where BOTH sides have genuine errors, with identical visit order and counts. All correct skips, one repeated wrong is sufficient. Same32realDPO/91pairs+KL256;24target+8sharedanchors,4steps. No checkpoint promotion unless original real gate passes. More errors/pairs alone is not benefit. Explain what a negative DPO result would mean.')
    base.save(ROOT/'source_protocol.json',p.model_dump(mode='json'))
    base.save(ROOT/'request_context.json',context)
    files=[Path(__file__),PROMPT,base.REPO/'examples/generate_reviewed_sg_semantic_probe_v1.py',
        PREVIOUS/'source_protocol.json',PREVIOUS/'results.json',PREVIOUS/'delivery_report.json',
        PREVIOUS/'stimuli.json',PREVIOUS/'request_context.json']
    hashes={**base.read(PREVIOUS/'sources.json'),**{str(f):base.sha256_file(f) for f in files}}
    assert all(base.sha256_file(Path(f))==h for f,h in hashes.items())
    base.save(ROOT/'sources.json',hashes)
    print(json.dumps(dict(status='prepared',root=str(ROOT),audio=48)),flush=True)


def propose():
    p=base.verify();context=base.read(ROOT/'request_context.json')
    request=dict(model=p.agent.model,reasoning={'effort':'high'},store=False,stream=True,max_output_tokens=5000,
        input=[dict(role='system',content='你是实际MAS proposal Agent。根据最新失败结果，落实用户提出的语义上下文假设。为每个给定自然句设计同字数、同用字但语义关系古怪的粤语对照句。保持词语可读，降低常识补词线索，不把字随机打散。不要重复旧的reference或超参扫描。'),
               dict(role='user',content=json.dumps(context,ensure_ascii=False))],
        text={'format':dict(type='json_schema',name='semantic_context_contrast',strict=True,schema=to_strict_json_schema(Proposal))})
    base.save(ROOT/'request.json',request)
    response=base.execute_therapist_request(p,request);base.save(ROOT/'response.json',base.serialize_response(response))
    assert response.status=='completed'
    proposal=Proposal.model_validate_json(base.terminal_output_text(response))
    base.save(ROOT/'proposal.json',proposal.model_dump())
    compile_proposal(proposal,context,p)
    print(proposal.model_dump_json(indent=2),flush=True)


def compile_proposal(proposal,context,p):
    cases={x['case_id']:x for x in context['cases']}
    assert Counter(x.case_id for x in proposal.contrasts)==Counter(cases)
    patients={x['voice_group']:x for x in context['patients']}
    recipes={x['voice_group']:x for x in base.read(previous.PREVIOUS/'proposal.json')['voices']}
    refs={x['reference_id']:x for x in base.read(base.B/'reference_catalog.json')['cards']}
    forbidden=[base.identity_keys(x) for x in context['forbidden_whole_real_sentences']]
    by_id={x.case_id:x for x in proposal.contrasts}
    validations=[]
    for x in proposal.contrasts:
        assert x.natural_text==cases[x.case_id]['natural_text']
        natural,odd=base.text_normalize(x.natural_text),base.text_normalize(x.odd_text)
        assert natural!=odd and Counter(natural)==Counter(odd),(x.case_id,natural,odd)
        v=cases[x.case_id]['voice_group'];carriers=patients[v]['actual_carrier_characters']
        counts=[]
        for text in [x.natural_text,x.odd_text]:
            keys=base.identity_keys(text)
            assert not any(any(keys[k]==f[k] for k in keys) for f in forbidden)
            mapping=map_text_to_syllables(text)
            counts.append(Counter(pos.character for pos in mapping.positions if pos.resolved and pos.coda=='ng' and pos.character in carriers))
        assert counts[0]==counts[1] and sum(counts[0].values())>=2
        validations.append(dict(case_id=x.case_id,characters=len(natural),carrier_counts=dict(counts[0]),same_character_bag=True))
    for arm in ['natural','odd']:
        out=ROOT/arm;out.mkdir();(out/'audio').mkdir()
        manifest,choices,sidecar=[],[],[]
        for wave in range(3):
            for i in range(2):
                for j,v in enumerate(VOICES):
                    case=f'semctx.{v}.t{i}';x=by_id[case];text=x.natural_text if arm=='natural' else x.odd_text
                    recipe=recipes[v];ref=refs[recipe['reference_id']];sid=f'{case}.w{wave}'
                    manifest.append(dict(sample_id=sid,target_text=text,reference_text=ref['reference_text'],
                        reference_wav=ref['reference_wav'],reference_wav_sha256=ref['reference_wav_sha256'],
                        output_wav=str(out/'audio'/f'{sid}.wav'),seed=2042+wave*8+i*4+j))
                    choices.append(dict(sample_id=sid,reference_id=ref['reference_id'],mode=recipe['mode'],instruction_id='cantonese_zh',
                        reference_wav_sha256=ref['reference_wav_sha256'],target_text_sha256=hashlib.sha256(text.encode()).hexdigest()))
                    sidecar.append(dict(sample_id=sid,case_id=case,wave=wave,voice_group=v,speaker_id=ref['speaker_id'],role='target',
                        literal_carriers=patients[v]['actual_carrier_characters'],reference_id=ref['reference_id'],content_condition=arm))
        base.jsonl(out/'manifest.jsonl',manifest)
        base.save(out/'generation_plan.json',dict(schema_version='generation_plan_v1',choices=choices))
        base.save(out/'role_sidecar.json',dict(samples=sidecar))
        base.load_generation_choices(out/'generation_plan.json',{x['sample_id']:x for x in manifest})
    base.save(ROOT/'plan.json',dict(version='semantic_context_matched_v1',total_new_audio=48,validations=validations,
        mode_labels=dict(MAS='odd weak-semantic context',fixed='natural coherent context'),
        initial_parent_sha256=p.initial_adapter.sha256,
        review_prompt_sha256=base.sha256_file(PROMPT),
        training_policy='Use matched slots with actual preferences on both sides. Round-robin same slot order to24targetvisits plus identical8previousanchors. No error-severity selection or fake negatives. If <3matchedslots, report insufficiency rather than force training. Three-text pass is not word fidelity proof. Same initial parent and frozen DPOreference,real32DPO/91pairs+KL256,4steps and originalgate.',
        manifest_hashes={str(f):base.sha256_file(f) for arm in ['natural','odd'] for f in (ROOT/arm).glob('*.json*')}))


def generate(arm,smoke):
    p=base.verify();c=p.preference_loop
    assert all(base.sha256_file(Path(f))==h for f,h in base.read(ROOT/'plan.json')['manifest_hashes'].items())
    parent=ROOT/arm;out=ROOT/(f'smoke_{arm}' if smoke else arm)
    if smoke:
        out.mkdir(exist_ok=False);(out/'audio').mkdir()
        manifest=base.rows(parent/'manifest.jsonl')[:4]
        for x in manifest:x['output_wav']=str(out/'audio'/Path(x['output_wav']).name)
        base.jsonl(out/'manifest.jsonl',manifest)
        ids={x['sample_id'] for x in manifest};plan=base.read(parent/'generation_plan.json')
        plan['choices']=[x for x in plan['choices'] if x['sample_id'] in ids]
        base.save(out/'generation_plan.json',plan)
    command=[sys.executable,str(base.REPO/'examples/generate_reviewed_sg_semantic_probe_v1.py'),
        '--manifest',str(out/'manifest.jsonl'),'--out-dir',str(out/'generation'),'--generation-plan',str(out/'generation_plan.json'),
        '--review-mode','asr_transcript','--review-asr-worker','--review-allow-pending',
        '--audio-review-model',c.audio_review_model,'--audio-review-base-url',c.audio_review_base_url,
        '--review-asr-model',p.base_model_dir,'--review-asr-adapter',c.review_asr_adapter,
        '--review-asr-adapter-sha256',c.review_asr_adapter_sha256,'--review-asr-base-model-sha256',c.review_asr_base_model_sha256]
    if not smoke:
        source=ROOT/f'smoke_{arm}'/'generation';files=sorted(source.rglob('assignment_result.json'));assert len(files)==4
        base.save(out/'reuse.json',dict(assignments=[dict(sample_id=base.read(f)['sample_id'],assignment_result=str(f),
            assignment_result_sha256=base.sha256_file(f),provenance='Same semantic-context execution smoke; identical new review prompt. Identity/audio/model hashes verified.') for f in files]))
        command+=['--reuse-manifest',str(out/'reuse.json')]
    base.save(out/'generation_launch.json',dict(command=command,start=time.time()))
    with (out/'generation.log').open('x') as log:subprocess.run(command,cwd=base.REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    base.verify();print(json.dumps(dict(stage='generated',arm=arm,smoke=smoke)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','propose','generate','sample'])
    parser.add_argument('--arm',choices=['natural','odd']);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.stage in ['prepare','propose']:globals()[args.stage]()
    elif args.stage=='generate':assert args.arm;generate(args.arm,args.smoke)
    else:assert args.arm and not args.smoke;base.sample(args.arm)
