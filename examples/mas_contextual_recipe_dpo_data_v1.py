#!/usr/bin/env python3
"""New error-context/recipe-aware MAS data for a matched-budget DPO experiment."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel,ConfigDict,Field
from openai.lib._pydantic import to_strict_json_schema
import mas_literal_matched_sg_probe_v1 as base
from therapist_harness.evidence import _read_snapshot,_cluster_id,_encode_unit
from therapist_harness.phonology import map_text_to_syllables
from therapist_harness.schema import Stimulus

ROOT=base.B/'contextual_recipe_DPO24_v1'
base.ROOT=ROOT
FAMILIES=[('onset','k','g'),('onset','s','z'),('nucleus','aa','a'),('nucleus','yu','i'),
          ('coda','ng','n'),('coda','n','ng'),('tone','3','6'),('tone','6','3')]


class Exercise(BaseModel):
    model_config=ConfigDict(extra='forbid')
    role:Literal['target','contrast','protection','maintenance']
    text:str
    evidence_ids:list[str]
    reason_zh:str


class VoiceRecipe(BaseModel):
    model_config=ConfigDict(extra='forbid')
    voice_group:str
    reference_id:str
    mode:Literal['zero_shot','instruct2']
    instruction_id:Literal['cantonese_en','cantonese_zh']
    evidence_basis_zh:str


class RoleBudget(BaseModel):
    model_config=ConfigDict(extra='forbid')
    target:int=Field(ge=1)
    contrast:int=Field(ge=1)
    protection:int=Field(ge=1)
    maintenance:int=Field(ge=1)


class Proposal(BaseModel):
    model_config=ConfigDict(extra='forbid')
    reflection_zh:str
    primary_evidence_id:str
    hypothesis_zh:str
    exercises:list[Exercise]=Field(min_length=6,max_length=6)
    voices:list[VoiceRecipe]=Field(min_length=4,max_length=4)
    intended_role_audio_visits:RoleBudget
    observation_against_hypothesis_zh:str


def prepare():
    ROOT.mkdir(exist_ok=False)
    p=base.RunProtocol.model_validate(base.read(base.DONOR/'protocol.json'))
    evidence=base.EvidencePack.model_validate(base.read(base.DONOR/'round_001/evidence.json'))
    assert evidence.snapshots.current.checkpoint_sha256==p.initial_adapter.sha256
    recipe_path=base.B/'reference_recipe_experience_v3/MAS_context.json'
    recipe=base.read(recipe_path);receipt=base.read(recipe_path.parent/'receipt.json')
    assert all(base.sha256_file(Path(f))==h for f,h in {**receipt['source_sha256'],**receipt['output_sha256']}.items())
    catalog_path=base.B/'reference_catalog.json'
    real,dev=base.rows(p.real_train.path),base.rows(p.feedback_split.path)
    cards=base.load_catalog(dict(catalog_path=str(catalog_path),catalog_sha256=base.sha256_file(catalog_path)),real)
    by_speaker={x['speaker_id']:x['voice_group'] for x in cards.values()}
    before_path=base.B/'parent_eval_check/predictions.jsonl';before=base.rows(before_path)
    utterances=_read_snapshot(before,role='current',prediction_column='prediction',critical_cer_threshold=p.evaluation.critical_cer_threshold)
    assert sum(x.edit_count for x in utterances)==1384
    known={x.evidence_id:x for x in evidence.error_clusters}
    family_ids={_cluster_id(('substitution',level,a,b,None,None)) for level,a,b in FAMILIES}
    contexts=defaultdict(list)
    for row in utterances:
        for event in row.phone_events:
            if event.context_sensitive is not False:continue
            eid=_cluster_id((event.operation,event.level,_encode_unit(event.reference_unit),_encode_unit(event.hypothesis_unit),None,None))
            if eid not in family_ids:continue
            ref,hyp=row.phonology.reference.text,row.phonology.hypothesis.text
            contexts[eid].append(dict(voice_group=by_speaker.get(row.speaker_id,'no_reference'),
                content_id=row.content_id,target=row.clean_gt,prediction=row.hypothesis,
                target_window=ref[max(0,event.reference_index-3):event.reference_index+4],
                hypothesis_window=hyp[max(0,event.hypothesis_index-3):event.hypothesis_index+4],
                reference_character=event.reference_character,hypothesis_character=event.hypothesis_character,
                reference_jyutping=event.reference_jyutping,hypothesis_jyutping=event.hypothesis_jyutping,
                local_event=asdict(event)))
    family_cards=[]
    for eid in sorted(family_ids):
        cluster=known[eid];events=contexts[eid]
        assert cluster.targetable and cluster.tts_actionable and len(events)==cluster.observations.current.error_count
        examples=[];counts=Counter()
        for event in events:
            key=(event['voice_group'],event['content_id'])
            if counts[key]<2:examples.append(event);counts[key]+=1
        family_cards.append(dict(cluster=cluster.model_dump(mode='json'),actual_contexts=examples,
            patient_event_counts=dict(Counter(x['voice_group'] for x in events)),
            literal_event_counts=[dict(reference=k[0],hypothesis=k[1],count=v) for k,v in Counter((x['reference_character'],x['hypothesis_character']) for x in events).most_common()],
            scope='These are conservative text-derived phone-component error signatures, not clinical/acoustic measurement. Repeated samecontent examples are labelled; allsupportcounts match coreEvidencePack.'))
    eligible={x['voice_group'] for values in contexts.values() for x in values}-{'no_reference'}
    patient_cards=[]
    for voice in sorted(eligible):
        card=next(x for x in cards.values() if x['voice_group']==voice)
        subset=[x for x in utterances if x.speaker_id==card['speaker_id']]
        patient_cards.append(dict(voice_group=voice,reference_difficulty_group=card['difficulty_group'],
            dev_audio=len(subset),dev_errors=sum(x.edit_count for x in subset),dev_characters=sum(len(x.reference) for x in subset),
            dev_utterance_difficulty_counts=dict(Counter(x.difficulty for x in subset)),
            training_audio=sum(x['speaker_id']==card['speaker_id'] for x in real)))
    compact=[]
    condition_fields=['mode','instruction_id','legacy_final_audio','legacy_final_verdicts','legacy_final_audio_with_prior_attempts',
        'new_actual_attempts','new_first_attempts','new_first_attempt_verdicts','new_all_attempt_verdicts','new_final_audio','new_final_admitted']
    for card in recipe['cards']:
        if card['voice_group'] not in eligible:continue
        conditions=[dict({k:c[k] for k in condition_fields},examples=c['examples'][:1]) for c in card['conditions']]
        compact.append(dict(reference_id=card['reference_id'],voice_group=card['voice_group'],conditions=conditions,
            unknown_unlisted_combinations=True,scope='Legacy final-only observations cannot establish first-attempt success. Do not transfer success across modes/instructions.'))
    context=dict(current_parent='initial R-only8c,1384/6656errors,hard688/1248; no endpoint promoted',
        family_cards=family_cards,patient_cards=patient_cards,
        reference_cards=base.public_cards({k:v for k,v in cards.items() if v['voice_group'] in eligible}),
        reference_recipe_conditions=compact,generation_path_capability=recipe['generation_path_capability'],
        capabilities=[x.model_dump(mode='json') for x in evidence.capability_anchors],
        forbidden_whole_sentences=sorted({x['clean_gt'] for x in real+dev}),
        closed='Do not resurrect literal主->指,oldfixed8SG,routinei->yu/m->n primary drills,or same尖/鉗/蠍 content. Two48drawliteral/contextualprobes deliveredzero主->指 despite109totalpairs. Existingmask/lr/loss/realheadroom/IPO/risk/audio-scope probesdidnotbeatinitial; preserve oldtraininghyperparameters here.',
        learned_recipe_failure='Observed voice009.ref1+zero_shot+cantonese_zh:4empty-token attempts; voice042.ref1+same recipe:4reference-leak attempts. Theirinstruct2versionspass. Do not choose these exactrepeatedlyfailingcombinations. Originalref0+zero_shot+cantonese_zh passed forbothpatients, but this aloneisnotproofDPOutility. Recipeexperiencepath suppliedinstead ofgenericreferencequalityranking.',
        design='Choose ONE suppliednewerrorfamily,6newnaturalTraditionalCantonese sentences (3target,1contrast,1protection,1maintenance),4distinctpatients actuallyexhibitingthisfamily,atleast2reference difficulty groups. EachpatientgetsoneMAS-selectedreference/mode/instruction for all6sentences=24audio. Matchedfixedrecipearm usesSAMEtexts/patients/TTS andASRseeds, originalref0+instruct2+cantonese_zh=24audio. OnlySGrecipepolicychangesbetweenarms. Primarytargets shoulduse actualobservedwordcontexts, notonly arbitrarycarriercharacters; keepnewwhole sentences. Aim10-18characters,max24surfacecharacters,terminalpunctuation. Keepcounterhypothesizedformsnaturallycorrect andprotection/maintenance meaningful.',
        training_plan='4steps,seed42,commoninitialR-onlyparent/frozenDPOreference,edit+EOSbeta.1,SGweight.25,realKL10,lr1e-4,r16alpha4,clip1. Same32realDPOaudio/91actualpairs and256realKLvisits/order as strict existingRcontrol; notall2117sampledforDPO. Total32SGaudio visits perarm. Propose positiveinteger rolebudgets summing32. Afteractual4draws/audio+greedy, MAS will allocate using relevantactualerrors. Eacharmusesitsowneligibleactualpairs atmatchedbudget; donotdropuniqueusefulMASpairs simplybecauseotherarmallcorrect. No fabricatednegative. Allnormalwrongerrorsforselectedaudio averaged; existingSGcoarsepassedintendedchosen remainsunverifiedwordfidelity. ReadymatchedDPOtest,notendlessnewprobes. Original511pooled/macro/hardgate unchanged; no winnerbySGCER/pairvolume.',
        review='Existingthree-text coarseAgent+finite rulebook unchanged: obviousreferenceleak,severetruncation,offtopiccollapse,mechanicalloops. NoCERthreshold,phonemereview orstrictwordaudit. Proposalgrounding/post-samplingdeliveryanalysisareseparatefromSGadmission.',
        prospective_test='Ifnewgeneratedtargetaudio offeractualsampleerrorsmatchingchosenfamily, consider boundedmatchedDPO. This proposal-level relevancecheck mustnotbecomeSGqualitythreshold. No claimreference/mode affectsclinicalimpairmentuntilmeasured; realCERbenefit needsactualtraining+evaluation.')
    base.save(ROOT/'source_protocol.json',p.model_dump(mode='json'))
    base.save(ROOT/'evidence.json',evidence.model_dump(mode='json'))
    base.save(ROOT/'request_context.json',context)
    files=[Path(__file__),base.DONOR/'protocol.json',base.DONOR/'round_001/evidence.json',before_path,catalog_path,recipe_path,
        recipe_path.parent/'receipt.json',Path(p.real_train.path),Path(p.feedback_split.path),
        base.B/'mode_reference_probe_v1/results.json',base.B/'contextual_matched_SG24_v2/delivery_report.json']
    files += [base.REPO/x for x in ['src/therapist_harness/evidence.py','src/therapist_harness/phonology.py',
        'src/therapist_harness/generation_controller.py','src/therapist_harness/generation_design.py',
        'examples/generate_reviewed_sg_recovery_v2.py','examples/generate_reviewed_sg_resumable.py',
        'examples/transcribe_sg_review.py','src/therapist_harness/review_worker.py','src/therapist_harness/transcript_review.py',
        'src/therapist_harness/prompts/transcript_review_v2.md','examples/sample_asr_preferences.py']]
    base.save(ROOT/'sources.json',{str(f):base.sha256_file(f) for f in files})
    print(json.dumps(dict(status='context_prepared',families=len(family_cards),eligible_patients=len(eligible),
        context_bytes=(ROOT/'request_context.json').stat().st_size)),flush=True)


def propose():
    p=base.verify();context=base.read(ROOT/'request_context.json')
    request=dict(model=p.agent.model,reasoning={'effort':'high'},store=False,stream=True,max_output_tokens=9000,
        input=[dict(role='system',content='你是实际MAS内容/reference/生成模式规划Agent。读取具体真实错误语境、患者对应关系及按模式和指令分组的经验卡，选择一个新错误族并给出将进入匹配DPO实验的6句和4个患者配方。全程自然粤语，不能以更多错误当作价值；保持原轻量SG审查。不要只做安全复述或再提一串超参数扫描。'),dict(role='user',content=json.dumps(context,ensure_ascii=False))],
        text={'format':dict(type='json_schema',name='contextual_recipe_DPO24_proposal',strict=True,schema=to_strict_json_schema(Proposal))})
    base.save(ROOT/'request.json',request)
    response=base.execute_therapist_request(p,request);base.save(ROOT/'response.json',base.serialize_response(response))
    assert response.status=='completed'
    decision=Proposal.model_validate_json(base.terminal_output_text(response));base.save(ROOT/'proposal.json',decision.model_dump())
    compile_proposal(p,decision,context)
    base.verify();print(decision.model_dump_json(indent=2),flush=True)


def compile_proposal(p,decision,context):
    families={x['cluster']['evidence_id']:x for x in context['family_cards']}
    family=families[decision.primary_evidence_id];cluster=family['cluster']
    assert Counter(x.role for x in decision.exercises)==dict(target=3,contrast=1,protection=1,maintenance=1)
    assert sum(decision.intended_role_audio_visits.model_dump().values())==32
    cards={x['reference_id']:x for x in base.read(base.B/'reference_catalog.json')['cards']}
    assert len({x.voice_group for x in decision.voices})==4
    assert all(x.voice_group in family['patient_event_counts'] for x in decision.voices)
    assert len({cards[x.reference_id]['difficulty_group'] for x in decision.voices})>=2
    for v in decision.voices:
        assert cards[v.reference_id]['voice_group']==v.voice_group
        assert not(v.reference_id in ['voice009.ref1','voice042.ref1'] and v.mode=='zero_shot' and v.instruction_id=='cantonese_zh')
    evidence=base.read(ROOT/'evidence.json');known={x['evidence_id'] for x in evidence['error_clusters']+evidence['capability_anchors']}
    forbidden=[base.identity_keys(t) for t in context['forbidden_whole_sentences']]
    stimuli=[]
    for i,item in enumerate(decision.exercises):
        s=Stimulus(stimulus_id=f'recipe24.t{i}',text=item.text,role=item.role,evidence_ids=item.evidence_ids,
            speaker_profile_id='all_patients_balanced',copies=4,rationale=item.reason_zh)
        assert set(s.evidence_ids)<=known
        assert len(base.text_normalize(s.text))<=24
        keys=base.identity_keys(s.text);assert not any(any(keys[k]==f[k] for k in keys) for f in forbidden)
        if s.role in ['target','contrast']:
            assert decision.primary_evidence_id in s.evidence_ids
            unit=cluster['reference_unit'] if s.role=='target' else cluster['hypothesis_unit']
            mapping=map_text_to_syllables(s.text)
            assert any(pos.resolved and getattr(pos,cluster['level'])==unit for pos in mapping.positions),(s.text,unit)
        stimuli.append(s.model_dump(mode='json'))
    assert len({base.text_normalize(x['text']) for x in stimuli})==6
    for arm in ['MAS','fixed']:
        out=ROOT/arm;out.mkdir();(out/'audio').mkdir();manifest=[];choices=[];sidecar=[]
        for i,s in enumerate(stimuli):
            for j,v in enumerate(decision.voices):
                rid=v.reference_id if arm=='MAS' else v.voice_group+'.ref0';ref=cards[rid]
                mode=v.mode if arm=='MAS' else 'instruct2';instruction=v.instruction_id if arm=='MAS' else 'cantonese_zh'
                sid=f'recipe24.t{i}.voice{j}'
                manifest.append(dict(sample_id=sid,target_text=s['text'],reference_text=ref['reference_text'],reference_wav=ref['reference_wav'],
                    reference_wav_sha256=ref['reference_wav_sha256'],output_wav=str(out/'audio'/f'{sid}.wav'),seed=42+i*4+j))
                choices.append(dict(sample_id=sid,reference_id=rid,mode=mode,instruction_id=instruction,
                    reference_wav_sha256=ref['reference_wav_sha256'],target_text_sha256=hashlib.sha256(s['text'].encode()).hexdigest()))
                sidecar.append(dict(sample_id=sid,stimulus_id=s['stimulus_id'],role=s['role'],evidence_ids=s['evidence_ids'],
                    speaker_id=ref['speaker_id'],voice_group=v.voice_group,reference_id=rid,reference_difficulty_group=ref['difficulty_group']))
        base.jsonl(out/'manifest.jsonl',manifest)
        base.save(out/'generation_plan.json',dict(schema_version='generation_plan_v1',choices=choices))
        base.save(out/'role_sidecar.json',dict(schema_version='contextual_recipe_roles_v1',samples=sidecar))
        base.load_generation_choices(out/'generation_plan.json',{x['sample_id']:x for x in manifest})
    base.save(ROOT/'stimuli.json',stimuli)
    base.save(ROOT/'plan.json',dict(version='contextual_recipe_DPO24_v1',primary_evidence_id=decision.primary_evidence_id,
        audio_per_arm=24,arms=['MAS','fixed'],initial_parent_sha256=p.initial_adapter.sha256,
        comparison='Same newMAStexts/patients/TTS andASRseeds; MASrecipeversusfixedoriginalref0/instruct2/cantonese_zh. WholeSGrecipepolicycomparison,notLLMversushumancontentcausality.',
        source_protocol_scope='Reusedfrozenmodel/data/review/decoding; thisexternal24audio/fourpatientplanoverridessource96all-patientmaterializationbudget. No fakecoreledgercommit.',
        intended_role_visits=decision.intended_role_audio_visits.model_dump(),
        training='Afteractualsampling:owneligibleSGpairsperarm,matched32SGaudiovisits/commonrealorder andinitialparent/ref; do notrestricttointersectionofpairedsyntheticIDs. Real32DPOaudio/91pairs+256KLvisits,4steps,r16a4,lr1e-4,editEOSbeta.1,SG.25,KL10. FinalactualMASallocation recordedseparately beforetraining.',
        acceptance='Originalstrictreal511pooledimprovement,patientmacro<=initial+.001,hard<=initial+.005; no automaticpromotion from one exploratoryseed.',
        manifest_hashes={str(f):base.sha256_file(f) for arm in ['MAS','fixed'] for f in (ROOT/arm).glob('*.json*')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','propose'])
    args=parser.parse_args();globals()[args.stage]()
