#!/usr/bin/env python3
"""Actual MAS target-wave allocation with common real replay and shared anchors."""
from collections import Counter
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from openai.lib._pydantic import to_strict_json_schema
import mas_literal_opportunity_selector_v1 as run
from therapist_harness.dpo_training import preference_key

base, ROOT, PREVIOUS = run.base, run.ROOT, run.PREVIOUS
OLD = Path('/tmp/mas_deterministic_matched_DPO_v1')


class Visit(BaseModel):
    model_config=ConfigDict(extra='forbid')
    audio_id:str
    visits:int=Field(ge=1,le=8)
    reason_zh:str


class Arm(BaseModel):
    model_config=ConfigDict(extra='forbid')
    mode:Literal['MAS','fixed']
    target_visits:list[Visit]


class Allocation(BaseModel):
    model_config=ConfigDict(extra='forbid')
    reflection_zh:str
    arms:list[Arm]
    limitations_zh:str


def main():
    p=base.verify();delivery=base.read(ROOT/'delivery_report.json')
    assert delivery['train_modes'], delivery['decision']
    assert all(base.sha256_file(Path(f))==h for f,h in delivery['source_sha256'].items())
    out=ROOT/'MAS_allocation';out.mkdir(exist_ok=False)
    context=dict(proposal=base.read(ROOT/'proposal.json'),immutable_recipe_selection=base.read(ROOT/'selection.json')['choices'],
        train_modes=delivery['train_modes'],actual_heldout_arms=delivery['arms'],
        policy_comparison='InternalMAS=opportunity; fixed=history. Recipe choices alreadyfrozen usingwave0 only. Cannotchange recipe usingheldoutresults now. Ifonlyfixed islisted, selectorfailedtocreateadifferenttrainingcondition; runONEnewliteral-dataDPO toexamine datautility, noMASselectoradvantageclaim.',
        instructions='Return oneallocation forEACHlistedtrainmode andNOothers. Exactly24targetaudio visits permode,max8peraudio,uniqueaudioIDs inallocation. Useonly listedheldouttargetcaseswithactualpairs. Prioritizeactualsamepatient/literalreferencecharacter/ng->n errors, then generalerrorsat relevanttargetcontexts tofillbudget. NoCERranking,fake negatives,wordfidelityassertion orSGreviewchanges. Allactualdistinctwrongpairs ofeachselectedaudio willbeaveraged. Keepcountsperaudio explicit. Everyarm alsouses EXACTsame8previouslygeneratedMASanchors:4protection+4maintenance,1visit each; compilerinterleaves6target+1protection+1maintenanceeachstep. Samecommon32realDPOaudio/91pairs and256KLvisits/order;4steps,initial8cparentandfrozenreference,editEOSbeta.1,SG.25,KL10,lr1e-4,r16a4. All2117realtrainand511feedback remainisolated; noextraASRsamplingsources invented. Realgateunchanged:pooled<initial,macro<=+.001,hard<=+.005. Literalopportunitiesreportmechanismseparately.')
    request=dict(model=p.agent.model,reasoning={'effort':'high'},store=False,stream=True,max_output_tokens=6500,
        input=[dict(role='system',content='你是实际MAS训练分配Agent。生成配方已经由独立selector音频决定，不可看held-out结果反过来改配方。现在仅依据真实候选偏好对制定明确训练访问次数。'),
            dict(role='user',content=json.dumps(context,ensure_ascii=False))],
        text={'format':dict(type='json_schema',name='literal_opportunity_training_allocation',strict=True,schema=to_strict_json_schema(Allocation))})
    base.save(out/'request.json',request)
    response=base.execute_therapist_request(p,request);base.save(out/'response.json',base.serialize_response(response));assert response.status=='completed'
    decision=Allocation.model_validate_json(base.terminal_output_text(response));base.save(out/'decision.json',decision.model_dump())
    assert Counter(x.mode for x in decision.arms)==Counter(delivery['train_modes'])
    original=base.read(OLD/'plan/plan.json');real=base.read(OLD/'plan/expected_DPO_exposures.json')['R_DPO_KL']
    kl=base.read(OLD/'plan/expected_common_KL_exposures.json')
    planned,selections={},{}
    for arm in decision.arms:
        mode=arm.mode;cases={x['sample_id']:x for x in delivery['arms'][mode]['cases']}
        assert sum(x.visits for x in arm.target_visits)==24
        assert len({x.audio_id for x in arm.target_visits})==len(arm.target_visits)
        assert all(cases[x.audio_id]['pairs'] and cases[x.audio_id]['metadata']['wave']>0 for x in arm.target_visits)
        counts={x.audio_id:x.visits for x in arm.target_visits};target_order=[]
        while any(counts.values()):
            for sid in cases:
                if counts.get(sid,0):target_order.append(sid);counts[sid]-=1
        metadata={x['sample_id']:x for x in base.read(ROOT/mode/'role_sidecar.json')['samples']}
        protection=sorted(k for k,v in metadata.items() if v['role']=='protection')
        maintenance=sorted(k for k,v in metadata.items() if v['role']=='maintenance')
        assert len(protection)==len(maintenance)==4
        prefs=base.rows(ROOT/mode/'training_preferences.jsonl');by_id={sid:[x for x in prefs if x['audio_id']==sid] for sid in metadata}
        selections[mode]=[];planned[mode]=[]
        for step in range(1,5):
            ids=target_order[(step-1)*6:step*6]+[protection[step-1],maintenance[step-1]]
            assert len(ids)==8;selections[mode]+=ids
            synthetic=[dict(audio_id=sid,source='synthetic',objective='dpo',step=step,occurrence=(step-1)*8+i,
                preference_key=preference_key(pair),loss_weight=.25/8/len(by_id[sid])) for i,sid in enumerate(ids) for pair in by_id[sid]]
            assert abs(sum(x['loss_weight'] for x in synthetic)-.25)<1e-12
            planned[mode].append(real[step-1]+synthetic)
    if delivery['train_modes']==['fixed']:
        # Frozen trainer smoke examines two aliases; no duplicate formal optimizer run is launched.
        planned['MAS']=planned['fixed'];selections['MAS']=selections['fixed']
        assert base.rows(ROOT/'MAS/training_preferences.jsonl')==base.rows(ROOT/'fixed/training_preferences.jsonl')
    files=[Path(__file__),ROOT/'delivery_report.json',ROOT/'selection.json',out/'decision.json',
        base.REPO/'examples/train_mas_literal_opportunity_v1.py',base.REPO/'examples/train_mas_contextual_recipe_dpo_v1.py',
        OLD/'plan/plan.json',OLD/'plan/expected_DPO_exposures.json',OLD/'plan/expected_common_KL_exposures.json']
    hashes={**base.read(ROOT/'sources.json'),**delivery['source_sha256'],**{str(f):base.sha256_file(f) for f in files}}
    real_receipt=original['source_sampling_receipts']['historical_real'];source_receipts={'real':real_receipt}
    for f,h in original['source_sha256'].items():
        if '/real96_dpo_probe/sampling/' in f or f==original['data_paths']['real_KL']:
            assert base.sha256_file(Path(f))==h;hashes[f]=h
    for name,path in [('new_history',ROOT/'history/sampling/sampling_receipt.json'),('new_alternate',ROOT/'alternate/sampling/sampling_receipt.json'),('shared_prior_MAS',PREVIOUS/'MAS/sampling/sampling_receipt.json')]:
        receipt=base.read(path)
        for key in ['current_asr_sha256','base_model_sha256','samples_per_audio','seed','temperature','top_p','top_k','language','prompt']:
            assert receipt[key]==real_receipt['decode'][key]
        source_receipts[name]=dict(path=str(path),sha256=base.sha256_file(path),decode=receipt,
            use='Fulloriginalreceipt retained; onlyexplicitselectedheldoutwaveforms or sharedanchorviews used. No mergedcomplete-sampling claim.')
    plan=ROOT/'training_plan';plan.mkdir(exist_ok=False)
    base.save(plan/'plan.json',dict(parent=original['parent'],source_sha256=hashes,
        data_paths=dict(real_KL=original['data_paths']['real_KL'],real_preferences=original['data_paths']['real_preferences'],
            synthetic_preferences={a:str(ROOT/a/'training_preferences.jsonl') for a in ['MAS','fixed']}),
        source_sampling_receipts=source_receipts,selections=selections,settings=original['settings'],gate=original['gate'],
        train_modes=delivery['train_modes'],mode_labels=dict(MAS='opportunity-selected recipe',fixed='frozen history recipe'),
        exposure_schedule='Eachstep sameoriginalrealDPO/KL, then6target+1sharedprotection+1sharedmaintenance. Rawselectorwave0excluded. SG32audiovisits each; sameanchorwaveforms andorder.',
        sampling_caps='Originalreal256versusSG128 preserved and separatelyreported; sameinitialASRsamplingdistributionandseeds.'))
    base.save(plan/'expected_DPO_exposures.json',planned);base.save(plan/'expected_common_KL_exposures.json',kl)
    base.verify();print(decision.model_dump_json(indent=2),flush=True)


if __name__=='__main__':main()
