#!/usr/bin/env python3
"""Actual MAS reflection on completed natural/odd matched DPO outcomes."""
import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from openai.lib._pydantic import to_strict_json_schema
import mas_semantic_context_probe_v1 as run


class Reflection(BaseModel):
    model_config=ConfigDict(extra='forbid')
    conclusion_zh:str
    supported_findings_zh:list[str]
    unproved_or_failed_assumptions_zh:list[str]
    effect_on_MAS_design_zh:str
    next_single_experiment_zh:str
    concrete_data_controls_and_updates_zh:str
    result_that_would_change_our_mind_zh:str
    checkpoint_decision_zh:str


def main():
    base,root=run.base,run.ROOT;p=base.verify()
    result=base.read(root/'results.json');delivery=base.read(root/'delivery_report.json')
    assert all(base.read(root/'training'/m/'result.json')['status']=='training_and_evaluation_complete' for m in ['MAS','fixed'])
    out=root/'MAS_reflection';out.mkdir(exist_ok=False)
    context=dict(proposal=base.read(root/'proposal.json'),real_results=result['summary'],
        paired_CER={k:{g:v for g,v in x['groups'].items() if g in ['overall','difficulty:hard','vlink247','other_patients']} for k,x in result['paired_CER'].items()},
        target_changes=result['target_changes'],actual_delivery=delivery['arms'],
        common_training_slots=len(delivery['common_eligible_slots']),training=base.read(root/'matched_training_checks.json'),
        important_scope='MAS alias=odd; fixed alias=natural.48newaudio, samecharacterbags/lengths,patients,reference/mode/instruction andinitialTTSseeds. Same新版simplethree-textreview accepts deliberateoddness; all48firstpass. ActualASR4draws+greedyinitial8c; sharedeligible19slots,24targetvisits+8oldcommonanchors; real32/91pairsandKL256exactoldRcontrol. NewwordorderalsochangeswordpositionandpotentialTTSprosody, so notpuresemanticisolation. Coarsepass isnotwordverification; textphonesignaturesnotacousticphonemechecks.2117trainpoolisolatedfrom511repeateddevelopment, NOTall2117DPOsampled.',
        prior_closed='Initial1384/h688; matchedR/DPO1399/h714. Prior fixed8SG42/43/44=1382/1390/1410; SG-only+realKL1382/h695 didnotimprovejoint andfailedgate, oldsubsetclosed. Mean-tokenDPO1402 vsIPO1408, realcharacter-risk1398 vsCE1385, allfailed. MASgradientutility8=1409, uniform/MASreal16=1394/1393. Oldliteral/contextual主->指 each48targetdrawszero. RecentnewrecipeDPO1401/1404; newpatientliteral-opportunityselectorDPO1402/1406 despiteheldouttargetopportunities, selectorchanged1/8 anddidnotincreaseheldoutliteralcoverage. No newadjacentgridonclosedoldpools or postprocessinggainasDPOsubstitute.',
        task='Reflect specifically on the user semantic-context hypothesis using actual new completeDPO. MoreSGerrors/pairs alone isnotbenefit; compare against BOTHnaturalDPO andR-only/R-DPO, andreal recovered/newerrors notsignaturecountalone. Give ONE concrete next experiment, notlist orhyperparametersweep. Ifpassesoriginalgate, prioritize checkingrepeatability with matchedcontrol; ifnot, saywhatfailedandchoosea bounded informative changealongMAS->TTS->DPO. Do not just add more increasinglyabsurd sentences orreopenclosedreference/wordscans. OldSG-onlyablationwasdifferentcloseddataandnegative; donotpretendnewmaskinghypothesisproved. No additionalSGqualityscore,CERadmission,phonemeaudit,manufacturednegative orsilentparentpromotion. Originalgateunchanged:pooledstrictlybetter,macro<=initial+.001,hard<=initial+.005 (harderrors<=694), noinventedstricterhard688. Explain howMAS proposal/reference/training decisions should change; do notinventunrunresults orclaim clinical transfer. User values practical performance andclearstory, notdefensiveengineering.')
    request=dict(model=p.agent.model,reasoning={'effort':'high'},store=False,stream=True,max_output_tokens=5000,
        input=[dict(role='system',content='你是实际MAS反思Agent。根据已完成的同字袋自然／古怪句DPO对照，诚实判断用户提出的语义上下文假设，提出一个具体下一步。'),dict(role='user',content=json.dumps(context,ensure_ascii=False))],
        text={'format':dict(type='json_schema',name='semantic_context_DPO_reflection',strict=True,schema=to_strict_json_schema(Reflection))})
    base.save(out/'request.json',request)
    response=base.execute_therapist_request(p,request);base.save(out/'response.json',base.serialize_response(response));assert response.status=='completed'
    reflection=Reflection.model_validate_json(base.terminal_output_text(response));base.save(out/'reflection.json',reflection.model_dump())
    files=[Path(__file__),root/'results.json',root/'proposal.json',root/'delivery_report.json',root/'matched_training_checks.json']
    base.save(out/'sources.json',{str(f):base.sha256_file(f) for f in files})
    print(reflection.model_dump_json(indent=2),flush=True)


if __name__=='__main__':main()
