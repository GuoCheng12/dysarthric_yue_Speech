#!/usr/bin/env python3
"""Reflect on two delivery failures and the verified SG conditioning paths."""
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel,ConfigDict
from openai.lib._pydantic import to_strict_json_schema
import mas_literal_matched_sg_probe_v1 as run

ROOT=run.B/'contextual_matched_SG24_v2'
OUT=ROOT/'MAS_reflection'


class Decision(BaseModel):
    model_config=ConfigDict(extra='forbid')
    next_action:Literal['mode_reference_probe','new_error_family_proposal']
    conclusion_zh:str
    hypothesis_zh:str
    minimal_execution_zh:list[str]
    observed_result_for_and_against_zh:str
    route_back_to_DPO_zh:str
    closed_current_literal_target:bool
    no_current_proposal_training:bool
    no_claim_more_errors_is_more_utility:bool


def main():
    old=run.read(run.ROOT/'delivery_report.json')
    current=run.read(ROOT/'delivery_report.json')
    audit_path=run.B/'instruct2_reference_conditioning_audit_v1/result.json'
    audit=run.read(audit_path)
    assert not old['MAS_preregistered_delivery_test']['supported']
    assert not current['MAS_preregistered_delivery_test']['supported']
    assert audit['status']=='passed_actual_fixed_SG_frontend_conditioning_audit'
    context=dict(initial='SameinitialR-only1384/6656,hard688/1248. No stable better model; no new endpoint promoted.',
        prior_training='FixedSGsubset42/43/44 R+SG1382/1390/1410, mean1394worse thaninitial1384. RemovingrealDPO1382/hard695failsgate. Fixedsubsetclosed. MASutilityselection1409failed. IPO,DPO,risk,realdataheadroom,audioscope probespreviouslyfailed; don’t return to adjacent loss/lr/freeze/ranking scans.',
        old_literal_probe={a:old['arms'][a]['aggregate'] for a in ['A','B']},
        contextual_probe={a:current['arms'][a]['aggregate'] for a in ['A','B']},
        stops=[old['MAS_preregistered_delivery_test'],current['MAS_preregistered_delivery_test']],
        actual_error_context=run.read(run.ROOT/'literal_target_real_context_audit.json'),
        content=run.read(ROOT/'proposal.json'),
        verified_runtime_audit={k:v for k,v in audit.items() if k not in ['runtime','source_sha256']},
        interpretive_limits='Both24probes allcoarsepassed and each had48targetdraws, zero主->指; close this narrow target and do NOTtrain these two proposals. Context+patient pool changed together, so no isolated context causality. Runtimeaudit verifies that instruct2removesLMreference speech tokens, actualinheritedLMdoesnotuseembeddingargument; Flow stillusesreference. This doesNOTprove no impairmenttransfer, identicaloutputtokens, or zero_shotbetter. We already observed27/48pairedtranscriptionschangedbetweenrefs underinstruct2.',
        option_mode_probe='One NEW mechanism diagnostic of mode xreference, justified by actualinputaudit. Firsttarget+countertext(pair0) ofcurrentproposal xsame2patients xsame2references x2modes=16cells. Reuse8existinginstruct2audios/reviews/candidates with separate verifiedprovenance; generate only8newzero_shot+cantonese_zh with same logicalcase/TTS andASR seeds,4actualdraws+greedy frominitial. No ASRtraining in thisdiagnostic, no resurrection ofclosed主->指DPOtarget. Comparefirstattemptreview/fallbacks and actualASRerrorpatterns/reference-sensitivity; finalrulebookoutcomesseparate. Before generation optionally transcribe4actualtrainreferenceaudios usingfrozenR-onlyaudio-onlyworker to annotate reference-associated transcript error patterns, not clinicalphonetictruth. Do NOTchoose winnerjustbymoreerrors/pairs orlessCER onintendedSGtext. NewactualMAS proposal foranotherwell-supportedrealerrorfamily follows if this diagnostic informsrecipechoice; a matchedDPOpilot then testsrealutility.',
        option_new_error='Skipfurthercurrentcontentgeneration; make one newproposal for another currentlysupportedrealerrorfamily with actualwordcontexts andpatientmembership, including explicitgeneration-path capabilitycard. BothTTSreference andmodechoiceareMAScontrols. This needs a concretefreshproposal beforeanynewgeneration; no automaticpositiveclaims.',
        constraints='Choose one boundednextaction. MainrouteMAS->TTS->DPO stays. User permitsprobesandautonomouschanges but doesnotwant defensivecomplexity or endlessknobs. Keep existing simplethree-text SGcoarseAgent/rulebook; noCERthreshold/no phonemereview/no word-fidelityassertion. Existing closednarrowtarget staysclosed. DPOreference initialR-onlyASR andreal2117/511audioisolationunchanged. Moreerrorsnotutility; onlymatchedreal511evaluationcan establishbenefit, finalindependentgeneralizationnotestablished.')
    p=run.RunProtocol.model_validate(run.read(ROOT/'source_protocol.json'))
    request=dict(model=p.agent.model,reasoning={'effort':'high'},store=False,stream=True,max_output_tokens=3500,
        input=[dict(role='system',content='你是MAS Reflection。两个主→指生成探针均零交付，遵守关闭条件。结合新验证的SG条件路径，选择一个小而具体的下一步，说明它如何回到新proposal和匹配DPO，不把模式差异或更多错误直接当训练价值。简洁中文。'),dict(role='user',content=json.dumps(context,ensure_ascii=False))],
        text={'format':dict(type='json_schema',name='after_contextual_SG_failure',strict=True,schema=to_strict_json_schema(Decision))})
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__),run.ROOT/'delivery_report.json',ROOT/'delivery_report.json',ROOT/'proposal.json',audit_path,
           run.ROOT/'literal_target_real_context_audit.json']
    hashes={str(f):run.sha256_file(f) for f in files}
    run.save(OUT/'sources.json',hashes);run.save(OUT/'request.json',request)
    response=run.execute_therapist_request(p,request);run.save(OUT/'response.json',run.serialize_response(response))
    assert response.status=='completed'
    decision=Decision.model_validate_json(run.terminal_output_text(response))
    assert decision.closed_current_literal_target and decision.no_current_proposal_training and decision.no_claim_more_errors_is_more_utility
    assert all(run.sha256_file(Path(f))==h for f,h in hashes.items())
    run.save(OUT/'decision.json',decision.model_dump())
    print(decision.model_dump_json(indent=2),flush=True)


if __name__=='__main__':main()
