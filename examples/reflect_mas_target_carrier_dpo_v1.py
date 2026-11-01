#!/usr/bin/env python3
"""Feed the completed scoring-scope comparison back into actual MAS design."""
import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from openai.lib._pydantic import to_strict_json_schema
import mas_literal_matched_sg_probe_v1 as base
import prepare_mas_target_carrier_dpo_v1 as design


class Reflection(BaseModel):
    model_config=ConfigDict(extra='forbid')
    conclusion_zh:str
    supported_findings_zh:list[str]
    remaining_design_problem_zh:str
    next_single_experiment_zh:str
    concrete_data_and_training_zh:str
    why_this_is_not_repeating_closed_probes_zh:str
    checkpoint_decision_zh:str


def main():
    root=design.ROOT
    p=base.RunProtocol.model_validate(base.read(design.PREVIOUS/'source_protocol.json'))
    plan=base.read(root/'training_plan/plan.json')
    assert all(base.sha256_file(Path(f))==h for f,h in plan['source_sha256'].items())
    assert all(base.read(root/'training'/m/'result.json')['status']=='training_and_evaluation_complete' for m in ['MAS','fixed'])
    result=base.read(root/'results.json')
    context=dict(completed_scoring_results=result['summary'],mode_labels=result['mode_labels'],
        paired_CER={k:{g:v for g,v in x['groups'].items() if g in ['overall','difficulty:hard','vlink247','other_patients']} for k,x in result['paired_CER'].items()},
        real_target_changes=result['target_changes'],target_coverage=base.read(root/'target_coverage_context.json'),
        previous_content_results=base.read(design.PREVIOUS/'results.json')['summary'],
        previous_actual_MAS_reflection=base.read(design.PREVIOUS/'MAS_reflection/reflection.json'),
        comparison='NewMAS=target-char-overlap+EOS, fixed=all-edit+EOS. SAME5odd-audio14actualtargetpairs plus8sharedanchors28pairs,24target+8anchorvisits; real32/91+KL256,4steps,seed42. All28continuationmasks strictlyreduced237->72positionsinclEOS; actualprocessorinputsidentical, real/anchor/KLmasksunchanged. Per-modeindependentfrozenreference16pairsmokespassed; originalfullSGscoresmatchpriorcache. Actualrejectedstringsunchanged; targetmetadataNOTaudiofidelityaudit. Freshsame-subsetfullcontrol, priorodd1408notmatchedcontrol.',
        actual_pipeline_scope='MainMAS->TTS->DPO goalstillactive.2117trainpooland511developmentisolated; repeated511notindependenttest. Currentrealcontrolonly32audio/91pairs fromexisting96audio sampling. No authenticatedinitial8c full2117-times4 samplingreceipt isavailable inthisexecution. Do notclaimfull2117DPOalreadytested. Source modelparent andDPOreference bothfrozeninitial8c, TTSreference distinct. MoreSGerrorsorattentiontofewrarewords isnotutility. Currentfocuspairs heavily盲 despiteonlyoneoriginalreal盲error; keep realerrorcoverage distinctfromnumberofgeneratedwaveforms.',
        closed_history='Initial1384/hard688; matchedR-DPO1399/h714. Oldfixed8SG42/43/44=1382/1390/1410 andSG-only1382/h695, subsetclosed. MASgradientutility8=1409;uniform/MASreal16=1394/1393;real audio-only1398. Mean-tokenDPO1402vsIPO1408;realcharacter-risk1398vsCE1385;allfailed. Literal/contextual主->指 generationeach48targetdraws0. NewcontextrecipeDPO1401/1404;literalopportunityselector1402/1406, didnotincreaseheldouttargetcoverage. Samecharbagnewnatural/oddDPO1423/1408; bothfailed, oddrealngn24 unchangedfrominitial. Do notcloseallTTSorallDPO basedontinycontrols, butdonotkeepdoingadjacentloss/lr/quota scans onthis5audio pool.',
        task='Judge whether narrowingtargetloss actuallyhelps matchedrealCER andliteralrealrecovery; give ONE practical nextstep toward the fullusergoal. Ifa candidatepassesoriginalgate,checkrepeatabilitybeforeclaimingstablegain. Ifnot, distinguishpartialrescueversusmatchedR-DPOfromnosignal. Ifthereisnorescue,closethis5audioadjacentmicroprobe. IfitrescuesrealCERagainstbothall-editandR-DPOwhileR-DPOitselfregresses, a SINGLEdirectinteractionablationremovingthe32realDPOupdateswhilepreservingallrealKL/newSGscoringmaybetestable; oldSG-onlynegativeisfromadifferentclosedpool,NOTevidenceitwillwork. Donotassumeadditivedeltas. OtherwiseproposealargerinformativeactualMAS/TTS/DPOexecution. Do notrunanotheradjacentmask/lr/quota scan. Consider whetherlimitedreallearningcontrol/coverage callsforlargerrealtraining support, or a concretegeneration change; chooseONE withexplicitdata/controlscope and whatcanbe reused vsneedsnewactualsampling. Do not inventtrained models/results/ASRreceipts. No CER SGadmission, word/phoneaudioaudit, fake negatives orpostprocessingasDPOgain. Originalgateunchanged:pooled<initial,macro<=+.001,hard<=+.005(hard<=694); no silentparentpromotion. Preferdecisiveimplementationandrunningoverdefensiveengineering.')
    out=root/'MAS_reflection';out.mkdir(exist_ok=False)
    request=dict(model=p.agent.model,reasoning={'effort':'high'},store=False,stream=True,max_output_tokens=5000,
        input=[dict(role='system',content='你是实际MAS反思Agent。依据完成的相同偏好对损失范围对照，更新下一次可执行实验，推进真实ASR性能。'),
               dict(role='user',content=json.dumps(context,ensure_ascii=False))],
        text={'format':dict(type='json_schema',name='target_scope_DPO_reflection',strict=True,schema=to_strict_json_schema(Reflection))})
    base.save(out/'request.json',request)
    response=base.execute_therapist_request(p,request);base.save(out/'response.json',base.serialize_response(response));assert response.status=='completed'
    reflection=Reflection.model_validate_json(base.terminal_output_text(response));base.save(out/'reflection.json',reflection.model_dump())
    files=[Path(__file__),root/'results.json',root/'target_coverage_context.json',root/'training_plan/plan.json',design.PREVIOUS/'results.json']
    base.save(out/'sources.json',{str(f):base.sha256_file(f) for f in files})
    print(reflection.model_dump_json(indent=2),flush=True)


if __name__=='__main__':main()
