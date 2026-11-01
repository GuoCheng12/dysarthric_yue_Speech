#!/usr/bin/env python3
"""Actual MAS reflection after the completed matched recipe DPO experiment."""
import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from openai.lib._pydantic import to_strict_json_schema
import mas_contextual_recipe_dpo_data_v1 as data

base, ROOT = data.base, data.ROOT


class Reflection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    conclusion_zh: str
    supported_findings_zh: list[str]
    failed_or_unproved_assumptions_zh: list[str]
    next_single_experiment_zh: str
    evidence_for_next_experiment_zh: str
    concrete_data_and_controls_zh: str
    observation_that_would_change_design_zh: str
    checkpoint_decision_zh: str


def main():
    p = base.verify(); out = ROOT / 'MAS_reflection'; out.mkdir(exist_ok=False)
    result = base.read(ROOT / 'results.json')
    for arm in ['MAS','fixed']:
        assert not base.read(ROOT / 'training' / arm / 'result.json')['passes_original_gate']
    delivery = base.read(ROOT / 'delivery_report.json')
    context = dict(proposal=base.read(ROOT / 'proposal.json'), allocation=base.read(ROOT / 'MAS_allocation/decision.json'),
        real_results=result['summary'], paired_CER=result['paired_CER'],
        target_changes=result['target_comparisons'], real_DPO_target_control=base.read(ROOT / 'real_DPO_target_control.json'),
        target_delivery={a: x['aggregate'] for a, x in delivery['arms'].items()},
        literal_overlap=base.read(ROOT / 'target_literal_overlap_diagnostic.json'),
        already_tested='Initial1384/hard688. StrictR-DPO1399/hard714. Prior fixed8SG seed42/43/44=1382/1390/1410 and SG-only+realKL1382/hard695; thatsubsetclosed, no adjacent loss/lr/freeze/quota scans. Prior MAS gradientutility8=1409. Mean-tokenDPO1402,IPO1408; character-riskreal1398 vsmatchedCE1385, bothfailed. Uniformrealheadroom1394, MAS1393, audio-tower-onlyreal1398. Literal主->指 and contextual versions each48targetdrawszero, closed. Runtimeinstruct2deletesLMreference_speech_tokens but retainsflowreference; unknownclinicaltransfer. ZeroShot voice009.ref1 emptytokens4,voice042.ref1 referenceleaks4; ref0modepassedboth. Currentnewrecipe24 eachbranchcompleted, all48firstpasscoarse review; currentMAS usedrecipe-conditionedpasshistory plusactualrealwordcontexts. Do not claimall2117realwereDPOsampled; actualcontrol32real/91pairs,KL96pool/256visits.',
        task='Reflect on actual real effects, including same-carrier recovery, unresolved states and matched R-DPO. Distinguish generictarget-context errors from literalrealerrorcarriers and from ASRtext-derivedphone-signatures. Do not claimsyntheticdataallnoise, or reducedsignaturecountsproveaccuratecorrection. No checkpointpromotion. Propose ONE next bounded experiment along MAS->TTS->DPO, with concrete reused/newdata and control, that resolves a design uncertainty exposed here; no openendedprobe/lr/seed scan or returntoclosedpools. User prioritizes performance and decisive practical experiments, not defensiveengineering. Explain whether MAS should learn from actual post-generation error opportunities rather than coarse-pass history alone, and whether real-update regression constrains attainablegain. Preserve simplethree-textSGreview, noCER or strictword/phoneaudits; repeated511remainsdevelopmentnotindependenttest. Do not pretend dataavailability or actualAPI/generation results not supplied.')
    request = dict(model=p.agent.model, reasoning={'effort':'high'}, store=False, stream=True, max_output_tokens=5000,
        input=[dict(role='system',content='你是实际MAS反思Agent。用完成的匹配DPO结果更新设计，不替失败结果编成功叙事。给出一个具体、有区分力、可执行的下一步。'),
            dict(role='user',content=json.dumps(context,ensure_ascii=False))],
        text={'format':dict(type='json_schema',name='completed_recipe_DPO_reflection',strict=True,schema=to_strict_json_schema(Reflection))})
    base.save(out / 'request.json', request)
    response = base.execute_therapist_request(p, request); base.save(out / 'response.json', base.serialize_response(response))
    assert response.status == 'completed'
    reflection = Reflection.model_validate_json(base.terminal_output_text(response))
    base.save(out / 'reflection.json', reflection.model_dump())
    files = [Path(__file__), ROOT / 'results.json', ROOT / 'real_DPO_target_control.json', ROOT / 'delivery_report.json',
        ROOT / 'proposal.json', ROOT / 'MAS_allocation/decision.json', ROOT / 'target_literal_overlap_diagnostic.json']
    base.save(out / 'sources.json', {str(f):base.sha256_file(f) for f in files})
    print(reflection.model_dump_json(indent=2), flush=True)


if __name__ == '__main__': main()
