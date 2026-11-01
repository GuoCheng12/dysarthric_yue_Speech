#!/usr/bin/env python3
"""Give MAS actual lexical error contexts and the corresponding patient pool."""
import argparse
import json
from pathlib import Path

from openai.lib._pydantic import to_strict_json_schema
import mas_literal_matched_sg_probe_v1 as base

PREVIOUS = base.ROOT
ROOT = base.B / 'contextual_matched_SG24_v2'
base.ROOT = ROOT


def propose():
    ROOT.mkdir(exist_ok=False)
    p = base.RunProtocol.model_validate(base.read(PREVIOUS / 'source_protocol.json'))
    evidence = base.EvidencePack.model_validate(base.read(PREVIOUS / 'evidence.json'))
    report = base.read(PREVIOUS / 'delivery_report.json')
    assert not report['MAS_preregistered_delivery_test']['supported']
    assert report['MAS_preregistered_delivery_test']['target_audio_with_actual_substitution'] == 0
    context_audit = base.read(PREVIOUS / 'literal_target_real_context_audit.json')
    assert len(context_audit['events']) == 3
    voices = {x['voice_group'] for x in context_audit['events']}
    assert voices == {'voice009','voice042'}
    catalog = base.B / 'reference_catalog.json'
    real, dev = base.rows(p.real_train.path), base.rows(p.feedback_split.path)
    all_cards = base.load_catalog(dict(catalog_path=str(catalog),catalog_sha256=base.sha256_file(catalog)),real)
    cards = {k:v for k,v in all_cards.items() if v['voice_group'] in voices}
    prior_context = base.read(PREVIOUS / 'request_context.json')
    glyphs = [x for x in prior_context['literal_error_options'] if x['evidence_id'] == 'character-sub-eb6dadec9faa']
    assert len(glyphs) == 1
    prior_texts = [t for x in base.read(PREVIOUS / 'proposal.json')['pairs'] for t in [x['target_text'],x['counter_text']]]
    context = dict(current_parent=prior_context['current_parent'],
        prior_probe_summary={arm:dict(aggregate=report['arms'][arm]['aggregate'],
            first_attempt_verdicts=report['arms'][arm]['first_attempt_verdicts'],truncations=report['arms'][arm]['truncations']) for arm in ['A','B']},
        prior_proposal=base.read(PREVIOUS / 'proposal.json'), prior_stop=report['MAS_preregistered_delivery_test'],
        literal_error_options=glyphs,
        actual_real_error_contexts=[{k:v for k,v in x.items() if k not in ['utt_id','speaker_id']} for x in context_audit['events']],
        patient_cards=[x for x in prior_context['patient_cards'] if x['voice_group'] in voices],
        reference_cards=base.public_cards(cards),
        reference_history=[x for x in prior_context['historical_reference_observations'] if x['reference_id'] in cards],
        reference_history_semantics=prior_context['history_semantics'],
        forbidden_whole_sentences=prior_context['forbidden_whole_feedback_sentences']+prior_context['forbidden_whole_training_sentences']+prior_texts,
        design='New24 audio:3target/counter pairs x2actual-error patients x2within-patient references. Same fixed instruct2+cantonese_zh, same-case TTS and ASR seeds,4actual stochastic candidates+greedy. A/B differ only within-patient reference. Compared with priorprobe, BOTH content policy and patient/reference pool change; do not attribute any difference to context alone. No efficacy claim from target delivery.',
        feedback_gap='Prior MAS saw only aggregate 主->指 and overall patient/reference histories, not the actual lexical contexts or patient membership. Its 店主/主廚/屋主 on voice005/032 produced71actualpairs but zero intended 主->指 in48targetdraws. This does not prove all otherpairs worthless; it refutes that proposal’s declared target delivery. All24coarsepassed and no review threshold changes. Priorprobeclosedbeforetraining.',
        requested_change='Use observed lexical contexts 主任 and 主張 in genuinely new natural Cantonese sentences; include BOTH across3targetsentences. Each has a coherent counter sentence where 指 is correct, preferably natural 指引/指示/指住. Do not force the invalid observed error string 指將 into a supposedly correct sentence. Do not copy any full train/dev/prior sentence or construct a concatenated word drill. Choose exactly both suppliedpatients and bothrefs each; prospectively choose preferredA versusB with factual reason. Unknown acoustic traits stay unknown. The two vlink247 errors repeat the samecontent and are not two independent contents. Newcontext may also be allcorrect; no manufactured wrong transcripts.',
        prospective_rule='Before training require intended 主->指 in at least2distinct targetaudios spanning bothselectedpatients, as in priorprobe; otherwise close this narrow contextual target and do not train it. This is a proposal-level delivery test, NOT a stricter SG admission gate. If delivered, actualMAS next selects a small matched training allocation and original511checkpoint gate remains. DPO reference fixedinitialR-only; real2117train/511feedback isolation unchanged.')
    base.save(ROOT / 'source_protocol.json',p.model_dump(mode='json'))
    base.save(ROOT / 'evidence.json',evidence.model_dump(mode='json'))
    base.save(ROOT / 'request_context.json',context)
    source_hashes = base.read(PREVIOUS / 'sources.json')
    files = [Path(__file__), PREVIOUS / 'delivery_report.json', PREVIOUS / 'proposal.json',
             PREVIOUS / 'request_context.json', PREVIOUS / 'literal_target_real_context_audit.json']
    base.save(ROOT / 'sources.json', {**source_hashes,**{str(f):base.sha256_file(f) for f in files}})
    request = dict(model=p.agent.model,reasoning={'effort':'high'},store=False,stream=True,max_output_tokens=5000,
        input=[dict(role='system',content='你是同一个MAS。上一24音频探针71个实际偏好对却零次指定主→指，按你的预声明不训练。现在给你此前缺失的真实错误词语上下文和对应患者，设计最后一个该字级目标的24音频探针。保持轻量审查；仅检验生成是否交付目标，不能借新数据量讲已有性能收益。简洁反思并提供具体新句子与reference选择。'),
               dict(role='user',content=json.dumps(context,ensure_ascii=False))],
        text={'format':dict(type='json_schema',name='contextual_matched_SG24_proposal',strict=True,
                            schema=to_strict_json_schema(base.MatchedProposal))})
    base.save(ROOT / 'request.json',request)
    response=base.execute_therapist_request(p,request)
    base.save(ROOT / 'response.json',base.serialize_response(response))
    assert response.status=='completed'
    decision=base.MatchedProposal.model_validate_json(base.terminal_output_text(response))
    base.save(ROOT / 'proposal.json',decision.model_dump())
    assert {x.voice_group for x in decision.voices}==voices
    targets=[x.target_text for x in decision.pairs]
    assert any('主任' in x for x in targets) and any('主張' in x for x in targets)
    assert all(x.evidence_id=='character-sub-eb6dadec9faa' for x in decision.pairs)
    assert not {base.text_normalize(t) for x in decision.pairs for t in [x.target_text,x.counter_text]} & {base.text_normalize(t) for t in prior_texts}
    base.compile_proposal(p,decision,cards,glyphs,real,dev)
    base.save(ROOT / 'execution_version.json',dict(version='contextual_matched_SG24_v2',
        source_proposal=str(PREVIOUS / 'proposal.json'),old_proposal_closed=True,
        changes='Actual lexical error contexts supplied; use 主任/主張 in new texts and actual-error patient pool009/042. Same24budget, samewithin-patient A/B reference comparison, fixedrecipe/currentASR/review/samplecounts.',
        comparison_scope='Whole content+patient/reference policy differs from priorprobe; not a content-only causal ablation.',
        uses_original_helper_without_file_changes=True,no_reused_old_audio=True,
        stop='No intended substitutions in >=2audio spanningbothpatients: close this narrow target withouttraining; no further adjacent context/reference scans.'))
    base.verify()
    print(decision.model_dump_json(indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['propose','generate','sample'])
    parser.add_argument('--arm',choices=['A','B'],default='A')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.stage=='propose':propose()
    elif args.stage=='generate':base.generate(args.arm,args.smoke)
    else:base.sample(args.arm)
