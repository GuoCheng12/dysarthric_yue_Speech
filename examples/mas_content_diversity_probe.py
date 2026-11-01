#!/usr/bin/env python3
"""Expand six MAS exercises to 24 contexts at the same 96-audio/96-exposure budget."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

from pydantic import BaseModel,ConfigDict,Field

import mas_matched_generation_control as runner
from mas_selection_dpo_probe import REPO,execute,jsonl,read,rows,save
from therapist_harness.agent import execute_therapist_request,serialize_response
from therapist_harness.generation_controller import load_catalog
from therapist_harness.grounding import validate_stimulus_grounding
from therapist_harness.materialize import materialize_proposal,build_materialization_sidecar
from therapist_harness.schema import RunProtocol,EvidencePack,Proposal,canonical_sha256,validate_proposal
from therapist_harness.store import sha256_file


class Variants(BaseModel):
    model_config=ConfigDict(extra='forbid')
    parent_stimulus_id: str
    texts: list[str]=Field(min_length=4,max_length=4)
    rationale: str


class Expansion(BaseModel):
    model_config=ConfigDict(extra='forbid')
    hypothesis: str
    expected_response: str
    exercises: list[Variants]=Field(min_length=6,max_length=6)


def prepare(base,root,reuse_response_root=None):
    from openai.lib._pydantic import to_strict_json_schema
    main=base/'formal_mas96_v3';source=main/'round_001'
    donor=base/'formal_random96_control_v2'
    root.mkdir(exist_ok=False)
    cfg=read(Path(read(main/'protocol.json')['preference_loop']['generation_design_path']))
    cfg['strategy']='paired_content_control'
    save(root/'generation_design.json',cfg)
    pdata=read(main/'protocol.json')
    pdata.update(run_id=root.name,round_count=1,unique_stimulus_count_per_round=24)
    pdata['proposal_constraints']['max_copies_per_stimulus']=4
    pdata['preference_loop'].update(generation_design_path=str(root/'generation_design.json'),
        generation_design_sha256=sha256_file(root/'generation_design.json'))
    p=RunProtocol.model_validate(pdata);save(root/'protocol.json',p.model_dump(mode='json'))
    edata=read(source/'evidence.json');edata['run_id']=p.run_id
    edata['action_contract'].update(unique_stimulus_count=24,
        proposal_constraints=p.proposal_constraints.model_dump(mode='json'))
    evidence=EvidencePack.model_validate(edata);save(root/'evidence.json',evidence.model_dump(mode='json'))
    original=read(source/'proposal.json')
    selected_ids={eid for x in original['stimuli'] for eid in x['evidence_ids']}
    selected_evidence=[x for x in edata['error_clusters']+edata['capability_anchors'] if x['evidence_id'] in selected_ids]
    request=dict(model=p.agent.model,reasoning={'effort':p.agent.reasoning_effort},store=False,stream=True,
        max_output_tokens=16384,input=[dict(role='system',content='''Design a controlled Cantonese ASR training-content probe for the MAS.
For EACH of the six original exercises, create exactly FOUR fresh, natural
Cantonese whole sentences with diverse lexical and semantic contexts. They
inherit that exercise's role and evidence IDs and receive four audio copies each.
Keep the SAME phonological target, comparator/protection requirements and
maintenance anchor. Contrast sentences must carry both the reference unit and
the cited comparator unit; capability sentences must carry their cited anchor.
Target the supplied real-error hypothesis, not generic difficult tongue twisters.
Aim for 12-20 spoken Chinese characters (absolute maximum 24). Use traditional
Chinese and natural Cantonese; do not issue speech-style directions in the text.
Do not repeat an original whole sentence or any other new sentence. Distinct
contexts are required, not four tiny rewrites of the same sentence. This expands
six texts x16 voices to 24 texts x4 voices at the same 96-audio budget. Roles,
per-role patient exposure, donor reference/recipe choices, DPO parent/reference,
12 steps and 96 audio exposures remain controlled. No extra real training data.
The six-content experiments did not beat the real-only parent; a hypothesis is
that narrow lexical coverage makes updates too concentrated. This is unproven.
Return the expansion with a prospective hypothesis and expected real outcome.
Written phonological carriers are content design, not proof of generated audio
fidelity; coarse review remains simple and has no CER rejection threshold.'''),
            dict(role='user',content=json.dumps(dict(original_proposal=original,
                actual_current_evidence=selected_evidence,
                observed_six_content_results=dict(parent_pooled_cer=.20793269230769232,
                    random_dpo=.21063701923076922,mas_dpo=.20928485576923078,
                    mas_allocated_dpo=.21153846153846154)),ensure_ascii=False))],
        text={'format':{'type':'json_schema','name':'content_expansion','strict':True,
                       'schema':to_strict_json_schema(Expansion)}})
    save(root/'request.json',request)
    if reuse_response_root is None:
        response=serialize_response(execute_therapist_request(p,request))
    else:
        if read(reuse_response_root/'request.json') != request:
            raise ValueError('reused response must answer the identical request')
        response=read(reuse_response_root/'response.json')
        save(root/'response_reuse.json',dict(actual_new_api_call=False,
            source_root=str(reuse_response_root),source_sha256={str(f):sha256_file(f)
                for f in [reuse_response_root/'request.json',reuse_response_root/'response.json']},
            reason='The previous parser concatenated commentary and final-answer messages. Reuse the actual final answer.'))
    save(root/'response.json',response)
    if response['status']!='completed':raise RuntimeError('content expansion did not complete')
    final_messages=[(i,x) for i,x in enumerate(response['output'])
        if x.get('type')=='message' and x.get('phase')=='final_answer']
    if len(final_messages)!=1:raise ValueError('expected one explicit final-answer message')
    index,message=final_messages[0]
    final_text=''.join(x['text'] for x in message['content'] if x['type']=='output_text')
    decision=Expansion.model_validate_json(final_text)
    save(root/'response_extraction.json',dict(output_index=index,message_id=message['id'],
        phase=message['phase'],decision=decision.model_dump(mode='json'),
        response_sha256=sha256_file(root/'response.json')))
    by_parent={x.parent_stimulus_id:x for x in decision.exercises}
    if len(by_parent)!=6 or set(by_parent)!={x['stimulus_id'] for x in original['stimuli']}:
        raise ValueError('expansion must cover each original exercise once')
    stimuli=[]
    for old in original['stimuli']:
        item=by_parent[old['stimulus_id']]
        for index,text in enumerate(item.texts):
            stimuli.append(dict(old,stimulus_id=old['stimulus_id']+f'.context{index+1}',
                text=text,copies=4,rationale=item.rationale))
    proposal=Proposal.model_validate(dict(original,evidence_sha256=canonical_sha256(evidence),
        hypothesis=decision.hypothesis,expected_response=decision.expected_response,stimuli=stimuli))
    validate_proposal(proposal,evidence,p);validate_stimulus_grounding(proposal,evidence)
    save(root/'proposal.json',proposal.model_dump(mode='json'))
    (root/'audio').mkdir()
    assignments=materialize_proposal(proposal,evidence,p,output_root=root/'audio')
    cards=load_catalog(cfg,rows(p.real_train.path))
    speakers={x['reference_wav_sha256']:x['speaker_id'] for x in cards.values()}
    donor_roles={x['sample_id']:x['role'] for x in read(donor/'materialization_sidecar.json')['samples']}
    donor_choices={x['sample_id']:x for x in read(donor/'generation_plan.json')['choices']}
    queues=defaultdict(list)
    for x in rows(donor/'manifest.jsonl'):
        queues[(donor_roles[x['sample_id']],speakers[x['reference_wav_sha256']])].append(x)
    roles=[s.role for s in proposal.stimuli for _ in range(s.copies)]
    choices=[];mapping=[]
    for row,role in zip(assignments,roles):
        speaker=speakers[row['reference_wav_sha256']]
        old=queues[(role,speaker)].pop(0);choice=donor_choices[old['sample_id']]
        for key in ['reference_wav','reference_wav_sha256','reference_text']:row[key]=old[key]
        choices.append(dict(choice,sample_id=row['sample_id'],
            target_text_sha256=hashlib.sha256(row['target_text'].encode()).hexdigest(),
            reason='Reuse the donor reference/recipe for the same role and patient; vary content coverage.'))
        mapping.append(dict(sample_id=row['sample_id'],donor_sample_id=old['sample_id'],role=role,
            reference_id=choice['reference_id'],donor_seed=old['seed'],new_seed=row['seed']))
    assert all(not q for q in queues.values())
    save(root/'generation_plan.json',dict(schema_version='generation_plan_v1',choices=choices,
        hypothesis='Control donor reference and recipe within each role/patient while expanding lexical contexts.',
        design_sha256=p.preference_loop.generation_design_sha256))
    jsonl(root/'manifest.jsonl',assignments)
    sidecar=build_materialization_sidecar(proposal,evidence,p,assignments,generation_plan_path=root/'generation_plan.json')
    save(root/'materialization_sidecar.json',sidecar.model_dump(mode='json'));save(root/'donor_mapping.json',mapping)
    files=[Path(__file__),REPO/'examples/mas_matched_generation_control.py',REPO/'examples/generate_reviewed_sg_recovery.py',
        main/'protocol.json',source/'proposal.json',source/'evidence.json',donor/'manifest.jsonl',
        donor/'generation_plan.json',donor/'materialization_sidecar.json',root/'protocol.json',root/'evidence.json',
        root/'proposal.json',root/'request.json',root/'response.json',root/'manifest.jsonl',root/'generation_plan.json',root/'donor_mapping.json']
    sources=read(main/'sources.json');sources.update({str(x):sha256_file(x) for x in files})
    save(root/'provenance.json',dict(source_sha256=sources,source_root=str(main),donor_root=str(donor),
        actual_mas_content_call=reuse_response_root is None,actual_new_generation_planner_call=False,
        content_decision_source=str(reuse_response_root/'response.json') if reuse_response_root else str(root/'response.json'),
        intervention='6 independent contents become 24; 96 audio, same role totals, matched per-role patients and exact donor references/recipes.',
        seed_caveat='Same global seed42 derivation policy; new stimulus IDs imply different per-audio seeds. This is not a paired same-waveform comparison.',
        review='same coarse three-text standard; observed empty-token recovery extension is recorded',
        expected_role_exposures=dict(target=32,contrast=16,protection=16,maintenance=32)))
    print(json.dumps(dict(status='prepared',independent_contents=24,audio=96)),flush=True)


def run(root):
    original=runner.execute
    def execute_with_recovery(directory,command,name):
        command=list(command)
        if command[1]==str(REPO/'examples/generate_reviewed_sg.py'):
            command[1]=str(REPO/'examples/generate_reviewed_sg_recovery.py')
        return original(directory,command,name)
    runner.execute=execute_with_recovery
    runner.run(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','all']);parser.add_argument('base',type=Path)
    parser.add_argument('--name',default='content24_probe')
    parser.add_argument('--reuse-response-root',type=Path)
    args=parser.parse_args();base=args.base.resolve();root=base/args.name
    if args.stage in ['prepare','all']:prepare(base,root,args.reuse_response_root)
    if args.stage in ['run','all']:run(root)
