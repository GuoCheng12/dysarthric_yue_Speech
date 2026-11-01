#!/usr/bin/env python3
"""Generic-text control matched to the actual lexical MAS probe's generation plan."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

import mas_matched_generation_control as runner
from mas_selection_dpo_probe import REPO,jsonl,read,rows,save
from therapist_harness.grounding import validate_stimulus_grounding
from therapist_harness.materialize import materialize_proposal,build_materialization_sidecar
from therapist_harness.schema import EvidencePack,Proposal,RunProtocol,canonical_sha256,validate_proposal
from therapist_harness.store import sha256_file


def prepare(base,root):
    donor=base/'lexical24_loop_v1';source=donor/'round_001';root.mkdir(exist_ok=False)
    pdata=read(donor/'protocol.json');pdata['run_id']=root.name
    cfg=read(Path(pdata['preference_loop']['generation_design_path']));cfg['strategy']='paired_content_control'
    save(root/'generation_design.json',cfg)
    pdata['preference_loop'].update(generation_design_path=str(root/'generation_design.json'),generation_design_sha256=sha256_file(root/'generation_design.json'))
    p=RunProtocol.model_validate(pdata);save(root/'protocol.json',p.model_dump(mode='json'))
    edata=read(source/'evidence.json');edata['run_id']=p.run_id;e=EvidencePack.model_validate(edata)
    save(root/'evidence.json',e.model_dump(mode='json'))
    original=read(source/'proposal.json');generic=read(base/'formal_mas96_v3/round_001/proposal.json')
    by_role=defaultdict(list)
    for x in generic['stimuli']:by_role[x['role']].append(x)
    selected=[];mapping=[]
    for stimulus in original['stimuli']:
        choices=by_role[stimulus['role']]
        compatible=[x for x in choices if set(x['evidence_ids'])==set(stimulus['evidence_ids'])]
        assert compatible
        old=compatible[0];choices.remove(old)
        selected.append(dict(stimulus,text=old['text'],rationale='Matched generic-carrier content control; reused actual earlier MAS text, not a fresh MAS decision.'))
        mapping.append(dict(stimulus_id=stimulus['stimulus_id'],lexical_text=stimulus['text'],generic_text=old['text'],generic_stimulus_source=old['stimulus_id']))
    q=Proposal.model_validate(dict(original,evidence_sha256=canonical_sha256(e),stimuli=selected,
        hypothesis='Generic phone-carrier texts under the lexical probe budget provide a control for grounding content in actual word confusions.',
        expected_response='Compare real development change to the lexical arm at the same four-step budget; no benefit assumed.'))
    validate_proposal(q,e,p);validate_stimulus_grounding(q,e)
    save(root/'proposal.json',q.model_dump(mode='json'));save(root/'text_mapping.json',mapping)
    (root/'audio').mkdir();assignments=materialize_proposal(q,e,p,output_root=root/'audio')
    donors=rows(source/'synthesis_manifest.jsonl');decisions={x['sample_id']:x for x in read(source/'generation_plan.json')['choices']}
    assert len(donors)==len(assignments)==24
    actions=[];links=[]
    for row,old in zip(assignments,donors):
        for key in ['reference_text','reference_wav','reference_wav_sha256','seed']:row[key]=old[key]
        decision=decisions[old['sample_id']]
        actions.append(dict(decision,sample_id=row['sample_id'],target_text_sha256=hashlib.sha256(row['target_text'].encode()).hexdigest(),
            reason='Exact donor reference/recipe/seed retained for a generic-versus-lexical content intervention.'))
        links.append(dict(sample_id=row['sample_id'],donor_sample_id=old['sample_id'],shared_tts_seed=old['seed'],reference_id=decision['reference_id']))
    save(root/'generation_plan.json',dict(schema_version='generation_plan_v1',choices=actions,
        design_sha256=p.preference_loop.generation_design_sha256,hypothesis=q.hypothesis))
    jsonl(root/'manifest.jsonl',assignments)
    sidecar=build_materialization_sidecar(q,e,p,assignments,generation_plan_path=root/'generation_plan.json')
    save(root/'materialization_sidecar.json',sidecar.model_dump(mode='json'));save(root/'donor_mapping.json',links)
    files=[Path(__file__),REPO/'examples/generate_reviewed_sg_recovery_v2.py',REPO/'examples/generate_reviewed_sg_resumable.py',
        donor/'protocol.json',source/'proposal.json',source/'prescribe/request.json',source/'prescribe/response.json',
        source/'synthesis_manifest.jsonl',source/'generation_plan.json',base/'formal_mas96_v3/round_001/proposal.json',
        root/'protocol.json',root/'evidence.json',root/'proposal.json',root/'manifest.jsonl',root/'generation_plan.json',root/'donor_mapping.json']
    sources=read(donor/'sources.json');sources.update({str(f):sha256_file(f) for f in files})
    save(root/'provenance.json',dict(source_sha256=sources,actual_new_mas_call=False,
        comparison='Generic versus actual-confusion lexical content; same patients, exact references, recipes and TTS seeds, role/copy budget, R-only parent/reference, decode parameters, four steps and 32 exposures.',
        seed_policy='Explicit donor TTS seed reuse overrides the normal new-run sample-ID hash derivation; independent ASR draws retain the shared seed42 decoding policy with their own unique audio IDs.',
        control_content_source='Actual earlier six-text MAS proposal; adapted into a control artifact, not an independent MAS response',
        review='Unchanged coarse three-text Agent; no lexical-CER or phoneme rejection rule'))


def run(root):
    execute=runner.execute
    def launch(directory,command,name):
        command=list(command)
        if name=='generation':command[1]=str(REPO/'examples/generate_reviewed_sg_recovery_v2.py')
        return execute(directory,command,name)
    runner.execute=launch;runner.run(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run','all']);parser.add_argument('base',type=Path)
    args=parser.parse_args();base=args.base.resolve();root=base/'lexical24_generic_control'
    if args.stage in ['prepare','all']:prepare(base,root)
    if args.stage in ['run','all']:run(root)
