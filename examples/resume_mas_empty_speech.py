#!/usr/bin/env python3
"""Migrate a stopped materialized execution without replacing its frozen ledger.

Data identity, actual MAS responses and generation plan are retained. Completed
waveforms receive a fresh, normally closed ASR receipt; identical three-text
judgments are explicitly reused. A separately hashed SG execution extension
handles the observed empty-speech error, leaving the running control untouched.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import mas_asr_dpo_loop as cli
from therapist_harness import preference_loop as loop
from therapist_harness.materialize import build_materialization_sidecar
from therapist_harness.memory import SkillMemory
from therapist_harness.review_worker import ReviewASRWorker
from therapist_harness.schema import EvidencePack, Proposal
from therapist_harness.store import RunStore, sha256_file, write_immutable_bytes
from therapist_harness.transcript_review import TranscriptReviewer,validate_transcript_review

REPO=Path(__file__).resolve().parents[1]
EXTENSION=REPO/'examples/generate_reviewed_sg_recovery.py'


def migrate(old,root):
    source=RunStore(old);assert source.current().phase=='materialized'
    p=source.protocol();c=p.preference_loop
    store=RunStore.initialize(p,root)
    for relative in ['initial_skill.md','prompts/therapist.md','prompts/reflection.md','real_selection.jsonl']:
        write_immutable_bytes(root/relative,(old/relative).read_bytes())
    SkillMemory.initialize(root/'skills',root/'initial_skill.md')
    SkillMemory(root/'skills').bind_round(1)
    loop.freeze_sources(root)
    d=root/'round_001';od=old/'round_001'
    for name in ['evidence.json','proposal.json','evaluation_plan.json','generation_plan.json',
                 'prescribe/request.json','prescribe/response.json',
                 'generation_prescribe/request.json','generation_prescribe/response.json']:
        write_immutable_bytes(d/name,(od/name).read_bytes())
    store.advance('assessed',{'evidence':d/'evidence.json'})
    store.advance('proposed',dict(agent_request=d/'prescribe/request.json',agent_response=d/'prescribe/response.json',
        proposal=d/'proposal.json',evaluation_plan=d/'evaluation_plan.json'))
    (d/'audio').mkdir()
    assignments=loop.rows(od/'synthesis_manifest.jsonl')
    for row in assignments:row['output_wav']=str(d/'audio'/Path(row['output_wav']).name)
    loop.jsonl(d/'synthesis_manifest.jsonl',assignments)
    sidecar=build_materialization_sidecar(loop.load(d/'proposal.json',Proposal),
        loop.load(d/'evidence.json',EvidencePack),p,assignments,generation_plan_path=d/'generation_plan.json')
    loop.save(d/'materialization_sidecar.json',sidecar)
    extension=dict(execution_id=root.name,data_run_id=p.run_id,version='observed_empty_speech_resume_v1',
        sources={str(x):sha256_file(x) for x in [Path(__file__),EXTENSION]},
        scope='Known empty SG output advances within the same finite seed/mode rulebook; other errors stop. Core source files are unchanged.')
    loop.save(root/'execution_extension.json',extension)
    loop.save(root/'migration.json',dict(source_root=str(old),source_phase='materialized',
        source_files={str(x):sha256_file(x) for x in [old/'protocol.json',old/'sources.json',
            od/'generation/generation_failure.json',od/'synthesis_manifest.jsonl',od/'proposal.json',od/'generation_plan.json']},
        reuse='Actual MAS response, proposal, generation decisions, identities and completed waveforms retained; no new MAS call is claimed.',
        transcription='Old interrupted worker did not close normally. Re-transcribe preserved audio with a normally closed worker; reuse the Agent judgment only for identical three-text inputs.',
        dpo_reference=p.initial_adapter.sha256))
    store.advance('materialized',dict(synthesis_manifest=d/'synthesis_manifest.jsonl',
        materialization_sidecar=d/'materialization_sidecar.json',generation_plan=d/'generation_plan.json',
        generation_execution=root/'execution_extension.json',migration=root/'migration.json'))
    originals=sorted((od/'generation/attempts').glob('*/assignment_result.json'))
    worker=ReviewASRWorker(base_model=p.base_model_dir,adapter=c.review_asr_adapter,
        session_path=root/'refreshed_reviews/worker.json')
    reviewer=TranscriptReviewer(model=c.audio_review_model,base_url=c.audio_review_base_url,
        transcription_model=p.base_model_dir,transcription_base_model_sha256=c.review_asr_base_model_sha256,
        transcription_adapter_sha256=c.review_asr_adapter_sha256,allow_pending=c.review_allow_pending)
    entries=[]
    try:
        for index,path in enumerate(originals):
            previous=loop.read(path)
            validate_transcript_review(previous['review'],c.review_asr_adapter_sha256,
                c.review_asr_base_model_sha256,require_worker_closed=False)
            assert sha256_file(Path(previous['output_wav']))==previous['output_wav_sha256']
            receipt=worker.transcribe(previous['final_attempt_wav'],root/f'refreshed_reviews/{index:03d}.transcription.json')
            actual=loop.read(receipt);old_review=previous['review']
            identical=actual['prediction']==old_review['asr_transcription'] and actual['raw_output']==old_review['asr_raw_output']
            if identical:
                review=dict(old_review,transcription_receipt_path=str(receipt),transcription_receipt_sha256=sha256_file(receipt))
            else:
                review=reviewer.review(previous['final_attempt_wav'],previous['target_text'],previous['reference_wav'],
                    previous['reference_text'],receipt).to_dict()
            validate_transcript_review(review,c.review_asr_adapter_sha256,c.review_asr_base_model_sha256,require_worker_closed=False)
            # Completed old samples in this migration all passed on the first attempt.
            assert len(previous['attempt_records'])==1 and review['verdict']=='pass'
            refreshed=copy.deepcopy(previous)
            refreshed['review']=review;refreshed['attempt_records'][-1]['review']=review
            provenance=dict(original_assignment_result=str(path),original_assignment_result_sha256=sha256_file(path),
                old_transcription_receipt=old_review['transcription_receipt_path'],
                new_transcription_receipt=str(receipt),new_transcription_receipt_sha256=sha256_file(receipt),
                actual_new_transcription=True,agent_judgment_reused=identical,three_text_inputs_identical=identical,
                actual_new_agent_call=not identical,actual_new_generation=False)
            refreshed['refresh_provenance']=provenance
            output=root/f'refreshed_reviews/{index:03d}.assignment_result.json'
            loop.save(output,refreshed)
            entries.append(dict(sample_id=previous['sample_id'],assignment_result=str(output),
                assignment_result_sha256=sha256_file(output),**provenance))
            print(json.dumps(dict(refreshed=index+1,total=len(originals),identical_agent_input=identical)),flush=True)
        worker.close()
    finally:
        if not worker.closed:worker.abort()
    for entry in entries:validate_transcript_review(loop.read(entry['assignment_result'])['review'],
        c.review_asr_adapter_sha256,c.review_asr_base_model_sha256)
    return entries


def main(args):
    old=args.old.resolve();root=args.root.resolve()
    entries=migrate(old,root)
    # The independently executed exact-failure recovery is also reusable only
    # after its generation, Agent review and worker final hashes have completed.
    recovery=args.recovery.resolve()
    started=time.time()
    while not (recovery/'generation_summary.json').exists():
        if (recovery/'generation_failure.json').exists():raise RuntimeError('exact failure recovery did not finish')
        if time.time()-started>600:raise RuntimeError('exact failure recovery is not ready')
        time.sleep(5)
    paths=list((recovery/'attempts').glob('*/assignment_result.json'))
    assert len(paths)==1
    result=loop.read(paths[0]);p=RunStore(root).protocol();c=p.preference_loop
    validate_transcript_review(result['review'],c.review_asr_adapter_sha256,c.review_asr_base_model_sha256)
    entries.append(dict(sample_id=result['sample_id'],assignment_result=str(paths[0]),
        assignment_result_sha256=sha256_file(paths[0]),source='actual exact-assignment recovery under the declared extension'))
    loop.save(root/'round_001/generation_reuse.json',dict(assignments=entries))
    extension=loop.read(root/'execution_extension.json')
    original=loop.run_command
    def execute(command,log,*,env_overrides=None):
        for path,digest in extension['sources'].items():
            if sha256_file(Path(path))!=digest:raise ValueError('execution extension changed')
        command=list(command)
        if command[1]==str(REPO/'examples/generate_reviewed_sg.py'):
            command[1]=str(EXTENSION)
        loop.save(Path(log).with_suffix('.command.json'),command)
        return original(command,log,env_overrides=env_overrides)
    loop.run_command=execute
    cli.main(['run',str(root)])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old',type=Path);parser.add_argument('root',type=Path)
    parser.add_argument('--recovery',type=Path,required=True)
    main(parser.parse_args())
