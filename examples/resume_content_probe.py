#!/usr/bin/env python3
"""Preserve completed content-probe audio across an observed SG failure."""

import argparse
import copy
import json
from pathlib import Path

import mas_matched_generation_control as runner
from mas_selection_dpo_probe import REPO,jsonl,read,rows,save
from therapist_harness.materialize import build_materialization_sidecar
from therapist_harness.review_worker import ReviewASRWorker
from therapist_harness.schema import EvidencePack,Proposal,RunProtocol
from therapist_harness.store import sha256_file,write_immutable_bytes
from therapist_harness.transcript_review import TranscriptReviewer,validate_transcript_review

EXTENSION=REPO/'examples/generate_reviewed_sg_recovery_v2.py'


def main(old,root):
    runner.verify(old);root.mkdir(exist_ok=False)
    p=RunProtocol.model_validate(read(old/'protocol.json'));c=p.preference_loop
    for name in ['protocol.json','evidence.json','proposal.json','request.json','response.json',
                 'response_reuse.json','response_extraction.json','donor_mapping.json','generation_plan.json','generation_design.json']:
        write_immutable_bytes(root/name,(old/name).read_bytes())
    (root/'audio').mkdir()
    assignments=rows(old/'manifest.jsonl')
    for row in assignments:row['output_wav']=str(root/'audio'/Path(row['output_wav']).name)
    jsonl(root/'manifest.jsonl',assignments)
    sidecar=build_materialization_sidecar(Proposal.model_validate(read(root/'proposal.json')),
        EvidencePack.model_validate(read(root/'evidence.json')),p,assignments,generation_plan_path=root/'generation_plan.json')
    save(root/'materialization_sidecar.json',sidecar.model_dump(mode='json'))
    save(root/'migration.json',dict(source_root=str(old),source_sha256={str(f):sha256_file(f)
        for f in [old/'provenance.json',old/'generation/generation_failure.json',old/'generation.log']},
        reason='Sample 72 emitted 0.08 seconds, rejected by the unchanged existing waveform validator. New execution advances only this observed short-output condition within the original finite retry rulebook.',
        actual_new_mas_call=False,original_data_run_id=p.run_id,
        preserved='Target text, reference, recipe, seed and IDs unchanged; prior audio retained and ASR receipts refreshed after interrupted worker'))
    sources=read(old/'provenance.json')['source_sha256']
    sources.update({str(f):sha256_file(f) for f in [Path(__file__),EXTENSION,REPO/'examples/generate_reviewed_sg_resumable.py',root/'migration.json',root/'manifest.jsonl',
        root/'protocol.json',root/'materialization_sidecar.json',root/'generation_plan.json']})
    save(root/'provenance.json',dict(source_sha256=sources,source_root=str(old),actual_new_mas_call=False,
        intervention='Same 24-content probe continued with explicitly recorded short-SG-output recovery'))
    originals=sorted((old/'generation/attempts').glob('*/assignment_result.json'))
    worker=ReviewASRWorker(base_model=p.base_model_dir,adapter=c.review_asr_adapter,
        session_path=root/'refreshed_reviews/worker.json')
    reviewer=TranscriptReviewer(model=c.audio_review_model,base_url=c.audio_review_base_url,
        transcription_model=p.base_model_dir,transcription_base_model_sha256=c.review_asr_base_model_sha256,
        transcription_adapter_sha256=c.review_asr_adapter_sha256,allow_pending=c.review_allow_pending)
    entries=[]
    try:
        for index,path in enumerate(originals):
            previous=read(path);review=previous['review']
            validate_transcript_review(review,c.review_asr_adapter_sha256,c.review_asr_base_model_sha256,require_worker_closed=False)
            assert sha256_file(Path(previous['output_wav']))==previous['output_wav_sha256']
            receipt=worker.transcribe(previous['final_attempt_wav'],root/f'refreshed_reviews/{index:03d}.transcription.json')
            actual=read(receipt)
            same=actual['prediction']==review['asr_transcription'] and actual['raw_output']==review['asr_raw_output']
            if same:review=dict(review,transcription_receipt_path=str(receipt),transcription_receipt_sha256=sha256_file(receipt))
            else:review=reviewer.review(previous['final_attempt_wav'],previous['target_text'],previous['reference_wav'],previous['reference_text'],receipt).to_dict()
            assert review['verdict']==previous['review']['verdict']
            refreshed=copy.deepcopy(previous);refreshed['review']=review;refreshed['attempt_records'][-1]['review']=review
            provenance=dict(original_assignment_result=str(path),original_assignment_result_sha256=sha256_file(path),
                actual_new_transcription=True,actual_new_generation=False,three_text_inputs_identical=same,
                actual_new_agent_call=not same,agent_judgment_reused=same)
            refreshed['refresh_provenance']=provenance;output=root/f'refreshed_reviews/{index:03d}.assignment_result.json'
            save(output,refreshed)
            entries.append(dict(sample_id=previous['sample_id'],assignment_result=str(output),assignment_result_sha256=sha256_file(output),**provenance))
            if index%10==0:print(json.dumps(dict(refreshed=index+1,total=len(originals),identical_agent_input=same)),flush=True)
        worker.close()
    finally:
        if not worker.closed:worker.abort()
    for entry in entries:validate_transcript_review(read(entry['assignment_result'])['review'],c.review_asr_adapter_sha256,c.review_asr_base_model_sha256)
    save(root/'generation_reuse.json',dict(assignments=entries))
    execute=runner.execute
    def run(directory,command,name):
        command=list(command)
        if name=='generation':
            command[1]=str(EXTENSION);command+=['--reuse-manifest',str(root/'generation_reuse.json')]
        return execute(directory,command,name)
    runner.execute=run;runner.run(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('old',type=Path);parser.add_argument('root',type=Path)
    args=parser.parse_args();main(args.old.resolve(),args.root.resolve())
