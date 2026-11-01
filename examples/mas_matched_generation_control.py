#!/usr/bin/env python3
"""Explicit matched random-generation control for a saved first-round MAS proposal.

The proposal is reused with provenance. This is an experimental control, not an
independent MAS response or a second closed-loop journal.
"""

import argparse
import json
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO, execute, jsonl, read, rows, save
from therapist_harness.generation_controller import materialize_control
from therapist_harness.materialize import build_materialization_sidecar, materialize_proposal
from therapist_harness.schema import EvidencePack, Proposal, RunProtocol
from therapist_harness.store import sha256_file, sha256_directory
from therapist_harness.training_allocation import compile_role_allocation
from therapist_harness.generation_design import role_delivery


def verify(root):
    for path,digest in read(root/'provenance.json')['source_sha256'].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('matched control source changed: '+path)


def prepare(main_root,root):
    root.mkdir(exist_ok=False)
    source=main_root/'round_001'
    p=RunProtocol.model_validate(read(main_root/'protocol.json'))
    assert p.preference_loop.comparison=='synthetic_only'
    config=read(Path(p.preference_loop.generation_design_path))
    config['strategy']='balanced_random'
    save(root/'generation_design.json',config)
    data=p.model_dump(mode='json')
    data['preference_loop'].update(generation_design_path=str(root/'generation_design.json'),
        generation_design_sha256=sha256_file(root/'generation_design.json'))
    p=RunProtocol.model_validate(data)
    save(root/'protocol.json',data)
    evidence=EvidencePack.model_validate(read(source/'evidence.json'))
    proposal=Proposal.model_validate(read(source/'proposal.json'))
    (root/'audio').mkdir()
    original=materialize_proposal(proposal,evidence,p,output_root=root/'audio')
    assignments=materialize_control(p,proposal,original,directory=root,history=[])
    main_assignments=rows(source/'synthesis_manifest.jsonl')
    assert len(assignments)==len(main_assignments)
    for a,b in zip(assignments,main_assignments):
        for key in ['sample_id','target_text','seed']:
            assert a[key]==b[key],(key,a['sample_id'])
    sidecar=build_materialization_sidecar(proposal,evidence,p,assignments,
                                         generation_plan_path=root/'generation_plan.json')
    jsonl(root/'manifest.jsonl',assignments)
    save(root/'materialization_sidecar.json',sidecar.model_dump(mode='json'))
    sources=read(main_root/'sources.json')
    sources.update({str(x):sha256_file(x) for x in [Path(__file__),main_root/'protocol.json',
        source/'evidence.json',source/'proposal.json',source/'synthesis_manifest.jsonl',
        source/'generation_plan.json',source/'materialization_sidecar.json',
        root/'generation_design.json',root/'generation_plan.json',root/'manifest.jsonl']})
    save(root/'provenance.json',dict(control_id=root.name,source_root=str(main_root),
        source_round=1,source_sha256=sources,proposal_source='actual main MAS response reused explicitly',
        same_sample_ids_for_pairing=True,independent_mas_call=False,independent_sampling_receipt=True,
        matched=['target texts','assigned patients','TTS seeds','intended roles','audio budget',
                 'current parent','frozen DPO reference','decode settings','optimizer and exposure budget'],
        intervention='fixed-seed random same-patient reference and permitted recipe',
        scope='Control only, no checkpoint promotion or independent closed-loop journal'))
    verify(root)
    print(json.dumps(dict(status='prepared',audio=len(assignments),actual_generation=False)),flush=True)


def run(root):
    verify(root)
    p=RunProtocol.model_validate(read(root/'protocol.json'));c=p.preference_loop
    command=[sys.executable,str(REPO/'examples/generate_reviewed_sg.py'),
        '--manifest',str(root/'manifest.jsonl'),'--out-dir',str(root/'generation'),
        '--generation-plan',str(root/'generation_plan.json'),'--review-mode','asr_transcript',
        '--review-asr-worker','--audio-review-model',c.audio_review_model,
        '--audio-review-base-url',c.audio_review_base_url,
        '--review-asr-model',p.base_model_dir,'--review-asr-adapter',c.review_asr_adapter,
        '--review-asr-adapter-sha256',c.review_asr_adapter_sha256,
        '--review-asr-base-model-sha256',c.review_asr_base_model_sha256]
    if c.review_allow_pending:command+=['--review-allow-pending']
    execute(root,command,'generation')
    verify(root)
    if not rows(root/'generation/accepted.jsonl'):
        save(root/'results.json',dict(status='no_admitted_audio',training_executed=False));return
    command=[sys.executable,str(REPO/'examples/sample_asr_preferences.py'),'sample',
        '--inputs',str(root/'generation/accepted.jsonl'),'--real-train',p.real_train.path,
        '--feedback',p.feedback_split.path,'--out-dir',str(root/'sampling'),
        '--base-model',p.base_model_dir,'--current-adapter',p.initial_adapter.path,
        '--expected-parent-sha256',p.initial_adapter.sha256,'--samples',str(c.samples_per_audio),
        '--seed',str(p.lora.seed),'--temperature',str(c.temperature),'--top-p',str(c.top_p),
        '--max-new-tokens',str(c.max_new_tokens)]
    if c.include_greedy:command+=['--include-greedy']
    execute(root,command,'sampling')
    pairs=rows(root/'sampling/preferences.jsonl')
    if not pairs:
        save(root/'results.json',dict(status='all_correct_no_update',training_executed=False));return
    assert {x['current_asr_sha256'] for x in pairs}=={p.initial_adapter.sha256}
    assert sha256_directory(Path(p.initial_adapter.path))==p.initial_adapter.sha256
    allocation=compile_role_allocation(read(root/'materialization_sidecar.json'),pairs,
        steps=p.lora.optimizer_steps_per_round,per_step=c.synthetic_per_step,seed=p.lora.seed)
    save(root/'allocation.json',allocation)
    command=[sys.executable,str(REPO/'finetune/scripts/qwen3_asr_audio_dpo.py'),
        '--model_path',p.base_model_dir,'--policy_adapter_path',p.initial_adapter.path,
        '--reference_adapter_path',p.initial_adapter.path,
        '--train_file',str(root/'sampling/preferences.jsonl'),'--output_dir',str(root/'training'),
        '--max_steps',str(p.lora.optimizer_steps_per_round),'--real_per_step','0',
        '--synthetic_per_step',str(c.synthetic_per_step),'--synthetic_weight',str(c.synthetic_weight),
        '--lr',str(p.lora.learning_rate),'--beta',str(c.beta),'--seed',str(p.lora.seed)]
    if c.training_allocation=='role_budget_v1':
        command+=['--synthetic_allocation_file',str(root/'allocation.json')]
    execute(root,command,'training')
    execute(root,[sys.executable,str(REPO/'finetune/scripts/evaluate_asr_checkpoint.py'),
        '--base-model-path',p.base_model_dir,'--adapter-path',str(root/f'training/checkpoint-{p.lora.optimizer_steps_per_round}'),
        '--split-jsonl',p.feedback_split.path,'--out-dir',str(root/'evaluation'),
        '--batch-size',str(p.evaluation.batch_size),'--critical-cer-threshold',str(p.evaluation.critical_cer_threshold),
        '--max-new-tokens','256'],'evaluation')
    verify(root)
    result=dict(status='complete_matched_control',metrics=read(root/'evaluation/summary.json')['metrics'],
        sampling=read(root/'sampling/preference_audit.json'),training=read(root/'training/training_result.json'),
        actual_delivery=role_delivery(read(root/'materialization_sidecar.json')['samples'],
            [dict(x['review'],sample_id=x['sample_id']) for x in read(root/'generation/generation_summary.json')['outputs']],
            pairs,rows(root/'training/training_exposures.jsonl')))
    save(root/'results.json',result)
    print(json.dumps(result['metrics']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','all'])
    parser.add_argument('root',type=Path)
    parser.add_argument('--main-root',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    if args.stage in ['prepare','all']:
        if args.main_root is None:parser.error('prepare needs --main-root')
        prepare(args.main_root.resolve(),root)
    if args.stage in ['run','all']:run(root)
