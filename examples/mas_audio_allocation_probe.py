#!/usr/bin/env python3
"""Let MAS allocate actual audio exposures after seeing realized preference data.

The shared audio, reviews, preference pairs, role counts, per-step role order,
parent and DPO objective stay fixed. Only audio selection within each role changes.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

from pydantic import BaseModel, ConfigDict, Field

from mas_selection_dpo_probe import REPO, execute, read, rows, save
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file


class AudioCount(BaseModel):
    model_config=ConfigDict(extra='forbid')
    audio_key: str
    exposures: int=Field(ge=1)
    reason: str=Field(min_length=1,max_length=200)


class Allocation(BaseModel):
    model_config=ConfigDict(extra='forbid')
    hypothesis: str
    allocations: list[AudioCount]
    expected_observation: str


def prepare(main,root):
    from openai.lib._pydantic import to_strict_json_schema
    root.mkdir(exist_ok=False)
    p=RunProtocol.model_validate(read(main/'protocol.json'))
    assert p.preference_loop.comparison=='synthetic_only'
    d=main/'round_001'
    pairs=rows(d/'mas_dpo/train.jsonl')
    baseline=read(d/'training_allocation.json')
    roles={x['sample_id']:x for x in read(d/'materialization_sidecar.json')['samples']}
    audit={x['audio_id']:x for x in read(d/'sampling/preference_audit.json')['audio_results']}
    reviews={x['sample_id']:x for x in rows(d/'generation/reviews.jsonl')}
    greedy={x['audio_id']:x for x in rows(d/'sampling/greedy.jsonl')}
    choices={x['sample_id']:x for x in read(d/'generation_plan.json')['choices']}
    generated={x['sample_id']:x['attempt_records'][-1]['generation']
               for x in read(d/'generation/generation_summary.json')['outputs']}
    by_audio=defaultdict(list)
    for x in pairs:
        assert x['source']=='synthetic' and x['current_asr_sha256']==p.initial_adapter.sha256
        by_audio[x['audio_id']].append(x)
    keys={f'a{i:03d}':audio_id for i,audio_id in enumerate(sorted(by_audio))}
    cards=[]
    for key,audio_id in keys.items():
        pp=by_audio[audio_id];rr=reviews[audio_id]
        assert rr['verdict']=='pass'
        cards.append(dict(audio_key=key,role=roles[audio_id]['role'],
            difficulty_group=roles[audio_id]['reference_difficulty_group'],
            reference_id=choices[audio_id]['reference_id'],
            voice_group=choices[audio_id]['reference_id'].split('.')[0],
            actual_generation_mode=generated[audio_id]['generation_mode'],
            actual_instruction_id=generated[audio_id]['effective_instruction_id'],
            evidence_ids=roles[audio_id]['evidence_ids'],target=pp[0]['chosen'],
            reference_text=rr['reference_text'],coarse_review_reason=rr['reason'],
            greedy=greedy[audio_id]['prediction'],correct_random_draws=audit[audio_id]['correct_samples'],
            error_origin=audit[audio_id]['error_origin'],distinct_actual_errors=[x['rejected'] for x in pp]))
    request=dict(model=p.agent.model,reasoning={'effort':p.agent.reasoning_effort},
        store=False,stream=True,max_output_tokens=32768,
        input=[dict(role='system',content='''You allocate realized training exposures for the Cantonese patient-ASR MAS.
Every supplied audio already passed the same simple three-text generation review.
You inspect text evidence, not audio. Do not reopen review, invent word fidelity,
use a CER cutoff, or call difficult patient speech a failure. Select audio keys
and positive integer exposure counts; unlisted keys receive zero exposure in this
allocation probe, while their reviews and preference pairs remain unchanged.
Meet EACH role exposure budget exactly. Prefer useful corrective and contrasting
examples, using the actual deployment (greedy) and four stochastic transcripts.
More errors or hard speech does not automatically mean more value. Explain the
learning reason for selection; distinguish localized confusions from broad,
uncertain transcript differences without claiming which words were spoken.
Consider coverage and repetition rather than concentrating all exposure on one
example. Use only supplied keys. The model, DPO loss, pairs, role budgets and
per-step role order are fixed; only audio selection within a role can change.
This is a prospective hypothesis tested on real development data afterward.'''),
            dict(role='user',content=json.dumps(dict(proposal=read(d/'proposal.json'),
                role_exposure_budgets=baseline['role_exposures'],audio_cards=cards,
                total_exposures=baseline['total_audio_exposures']),ensure_ascii=False))],
        text={'format':{'type':'json_schema','name':'audio_allocation','strict':True,
                       'schema':to_strict_json_schema(Allocation)}})
    save(root/'request.json',request);save(root/'audio_keys.json',keys)
    response=execute_therapist_request(p,request)
    save(root/'response.json',serialize_response(response))
    if response.status!='completed':raise RuntimeError('allocation decision did not complete')
    decision=Allocation.model_validate_json(response.output_text)
    selected={x.audio_key:x for x in decision.allocations}
    if len(selected)!=len(decision.allocations) or not selected.keys()<=keys.keys():
        raise ValueError('allocation contains duplicate or unavailable audio keys')
    queues=defaultdict(list)
    for key,item in selected.items():
        audio_id=keys[key]
        queues[roles[audio_id]['role']]+=[audio_id]*item.exposures
    if {r:len(q) for r,q in queues.items()}!=baseline['role_exposures']:
        raise ValueError('MAS counts do not match actual role exposure budgets')
    rng=random.Random(p.lora.seed+3017)
    for role in sorted(queues):rng.shuffle(queues[role])
    schedule=[[queues[roles[x]['role']].pop() for x in step] for step in baseline['steps']]
    assert all(not x for x in queues.values())
    allocation=dict(baseline,rule='MAS actual-audio allocation; identical per-step role order and role totals',steps=schedule)
    save(root/'decision.json',decision.model_dump(mode='json'));save(root/'allocation.json',allocation)
    source_paths=[Path(__file__),main/'protocol.json',d/'proposal.json',d/'mas_dpo/train.jsonl',
        d/'sampling/sampling_receipt.json',d/'sampling/greedy.jsonl',d/'generation/reviews.jsonl',
        d/'generation_plan.json',d/'generation/generation_summary.json',
        d/'materialization_sidecar.json',d/'training_allocation.json',root/'allocation.json',
        REPO/'finetune/scripts/qwen3_asr_audio_dpo.py',REPO/'src/therapist_harness/dpo_training.py']
    save(root/'protocol.json',dict(source_protocol=p.model_dump(mode='json'),source_root=str(main),
        source_preferences=str(d/'mas_dpo/train.jsonl'),
        source_sha256={str(x):sha256_file(x) for x in source_paths},
        actual_new_sampling=False,actual_new_generation=False,
        selected_audio=len(selected),actual_audio_exposures=sum(map(len,schedule)),
        scope='Matched within-role exposure allocation only; no reclassification of coarse review or preference labels'))
    print(json.dumps(dict(status='allocated',selected_audio=len(selected))),flush=True)


def run(root):
    cfg=read(root/'protocol.json');p=RunProtocol.model_validate(cfg['source_protocol']);c=p.preference_loop
    for path,digest in cfg['source_sha256'].items():
        if sha256_file(Path(path))!=digest:raise ValueError('allocation probe source changed: '+path)
    execute(root,[sys.executable,str(REPO/'finetune/scripts/qwen3_asr_audio_dpo.py'),
        '--model_path',p.base_model_dir,'--policy_adapter_path',p.initial_adapter.path,
        '--reference_adapter_path',p.initial_adapter.path,'--train_file',cfg['source_preferences'],
        '--output_dir',str(root/'training'),'--max_steps',str(p.lora.optimizer_steps_per_round),
        '--real_per_step','0','--synthetic_per_step',str(c.synthetic_per_step),
        '--synthetic_weight',str(c.synthetic_weight),'--lr',str(p.lora.learning_rate),
        '--beta',str(c.beta),'--seed',str(p.lora.seed),'--synthetic_allocation_file',str(root/'allocation.json')],'training')
    execute(root,[sys.executable,str(REPO/'finetune/scripts/evaluate_asr_checkpoint.py'),
        '--base-model-path',p.base_model_dir,
        '--adapter-path',str(root/f'training/checkpoint-{p.lora.optimizer_steps_per_round}'),
        '--split-jsonl',p.feedback_split.path,'--out-dir',str(root/'evaluation'),
        '--batch-size',str(p.evaluation.batch_size),'--critical-cer-threshold',str(p.evaluation.critical_cer_threshold),
        '--max-new-tokens','256'],'evaluation')
    save(root/'results.json',dict(status='complete',metrics=read(root/'evaluation/summary.json')['metrics'],
        training=read(root/'training/training_result.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','all']);parser.add_argument('root',type=Path)
    parser.add_argument('--main-root',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    if args.stage in ['prepare','all']:
        if args.main_root is None:parser.error('prepare requires --main-root')
        prepare(args.main_root.resolve(),root)
    if args.stage in ['run','all']:run(root)
