#!/usr/bin/env python3
"""Same-parent short DPO comparison on the completed paired generation probe."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from pydantic import BaseModel, ConfigDict

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.schema import RunProtocol
from therapist_harness.store import sha256_file, write_immutable_json, write_immutable_bytes
from therapist_harness.training_allocation import compile_role_allocation


class Choice(BaseModel):
    model_config=ConfigDict(extra='forbid')
    case_id: str
    audio_id: str
    reason: str


class Selection(BaseModel):
    model_config=ConfigDict(extra='forbid')
    choices: list[Choice]
    hypothesis: str
    limitation: str


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def save(path,value):write_immutable_json(Path(path),value)
def jsonl(path,value):write_immutable_bytes(Path(path),''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in value).encode())


def prepare(base,root):
    from openai.lib._pydantic import to_strict_json_schema
    old=base.parent/'mas_tts_only_dpo_seed42_20260908_retry_v2'
    p=RunProtocol.model_validate_json((old/'protocol.json').read_text())
    audit=read(base/'probe24_sampling/sampling/preference_audit.json')
    audit_by_id={x['audio_id']:x for x in audit['audio_results']}
    cases=read(base/'probe24/cases.json')
    outputs=read(base/'probe24/generation/generation_summary.json')['outputs']
    available=[]
    for case in cases:
        options=[]
        for row in outputs:
            if not row['sample_id'].startswith(case['case_id']+'.'):continue
            options.append(dict(audio_id=row['sample_id'],final_review=row['review']['verdict'],
                final_asr=row['review']['asr_transcription'],sampling=audit_by_id[row['sample_id']],
                attempts=[dict(mode=x['generation']['generation_mode'],instruction=x['generation']['effective_prompt_text'],
                    asr=x['review']['asr_transcription'],verdict=x['review']['verdict'],reason=x['review']['reason']) for x in row['attempt_records']]))
        available.append(dict(case_id=case['case_id'],role=case['role'],target_text=case['target_text'],options=options))
    request=dict(model=p.agent.model,reasoning={'effort':p.agent.reasoning_effort},store=False,stream=True,
        max_output_tokens=12000,input=[dict(role='system',content='''Select exactly one already-generated audio per case for a small Cantonese patient-ASR DPO probe.
Use only supplied audio IDs from that case. Keep the same six contents and patients.
Choose plausible training examples using actual generation/review history and
current-ASR sampling. ASR error is not proof of TTS error or of word fidelity;
more errors is not automatically more useful. Do not invent acoustic properties.
All-correct choices are permitted and honestly contribute no DPO pair. Explain
how the choice may support the content role, without claiming downstream benefit
before training. Real-development results will decide. No scores or CER cutoff.
Prefer a stable, natural realization; failed initial generations recovered by
the rulebook are identified as such. You may retain the original reference when
alternatives lack evidence. Return the structured selection only.'''),
        dict(role='user',content=json.dumps(dict(cases=available,training=dict(parent='initial R-only',
            steps=4,audio_per_step=8,objective='standard DPO',role_allocation='same requested roles adjusted for pair availability')),
            ensure_ascii=False))],text={'format':{'type':'json_schema','name':'mas_probe_selection','strict':True,
            'schema':to_strict_json_schema(Selection)}})
    save(root/'selection_request.json',request)
    response=execute_therapist_request(p,request)
    save(root/'selection_response.json',serialize_response(response))
    if response.status!='completed':raise RuntimeError('MAS selection did not complete')
    selection=Selection.model_validate_json(response.output_text)
    choices={x.case_id:x.audio_id for x in selection.choices}
    if len(choices)!=len(selection.choices) or set(choices)!={x['case_id'] for x in cases}:
        raise ValueError('selection must cover each case once')
    for case in available:
        if choices[case['case_id']] not in {x['audio_id'] for x in case['options']}:
            raise ValueError('MAS chose unavailable audio')
    rng=random.Random(42)
    arms=dict(original=[x['case_id']+'.ref0.default' for x in cases],
              mas_selected=[choices[x['case_id']] for x in cases],
              balanced_random=[rng.choice([y['audio_id'] for y in x['options']]) for x in available])
    save(root/'selection.json',selection.model_dump(mode='json'))
    save(root/'arms.json',arms)
    all_pairs=rows(base/'probe24_sampling/sampling/preferences.jsonl')
    inputs={x['audio_id']:x for x in rows(base/'probe24_sampling/sampling/validated_inputs.jsonl')}
    for name,ids in arms.items():
        pairs=[row for audio_id in ids for row in all_pairs if row['audio_id']==audio_id]
        jsonl(root/name/'preferences.jsonl',pairs)
        jsonl(root/name/'selected_inputs.jsonl',[inputs[x] for x in ids])
        sidecar=dict(samples=[dict(sample_id=audio_id,role=case['role']) for audio_id,case in zip(ids,cases)])
        allocation=compile_role_allocation(sidecar,pairs,steps=4,per_step=8,seed=42)
        save(root/name/'allocation.json',allocation)
        save(root/name/'selection_receipt.json',dict(selected_audio_ids=ids,
            shared_sampling_receipt=str(base/'probe24_sampling/sampling/sampling_receipt.json'),
            shared_sampling_receipt_sha256=sha256_file(base/'probe24_sampling/sampling/sampling_receipt.json'),
            shared_preference_manifest_sha256=sha256_file(base/'probe24_sampling/sampling/preferences.jsonl'),
            selected_preference_manifest_sha256=sha256_file(root/name/'preferences.jsonl'),
            audio_count=len(ids),paired_audio=len({x['audio_id'] for x in pairs}),pair_count=len(pairs),
            source='explicit subset of the shared actual 96-draw receipt; no new sampling is claimed'))
    sources=[Path(__file__),REPO/'src/therapist_harness/dpo_training.py',REPO/'src/therapist_harness/training_allocation.py',
             REPO/'finetune/scripts/qwen3_asr_audio_dpo.py',REPO/'finetune/scripts/evaluate_asr_checkpoint.py']
    save(root/'protocol.json',dict(source_protocol=p.model_dump(mode='json'),
        source_sha256={str(x):sha256_file(x) for x in sources},
        steps=4,synthetic_per_step=8,seed=42,lr=1e-5,reference='frozen initial R-only',
        criterion='compare pooled and patient-macro real-development CER to the common parent and matched controls',
        scope='small mechanism/dose probe; same six intended contents and patients; actual paired pools may differ'))
    print(json.dumps(dict(status='selected',arms=arms),ensure_ascii=False),flush=True)


def execute(root,command,name):
    save(root/(name+'.command.json'),command)
    with (root/(name+'.log')).open('x') as log:
        subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)


def run(base,root):
    cfg=read(root/'protocol.json');p=cfg['source_protocol']
    for path,digest in cfg['source_sha256'].items():
        if sha256_file(Path(path))!=digest:raise ValueError('frozen probe source changed: '+path)
    completed={}
    for name in read(root/'arms.json'):
        d=root/name
        signature=(sha256_file(d/'preferences.jsonl'),sha256_file(d/'allocation.json'))
        if signature in completed:
            save(d/'reuse.json',dict(source_arm=completed[signature],reason='identical actual preferences and exposure schedule; one training execution'))
            continue
        execute(d,[sys.executable,str(REPO/'finetune/scripts/qwen3_asr_audio_dpo.py'),
            '--model_path',p['base_model_dir'],'--policy_adapter_path',p['initial_adapter']['path'],
            '--reference_adapter_path',p['initial_adapter']['path'],'--train_file',str(d/'preferences.jsonl'),
            '--output_dir',str(d/'training'),'--max_steps','4','--real_per_step','0','--synthetic_per_step','8',
            '--synthetic_weight','1','--lr','1e-5','--seed','42','--synthetic_allocation_file',str(d/'allocation.json')],'training')
        execute(d,[sys.executable,str(REPO/'finetune/scripts/evaluate_asr_checkpoint.py'),
            '--base-model-path',p['base_model_dir'],'--adapter-path',str(d/'training/checkpoint-4'),
            '--split-jsonl',p['feedback_split']['path'],'--out-dir',str(d/'evaluation'),
            '--batch-size','8','--critical-cer-threshold','.5','--max-new-tokens','256'],'evaluation')
        completed[signature]=name
        print(json.dumps(dict(arm=name,metrics=read(d/'evaluation/summary.json')['metrics'])),flush=True)
    report(root)


def report(root):
    p=read(root/'protocol.json')['source_protocol']
    parent=read(Path(p['initial_adapter_dev_predictions']['path']).parent/'summary.json')['metrics']
    result={'parent':parent}
    for name in read(root/'arms.json'):
        d=root/name;source=read(d/'reuse.json')['source_arm'] if (d/'reuse.json').exists() else name
        result[name]=dict(metrics=read(root/source/'evaluation/summary.json')['metrics'],
                         selection=read(d/'selection_receipt.json'),training_source=source)
    save(root/'results.json',result)
    lines=['# 同parent生成选择DPO短步probe','',
           '三策略共享原始24音频/96次采样来源，各选同六个内容—患者组合。每支4步×8音频曝光。511为开发反馈。','',
           '|策略|pooled CER|句均 CER|患者宏 CER|相对parent pooled变化|','|---|---:|---:|---:|---:|']
    for name,data in result.items():
        m=(data if name=='parent' else data['metrics'])['overall']
        lines.append(f"|{name}|{m['pooled_cer']*100:.4f}%|{m['mean_utterance_cer']*100:.4f}%|{m['patient_macro_cer']*100:.4f}%|{(m['pooled_cer']-parent['overall']['pooled_cer'])*100:+.4f}pp|")
    write_immutable_bytes(root/'report.zh.md',('\n'.join(lines)+'\n').encode())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path);parser.add_argument('--stage',choices=['all','prepare','run','report'],default='all')
    args=parser.parse_args();base=args.base.resolve();root=base/'selection_dpo_probe'
    if args.stage in ('all','prepare'):
        root.mkdir(exist_ok=False);prepare(base,root)
    if args.stage in ('all','run'):run(base,root)
    if args.stage=='report':report(root)
