#!/usr/bin/env python3
"""Eight new zero-shot cells against verified existing instruct2 counterparts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import mas_literal_matched_sg_probe_v1 as base

PREVIOUS=base.B/'contextual_matched_SG24_v2'
ROOT=base.B/'mode_reference_probe_v1'
base.ROOT=ROOT


def prepare():
    decision=base.read(PREVIOUS/'MAS_reflection/decision.json')
    assert decision['next_action']=='mode_reference_probe' and decision['no_current_proposal_training']
    p=base.RunProtocol.model_validate(base.read(PREVIOUS/'source_protocol.json'))
    ROOT.mkdir(exist_ok=False)
    base.save(ROOT/'source_protocol.json',p.model_dump(mode='json'))
    base.save(ROOT/'MAS_decision.json',decision)
    inherited=base.read(PREVIOUS/'sources.json')
    files=[Path(__file__),PREVIOUS/'MAS_reflection/decision.json',PREVIOUS/'delivery_report.json',
           base.B/'instruct2_reference_conditioning_audit_v1/result.json']
    for arm in ['A','B']:
        old=PREVIOUS/arm;out=ROOT/arm;out.mkdir();(out/'audio').mkdir()
        subset=ROOT/'reused_instruct2'/arm;subset.mkdir(parents=True)
        original=base.rows(old/'manifest.jsonl')
        manifest=[dict(x,output_wav=str(out/'audio'/f'{x["sample_id"]}.wav')) for x in original if x['sample_id'].startswith('literal24.p0.')]
        ids={x['sample_id'] for x in manifest};assert len(ids)==4
        plan=base.read(old/'generation_plan.json')
        choices=[dict(x,mode='zero_shot') for x in plan['choices'] if x['sample_id'] in ids]
        assert all(x['instruction_id']=='cantonese_zh' for x in choices)
        base.jsonl(out/'manifest.jsonl',manifest)
        base.save(out/'generation_plan.json',dict(schema_version='generation_plan_v1',choices=choices))
        sidecar=base.read(old/'role_sidecar.json')
        base.save(out/'role_sidecar.json',dict(schema_version='mode_reference_roles_v1',samples=[x for x in sidecar['samples'] if x['sample_id'] in ids]))
        base.load_generation_choices(out/'generation_plan.json',{x['sample_id']:x for x in manifest})
        receipt=base.read(old/'sampling/sampling_receipt.json')
        assert receipt['status']=='complete' and receipt['current_asr_sha256']==p.initial_adapter.sha256
        generated=[x for x in base.read(old/'generation/generation_summary.json')['outputs'] if x['sample_id'] in ids]
        assert len(generated)==4 and all(x['admitted'] and x['generation_mode']=='instruct2' and len(x['attempt_records'])==1 for x in generated)
        base.save(subset/'generation_outputs.json',generated)
        for filename in ['validated_inputs.jsonl','candidates.jsonl','greedy.jsonl','preferences.jsonl']:
            values=[x for x in base.rows(old/'sampling'/filename) if x['audio_id'] in ids]
            base.jsonl(subset/filename,values)
        inputs=base.rows(subset/'validated_inputs.jsonl');draws=base.rows(subset/'candidates.jsonl')
        assert len(inputs)==4 and len(draws)==16
        assert all(base.sha256_file(Path(x['audio']))==x['audio_sha256'] for x in inputs)
        for key,filename in [('validated_inputs_sha256','validated_inputs.jsonl'),('candidates_sha256','candidates.jsonl'),('greedy_sha256','greedy.jsonl')]:
            assert receipt[key]==base.sha256_file(old/'sampling'/filename)
        upstream=[old/'manifest.jsonl',old/'generation_plan.json',old/'role_sidecar.json',old/'generation/generation_summary.json']
        upstream += [old/'sampling'/f for f in ['validated_inputs.jsonl','candidates.jsonl','greedy.jsonl','preferences.jsonl','sampling_receipt.json']]
        base.save(subset/'reuse_receipt.json',dict(actual_new_audio=False,actual_new_sampling=False,selected_case_ids=sorted(ids),
            selected_audio=4,selected_stochastic_draws=16,upstream_full_audio=12,upstream_full_stochastic_draws=48,
            original_complete_receipt=receipt,source_sha256={str(f):base.sha256_file(f) for f in upstream},
            subset_sha256={str(f):base.sha256_file(f) for f in subset.glob('*.json*')},
            scope='Explicit subset views of an actual12audio/48drawreceipt; not a fabricated complete4audiosampling receipt. Old files and sources untouched.'))
        files+=upstream+list(subset.glob('*.json*'))+list(out.glob('*.json*'))
    audit=base.read(base.B/'instruct2_reference_conditioning_audit_v1/result.json')
    base.save(ROOT/'sources.json',{**inherited,**audit['source_sha256'],**{str(f):base.sha256_file(f) for f in files}})
    base.save(ROOT/'plan.json',dict(version='mode_reference_probe_v1',new_audio=8,reused_audio=8,total_cells=16,
        factors='2texts x2patients x2within-patient references x2modes',
        intervention='Newzero_shot+cantonese_zh versus existinginstruct2+cantonese_zh; exactsame target/patient/reference/TTSseed/logicalcase/ASRseed. Mode intrinsically changes both LM reference-speech tokens and reference text conditioning; no isolated audio-token-only claim.',
        first_attempts='Primarymechanism observations; existingrulebookfallbacks reportedseparately, actualfinalmode always recorded.',
        no_ASR_training=True,current_literal_target_remains_closed=True,
        no_winner_by_errors_pairs_or_SG_CER=True,
        review='Same three-text coarseAgent/rulebook, noCERthreshold/phonemeaudit/wordfidelityclaim.',
        provenance='Existinginstruct2 subset views keepfulloriginalsampling receipts andexplicitselection. Newzero_shot arms haveownactualsampling receipts.',
        smoke='This8newaudio boundedmechanismprobe usesunchangedproductionSG/review/ASR; manifest/provenance preflight beforegeneration. It is not a formal ASRtraining run.'))
    base.verify()
    print(json.dumps(dict(status='prepared',new_audio=8,reused_audio=8,total_cells=16)),flush=True)


def generate(arm):
    p=base.verify();c=p.preference_loop;out=ROOT/arm
    command=[sys.executable,str(base.REPO/'examples/generate_reviewed_sg_recovery_v2.py'),
        '--manifest',str(out/'manifest.jsonl'),'--out-dir',str(out/'generation'),
        '--generation-plan',str(out/'generation_plan.json'),'--review-mode','asr_transcript',
        '--review-asr-worker','--review-allow-pending','--audio-review-model',c.audio_review_model,
        '--audio-review-base-url',c.audio_review_base_url,'--review-asr-model',p.base_model_dir,
        '--review-asr-adapter',c.review_asr_adapter,'--review-asr-adapter-sha256',c.review_asr_adapter_sha256,
        '--review-asr-base-model-sha256',c.review_asr_base_model_sha256]
    base.save(out/'generation_launch.json',dict(command=command,start=time.time()))
    with (out/'generation.log').open('x') as log:
        subprocess.run(command,cwd=base.REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    base.verify()
    print(json.dumps(dict(stage='generation_complete',arm=arm)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','generate','sample'])
    parser.add_argument('--arm',choices=['A','B'],default='A')
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='generate':generate(args.arm)
    else:base.sample(args.arm)
