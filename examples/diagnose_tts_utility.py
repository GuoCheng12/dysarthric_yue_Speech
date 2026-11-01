"""Frozen full-batch TTS content screening for the utility investigation."""
import argparse
from collections import Counter
from pathlib import Path
import json
import os
import statistics
import subprocess
import sys

from tts_increment_control import REPO, ROOT as CONTROL, rows, read, jsonl, sha256_file
from tts_increment_control import write_immutable_json as save

ROOT = CONTROL.parent / 'tts_utility_diagnosis_v1'
MAS = CONTROL.parent / 'real2117_mas576_hardref_v1'
REF = CONTROL.parent / 'tts_reference_policy_v1'
ADAPTER = CONTROL.parent / 'tts_increment_r_only1541_v1/r_only/seed_42/training/checkpoint-50'


def prepare():
    import soundfile as sf
    assert not ROOT.exists()
    records = []
    pairs = read(MAS/'pairs.json')
    for p in pairs:
        new = read(MAS/'selected'/f"{p['sample_id']}.json")['record']
        for variant, rec in [('mas_old', p['old_record']), ('mas_hard', new)]:
            records.append({'utt_id':variant+'::'+p['sample_id'], 'speaker_id':p['speaker_id'],
                'prompt_id':'mas::'+p['target_text'], 'clean_gt':p['target_text'],
                'zero_shot_bucket':'medium', 'audio':rec['output_wav'], 'variant':variant,
                'pair_id':p['sample_id'], 'reference_text':rec['reference_text'],
                'reference_audio':rec['reference_wav']})
    q = {r['utt_id']:r for r in read(CONTROL/'selection.json')['Q']}
    for p in read(REF/'pairs.json'):
        original=q[p['target']['utt_id']]
        rec=read(REF/'selected'/f"{p['sample_id']}.json")['records']['hard_multi']
        for variant,path in [('q_real',original['audio']),('q_hard',rec['output_wav'])]:
            records.append({**original,'utt_id':variant+'::'+p['sample_id'], 'audio':path,
                'variant':variant,'pair_id':p['sample_id'],'reference_text':rec['reference_text'],
                'reference_audio':rec['reference_wav']})
    for r in records:
        r['audio_sha256']=sha256_file(Path(r['audio']))
        info=sf.info(r['audio']); r['seconds']=info.duration
    assert Counter(r['variant'] for r in records)==dict.fromkeys(['mas_old','mas_hard','q_real','q_hard'],576)
    jsonl(ROOT/'screening/input.jsonl', records)
    save(ROOT/'screening/protocol.json',{'purpose':'ASR disagreement screen; not ground-truth audio transcription',
        'adapter':str(ADAPTER),'rows':len(records),'selection':'all576 per variant, no filtering',
        'mas_bucket':'placeholder medium; no MAS utterance-severity inference',
        'input_sha256':sha256_file(ROOT/'screening/input.jsonl'),
        'source_sha256':sha256_file(Path(__file__))})


def decode():
    command=[sys.executable,str(REPO/'finetune/scripts/evaluate_asr_checkpoint.py'),
        '--base-model-path','/data/qwen3-asr/models/Qwen3-ASR-1.7B',
        '--adapter-path',str(ADAPTER),'--split-jsonl',str(ROOT/'screening/input.jsonl'),
        '--out-dir',str(ROOT/'screening/asr'),'--batch-size','8',
        '--critical-cer-threshold','.5','--max-new-tokens','256']
    save(ROOT/'screening/execution.json',{'command':command,'pid':os.getpid()})
    subprocess.run(command,check=True)
    summarize()


def summarize():
    from therapist_harness.text import text_normalize,edit_distance
    records=rows(ROOT/'screening/asr/predictions.jsonl')
    source={r['utt_id']:r for r in rows(ROOT/'screening/input.jsonl')}
    assert len(records)==len(source)==2304
    groups={}
    audit=[]
    for r in records:
        s=source[r['utt_id']]
        a,b=text_normalize(s['clean_gt']),text_normalize(r['prediction'])
        cer=edit_distance(a,b)/len(a)
        assert abs(cer-r['cer'])<1e-12
        ref=text_normalize(s['reference_text'])
        # Exact >=4-character reference fragments absent from the target: a clue,
        # not a proof of acoustic prompt leakage.
        fragments=sorted({ref[i:i+4] for i in range(max(0,len(ref)-3))
                          if ref[i:i+4] in b and ref[i:i+4] not in a})
        audit.append({**s,'prediction':r['prediction'],'cer':cer,
            'length_ratio':len(b)/len(a),'reference_fragment_matches':fragments})
    for variant in sorted({r['variant'] for r in audit}):
        group=[r for r in audit if r['variant']==variant]
        groups[variant]={'rows':len(group),'mean_cer':statistics.mean(r['cer'] for r in group),
            'perfect':sum(r['cer']==0 for r in group),
            'cer_over_25pct':sum(r['cer']>.25 for r in group),
            'cer_over_50pct':sum(r['cer']>.5 for r in group),
            'under_one_second':sum(r['seconds']<1 for r in group),
            'reference_fragment_matches':sum(bool(r['reference_fragment_matches']) for r in group),
            'median_seconds':statistics.median(r['seconds'] for r in group)}
    jsonl(ROOT/'screening/audit.jsonl',audit)
    save(ROOT/'screening/results.json',groups)
    print(json.dumps(groups,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','decode','summarize'],required=True)
    globals()[p.parse_args().stage]()
