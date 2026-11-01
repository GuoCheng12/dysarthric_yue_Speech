"""Paired diagnostic pilot: Cantonese instruction and LLM reference-token removal.

This creates diagnostic audio only. It does not add anything to ASR training.
The original frozen generator and all old artifacts remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
BASE=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1')
ROOT=BASE/'tts_utility_diagnosis_v1/instruction_pilot'
GENERIC='You are a helpful assistant.<|endofprompt|>'
CANTONESE='You are a helpful assistant. 请用广东话表达。<|endofprompt|>'
MODES=('zero_shot_cantonese','instruct2_generic','instruct2_cantonese')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def prepare():
    source=BASE/'real2117_mas576_hardref_v1/pairs.json'
    pairs=json.loads(source.read_text())
    selected=[r for r in pairs if r['target_text']=='三嬸啱啱飲完杯淡茶。' or r['sample_id'] in ('mas0042','mas0022','mas0333')]
    assert len(selected)==19
    save(ROOT/'protocol.json',{'selection':'all16 三嬸 examples plus mas0042/mas0022/mas0333 observed failures',
        'population':'targeted failure cases; not an unbiased generator-quality estimate',
        'modes':list(MODES),'generic_instruction':GENERIC,'cantonese_instruction':CANTONESE,
        'reference':'same hard reference waveform and text, same original seed per pair',
        'instruct2_difference':'LLM reference speech tokens removed by native frontend; acoustic reference retained',
        'technical_retries':3,'retry_seed':'original seed + attempt index, all attempts retained',
        'training':'none; pilot audio is not an ASR training manifest',
        'source_sha256':{str(source):digest(source),str(Path(__file__)):digest(__file__)},'pairs':selected})


class ModeProxy:
    def __init__(self,model,mode):self.model=model;self.mode=mode;self.sample_rate=model.sample_rate
    def inference_zero_shot(self,target,prompt,reference,**kwargs):
        assert prompt.startswith(GENERIC)
        if self.mode=='zero_shot_cantonese':
            return self.model.inference_zero_shot(target,CANTONESE+prompt[len(GENERIC):],reference,**kwargs)
        instruction=CANTONESE if self.mode=='instruct2_cantonese' else GENERIC
        return self.model.inference_instruct2(target,instruction,reference,**kwargs)


def synthesize():
    cfg=json.loads((ROOT/'protocol.json').read_text())
    for path,sha in cfg['source_sha256'].items():assert digest(path)==sha,path
    # Load the pinned TTS Transformers overlay before any ASR imports.
    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts
    model,runtime=tts.load_fixed_runtime()
    from resume_tts_reference_policy import install_empty_token_guard
    install_empty_token_guard(model.model)
    save(ROOT/'runtime.json',runtime)
    predictions=[];attempts=[]
    original={r['pair_id']:r for r in rows(ROOT.parent/'screening/input.jsonl') if r['variant']=='mas_hard'}
    for pair in cfg['pairs']:
        ref=pair['new_reference'];assert digest(ref['reference_wav'])==ref['reference_wav_sha256']
        for mode in MODES:
            out=ROOT/'audio'/f"{pair['sample_id']}_{mode}.wav";out.parent.mkdir(exist_ok=True)
            assert not out.exists()
            for attempt in range(3):
                sample=tts.ManifestSample(sample_id=f"{pair['sample_id']}_{mode}",target_text=pair['target_text'],
                    reference_text=ref['reference_text'],reference_wav=Path(ref['reference_wav']),
                    reference_wav_sha256=ref['reference_wav_sha256'],output_wav=out,seed=pair['seed']+attempt)
                try:
                    record,temp=tts._stage_audio_once(sample,ModeProxy(model,mode))
                    os.link(temp,out);temp.unlink()
                    attempts.append({'pair_id':pair['sample_id'],'mode':mode,'attempt':attempt,'status':'complete','output':record})
                    row=original[pair['sample_id']].copy()
                    row.update(utt_id=f"{mode}::{pair['sample_id']}",variant=mode,audio=str(out),
                        audio_sha256=digest(out),seconds=record['duration_seconds'])
                    predictions.append(row)
                    break
                except Exception as error:
                    attempts.append({'pair_id':pair['sample_id'],'mode':mode,'attempt':attempt,'status':'failed',
                        'error':f'{type(error).__name__}: {error}'})
            save(ROOT/'receipts'/f"{pair['sample_id']}_{mode}.json",attempts[-(attempt+1):])
            print(f"[instruction-pilot] {pair['sample_id']} {mode} {attempts[-1]['status']}",flush=True)
    save(ROOT/'summary.json',{'expected':len(cfg['pairs'])*len(MODES),'complete':len(predictions),'attempts':attempts})
    with (ROOT/'input.jsonl').open('x') as f:
        for row in predictions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    assert len(predictions)==len(cfg['pairs'])*len(MODES),'Partial diagnostic batch: inspect receipts before evaluation'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','synthesize'],required=True)
    args=p.parse_args()
    prepare() if args.stage=='prepare' else synthesize()
