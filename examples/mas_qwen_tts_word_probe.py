#!/usr/bin/env python3
"""Eight rare-word generation diagnostics through the already installed Qwen TTS."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

from tts_model_comparison_worker import load_qwen,set_seed,save_wave,MODELS,SOURCES
from tts_model_comparison import sha256_file,write_immutable_json


def main(base):
    root=base/'lexical_qwen_tts_probe';root.mkdir(exist_ok=False);(root/'audio').mkdir()
    source=base/'lexical24_loop_v1/round_001/synthesis_manifest.jsonl'
    selected=[json.loads(x) for x in source.read_text().splitlines()][:8]
    weight_manifest=base.parent/'tts_model_comparison24_v1/download_qwen3tts.json'
    weights=json.loads(weight_manifest.read_text())
    for x in weights['files']:assert sha256_file(MODELS/'qwen3tts'/x['path'])==x['sha256']
    sources={str(p):sha256_file(p) for p in [Path(__file__),Path(__file__).with_name('tts_model_comparison_worker.py'),
        source,weight_manifest,*sorted((SOURCES['qwen3tts']/'qwen_tts').rglob('*.py'))]}
    started=time.time();generate,_,config=load_qwen();loaded=time.time();outputs=[]
    for index,row in enumerate(selected):
        seed=row['seed']%(2**32);set_seed(seed);begin=time.time()
        wave,sr,metadata=generate(dict(pair=row,seed=seed))
        record=save_wave(root/'audio'/f'{index:02d}.wav',wave,sr,time.time()-begin,metadata)
        result=dict(record,sample_id=row['sample_id']+'.qwen3tts',original_sample_id=row['sample_id'],
            target_text=row['target_text'],reference_text=row['reference_text'],reference_wav=row['reference_wav'],
            reference_wav_sha256=row['reference_wav_sha256'],donor_seed=row['seed'],actual_seed=seed,
            coarse_review_executed=False,training_eligible=False)
        outputs.append(result);write_immutable_json(root/f'{index:02d}.generation.json',result)
        print(json.dumps(dict(generated=index+1,total=8,seconds=record['seconds'])),flush=True)
    write_immutable_json(root/'generation.json',dict(status='complete',outputs=outputs,config=config,
        model=weights['repository'],revision=weights['revision'],source_sha256=sources,
        versions={k:importlib.metadata.version(k) for k in ['torch','transformers']},
        model_load_seconds=loaded-started,elapsed_seconds=time.time()-started,
        scope='Independent generator feasibility comparison: same target/reference; Qwen official ICL Auto recipe and its own sampling defaults differ from CosyVoice. Seed mapped modulo 2^32 for the existing Qwen probe runtime. Full first outputs retained; no ASR training or admission claim.'))
    manifest=root/'asr_inputs.jsonl'
    with manifest.open('x') as f:
        for row in outputs:f.write(json.dumps(dict(audio_id=row['sample_id'],audio=row['audio'],audio_sha256=row['sha256']))+'\n')
    command=['/data/qwen3-asr/venvs/qwen3-asr/bin/python',str(Path(__file__).with_name('transcribe_sg_review.py')),
        '--manifest',str(manifest),'--output',str(root/'transcriptions.json'),
        '--base-model','/data/qwen3-asr/models/Qwen3-ASR-1.7B',
        '--adapter','/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_utility_diagnosis_v1/additive/real_w0/seed_42/training/checkpoint-50']
    write_immutable_json(root/'transcription_command.json',command)
    subprocess.run(command,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('base',type=Path)
    main(parser.parse_args().base.resolve())
