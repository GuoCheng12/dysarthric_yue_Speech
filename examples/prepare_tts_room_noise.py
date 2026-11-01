"""Fixed generic room/noise perturbation for the strict-QC synthetic set."""
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.signal import fftconvolve,resample_poly
import soundfile as sf
from math import gcd

ROOT=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_utility_diagnosis_v1')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]


def run():
    out=ROOT/'room_noise';out.mkdir(exist_ok=False);(out/'audio').mkdir()
    source=ROOT/'consistency/strict_qc_clean.jsonl';data=rows(source);assert len(data)==514
    features=ROOT/'acoustics/features.jsonl'
    target_db=statistics.median(r['features']['rms_db'] for r in rows(features) if r['variant']=='q_real')
    protocol={'seed':20260907,'rows':514,'sample_rate':16000,'t60_seconds':[.10,.25],'snr_db':[24,32],
        'late_reverberation_energy_ratio':.12,'target_rms_db':target_db,
        'noise':'Gaussian white noise; generic perturbation, not a measured patient-room response',
        'training':'strict QC clean labels; lambda .05; fixed real exposure; seeds42/43/44; step50',
        'comparison':'strict_qc_clean_w005, same514 pairs and labels without waveform perturbation',
        'interpretation':'combined room/noise/gain treatment; not isolated attribution',
        'hashes':{str(p):digest(p) for p in [source,features,Path(__file__)]}}
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    records=[];manifest=[]
    for i,row in enumerate(data):
        seed=20260907+i;rng=np.random.default_rng(seed);wav,sr=sf.read(row['audio'],dtype='float64')
        if wav.ndim==2:wav=wav.mean(axis=1)
        if sr!=16000:
            divisor=gcd(sr,16000);wav=resample_poly(wav,16000//divisor,sr//divisor)
        t60=float(rng.uniform(.1,.25));snr=float(rng.uniform(24,32));n=int(np.ceil(t60*16000))
        tail=rng.standard_normal(n)*10**(-3*np.arange(n)/16000/t60);tail[:160]=0
        tail*=np.sqrt(.12)/np.linalg.norm(tail);impulse=tail;impulse[0]=1
        wet=fftconvolve(wav,impulse);rms=np.sqrt(np.mean(wet**2));assert rms>1e-5
        noise=rng.standard_normal(len(wet));noise*=rms*10**(-snr/20)/np.sqrt(np.mean(noise**2))
        signal=wet+noise;signal*=10**(target_db/20)/np.sqrt(np.mean(signal**2))
        peak=float(np.max(np.abs(signal)));limited=peak>.95
        if limited:signal*=.95/peak
        assert np.isfinite(signal).all() and .25<=len(signal)/16000<=30
        dest=out/'audio'/f'{i:04d}.wav';sf.write(dest,signal,16000,subtype='PCM_16')
        check,rate=sf.read(dest);assert rate==16000 and len(check)==len(signal) and np.max(np.abs(check))<1
        manifest.append({**row,'audio':str(dest)})
        records.append({'index':i,'seed':seed,'source_audio':row['audio'],'source_sha256':digest(row['audio']),
            'audio':str(dest),'audio_sha256':digest(dest),'t60':t60,'snr_db':snr,'peak_limited':limited,
            'seconds':len(check)/16000,'rms_db':float(20*np.log10(np.sqrt(np.mean(check**2))))})
    with (out/'train.jsonl').open('x') as f:
        for row in manifest:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    assert [r['text'] for r in manifest]==[r['text'] for r in data]
    (out/'receipt.json').write_text(json.dumps({'rows':len(records),'manifest_sha256':digest(out/'train.jsonl'),
        'source_labels_identical':True,'outputs':records},ensure_ascii=False,indent=2)+'\n')
    print('Prepared',len(records),'paired perturbed files; target RMS dB',target_db,flush=True)


if __name__=='__main__':run()
