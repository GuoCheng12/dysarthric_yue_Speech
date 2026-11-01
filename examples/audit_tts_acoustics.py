"""Paired training-only acoustic audit; no test data used to fit the probe."""
from collections import defaultdict
from pathlib import Path
import json
import math
import statistics

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly,welch

from diagnose_tts_utility import ROOT,rows,save,jsonl


def features(path):
    y,sr=sf.read(path,dtype='float32',always_2d=True);y=y.mean(axis=1)
    gcd=math.gcd(sr,16000)
    if sr!=16000:y=resample_poly(y,16000//gcd,sr//gcd)
    size=len(y)//320*320
    energy=np.sqrt(np.mean(y[:size].reshape(-1,320).astype('float64')**2,axis=1)+1e-16)
    f,p=welch(y,16000,nperseg=512)
    total=float(p.sum())+1e-16
    return {'seconds':len(y)/16000,'rms_db':20*np.log10(np.sqrt(np.mean(y.astype('float64')**2))+1e-12),
        'quiet_to_peak_db':20*np.log10((np.quantile(energy,.1)+1e-12)/(np.quantile(energy,.95)+1e-12)),
        'quiet_fraction_35db':float(np.mean(energy<np.quantile(energy,.95)*10**(-35/20))),
        'spectral_flatness':float(np.exp(np.mean(np.log(p+1e-16)))/(p.mean()+1e-16)),
        'below300_fraction':float(p[f<300].sum())/total,
        'above4000_fraction':float(p[f>4000].sum())/total,
        'centroid_hz':float((f*p).sum())/total}


def run():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    result=[]
    for i,row in enumerate(rows(ROOT/'screening/audit.jsonl')):
        if row['variant'] not in ('q_real','q_hard'):continue
        result.append({**{k:row[k] for k in ['variant','pair_id','speaker_id','clean_gt','audio']},
            'features':features(row['audio'])})
        if len(result)%200==0:print(f'[acoustics] {len(result)}/1152',flush=True)
    keys=list(result[0]['features'])
    X=np.array([[r['features'][k] for k in keys] for r in result]);y=np.array([r['variant']=='q_hard' for r in result])
    metrics={}
    for grouping in ['speaker_id','clean_gt']:
        group=[r[grouping] for r in result];pred=np.empty(len(y))
        for train,test in GroupKFold(5).split(X,y,group):
            model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
            model.fit(X[train],y[train]);pred[test]=model.predict_proba(X[test])[:,1]
        metrics[grouping+'_heldout_auc']=roc_auc_score(y,pred)
    medians={v:{k:statistics.median(r['features'][k] for r in result if r['variant']==v) for k in keys}
             for v in ['q_real','q_hard']}
    jsonl(ROOT/'acoustics/features.jsonl',result)
    save(ROOT/'acoustics/results.json',{'medians':medians,'probe':metrics,
        'meaning':'simple acoustic domain separability on paired train Q; not downstream ASR performance',
        'split':'5-fold held-out speaker and separately held-out content; each real/synthetic pair stays together'})
    print(json.dumps({'medians':medians,'probe':metrics},indent=2),flush=True)


if __name__=='__main__':run()
