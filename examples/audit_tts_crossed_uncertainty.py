"""Sensitivity to repeated patients AND repeated content in the 511-row panel."""
import json
from pathlib import Path

import numpy as np
from therapist_harness.text import text_normalize,edit_distance

ROOT=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_utility_diagnosis_v1')


def read_rows(path):return [json.loads(x) for x in path.read_text().splitlines()]


def run():
    values={};meta=None
    for p in sorted((ROOT/'additive').glob('*/seed_*/result.json')):
        result=json.loads(p.read_text());rr=read_rows(p.parent/f"evaluation/merged_step_{result['step']}/predictions.jsonl")
        identity=[(r['utt_id'],r['speaker_id'],r['clean_gt']) for r in rr]
        if meta is None:meta=identity
        else:assert meta==identity
        vals=[]
        for row in rr:
            a,b=map(text_normalize,[row['clean_gt'],row['prediction']]);vals.append(edit_distance(a,b)/len(a))
        values[result['arm'],result['seed']]=np.asarray(vals)
    patients=sorted({r[1] for r in meta});texts=sorted({r[2] for r in meta})
    pi={p:i for i,p in enumerate(patients)};ti={t:i for i,t in enumerate(texts)}
    counts=np.zeros((len(patients),len(texts)))
    for _,p,t in meta:counts[pi[p],ti[t]]+=1
    rng=np.random.default_rng(20260907);draws=20000
    pw=rng.multinomial(len(patients),np.ones(len(patients))/len(patients),size=draws)
    tw=rng.multinomial(len(texts),np.ones(len(texts))/len(texts),size=draws)
    joint_den=np.einsum('bp,pt,bt->b',pw,counts,tw)
    text_den=tw@counts.sum(axis=0);assert joint_den.min()>0 and text_den.min()>0
    out={'method':'Two-way pigeonhole bootstrap: independently resample patient and text clusters, multiplying row weights; 20000 draws; fitted models and seed-average fixed',
        'n_utterances':len(meta),'n_patients':len(patients),'n_texts':len(texts),
        'limits':'Post-hoc sensitivity, not independent confirmation or a correction for selecting among many experimental settings','comparisons':{}}
    for arm in sorted({a for a,s in values}):
        if arm in ['real_w0','real_full72_w0']:continue
        baseline={'mas_period_w025':'real_period_w0','room_full72_w005':'real_full72_w0'}.get(arm,'real_w0')
        seeds=[s for s in [42,43,44] if (arm,s) in values and (baseline,s) in values]
        if not seeds:continue
        delta=np.mean([values[arm,s]-values[baseline,s] for s in seeds],axis=0)
        sums=np.zeros_like(counts)
        for d,(_,p,t) in zip(delta,meta):sums[pi[p],ti[t]]+=d
        joint=np.einsum('bp,pt,bt->b',pw,sums,tw)/joint_den*100
        text_only=(tw@sums.sum(axis=0))/text_den*100
        out['comparisons'][arm]={'baseline':baseline,'seeds':seeds,'delta_pp':float(delta.mean()*100),
            'patient_and_text_95ci_pp':np.quantile(joint,[.025,.975]).tolist(),
            'text_only_95ci_pp':np.quantile(text_only,[.025,.975]).tolist()}
    (ROOT/'cluster_sensitivity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
