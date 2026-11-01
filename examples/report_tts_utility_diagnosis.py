"""Independent CER reconstruction and paired comparisons of completed runs."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from tts_additive_control import ROOT,TEST,read,rows,sha256_file
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.text import text_normalize


def distance(a,b):
    # Full matrix, independent of the project's rolling-row implementation.
    d=np.zeros((len(a)+1,len(b)+1),dtype=np.int32)
    d[:,0]=np.arange(len(a)+1);d[0,:]=np.arange(len(b)+1)
    for i,c in enumerate(a,1):
        for j,e in enumerate(b,1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(c!=e))
    return int(d[-1,-1])


def run():
    scope=rows(TEST);results=defaultdict(dict);cer={};orders={}
    for path in sorted(ROOT.glob('*/seed_*/result.json')):
        saved=read(path);arm,seed=saved['arm'],saved['seed'];directory=path.parent
        cfg=read(directory/'protocol.json')
        predicted=rows(directory/f"evaluation/merged_step_{saved['step']}/predictions.jsonl")
        assert len(predicted)==511
        values=[]
        for a,b in zip(predicted,scope):
            assert all(a[k]==b[k] for k in ('utt_id','audio','speaker_id','clean_gt','zero_shot_bucket'))
            ref,hyp=text_normalize(a['clean_gt']),text_normalize(a['prediction'])
            values.append(distance(ref,hyp)/len(ref))
        metrics=evaluate_snapshot(predicted,prediction_column='prediction',critical_cer_threshold=.5).model_dump(mode='json')
        assert abs(statistics.mean(values)-metrics['overall']['mean_utterance_cer'])<1e-12
        for group in ('overall','easy','medium','hard'):
            indexes=[i for i,r in enumerate(scope) if group=='overall' or r['zero_shot_bucket']==group]
            assert abs(statistics.mean(values[i] for i in indexes)-saved['metrics'][group]['mean_utterance_cer'])<1e-12
        compact={g:metrics[g]['mean_utterance_cer']*100 for g in ('overall','easy','medium','hard')}
        compact['patient_macro']=metrics['overall']['patient_macro_cer']*100
        for group in ('vlink247','other_patients'):
            compact[group]=statistics.mean(v for v,r in zip(values,scope) if (r['speaker_id']=='vlink247')==(group=='vlink247'))*100
        compact['synthetic_weight']=cfg['synthetic_weight']
        results[arm][seed]=compact;cer[arm,seed]=np.array(values)
        orders[arm,seed]=[r for r in read(directory/'index.json') if r['domain']=='real']
    comparisons={};patients=sorted({r['speaker_id'] for r in scope})
    counts=np.array([sum(r['speaker_id']==p for r in scope) for p in patients])
    rng=np.random.default_rng(20260907);draws=rng.integers(0,len(patients),(20000,len(patients)))
    for arm,runs in results.items():
        if arm in ['real_w0','real_full72_w0']:continue
        baseline={'mas_period_w025':'real_period_w0','room_full72_w005':'real_full72_w0'}.get(arm,'real_w0')
        common=sorted(set(runs)&set(results.get(baseline,{})))
        if not common:continue
        for seed in common:assert orders[arm,seed]==orders[baseline,seed]
        delta=np.mean([cer[arm,s]-cer[baseline,s] for s in common],axis=0)
        sums=np.array([sum(v for v,r in zip(delta,scope) if r['speaker_id']==p) for p in patients])
        bootstrap=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)*100
        comparisons[arm]={'baseline':baseline,'seeds':common,'mean_delta_pp':float(delta.mean()*100),
            'seed_deltas_pp':{s:float((cer[arm,s]-cer[baseline,s]).mean()*100) for s in common},
            'patient_cluster_bootstrap_95ci_pp':np.quantile(bootstrap,[.025,.975]).tolist(),
            'bootstrap_scope':'seed-averaged paired utterance difference; resample patients; conditional on fitted models'}
    means={a:{g:statistics.mean(v[g] for v in runs.values()) for g in next(iter(runs.values()))} for a,runs in results.items()}
    mechanisms={}
    for name,arm,baseline in [
        ('reference_quality_selection','consistency_best_w025','consistency_shuffled_w025'),
        ('synthetic_label_format','mas_labels_clean_w025','mas_hard_w025'),
        ('acoustic_perturbation','strict_qc_room_w005','strict_qc_clean_w005'),
        ('practical_quality_package','strict_qc_clean_w005','mas_hard_w025'),
        ('real_label_format','real_period_w0','real_w0')]:
        common=sorted(set(results.get(arm,{}))&set(results.get(baseline,{})))
        if not common:continue
        for s in common:assert orders[arm,s]==orders[baseline,s]
        delta=np.mean([cer[arm,s]-cer[baseline,s] for s in common],axis=0)
        sums=np.array([sum(v for v,r in zip(delta,scope) if r['speaker_id']==p) for p in patients])
        bootstrap=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)*100
        mechanisms[name]={'arm':arm,'baseline':baseline,'seeds':common,'mean_delta_pp':float(delta.mean()*100),
            'seed_deltas_pp':{s:float((cer[arm,s]-cer[baseline,s]).mean()*100) for s in common},
            'patient_cluster_bootstrap_95ci_pp':np.quantile(bootstrap,[.025,.975]).tolist()}
    report={'results':results,'means':means,'paired_vs_real':comparisons,'mechanism_comparisons':mechanisms,
        'verified_runs':sum(map(len,results.values()))}
    (ROOT.parent/'current_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# 固定真实曝光：TTS 效用诊断','',
        'CER（%）；原有511条开发反馈集；每组独立列明已完成seed，不混用旧设置基线。','',
        '| 组别 | Seeds | Overall | Patient macro | vlink247 | 其他患者 | Easy | Medium | Hard |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    groups=['overall','patient_macro','vlink247','other_patients','easy','medium','hard']
    for arm in means:
        lines.append(f'| {arm} | '+','.join(map(str,sorted(results[arm])))+' | '+' | '.join(f'{means[arm][g]:.4f}' for g in groups)+' |')
    lines+=['','## 配对差值（增强 − 新纯真实基线；负值改善）','']
    for arm,c in comparisons.items():lines.append(f"- {arm} vs {c['baseline']}: {c['mean_delta_pp']:+.4f} pp；逐seed {c['seed_deltas_pp']}；患者聚类95%区间 {c['patient_cluster_bootstrap_95ci_pp']}")
    lines+=['','## 干预与其直接对照','']
    for name,c in mechanisms.items():
        lines.append(f"- {name}: {c['arm']} vs {c['baseline']}，seed {c['seeds']}，差值 {c['mean_delta_pp']:+.4f} pp；逐seed {c['seed_deltas_pp']}")
    lines+=['','全部已完成结果按独立完整编辑距离矩阵复算；评测身份、共同真实训练序列已核验。','']
    (ROOT.parent/'current_report.zh.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':run()
