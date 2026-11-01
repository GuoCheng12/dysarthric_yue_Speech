"""Compare one-step real/synthetic continuations with their exact parent models."""
import json
from pathlib import Path
import statistics

from tts_increment_control import TEST,rows
from therapist_harness.text import text_normalize,edit_distance

ROOT=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_utility_diagnosis_v1')


def run():
    scope=rows(TEST);results={};base={}
    for seed in [42,43,44]:
        p=ROOT/'additive/real_w0'/f'seed_{seed}/result.json'
        base[seed]=json.loads(p.read_text())['metrics']['overall']['mean_utterance_cer']*100
    for p in sorted((ROOT/'post_small').glob('*/seed_*/result.json')):
        saved=json.loads(p.read_text());rr=rows(p.parent/'evaluation/merged_step_1/predictions.jsonl')
        assert len(rr)==len(scope)==511
        values=[]
        for a,b in zip(rr,scope):
            assert all(a[k]==b[k] for k in ['utt_id','audio','speaker_id','clean_gt'])
            ref,hyp=map(text_normalize,[a['clean_gt'],a['prediction']]);values.append(edit_distance(ref,hyp)/len(ref)*100)
        overall=statistics.mean(values)
        assert abs(overall-saved['metrics']['overall']['mean_utterance_cer']*100)<1e-10
        results.setdefault(saved['arm'],{})[saved['seed']]={'cer_pct':overall,
            'parent_cer_pct':base[saved['seed']],'delta_pp':overall-base[saved['seed']],
            'patient_macro_pct':saved['metrics']['overall']['patient_macro_cer']*100,
            'vlink247_pct':statistics.mean(v for v,r in zip(values,rr) if r['speaker_id']=='vlink247'),
            'others_pct':statistics.mean(v for v,r in zip(values,rr) if r['speaker_id']!='vlink247')}
    means={a:{k:statistics.mean(r[k] for r in runs.values()) for k in next(iter(runs.values()))} for a,runs in results.items()}
    common=sorted(set(results.get('real32',{}))&set(results.get('synthetic32',{})))
    comparison={s:results['synthetic32'][s]['cer_pct']-results['real32'][s]['cer_pct'] for s in common}
    out={'results':results,'means':means,'synthetic_minus_real_pp':comparison,
        'mean_synthetic_minus_real_pp':statistics.mean(comparison.values()) if comparison else None,
        'scope':'one added step, identical parent within seed; fresh optimizer; fixed historical S32 set; existing511 exploratory feedback'}
    folder=ROOT/'post_small';folder.mkdir(exist_ok=True)
    (folder/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
