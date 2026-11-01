"""Content/split sensitivity and dictionary-Jyutping proxy for completed ASR runs."""
from collections import Counter,defaultdict
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import statistics

import pycantonese
from therapist_harness.text import text_normalize,edit_distance
from therapist_harness.text_scoring_v2 import text_normalize_v2

ROOT=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_utility_diagnosis_v1')


@lru_cache(None)
def syllables(text):
    out=[];unknown=0
    for surface,pronunciation in pycantonese.characters_to_jyutping(text):
        tokens=pronunciation.split() if pronunciation else []
        if len(tokens)==len(surface):out.extend(tokens)
        else:
            # Do not silently drop unresolved characters or merge them into UNK.
            out.extend('UNK:'+char for char in surface);unknown+=len(surface)
    assert len(out)==len(text)
    return tuple(out),unknown


def run():
    results={};items={}
    for p in sorted((ROOT/'additive').glob('*/seed_*/result.json')):
        saved=json.loads(p.read_text());arm,seed=saved['arm'],saved['seed'];rr=[]
        for split in ['dev','test']:
            f=p.parent/f"evaluation/step_{saved['step']}_{split}_shard/predictions.jsonl"
            for row in map(json.loads,f.read_text().splitlines()):
                ref,hyp=map(text_normalize,[row['clean_gt'],row['prediction']])
                a,b=syllables(ref),syllables(hyp)
                vref,vhyp=map(text_normalize_v2,[row['clean_gt'],row['prediction']])
                rr.append({'utt_id':row['utt_id'],'speaker_id':row['speaker_id'],'text':row['clean_gt'],'split':split,
                    'cer':edit_distance(ref,hyp)/len(ref),'cer_v2':edit_distance(vref,vhyp)/len(vref),
                    'syllable_error':edit_distance(a[0],b[0])/len(a[0]),
                    'ref_unknown':a[1],'hyp_unknown':b[1],
                    'ref_chars':len(ref),'hyp_chars':len(hyp)})
        groups={}
        for group in ['overall','dev','test']:
            xs=[r for r in rr if group=='overall' or r['split']==group]
            groups[group]={k:100*statistics.mean(r[k] for r in xs) for k in ['cer','cer_v2','syllable_error']}
            groups[group]['n']=len(xs)
        for text in sorted({r['text'] for r in rr}):
            xs=[r for r in rr if r['text']==text]
            groups[text]={k:100*statistics.mean(r[k] for r in xs) for k in ['cer','cer_v2','syllable_error']}
            groups[text].update(n=len(xs),split=xs[0]['split'])
        results.setdefault(arm,{})[seed]=groups;items[arm,seed]=rr
    comparisons={}
    for arm,seeds in results.items():
        if arm=='real_w0':continue
        baseline='real_period_w0' if arm=='mas_period_w025' else 'real_w0'
        common=sorted(set(seeds)&set(results.get(baseline,{})))
        if not common:continue
        groups={}
        for group in results[arm][common[0]]:
            groups[group]={k:statistics.mean(results[arm][s][group][k]-results[baseline][s][group][k] for s in common)
                for k in ['cer','cer_v2','syllable_error']}
        comparisons[arm]={'baseline':baseline,'seeds':common,'delta_pp':groups}
    out={'limits':'Existing development feedback; split/text breakdowns are post hoc. Dictionary syllable error is a text-derived proxy, not an audio/clinical phoneme annotation and does not replace CER.',
        'pycantonese_version':importlib.metadata.version('pycantonese'),'results':results,'comparisons':comparisons,
        'dictionary_coverage':{'unique_strings':syllables.cache_info().currsize,
            'reference_unresolved_characters_per_run':{a+'__'+str(s):sum(x['ref_unknown'] for x in rr) for (a,s),rr in items.items()},
            'hypothesis_unresolved_characters_per_run':{a+'__'+str(s):sum(x['hyp_unknown'] for x in rr) for (a,s),rr in items.items()}}}
    folder=ROOT/'content_transfer';folder.mkdir(exist_ok=True)
    (folder/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    for arm,c in comparisons.items():print(arm,c['seeds'],{k:c['delta_pp'][k] for k in ['overall','dev','test']},flush=True)


if __name__=='__main__':run()
