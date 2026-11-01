"""Select between existing reference candidates without changing content/patients."""
from collections import defaultdict,Counter
import random
import statistics

from diagnose_tts_utility import ROOT,MAS,read,rows,jsonl,save,sha256_file


def run():
    audit=rows(ROOT/'screening/audit.jsonl')
    paired=defaultdict(dict)
    for r in audit:
        if r['variant'] in ('mas_old','mas_hard'):paired[r['pair_id']][r['variant']]=r
    target=rows(MAS/'mas_hard_reference/train.jsonl')[2117:]
    pairs=read(MAS/'pairs.json');selected={};by_patient=defaultdict(list)
    for p in pairs:
        uid=p['sample_id'];candidates=paired[uid]
        # Tie retains hard. Rank only with a frozen ASR that never trained on MAS.
        selected[uid]=min(('mas_hard','mas_old'),key=lambda v:candidates[v]['cer'])
        by_patient[p['speaker_id']].append(uid)
    rng=random.Random(20260907);control={}
    for speaker,uids in sorted(by_patient.items()):
        policies=[selected[u] for u in uids];rng.shuffle(policies)
        control.update(zip(uids,policies))
        assert Counter(selected[u] for u in uids)==Counter(control[u] for u in uids)
    summary={};provenance={}
    for arm,choices in [('consistency_best',selected),('consistency_shuffled',control)]:
        data=[];index=[]
        for p,t in zip(pairs,target):
            row=paired[p['sample_id']][choices[p['sample_id']]]
            data.append({**t,'audio':row['audio']})
            index.append({'sample_id':p['sample_id'],'speaker_id':p['speaker_id'],
                'target_text':p['target_text'],'choice':choices[p['sample_id']],
                'screening_cer':row['cer'],'audio':row['audio'],'audio_sha256':row['audio_sha256']})
        jsonl(ROOT/'consistency'/f'{arm}.jsonl',data)
        provenance[arm]=index
        summary[arm]={'rows':len(data),'mean_screening_cer':statistics.mean(x['screening_cer'] for x in index),
            'above_50pct':sum(x['screening_cer']>.5 for x in index),'choices':dict(Counter(choices.values()))}
    save(ROOT/'consistency/provenance.json',provenance)
    save(ROOT/'consistency/protocol.json',{'purpose':'ASR-consistency candidate selection; not clinical severity selection',
        'rule':'choose lower frozen R-only42 CER of old vs hard; ties retain hard',
        'control':'shuffle policy choices within patient, preserving per-patient old/hard counts',
        'same':'576 labels,51 patients,36 texts,per-patient content assignments',
        'different_choices':sum(selected[u]!=control[u] for u in selected),
        'new_synthesis':0,'test_data_used_for_selection':False,
        'audit_sha256':sha256_file(ROOT/'screening/audit.jsonl'),'summary':summary})
    print(summary,flush=True)


if __name__=='__main__':run()
