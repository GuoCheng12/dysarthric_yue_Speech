#!/usr/bin/env python3
"""Paired real-development comparison, including patient-cluster variability."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from therapist_harness.store import sha256_file, write_immutable_json
from therapist_harness.text import edit_distance, text_normalize


def rows(path):
    result=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    for row in result:
        reference=text_normalize(row['clean_gt'])
        edits=edit_distance(reference,text_normalize(row['prediction']))
        assert row.get('normalized_reference_characters',len(reference))==len(reference)
        assert row.get('edit_count',edits)==edits
        row.update(normalized_reference_characters=len(reference),edit_count=edits)
    return result


def compare(before_path,after_path):
    before={x['utt_id']:x for x in rows(before_path)}
    after={x['utt_id']:x for x in rows(after_path)}
    assert before.keys()==after.keys()
    groups=defaultdict(list)
    for key,b in before.items():
        a=after[key]
        assert all(a[k]==b[k] for k in ['audio','speaker_id','clean_gt','normalized_reference_characters'])
        for g in ['overall','vlink247' if b['speaker_id'].startswith('vlink247') else 'other_patients',
                  'difficulty:'+b['zero_shot_bucket'],'diagnosis:'+b['disease_tag']]:
            groups[g].append((b,a))
    result={}
    for group,pairs in groups.items():
        patients=defaultdict(lambda:np.zeros(3,dtype=float))
        for b,a in pairs:
            patients[b['speaker_id']]+=np.array([b['edit_count'],a['edit_count'],b['normalized_reference_characters']])
        values=np.stack([patients[k] for k in sorted(patients)])
        old,new,chars=values.sum(axis=0)
        rng=np.random.default_rng(42)
        indices=rng.integers(len(values),size=(2000,len(values)))
        samples=values[indices].sum(axis=1)
        delta=(samples[:,1]-samples[:,0])/samples[:,2]
        lo,hi=np.quantile(delta,[.025,.975])
        result[group]=dict(audio=len(pairs),patients=len(values),before_pooled=old/chars,
            after_pooled=new/chars,delta_pp=100*(new-old)/chars,
            before_patient_macro=float(np.mean(values[:,0]/values[:,2])),
            after_patient_macro=float(np.mean(values[:,1]/values[:,2])),
            patient_cluster_bootstrap_delta_pp_95=[100*lo,100*hi],
            improved_audio=sum(a['edit_count']<b['edit_count'] for b,a in pairs),
            worsened_audio=sum(a['edit_count']>b['edit_count'] for b,a in pairs))
    return dict(before=str(before_path),after=str(after_path),before_sha256=sha256_file(Path(before_path)),
        after_sha256=sha256_file(Path(after_path)),groups=result,
        interpretation='Exploratory 511-row development feedback, not independent test. Paired patient bootstrap (2000, seed42) describes variation; it does not correct repeated model selection. One-patient subgroup intervals cannot estimate between-patient uncertainty.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before',type=Path);parser.add_argument('after',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=compare(args.before.resolve(),args.after.resolve())
    write_immutable_json(args.output,result)
    print(json.dumps(result['groups'],ensure_ascii=False))
