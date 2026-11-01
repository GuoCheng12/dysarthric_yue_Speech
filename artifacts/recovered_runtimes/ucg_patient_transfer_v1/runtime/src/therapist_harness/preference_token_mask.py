"""Optional edit-token scoring for an explicit audio preference-loss ablation.

This changes the scored continuation positions, never the model's inputs or
the chosen/rejected strings. It is not standard full-sequence DPO.
"""

import torch

from therapist_harness.dpo_training import AudioPreferenceCollator, preference_key


def edit_positions(chosen, rejected):
    """Unit-cost token alignment; ties prefer diagonal, deletion, insertion."""
    n,m=len(chosen),len(rejected)
    costs=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):costs[i][0]=i
    for j in range(m+1):costs[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            costs[i][j]=min(costs[i-1][j-1]+(chosen[i-1]!=rejected[j-1]),
                costs[i-1][j]+1,costs[i][j-1]+1)
    left,right=[],[];i,j=n,m
    while i or j:
        if i and j and costs[i][j]==costs[i-1][j-1]+(chosen[i-1]!=rejected[j-1]):
            if chosen[i-1]!=rejected[j-1]:left.append(i-1);right.append(j-1)
            i-=1;j-=1
        elif i and costs[i][j]==costs[i-1][j]+1:
            left.append(i-1);i-=1
        else:
            assert j and costs[i][j]==costs[i][j-1]+1
            right.append(j-1);j-=1
    return sorted(left),sorted(right),costs[n][m]


def edit_eos_labels(labels, pair_count, eos_token_id):
    assert labels.ndim==2 and labels.shape[0]==2*pair_count
    result=torch.full_like(labels,-100);records=[]
    for index in range(pair_count):
        indices=[index,index+pair_count]
        positions=[labels[k].ne(-100).nonzero(as_tuple=True)[0] for k in indices]
        ids=[labels[k,p].tolist() for k,p in zip(indices,positions)]
        assert all(tokens and tokens[-1]==eos_token_id for tokens in ids)
        edits0,edits1,distance=edit_positions(ids[0][:-1],ids[1][:-1])
        kept=[edits0+[len(ids[0])-1],edits1+[len(ids[1])-1]]
        for k,p,selected in zip(indices,positions,kept):
            result[k,p[selected]]=labels[k,p[selected]]
        records.append(dict(chosen_token_ids=ids[0],rejected_token_ids=ids[1],
            chosen_edit_eos_indices=kept[0],rejected_edit_eos_indices=kept[1],
            token_edit_distance=distance))
    return result,records


def collator_type(mode, audit):
    assert mode in {'full','edit_eos'}

    class ScoringCollator(AudioPreferenceCollator):
        def __call__(self, rows, *, include_rejected=True):
            batch=super().__call__(rows,include_rejected=include_rejected)
            # Real teacher distributions must retain the complete original mask.
            if not include_rejected:
                return batch
            assert all(row['source']=='synthetic' for row in rows)
            edited,records=edit_eos_labels(batch['labels'],len(rows),self.processor.tokenizer.eos_token_id)
            for row,record in zip(rows,records):
                assert record['token_edit_distance']>0
                record=dict(record,mode=mode,audio_id=row['audio_id'],
                    chosen_scored_indices=record['chosen_edit_eos_indices'] if mode=='edit_eos' else list(range(len(record['chosen_token_ids']))),
                    rejected_scored_indices=record['rejected_edit_eos_indices'] if mode=='edit_eos' else list(range(len(record['rejected_token_ids']))))
                key=preference_key(row)
                if key in audit:assert audit[key]==record
                else:audit[key]=record
            if mode=='edit_eos':batch['labels']=edited
            return batch

    return ScoringCollator
