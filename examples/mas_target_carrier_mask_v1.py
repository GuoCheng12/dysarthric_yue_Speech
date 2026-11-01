"""Explicit target-character token scoring, with full audio/text conditioning.

This is a masked preference-loss ablation, not standard full-sequence DPO and
not an audio fidelity or SG admission check. Tokens can span neighboring chars.
"""
import torch
from therapist_harness.dpo_training import AudioPreferenceCollator, preference_key
from therapist_harness.preference_token_mask import edit_eos_labels


def token_plan(tokenizer, pair, events):
    result={}
    for side,index_key,char_key in [('chosen','reference_index','reference_character'),
                                   ('rejected','hypothesis_index','hypothesis_character')]:
        text=pair[side]
        indices=sorted({e[index_key] for e in events})
        assert indices and all(text[e[index_key]]==e[char_key] for e in events)
        encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        ids=encoded['input_ids'];offsets=encoded['offset_mapping']
        assert len(ids)==len(offsets)
        selected=sorted({i for char in indices for i,(start,end) in enumerate(offsets) if start<=char<end})
        assert selected and all(any(offsets[i][0]<=char<offsets[i][1] for i in selected) for char in indices)
        result[side]=dict(text=text,character_indices=indices,token_ids=ids+[tokenizer.eos_token_id],
            offsets=[list(x) for x in offsets],scored_indices=selected+[len(ids)],
            tokens=[dict(index=i,ids=ids[i],offset=list(offsets[i]),surface=text[offsets[i][0]:offsets[i][1]]) for i in selected])
    return result


def targeted_labels(labels, indices, plan):
    """Map continuation-relative token indices onto actual padded chat labels."""
    result=torch.full_like(labels,-100)
    for side,k in zip(['chosen','rejected'],indices):
        positions=labels[k].ne(-100).nonzero(as_tuple=True)[0]
        assert labels[k,positions].tolist()==plan[side]['token_ids']
        scored=plan[side]['scored_indices']
        assert scored[-1]==len(positions)-1 and len(scored)>1
        result[k,positions[scored]]=labels[k,positions[scored]]
    return result


class TargetCarrierCollator(AudioPreferenceCollator):
    def __init__(self,processor,audit,plans,focus):
        super().__init__(processor)
        self.audit,self.plans,self.focus=audit,plans,focus

    def __call__(self,rows,*,include_rejected=True):
        batch=super().__call__(rows,include_rejected=include_rejected)
        if not include_rejected:return batch
        full=batch['labels']
        edited,records=edit_eos_labels(full,len(rows),self.processor.tokenizer.eos_token_id)
        for i,(row,record) in enumerate(zip(rows,records)):
            assert row['source'] in {'real','synthetic'} and record['token_edit_distance']>0
            key=preference_key(row)
            record=dict(record,mode='edit_eos',audio_id=row['audio_id'],
                chosen_scored_indices=record['chosen_edit_eos_indices'],
                rejected_scored_indices=record['rejected_edit_eos_indices'])
            if key in self.plans:
                assert row['source']=='synthetic'
                plan=self.plans[key]
                assert all(row[s]==plan[s]['text'] for s in ['chosen','rejected'])
                indices=[i,i+len(rows)]
                target=targeted_labels(full,indices,plan)
                if self.focus:
                    for k in indices:edited[k]=target[k]
                    record=dict(record,mode='target_character_eos',
                        chosen_scored_indices=plan['chosen']['scored_indices'],
                        rejected_scored_indices=plan['rejected']['scored_indices'],
                        target_character_indices={s:plan[s]['character_indices'] for s in ['chosen','rejected']})
            if key in self.audit:assert self.audit[key]==record
            else:self.audit[key]=record
        batch['labels']=edited
        return batch
