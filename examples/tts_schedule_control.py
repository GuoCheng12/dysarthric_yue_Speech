"""Use synthetic CE early, then finish with the same real-only updates."""
import argparse
from pathlib import Path
import sys

import tts_additive_control as base


def synthetic_weight(step):
    return .25*max(0.,(21-step)/20)


class ScheduledTrainer(base.AdditiveTrainer):
    def training_step(self,model,inputs,num_items_in_batch=None):
        self.synthetic_weight=synthetic_weight(self.state.global_step+1)
        return super().training_step(model,inputs,num_items_in_batch)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=base.ROOT)
    p.add_argument('--arm',default='mas_hard_decay20')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=50)
    p.add_argument('--synthetic-file',type=Path,default=base.MAS/'mas_hard_reference/train.jsonl')
    args=p.parse_args();args.weight=.25
    directory=base.prepare(args)
    base.save(directory/'intervention.json',{'scope':'synthetic CE weight schedule; real schedule unchanged',
        'weights':{step:synthetic_weight(step) for step in range(1,args.steps+1)},
        'rule':'lambda=.25*(21-step)/20 for steps1..20; zero for21..50',
        'matched_integrated_weight_control':.0525,
        'command':sys.argv,'source_sha256':base.sha256_file(Path(__file__))})
    base.AdditiveTrainer=ScheduledTrainer
    base.train(directory)
