"""Restrict the synthetic loss to language-side LoRA; real updates unchanged."""
import argparse
from pathlib import Path
import sys

import tts_additive_control as base


class LanguageSyntheticTrainer(base.AdditiveTrainer):
    def training_step(self,model,inputs,num_items_in_batch=None):
        frozen=[p for name,p in model.named_parameters()
                if inputs['_synthetic'] and p.requires_grad and 'audio_tower' in name.split('.')]
        for p in frozen:p.requires_grad_(False)
        try:
            return super().training_step(model,inputs,num_items_in_batch)
        finally:
            for p in frozen:p.requires_grad_(True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=base.ROOT)
    p.add_argument('--arm',default='mas_hard_language_w025')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--weight',type=float,default=.25)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--synthetic-file',type=Path,default=base.MAS/'mas_hard_reference/train.jsonl')
    args=p.parse_args()
    directory=base.prepare(args)
    base.save(directory/'intervention.json',{'scope':'synthetic: language only; real: all existing LoRA',
        'command':sys.argv,'source_sha256':base.sha256_file(Path(__file__)),
        'keeps_real_gradients':'requires_grad toggled only for synthetic forward/backward; existing grads retained'})
    base.AdditiveTrainer=LanguageSyntheticTrainer
    base.train(directory)
