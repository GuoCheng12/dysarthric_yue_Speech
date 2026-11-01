"""Align both real and synthetic targets with the base model's terminal period."""
import argparse
from pathlib import Path
import sys

import tts_additive_control as base


def run(args):
    original=base.CONTROL/'real_q/train.jsonl'
    transformed=base.ROOT.parent/'format_all/real_q/train.jsonl'
    expected=[{**r,'text':r['text'].rstrip('。！？.!?')+'。'} for r in base.rows(original)]
    if transformed.exists():assert base.rows(transformed)==expected
    else:base.jsonl(transformed,expected)
    base.CONTROL=transformed.parent.parent
    directory=base.prepare(args)
    base.save(directory/'intervention.json',{'source_sha256':base.sha256_file(Path(__file__)),
        'command':sys.argv,'real_source':str(original),'real_source_sha256':base.sha256_file(original),
        'change':'append terminal 。 to every real label; original MAS hard labels already use 。',
        'paired_baseline':'real_period_w0','synthetic_arm':'mas_period_w025',
        'audio':'unchanged','selection':'all three seeds, fixed step50, existing511 development feedback'})
    base.train(directory)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=base.ROOT)
    p.add_argument('--arm',choices=['real_period_w0','mas_period_w025'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--weight',type=float,required=True)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--synthetic-file',type=Path,default=base.MAS/'mas_hard_reference/train.jsonl')
    args=p.parse_args()
    assert args.weight==({'real_period_w0':0.,'mas_period_w025':.25}[args.arm])
    run(args)
