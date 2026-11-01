#!/usr/bin/env python3
"""Independent real-speech DPO control; never part of the synthetic-only arm."""

import argparse
import json
from pathlib import Path
import random
import sys

from mas_selection_dpo_probe import REPO, execute, jsonl, read, rows, save
from therapist_harness.store import sha256_file


def main(base):
    root=base/'real96_dpo_probe'
    root.mkdir(exist_ok=False)
    p=read(base/'selection_dpo_probe/protocol.json')['source_protocol']
    selected=sorted(rows(p['real_train']['path']),key=lambda x:x['utt_id'])
    random.Random(42).shuffle(selected)
    jsonl(root/'inputs.jsonl',[dict(x,source='real',audio_id=x['utt_id']) for x in selected[:96]])
    save(root/'protocol.json',dict(source_protocol=p,seed=42,real_audio=96,
        sampling='four stochastic draws plus a separately recorded greedy decode from the initial R-only parent',
        source_sha256={str(x):sha256_file(x) for x in [Path(__file__),REPO/'examples/sample_asr_preferences.py',
            REPO/'finetune/scripts/qwen3_asr_audio_dpo.py',REPO/'src/therapist_harness/dpo_training.py']},
        steps=4,real_per_step=8,synthetic_per_step=0,
        scope='Independent DPO feasibility control on real training audio; not replay in the main MAS arm'))
    execute(root,[sys.executable,str(REPO/'examples/sample_asr_preferences.py'),'sample',
        '--inputs',str(root/'inputs.jsonl'),'--real-train',p['real_train']['path'],
        '--feedback',p['feedback_split']['path'],'--out-dir',str(root/'sampling'),
        '--base-model',p['base_model_dir'],'--current-adapter',p['initial_adapter']['path'],
        '--expected-parent-sha256',p['initial_adapter']['sha256'],'--samples','4','--seed','42',
        '--include-greedy'],'sampling')
    audit=read(root/'sampling/preference_audit.json')
    print(json.dumps({k:v for k,v in audit.items() if k!='audio_results'}),flush=True)
    if not audit['pair_count']:
        save(root/'results.json',dict(status='all_correct_no_update',audit=audit));return
    execute(root,[sys.executable,str(REPO/'finetune/scripts/qwen3_asr_audio_dpo.py'),
        '--model_path',p['base_model_dir'],'--policy_adapter_path',p['initial_adapter']['path'],
        '--reference_adapter_path',p['initial_adapter']['path'],
        '--train_file',str(root/'sampling/preferences.jsonl'),'--output_dir',str(root/'training'),
        '--max_steps','4','--real_per_step','8','--synthetic_per_step','0',
        '--lr','1e-5','--seed','42'],'training')
    execute(root,[sys.executable,str(REPO/'finetune/scripts/evaluate_asr_checkpoint.py'),
        '--base-model-path',p['base_model_dir'],'--adapter-path',str(root/'training/checkpoint-4'),
        '--split-jsonl',p['feedback_split']['path'],'--out-dir',str(root/'evaluation'),
        '--batch-size','8','--critical-cer-threshold','.5','--max-new-tokens','256'],'evaluation')
    result=dict(status='complete',metrics=read(root/'evaluation/summary.json')['metrics'],
                audit=audit,training=read(root/'training/training_result.json'))
    save(root/'results.json',result)
    print(json.dumps(result['metrics']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path)
    main(parser.parse_args().base.resolve())
