#!/usr/bin/env python3
"""SFT diagnostic with exactly the DPO control's audio exposure schedule."""

import argparse
from pathlib import Path
import sys
import time

from mas_selection_dpo_probe import REPO,execute,jsonl,read,rows,save
from therapist_harness.store import sha256_file


def main(base):
    # The beta probe was already using GPU 1; avoid overlapping GPU jobs.
    while not (base/'beta05_probe/results.json').exists():
        if not Path('/proc/149092').exists():raise RuntimeError('beta diagnostic stopped before releasing a completed result')
        time.sleep(5)
    source=base/'formal_random96_control_v2';root=base/'paired_sft_probe';root.mkdir(exist_ok=False)
    p=read(source/'protocol.json');c=p['preference_loop']
    used={x['audio_id'] for x in rows(source/'sampling/preferences.jsonl')}
    selected=[x for x in rows(source/'sampling/sft.jsonl') if x['audio_id'] in used]
    assert {x['audio_id'] for x in selected}==used
    jsonl(root/'sft.jsonl',selected)
    save(root/'protocol.json',dict(source_protocol=p,objective='sft',
        source_sha256={str(x):sha256_file(x) for x in [Path(__file__),source/'protocol.json',
            source/'sampling/preferences.jsonl',source/'sampling/sft.jsonl',source/'sampling/sampling_receipt.json',
            source/'allocation.json',root/'sft.jsonl',REPO/'finetune/scripts/qwen3_asr_audio_dpo.py',
            REPO/'src/therapist_harness/dpo_training.py']},
        diagnostic='Same actual audio, chosen labels and per-step exposure schedule as DPO; use chosen-text SFT to separate data usefulness from preference optimization.',
        selected_audio=len(selected),matched_steps=12,matched_exposures=96,
        new_generation=False,new_sampling=False,main_dpo_protocol_modified=False))
    execute(root,[sys.executable,str(REPO/'finetune/scripts/qwen3_asr_audio_dpo.py'),
        '--objective','sft','--model_path',p['base_model_dir'],
        '--policy_adapter_path',p['initial_adapter']['path'],'--reference_adapter_path',p['initial_adapter']['path'],
        '--train_file',str(root/'sft.jsonl'),'--output_dir',str(root/'training'),
        '--max_steps',str(p['lora']['optimizer_steps_per_round']),'--real_per_step','0',
        '--synthetic_per_step',str(c['synthetic_per_step']),'--synthetic_weight',str(c['synthetic_weight']),
        '--lr',str(p['lora']['learning_rate']),'--seed',str(p['lora']['seed']),
        '--synthetic_allocation_file',str(source/'allocation.json')],'training')
    execute(root,[sys.executable,str(REPO/'finetune/scripts/evaluate_asr_checkpoint.py'),
        '--base-model-path',p['base_model_dir'],
        '--adapter-path',str(root/f"training/checkpoint-{p['lora']['optimizer_steps_per_round']}"),
        '--split-jsonl',p['feedback_split']['path'],'--out-dir',str(root/'evaluation'),
        '--batch-size',str(p['evaluation']['batch_size']),'--critical-cer-threshold',str(p['evaluation']['critical_cer_threshold']),
        '--max-new-tokens','256'],'evaluation')
    save(root/'results.json',dict(status='complete',metrics=read(root/'evaluation/summary.json')['metrics'],
        dpo_metrics=read(source/'evaluation/summary.json')['metrics'],training=read(root/'training/training_result.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('base',type=Path)
    main(parser.parse_args().base.resolve())
