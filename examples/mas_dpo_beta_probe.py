#!/usr/bin/env python3
"""One predeclared beta comparison on the completed matched-control data."""

import argparse
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO,execute,read,save
from therapist_harness.store import sha256_file


def main(base):
    source=base/'formal_random96_control_v2'
    root=base/'beta05_probe';root.mkdir(exist_ok=False)
    p=read(source/'protocol.json');c=p['preference_loop']
    save(root/'protocol.json',dict(source_protocol=p,beta=.5,
        source_sha256={str(x):sha256_file(x) for x in [Path(__file__),source/'protocol.json',
            source/'sampling/preferences.jsonl',source/'sampling/sampling_receipt.json',source/'allocation.json',
            REPO/'finetune/scripts/qwen3_asr_audio_dpo.py',REPO/'src/therapist_harness/dpo_training.py']},
        baseline_beta=c['beta'],baseline_evaluation=str(source/'evaluation/summary.json'),
        hypothesis='Test whether stronger reference regularization at beta 0.5 reduces the hard-speech regression observed at beta 0.1.',
        matched='Same actual audio, preference pairs, full per-step exposure schedule, parent, frozen reference, optimizer, learning rate and 12 steps; only beta changes.',
        new_generation=False,new_sampling=False,objective='standard audio DPO'))
    execute(root,[sys.executable,str(REPO/'finetune/scripts/qwen3_asr_audio_dpo.py'),
        '--model_path',p['base_model_dir'],'--policy_adapter_path',p['initial_adapter']['path'],
        '--reference_adapter_path',p['initial_adapter']['path'],
        '--train_file',str(source/'sampling/preferences.jsonl'),'--output_dir',str(root/'training'),
        '--max_steps',str(p['lora']['optimizer_steps_per_round']),'--real_per_step','0',
        '--synthetic_per_step',str(c['synthetic_per_step']),'--synthetic_weight',str(c['synthetic_weight']),
        '--lr',str(p['lora']['learning_rate']),'--beta','.5','--seed',str(p['lora']['seed']),
        '--synthetic_allocation_file',str(source/'allocation.json')],'training')
    execute(root,[sys.executable,str(REPO/'finetune/scripts/evaluate_asr_checkpoint.py'),
        '--base-model-path',p['base_model_dir'],
        '--adapter-path',str(root/f"training/checkpoint-{p['lora']['optimizer_steps_per_round']}"),
        '--split-jsonl',p['feedback_split']['path'],'--out-dir',str(root/'evaluation'),
        '--batch-size',str(p['evaluation']['batch_size']),'--critical-cer-threshold',str(p['evaluation']['critical_cer_threshold']),
        '--max-new-tokens','256'],'evaluation')
    save(root/'results.json',dict(status='complete',metrics=read(root/'evaluation/summary.json')['metrics'],
        baseline_metrics=read(source/'evaluation/summary.json')['metrics'],training=read(root/'training/training_result.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('base',type=Path)
    main(parser.parse_args().base.resolve())
