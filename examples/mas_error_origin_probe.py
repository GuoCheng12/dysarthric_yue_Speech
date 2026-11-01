#!/usr/bin/env python3
"""Same-dose real DPO probe separating deployed errors from sampling-only errors."""

import argparse
import json
from pathlib import Path
import sys

from mas_selection_dpo_probe import REPO, execute, jsonl, read, rows, save
from therapist_harness.store import sha256_file


def main(base):
    root=base/'error_origin_probe';root.mkdir(exist_ok=False)
    source=base/'real96_dpo_probe'
    p=read(source/'protocol.json')['source_protocol']
    audit=read(source/'sampling/preference_audit.json')
    origin={x['audio_id']:x['error_origin'] for x in audit['audio_results']}
    all_pairs=rows(source/'sampling/preferences.jsonl')
    save(root/'protocol.json',dict(source_protocol=p,steps=4,real_per_step=8,seed=42,
        source_sha256={str(x):sha256_file(x) for x in [Path(__file__),source/'sampling/preferences.jsonl',
            source/'sampling/sampling_receipt.json',source/'sampling/greedy.jsonl',source/'sampling/preference_audit.json',
            REPO/'finetune/scripts/qwen3_asr_audio_dpo.py',REPO/'src/therapist_harness/dpo_training.py']},
        intervention='Actual normal-decode error versus stochastic-only error; same optimizer and exposure count, different eligible audio pools',
        scope='Independent real-data selection mechanism probe; no new sampling, synthetic data, CER cutoff or invented errors'))
    results={'all_pairs_existing':dict(metrics=read(source/'evaluation/summary.json')['metrics'],
        source=str(source),reused=True)}
    for name in ['greedy_error','stochastic_only']:
        d=root/name
        pairs=[x for x in all_pairs if origin[x['audio_id']]==name]
        jsonl(d/'preferences.jsonl',pairs)
        save(d/'selection_receipt.json',dict(rule=name,pairs=len(pairs),
            audio=len({x['audio_id'] for x in pairs}),
            source_preferences_sha256=sha256_file(source/'sampling/preferences.jsonl'),
            selected_preferences_sha256=sha256_file(d/'preferences.jsonl'),
            source_sampling_receipt=str(source/'sampling/sampling_receipt.json'),new_sampling=False))
        if not pairs:
            results[name]=dict(status='no_eligible_pairs_no_update');continue
        execute(d,[sys.executable,str(REPO/'finetune/scripts/qwen3_asr_audio_dpo.py'),
            '--model_path',p['base_model_dir'],'--policy_adapter_path',p['initial_adapter']['path'],
            '--reference_adapter_path',p['initial_adapter']['path'],
            '--train_file',str(d/'preferences.jsonl'),'--output_dir',str(d/'training'),
            '--max_steps','4','--real_per_step','8','--synthetic_per_step','0',
            '--lr','1e-5','--seed','42'],'training')
        execute(d,[sys.executable,str(REPO/'finetune/scripts/evaluate_asr_checkpoint.py'),
            '--base-model-path',p['base_model_dir'],'--adapter-path',str(d/'training/checkpoint-4'),
            '--split-jsonl',p['feedback_split']['path'],'--out-dir',str(d/'evaluation'),
            '--batch-size','8','--critical-cer-threshold','.5','--max-new-tokens','256'],'evaluation')
        results[name]=dict(metrics=read(d/'evaluation/summary.json')['metrics'],
                           selection=read(d/'selection_receipt.json'))
        print(json.dumps(dict(arm=name,overall=results[name]['metrics']['overall'])),flush=True)
    save(root/'results.json',results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('base',type=Path)
    main(parser.parse_args().base.resolve())
