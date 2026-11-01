#!/usr/bin/env python3
"""Matched MAS/random DPO dose probe on existing first-round data."""

import argparse
from pathlib import Path
import sys
import time

from mas_selection_dpo_probe import REPO,execute,read,save
from therapist_harness.store import sha256_file


def main(base):
    root=base/'checkpoint_dose_probe';root.mkdir(exist_ok=False)
    p=read(base/'formal_mas96_v3/protocol.json');c=p['preference_loop']
    arms=dict(mas=base/'formal_mas96_v3/round_001',random=base/'formal_random96_control_v2')
    spec={}
    for name,d in arms.items():
        spec[name]=dict(pairs=str(d/'sampling/preferences.jsonl'),receipt=str(d/'sampling/sampling_receipt.json'),
            allocation=str(d/('training_allocation.json' if name=='mas' else 'allocation.json')),
            training_result=str(d/('mas_dpo/training/training_result.json' if name=='mas' else 'training/training_result.json')),
            evaluation=str(d/('mas_dpo/evaluation/summary.json' if name=='mas' else 'evaluation/summary.json')))
    files=[Path(__file__),REPO/'finetune/scripts/qwen3_asr_dpo_checkpoint_probe.py',
        REPO/'finetune/scripts/qwen3_asr_audio_dpo.py',REPO/'src/therapist_harness/dpo_training.py',
        REPO/'finetune/scripts/evaluate_asr_checkpoint.py']+[Path(v) for x in spec.values() for v in x.values()]
    hashes={str(f):sha256_file(f) for f in files}
    save(root/'protocol.json',dict(source_protocol=p,arms=spec,source_sha256=hashes,checkpoints=[3,6,9,12],
        hypothesis='An earlier update may retain a benefit lost by the 12-step endpoint.',
        matched='Unchanged actual data, per-step schedule, optimizer, parent and frozen reference. Post-step checkpoint saves add no training loss.',
        selection='Repeated-development exploratory checkpoint selection; compare both arms at each same step and retain every result. No independent-test claim.'))
    results={}
    for name,x in spec.items():
        d=root/name;d.mkdir();started=time.time()
        execute(d,[sys.executable,str(REPO/'finetune/scripts/qwen3_asr_dpo_checkpoint_probe.py'),
            '--model_path',p['base_model_dir'],'--policy_adapter_path',p['initial_adapter']['path'],
            '--reference_adapter_path',p['initial_adapter']['path'],'--train_file',x['pairs'],
            '--output_dir',str(d/'training'),'--max_steps','12','--real_per_step','0',
            '--synthetic_per_step','8','--synthetic_weight',str(c['synthetic_weight']),
            '--lr',str(p['lora']['learning_rate']),'--beta',str(c['beta']),'--seed','42',
            '--synthetic_allocation_file',x['allocation']],'training')
        actual=read(d/'training/training_result.json');previous=read(x['training_result'])
        identical=actual['policy_trainable_sha256_after']==previous['policy_trainable_sha256_after']
        assert actual['reference_files_unchanged'] and actual['policy_changed']
        save(d/'replay_check.json',dict(final_policy_tensors_identical=identical,original_training_result=x['training_result'],
            original_sha256=sha256_file(Path(x['training_result'])),actual_training_result=str(d/'training/training_result.json')))
        metrics={}
        for step in [3,6,9,12]:
            if step==12 and identical:
                metrics[str(step)]=read(x['evaluation'])['metrics']
                save(d/'step12_evaluation_reuse.json',dict(source=x['evaluation'],sha256=sha256_file(Path(x['evaluation'])),
                    reason='All final trainable adapter tensors exactly match the already evaluated checkpoint; no new evaluation claimed'))
                continue
            execute(d,[sys.executable,str(REPO/'finetune/scripts/evaluate_asr_checkpoint.py'),
                '--base-model-path',p['base_model_dir'],'--adapter-path',str(d/f'training/checkpoint-{step}'),
                '--split-jsonl',p['feedback_split']['path'],'--out-dir',str(d/f'evaluation_{step}'),
                '--batch-size','8','--critical-cer-threshold','.5','--max-new-tokens','256'],f'evaluation_{step}')
            metrics[str(step)]=read(d/f'evaluation_{step}/summary.json')['metrics']
            print(name,step,metrics[str(step)]['overall']['pooled_cer'],flush=True)
        results[name]=dict(metrics_by_step=metrics,elapsed_seconds=time.time()-started)
        save(d/'results.json',results[name])
        assert all(sha256_file(Path(f))==digest for f,digest in hashes.items())
    save(root/'results.json',dict(status='complete',arms=results))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('base',type=Path)
    main(parser.parse_args().base.resolve())
