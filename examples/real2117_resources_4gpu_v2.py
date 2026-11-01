#!/usr/bin/env python3
"""Versioned four-GPU supervisor; preserves the interrupted two-GPU ledger."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import real2117_sampling_4gpu_v2 as sampling

old=sampling.old;ROOT=sampling.ROOT;old.ROOT=ROOT
TRAINER=old.REPO/'examples/train_real2117_dpo_kl_3gpu_v2.py'
SAMPLER=Path(sampling.__file__)


def prepare():
    sampling.verify()
    plan=old.read(sampling.PREVIOUS/'protocol.json')
    sources=[Path(__file__),TRAINER,SAMPLER,ROOT/'migration.json',ROOT/'real_inputs.jsonl']
    sources += [ROOT/f'inputs_shard{i}.jsonl' for i in range(4)]
    plan.update(execution_version='real2117_DPO_KL_seed42_3epochs_resource4gpu_v2',
        previous_protocol=str(sampling.PREVIOUS/'protocol.json'),
        GPUs='Sampling GPU0,1,2,3; then training replicasGPU0,1,2 and evaluationGPU3.',
        distributed_training='Each original global8audio window assigned chunk[rank::3]. SUM globally weighted gradients once, preserve globally unused parameters as None, clip globalnorm1 then identical AdamW step on all replicas. No world-size averaging or extra optimizer steps.',
        distributed_numerics='Parallel summation may differ in floating-point order from serial. Actual5audio/two-update smoke compares full serial and3rank gradients/loss at identical parameters, then verifies identical replica weights.',
        KL_storage='Each training replica computes detached frozen-reference fullfloat32KL on its assigned audio; no whole-pool teacher cache.',
        original_smoke=str(sampling.PREVIOUS/'smoke/result.json'),resource_migration=str(ROOT/'migration.json'),
        gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv'],text=True),
        source_sha256={**plan['source_sha256'],**{str(p):old.sha256_file(p) for p in sources}})
    old.write_immutable_json(ROOT/'protocol.json',plan)
    old.emit('resource_plan_ready',sampling_gpus=4,training_gpus=3,evaluation_gpus=1,global_audio_per_step=8,total_steps=795)


def spawn(command,gpus,log):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (ROOT/log).open('x') as f:
        return subprocess.Popen(command,cwd=old.REPO,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)


def supervise():
    old.verify()
    assert old.read(ROOT/'sampling_smoke/result.json')['passed']
    smoke=old.read(ROOT/'smoke_3gpu/result.json')
    assert smoke['passed'] and smoke['trainer_sha256']==old.sha256_file(TRAINER)
    jobs=[]
    try:
        jobs=[spawn([old.PY,str(SAMPLER),'sample','--shard',str(i)],str(i),f'sampling_gpu{i}.log') for i in range(4)]
        old.write_immutable_json(ROOT/'sampling_processes.json',dict(pids=[j.pid for j in jobs],supervisor=os.getpid(),created_utc=datetime.now(timezone.utc).isoformat()))
        old.emit('sampling_started',pids=[j.pid for j in jobs],sampling_gpus=4,recovered_draws=514,remaining_new_draws=7954)
        while any(j.poll() is None for j in jobs):
            assert all(j.poll() in (None,0) for j in jobs),'A sampling shard failed'
            time.sleep(5)
        assert all(j.returncode==0 for j in jobs)
        sampling.merge()
        command=[old.PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=3',str(TRAINER),'train']
        jobs=[spawn(command,'0,1,2','training.log'),spawn([old.PY,str(Path(__file__)),'evaluate'],'3','evaluation.log')]
        old.write_immutable_json(ROOT/'training_processes.json',dict(pids=[j.pid for j in jobs],supervisor=os.getpid(),created_utc=datetime.now(timezone.utc).isoformat()))
        old.emit('DPO_and_evaluator_started',pids=[j.pid for j in jobs],training_gpus=3,evaluation_gpu=3)
        while any(j.poll() is None for j in jobs):
            assert all(j.poll() in (None,0) for j in jobs),'Training or evaluation failed'
            time.sleep(5)
        assert all(j.returncode==0 for j in jobs)
        old.write_immutable_json(ROOT/'complete.json',dict(status='complete',epochs=3,evaluations=15,seed=42))
        old.emit('experiment_complete',epochs=3,evaluations=15)
    except BaseException as error:
        for job in jobs:
            if job.poll() is None:os.killpg(job.pid,signal.SIGTERM)
        old.write_immutable_json(ROOT/'failure.json',dict(error=repr(error),utc=datetime.now(timezone.utc).isoformat()))
        old.emit('failed',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','supervise','evaluate']);a=p.parse_args()
    if a.stage=='evaluate':old.evaluate()
    else:globals()[a.stage]()
