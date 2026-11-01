#!/usr/bin/env python3
"""Repeat the first gate-passing MAS DPO allocation with the same seed and data."""

import argparse
import os
from pathlib import Path

import mas_patient_retention_probe as patient
from mas_selection_dpo_probe import REPO, read, rows, save
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file

TRAINER = REPO / 'finetune/scripts/qwen3_asr_weights_only_patient_kl_dpo.py'


def prepare(base, root):
    previous = base / 'patient_retention_dose_probe_v1'
    cfg = read(previous / 'protocol.json')
    assert read(previous / 'results.json')['eligible_for_next_round_parent']
    assert read(previous / 'verified.json')['status'] == 'passed'
    assert all(sha256_file(Path(f)) == h for f, h in cfg['source_sha256'].items())
    root.mkdir(exist_ok=False)
    os.link(previous / 'real_loss_factors.json', root / 'real_loss_factors.json')
    save(root / 'cpu_smoke.json', dict(read(previous / 'cpu_smoke.json'), actual_new_cpu_smoke=False,
        reused_smoke=str(previous / 'cpu_smoke.json'),
        scope='Identical weighting/data/order reuse the completed full12-window CPU smoke and actual12-step GPU training. Only optimizer persistence changes; its separate CPU serialization smoke is recorded.'))
    from qwen3_asr_weights_only_patient_kl_dpo import optimizer_manifest
    import torch
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0.)
    parameter.square().sum().backward(); optimizer.step()
    record = optimizer_manifest(optimizer.state_dict())
    assert not record['optimizer_state_persisted'] and not record['exact_optimizer_resume_available']
    assert len(record['parameter_states']) == 1
    assert set(record['parameter_states']['0']) == {'step', 'exp_avg', 'exp_avg_sq'}
    save(root / 'storage_smoke.json', dict(status='passed', actual_new_GPU_training=False,
        actual_CPU_AdamW_step=True, optimizer_manifest=record,
        scope='Check honest omission metadata on actual AdamW state. The DPO/KL arithmetic and optimizer step are unchanged.'))
    save(root / 'design.json', dict(actual_new_MAS_decision=False, actual_new_training=True,
        replicate_of=str(previous), seed=42, GPU=0,
        purpose='Verify repeatability of the first raw gate-passing result before advancing the MAS loop. This is same-seed replay, not an independent seed result.',
        changes='Checkpoint persistence only: keep model weights and final optimizer tensor hashes, omit optimizer.pt; all optimization and teacher computation unchanged.',
        acceptance='Original raw pooled/macro/hard gate, same511 development recordings.'))
    files = [Path(__file__), TRAINER, previous / 'protocol.json', previous / 'results.json',
        previous / 'verified.json', previous / 'training/training_exposures.jsonl',
        previous / 'training/reference_log_probs.json', previous / 'evaluation_run/evaluation/predictions.jsonl',
        root / 'real_loss_factors.json', root / 'cpu_smoke.json', root / 'storage_smoke.json', root / 'design.json']
    save(root / 'protocol.json', dict(cfg, source_sha256={**cfg['source_sha256'], **{str(f): sha256_file(f) for f in files}},
        arguments=dict(cfg['arguments'], output_dir=str(root / 'training')), replication_control=str(previous)))


def run(base, root):
    patient.TRAINER = TRAINER
    patient.run(base, root)
    control = Path(read(root / 'protocol.json')['replication_control'])
    assert rows(root / 'training/training_exposures.jsonl') == rows(control / 'training/training_exposures.jsonl')
    assert read(root / 'training/reference_log_probs.json')['values'] == read(control / 'training/reference_log_probs.json')['values']
    checkpoint = root / 'training/checkpoint-12'
    assert not (checkpoint / 'optimizer.pt').exists()
    assert not read(checkpoint / 'optimizer_omission.json')['optimizer_state_persisted']
    assert not read(root / 'training/teacher_storage_reuse.json')['unchanged_optimizer_checkpoint_serialization']
    save(root / 'paired_replication_control.json', compare(control / 'evaluation_run/evaluation/predictions.jsonl',
        root / 'evaluation_run/evaluation/predictions.jsonl'))
    save(root / 'replication_verified.json', dict(status='passed', exact_actual_exposure_records=True,
        exact_actual_frozen_reference_scores=True,
        identical_learned_adapter_bytes=sha256_file(checkpoint / 'adapter_model.safetensors') == sha256_file(control / 'training/checkpoint-12/adapter_model.safetensors'),
        optimizer_omission_explicit=True, seed=42, independent_seed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('base', type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    root = base / 'patient_retention_replication_v1'
    if args.stage == 'prepare':
        prepare(base, root)
    else:
        run(base, root)
