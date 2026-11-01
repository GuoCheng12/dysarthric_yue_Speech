#!/usr/bin/env python3
"""Learning-rate-only comparison of the matched real/MAS DPO bridge."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

from mas_real_preference_bridge import ARMS, expected, run
from mas_selection_dpo_probe import read, save
from therapist_harness.store import sha256_file

SCRATCH = Path('/tmp/qwen3_asr_real_preference_bridge_strong_v1')


def prepare(base, root):
    previous = base / 'real_preference_bridge_v1'
    cfg = read(previous / 'protocol.json')
    assert all(sha256_file(Path(f)) == h for f, h in cfg['source_sha256'].items())
    assert read(previous / 'smoke/verified.json')['status'] == 'passed'
    root.mkdir(exist_ok=False)
    assert shutil.disk_usage('/tmp').free > 250 * 1024 ** 2
    SCRATCH.mkdir(exist_ok=False)
    arguments, source_files, diagnostics = {}, [], {}
    for arm in ARMS:
        out = root / arm; out.mkdir(); (SCRATCH / arm).mkdir()
        old = cfg['arguments'][arm]
        args = dict(old, lr=1e-4, output_dir=str(SCRATCH / arm / 'training'))
        assert {k for k in old if old[k] != args[k]} == {'lr', 'output_dir'}
        plan = expected(args)
        assert plan == read(previous / arm / 'expected_exposures.json')
        save(out / 'expected_exposures.json', plan)
        arguments[arm] = args
        assert read(previous / arm / 'verified.json')['status'] == 'passed'
        training = read(previous / arm / 'training_receipts/training_result.json')
        diagnostics[arm] = dict(first_step_loss=training['history'][0]['loss'],
            last_step_loss=training['history'][-1]['loss'], probe_before=training['probe_before'],
            probe_after=training['probe_after'], evaluation_result_available_at_prepare=(previous / arm / 'results.json').exists())
        source_files += [previous / arm / n for n in ['verified.json', 'expected_exposures.json', 'training_receipts/training_result.json']]
        source_files.append(out / 'expected_exposures.json')
    (root / 'smoke').mkdir()
    shutil.copyfile(previous / 'smoke/verified.json', root / 'smoke/verified.json')
    shutil.copyfile(previous / 'data_receipt.json', root / 'data_receipt.json')
    save(root / 'cpu_smoke.json', dict(status='passed', exact_previous_exposure_plan=True,
        only_learning_rate_and_output_path_changed=True, actual_new_GPU_smoke=False,
        reused_actual_GPU_smoke=str(previous / 'smoke/verified.json'), reused_smoke_lr=1e-5,
        scope='Same standard residual DPO engine; actual12-step runs also verify initial log2, frozen base/reference and exact exposures.'))
    save(root / 'decision_before_evaluation.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        rationale='Both low-rate runs show small preference-margin movement. Prespecify10x learning rate with all data, actual pairs, loss weights,initialization and exposure records fixed. No development CER is used to choose this comparison.',
        training_diagnostics=diagnostics))
    source_files += [Path(__file__), previous / 'protocol.json', root / 'cpu_smoke.json', root / 'smoke/verified.json',
        root / 'data_receipt.json', root / 'decision_before_evaluation.json']
    save(root / 'protocol.json', dict(cfg, arguments=arguments,
        source_sha256={**cfg['source_sha256'], **{str(f): sha256_file(f) for f in source_files}},
        previous_comparison=str(previous),
        motivation='Prespecified learning-rate-only comparison based on actual training loss/probe movement; current evaluation results were not read for this decision.',
        settings='Same seed42,12steps,beta0.1,192 real exposures and mixed96 synthetic exposures at weight0.25; only lr changes1e-5→1e-4.',
        storage='New temporary /tmp checkpoints, persistent workspace receipts; legacy outputs retained.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('base', type=Path)
    parser.add_argument('--arm', choices=ARMS, default='R_MAS_DPO')
    args = parser.parse_args(); base = args.base.resolve(); root = base / 'real_preference_bridge_strong_v1'
    if args.stage == 'prepare': prepare(base, root)
    else: run(base, root, args.arm)
