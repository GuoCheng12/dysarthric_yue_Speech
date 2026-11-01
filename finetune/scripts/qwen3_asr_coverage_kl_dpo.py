#!/usr/bin/env python3
"""Explicit real coverage schedule, residual DPO, and optional cache persistence."""

import argparse
import json
from pathlib import Path
import sys

import qwen3_asr_kl_dpo_storage_v2 as engine
import qwen3_asr_residual_kl_dpo as residual
from qwen3_asr_scheduled_patient_kl_dpo import scheduled_windows


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--real_schedule', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    plan = json.loads(args.real_schedule.read_text())
    files = [Path(__file__), args.real_schedule,
        Path(__file__).with_name('qwen3_asr_scheduled_patient_kl_dpo.py'),
        Path(__file__).with_name('qwen3_asr_weights_only_patient_kl_dpo.py')]
    hashes = {str(p.resolve()): engine.sha256(p) for p in files}
    original_write = engine.write_json

    def write_record(path, value):
        if Path(path).name == 'run_protocol.json':
            value = dict(value,
                objective='synthetic audio DPO + explicitly scheduled real symmetric KL',
                scope='Matched focus-patient budget and timing; real content coverage differs by arm. No real CE.',
                real_schedule_path=str(args.real_schedule.resolve()),
                real_schedule_sha256=hashes[str(args.real_schedule.resolve())],
                nominal_real_per_step_overridden_by_explicit_schedule=True,
                actual_real_exposures_per_step=[len(step) for step in plan['steps']],
                source_sha256={**value['source_sha256'], **hashes})
        original_write(path, value)

    engine.SourceBalancedWindows = scheduled_windows(plan['steps'])
    engine.write_json = write_record
    residual.trainer = engine
    sys.argv = [sys.argv[0]] + remaining
    residual.main()
    assert all(engine.sha256(Path(p)) == digest for p, digest in hashes.items())


if __name__ == '__main__':
    main()
