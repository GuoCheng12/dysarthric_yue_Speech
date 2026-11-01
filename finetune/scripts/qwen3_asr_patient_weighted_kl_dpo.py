#!/usr/bin/env python3
"""Apply explicit patient retention factors to the existing KL/DPO windows."""

import argparse
import json
import math
from pathlib import Path
import sys

import qwen3_asr_shared_teacher_kl_dpo as shared

BaseWindows = shared.residual.trainer.SourceBalancedWindows


def weighted_window_type(factors):
    class WeightedWindows(BaseWindows):
        def __init__(self, groups, *args, **kwargs):
            super().__init__(groups, *args, **kwargs)
            assert factors.keys() == groups['real'].keys()
            assert all(math.isfinite(x) and x > 0 for x in factors.values())
            assert math.isclose(sum(factors.values()), len(factors), abs_tol=1e-8)

        def next(self):
            window = super().next()
            for row in window:
                if row['source'] == 'real':
                    row['_loss_weight'] *= factors[row['_audio_group']]
            return window

    return WeightedWindows


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--real_loss_factors', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    engine = shared.residual.trainer
    factors = json.loads(args.real_loss_factors.read_text())['factors']
    hashes = {str(p.resolve()): engine.sha256(p) for p in [Path(__file__), Path(shared.__file__), args.real_loss_factors]}
    original_write = engine.write_json

    def write_record(path, value):
        if Path(path).name == 'run_protocol.json':
            value = dict(value,
                objective='synthetic audio DPO + kl_weight * patient-weighted real symmetric KL; real factors have pool mean one',
                scope='Independent weighted-retention branch; real order and total192-exposure KL weight unchanged',
                real_loss_factors_path=str(args.real_loss_factors.resolve()),
                real_loss_factors_sha256=hashes[str(args.real_loss_factors.resolve())],
                source_sha256={**value['source_sha256'], **hashes})
        original_write(path, value)

    engine.SourceBalancedWindows = weighted_window_type(factors)
    engine.write_json = write_record
    sys.argv = [sys.argv[0]] + remaining
    shared.main()
    assert all(engine.sha256(Path(p)) == h for p, h in hashes.items())


if __name__ == '__main__':
    main()
