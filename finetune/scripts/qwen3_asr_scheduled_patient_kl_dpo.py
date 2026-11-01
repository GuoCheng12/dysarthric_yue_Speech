#!/usr/bin/env python3
"""Schedule the same real KL loss mass across updates instead of rare bursts."""

import argparse
import json
from pathlib import Path
import sys

import qwen3_asr_weights_only_patient_kl_dpo as storage


def scheduled_windows(real_schedule):
    class ScheduledWindows:
        def __init__(self, groups, real_per_step, synthetic_per_step, synthetic_weight, seed, *, synthetic_schedule=None):
            self.groups, self.real_schedule, self.synthetic_schedule = groups, real_schedule, synthetic_schedule
            self.synthetic_weight, self.step = synthetic_weight, 0
            assert len(real_schedule) == len(synthetic_schedule)
            assert all(len(step) == synthetic_per_step for step in synthetic_schedule)
            assert all(len(group) == 1 for group in groups['real'].values())
            assert {x['audio_id'] for step in real_schedule for x in step} == set(groups['real'])
            assert all(x['loss_weight'] > 0 for step in real_schedule for x in step)
            self.counts = dict(real=real_per_step, synthetic=synthetic_per_step)

        def next(self):
            real = self.real_schedule[self.step];synthetic = self.synthetic_schedule[self.step]
            window = []
            for occurrence, item in enumerate(real):
                row = self.groups['real'][item['audio_id']][0]
                window.append(dict(row, _audio_group=item['audio_id'], _occurrence=occurrence, _loss_weight=item['loss_weight']))
            for occurrence, key in enumerate(synthetic):
                for row in self.groups['synthetic'][key]:
                    window.append(dict(row, _audio_group=key, _occurrence=occurrence,
                        _loss_weight=self.synthetic_weight * row['pair_weight'] / len(synthetic)))
            self.counts = dict(real=len(real), synthetic=len(synthetic));self.step += 1
            return window

    return ScheduledWindows


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--real_schedule', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    engine = storage.weighted.shared.residual.trainer
    plan = json.loads(args.real_schedule.read_text())
    files = [Path(__file__), Path(storage.__file__), args.real_schedule]
    hashes = {str(p.resolve()): engine.sha256(p) for p in files}
    original_write = engine.write_json

    def write_record(path, value):
        if Path(path).name == 'run_protocol.json':
            value = dict(value, objective='synthetic audio DPO + scheduled real symmetric KL with unchanged per-audio total loss weight',
                scope='Timing intervention: same real audio set and total per-audio weight; focus clips replayed every step at lower weight. Other real audio stay in their original step/order/weight. No real CE.',
                real_schedule_path=str(args.real_schedule.resolve()), real_schedule_sha256=hashes[str(args.real_schedule.resolve())],
                nominal_real_per_step_overridden_by_explicit_schedule=True,
                actual_real_exposures_per_step=[len(step) for step in plan['steps']],
                source_sha256={**value['source_sha256'], **hashes})
        original_write(path, value)

    engine.SourceBalancedWindows = scheduled_windows(plan['steps'])
    engine.write_json = write_record
    # Reuse the unchanged teacher/optimizer-storage hooks, bypassing the old
    # fixed-frequency patient-weight entry. The explicit schedule supplies weights.
    storage.weighted.main = storage.weighted.shared.main
    sys.argv = [sys.argv[0]] + remaining
    storage.main()
    assert all(engine.sha256(Path(p)) == h for p, h in hashes.items())


if __name__ == '__main__':
    main()
