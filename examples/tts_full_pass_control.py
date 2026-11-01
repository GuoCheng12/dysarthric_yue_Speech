"""Cover both pools and finish the LR schedule in a matched 72-step run."""
import argparse
from pathlib import Path
import sys

import tts_additive_control as base


def run(args):
    original_command = base.training_command

    def command(*values):
        result = original_command(*values)
        result[result.index('--max_steps') + 1] = '72'
        return result

    base.training_command = command
    directory = base.prepare(args)
    index = base.read(directory / 'index.json')
    real = [r for r in index if r['domain'] == 'real']
    synthetic = [r for r in index if r['domain'] == 'synthetic']
    assert len(real) == 2304 and len(synthetic) == 576
    assert len({r['source_index'] for r in real}) == 2117
    assert len({r['source_index'] for r in synthetic}) == 514
    protocol = base.read(directory / 'protocol.json')
    cmd = protocol['command']
    assert cmd[cmd.index('--max_steps') + 1] == '72'
    base.save(directory / 'intervention.json', {
        'source_sha256': base.sha256_file(Path(__file__)), 'command': sys.argv,
        'change': '72 updates, linear LR horizon72; unchanged real pool and repaired514 TTS pool',
        'paired_baseline': 'real_full72_w0', 'synthetic_arm': 'room_full72_w005',
        'real_exposures': 2304, 'real_unique': 2117,
        'synthetic_exposures': 576, 'synthetic_unique': 514,
        'selection': 'all seeds42/43/44 fixed endpoint72, no checkpoint selection; existing511 development feedback',
        'interpretation': 'joint data-coverage and LR-endpoint intervention; each TTS run compared only with its matched real72 run',
    })
    base.train(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=base.ROOT)
    parser.add_argument('--arm', choices=['real_full72_w0', 'room_full72_w005'], required=True)
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    args = parser.parse_args()
    args.steps = 72
    args.weight = 0.0 if args.arm == 'real_full72_w0' else 0.05
    args.synthetic_file = base.ROOT.parent / 'room_noise/train.jsonl'
    run(args)
