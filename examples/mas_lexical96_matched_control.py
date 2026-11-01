#!/usr/bin/env python3
"""Run the random-reference control after the actual MAS plan is materialized."""

import argparse
from pathlib import Path

import mas_matched_generation_control as runner
from mas_selection_dpo_probe import REPO, read, save
from therapist_harness.store import sha256_file


def main(base):
    main_root = base / 'lexical96_mas_v1'
    root = base / 'lexical96_random_control_v1'
    runner.prepare(main_root, root)
    files = [Path(__file__), REPO / 'examples/generate_reviewed_sg_recovery_v2.py',
             REPO / 'examples/generate_reviewed_sg_resumable.py']
    hashes = {str(p): sha256_file(p) for p in files}
    save(root / 'execution_extension.json', dict(source_sha256=hashes,
         scope='Same matched-control runner with the explicitly versioned observed empty/short SG recovery.'))
    execute = runner.execute

    def launch(directory, command, name):
        assert all(sha256_file(Path(p)) == digest for p, digest in hashes.items())
        command = list(command)
        if name == 'generation':
            command[1] = str(REPO / 'examples/generate_reviewed_sg_recovery_v2.py')
        return execute(directory, command, name)

    runner.execute = launch
    runner.run(root)
    assert read(root / 'provenance.json')['independent_sampling_receipt']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
