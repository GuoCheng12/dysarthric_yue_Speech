#!/usr/bin/env python3
"""New execution entry adding actual prior attempts to reference-selection memory."""

import argparse
from pathlib import Path

import run_mas_sg_extension as execution
from mas_generation_experience import latest_committed_experience
from mas_selection_dpo_probe import save
from therapist_harness import generation_controller
from therapist_harness.store import sha256_file


def main(root):
    files = [Path(__file__), Path(__file__).with_name('mas_generation_experience.py'),
             Path(execution.__file__)]
    hashes = {str(p): sha256_file(p) for p in files}
    binding = root / 'generation_memory_execution.json'
    save(binding, dict(source_sha256=hashes,
        extension='Pass latest committed generation attempts and actual sampling outcomes to the next reference/recipe decision. Frozen core, review and training objectives remain unchanged.'))
    original = generation_controller.choose_generation

    def choose(protocol, request_data, directory):
        assert all(sha256_file(Path(p)) == digest for p, digest in hashes.items())
        experience = latest_committed_experience(root)
        request_data = dict(request_data, actual_previous_round_generation=experience,
                            generation_memory_execution_sha256=sha256_file(binding))
        if experience is not None:
            save(Path(directory) / 'previous_generation_experience.json', experience)
        return original(protocol, request_data, directory)

    generation_controller.choose_generation = choose
    execution.main(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
