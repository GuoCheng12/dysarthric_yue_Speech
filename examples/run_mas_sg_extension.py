#!/usr/bin/env python3
"""Run a new prepared loop with explicitly bound observed-SG recovery code."""

import argparse
from pathlib import Path

import mas_asr_dpo_loop as cli
from therapist_harness import preference_loop as loop
from therapist_harness.store import RunStore,sha256_file

REPO=Path(__file__).resolve().parents[1]
EXTENSION=REPO/'examples/generate_reviewed_sg_recovery_v2.py'


def main(root):
    store=RunStore(root);assert store.current().phase=='initialized'
    files=[Path(__file__),EXTENSION,REPO/'examples/generate_reviewed_sg_resumable.py',root/'protocol.json',root/'sources.json']
    binding=dict(version='explicit_observed_sg_recovery_execution_v2',source_sha256={str(f):sha256_file(f) for f in files},
        protocol_run_id=store.protocol().run_id,scope='New prepared execution; existing core remains frozen. Empty/short output recovery and pending reuse are separately versioned; review and training objectives unchanged.')
    loop.save(root/'execution_extension.json',binding)
    original=loop.run_command
    def execute(command,log,*,env_overrides=None):
        assert all(sha256_file(Path(f))==digest for f,digest in binding['source_sha256'].items())
        command=list(command)
        if command[1]==str(REPO/'examples/generate_reviewed_sg.py'):command[1]=str(EXTENSION)
        loop.save(Path(log).with_suffix('.command.json'),dict(command=command,
            execution_extension_sha256=sha256_file(root/'execution_extension.json')))
        return original(command,log,env_overrides=env_overrides)
    loop.run_command=execute
    cli.main(['run',str(root)])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    main(parser.parse_args().root.resolve())
