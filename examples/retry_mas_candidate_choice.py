#!/usr/bin/env python3
"""Retry one identical MAS choice request with observable terminal handling."""

import argparse
import json
import os
from pathlib import Path
import time

from httpx import Timeout
from openai import OpenAI

import mas_candidate_coverage_probe as probe
from mas_selection_dpo_probe import read, save
from therapist_harness.store import sha256_file


def retry(root, arm, original_request):
    probe.verify(root)
    d = root / arm
    expected = read(original_request)
    hashes = {str(p): sha256_file(p) for p in [Path(__file__), Path(probe.__file__), original_request]}
    save(d / 'retry_protocol.json', dict(source_sha256=hashes, actual_new_call=True,
        unchanged_request_source=str(original_request),
        transport='Same model, reasoning, timeout and SDK retry parameters; record stream event types and return on an actual terminal event.'))

    def execute_traced(protocol, request):
        assert request == expected
        client = OpenAI(api_key=os.environ[protocol.agent.api_key_env],
            base_url=str(protocol.agent.base_url), max_retries=protocol.agent.max_retries,
            timeout=Timeout(float(protocol.agent.timeout_seconds), connect=20.0),
            default_headers={'User-Agent': 'curl/8.0'})
        start = time.monotonic()
        try:
            with (d / 'api_events.jsonl').open('x') as log:
                with client.responses.create(**request) as stream:
                    for event in stream:
                        record = dict(event_type=event.type, seconds=time.monotonic()-start,
                            sequence_number=getattr(event, 'sequence_number', None))
                        log.write(json.dumps(record)+'\n')
                        log.flush()
                        if event.type in {'response.completed', 'response.failed', 'response.incomplete'}:
                            save(d / 'api_terminal.json', dict(**record, response_id=event.response.id,
                                status=event.response.status, identical_request_verified=True))
                            return event.response
            raise RuntimeError('Response stream ended without a terminal event')
        except Exception as error:
            save(d / 'api_failure.json', dict(exception_type=type(error).__name__,
                seconds=time.monotonic()-start, message=str(error)[:500]))
            raise
        finally:
            client.close()

    probe.execute_therapist_request = execute_traced
    probe.choose(root, arm)
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('arm', choices=['first4', 'all16'])
    parser.add_argument('original_request', type=Path)
    args = parser.parse_args()
    retry(args.root.resolve(), args.arm, args.original_request.resolve())
