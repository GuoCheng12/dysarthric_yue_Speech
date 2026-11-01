#!/usr/bin/env python3
"""Recover a terminal API timeout without repeating completed ASR sampling."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import mas_candidate_coverage_probe as probe
from mas_selection_dpo_probe import REPO, read, rows, save
from therapist_harness.store import sha256_file


def live(pid):
    path = Path(f'/proc/{pid}/stat')
    if not path.exists():
        return False
    return path.read_text().split(') ', 1)[1].split()[0] != 'Z'


def link_files(source, destination, names):
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {}
    for name in names:
        src, dst = source / name, destination / name
        digest = sha256_file(src)
        os.link(src, dst)
        assert sha256_file(dst) == digest
        receipt[str(src)] = dict(destination=str(dst), sha256=digest)
    return receipt


def prepare(base, root):
    original = base / 'candidate_coverage_probe_v1'
    probe.verify(original)
    failed_pid = read(original / 'all16/choose.launch.json')['pid']
    assert not live(failed_pid)
    failure = original / 'all16/choose.supervisor.log'
    assert 'httpx.ReadTimeout: The read operation timed out' in failure.read_text()
    assert not (original / 'all16/response.json').exists()
    root.mkdir(exist_ok=False)
    imported = link_files(original, root, ['protocol.json', 'allocation.json', 'purposes.json',
        'inputs.jsonl', 'original_draw_comparison.json', 'available_target_coverage.json'])
    imported.update(link_files(original / 'sampling', root / 'sampling',
        [x.name for x in (original / 'sampling').iterdir() if x.is_file()]))
    for arm in ['first4', 'all16']:
        imported.update(link_files(original / arm, root / arm, ['candidate_view.jsonl',
            'available_preferences.jsonl', 'preference_audit.json', 'view_receipt.json']))
    save(root / 'migration.json', dict(original_root=str(original), imports=imported,
        source_sha256={str(f): sha256_file(f) for f in [Path(__file__), Path(probe.__file__), failure,
            original / 'all16/request.json', original / 'all16/choose.launch.json']},
        terminal_failed_pid=failed_pid, first4_original_pid=read(original / 'first4/choose.launch.json')['pid'],
        reason='all16 actual response stream failed with ReadTimeout; first4 original execution is preserved and awaited.',
        retry='Identical all16 request, model, reasoning and transport parameters. New client records event types/times without reasoning content and returns on the actual terminal event.',
        actual_new_sampling=False, actual_new_generation=False,
        training='Only the new recovery root will train once both real MAS decisions exist; original supervisor cannot pass its failed all16 child.'))


def verify(root):
    probe.verify(root)
    m = read(root / 'migration.json')
    assert all(sha256_file(Path(p)) == h for p, h in m['source_sha256'].items())
    for src, item in m['imports'].items():
        assert sha256_file(Path(src)) == sha256_file(Path(item['destination'])) == item['sha256']
    return m


def retry(root):
    from httpx import Timeout
    from openai import OpenAI

    m = verify(root)
    original_request = read(Path(m['original_root']) / 'all16/request.json')

    def execute_traced(protocol, request):
        assert request == original_request, 'Recovery must retry the identical actual request'
        client = OpenAI(api_key=os.environ[protocol.agent.api_key_env],
            base_url=str(protocol.agent.base_url), max_retries=protocol.agent.max_retries,
            timeout=Timeout(float(protocol.agent.timeout_seconds), connect=20.0),
            default_headers={'User-Agent': 'curl/8.0'})
        start = time.monotonic()
        with (root / 'all16/api_events.jsonl').open('x') as log:
            try:
                with client.responses.create(**request) as stream:
                    for event in stream:
                        record = dict(event_type=event.type, seconds=time.monotonic()-start,
                            sequence_number=getattr(event, 'sequence_number', None))
                        log.write(json.dumps(record) + '\n')
                        log.flush()
                        if event.type in {'response.completed', 'response.failed', 'response.incomplete'}:
                            save(root / 'all16/api_terminal.json', dict(**record,
                                response_id=event.response.id, status=event.response.status,
                                identical_request_verified=True))
                            return event.response
                raise RuntimeError('Response stream ended without a terminal event')
            except Exception as error:
                save(root / 'all16/api_failure.json', dict(exception_type=type(error).__name__,
                    seconds=time.monotonic()-start, message=str(error)[:500]))
                raise
            finally:
                client.close()

    probe.execute_therapist_request = execute_traced
    probe.choose(root, 'all16')
    verify(root)


def supervise(base, root):
    retry(root)
    m = verify(root)
    original = Path(m['original_root'])
    while live(m['first4_original_pid']):
        time.sleep(5)
    assert read(original / 'first4/response.json')['status'] == 'completed'
    imported = link_files(original / 'first4', root / 'first4', ['request.json', 'response.json',
        'preferences.jsonl', 'mas_decision.json', 'selection_receipt.json'])
    save(root / 'first4/reuse_receipt.json', dict(imports=imported, actual_new_mas_call=False,
        original_process_terminal=True, source='Completed original first4 MAS execution, unchanged.'))
    probe.train(base, root, 'first4', smoke=True)
    jobs = []
    for gpu, arm in enumerate(['first4', 'all16']):
        command = [sys.executable, str(Path(__file__).resolve()), 'train', str(base), '--arm', arm]
        with (root / arm / 'train.supervisor.log').open('x') as log:
            child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        save(root / arm / 'train.launch.json', dict(pid=child.pid, gpu=gpu, command=command))
        jobs.append(child)
    codes = [x.wait() for x in jobs]
    assert codes == [0, 0], codes
    data = [rows(root / arm / 'training/training_exposures.jsonl') for arm in ['first4', 'all16']]
    fields = ['step', 'source', 'audio_id', 'occurrence', 'loss_weight']
    assert [[tuple(x[k] for k in fields) for x in arm] for arm in data][0] == [tuple(x[k] for k in fields) for x in data[1]]
    assert Counter(x['audio_id'] for x in data[0]) == Counter(x for step in read(root / 'allocation.json')['steps'] for x in step)
    save(root / 'cross_arm_verification.json', dict(status='passed', same_actual_audio_exposures_order_weights=True,
        original_96_exposures_retained=True, same_parent_and_frozen_reference=True,
        actual_sampling_receipt=str(original / 'sampling/sampling_receipt.json'),
        all16_retried_identical_request=True, first4_actual_response_reused=True))
    save(root / 'results.json', dict(status='complete', arms={a: read(root / a / 'results.json') for a in ['first4', 'all16']}))
    verify(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'supervise', 'train'])
    parser.add_argument('base', type=Path)
    parser.add_argument('--arm', choices=['first4', 'all16'], default='first4')
    args = parser.parse_args()
    base = args.base.resolve()
    root = base / 'candidate_coverage_recovery_v1'
    if args.stage == 'prepare':
        prepare(base, root)
    elif args.stage == 'supervise':
        supervise(base, root)
    else:
        verify(root)
        probe.train(base, root, args.arm)
