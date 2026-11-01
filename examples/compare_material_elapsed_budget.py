"""Report latest completed evaluations at equal elapsed time, including API and recovery waits."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from therapist_harness.store import sha256_file, write_immutable_json


def compare(root, snapshot=False):
    protocol = json.loads((root / 'formal/protocol.json').read_text())
    branches = {'mas': root / 'formal/mas', 'uniform': Path(protocol['uniform_comparison_path'])}
    records = {}
    for name, path in branches.items():
        complete_path = path / 'complete.json'
        complete = json.loads(complete_path.read_text()) if complete_path.exists() else None
        if complete:
            assert complete['passed'] and complete['steps'] == 590 and complete['actual_audio_exposures'] == 18880
        elif not snapshot:
            raise ValueError(f'{name} has not completed its full budget; use --snapshot only for an interim report.')
        supervisor = json.loads((path / 'supervisor.json').read_text())
        assert len(supervisor['gpus']) == 2
        data = (path / 'events.jsonl').read_bytes()
        # A snapshot can encounter a final line still being appended.
        lines = data.splitlines(keepends=True)
        events = [json.loads(line) for line in lines if line.endswith(b'\n')]
        start = next(r for r in events if r['stage'] == 'process_started' and r['rank'] == 0)
        start_time = datetime.fromisoformat(start['utc'])
        evaluations = []
        for row in events:
            if row['stage'] != 'evaluation_complete':
                continue
            elapsed = (datetime.fromisoformat(row['utc']) - start_time).total_seconds()
            assert elapsed >= 0 and (not evaluations or elapsed >= evaluations[-1]['elapsed_seconds'])
            evaluations.append(dict(step=row['step'], utc=row['utc'], elapsed_seconds=elapsed,
                test=row['test'], dev=row['feedback'], timing=row['timing']))
        assert evaluations
        records[name] = dict(source_path=str(path), completed=complete is not None,
            source_events_sha256=hashlib.sha256(data).hexdigest(), first_process_event=start,
            latest_step=evaluations[-1]['step'], evaluations=evaluations,
            supervisor_sha256=sha256_file(path / 'supervisor.json'))
    budget = min(r['evaluations'][-1]['elapsed_seconds'] for r in records.values())
    matched = {name: [e for e in r['evaluations'] if e['elapsed_seconds'] <= budget][-1]
               for name, r in records.items()}
    result = dict(interim_snapshot=snapshot, common_elapsed_seconds=budget,
        matched=matched, branches=records,
        test_CER_delta_pp=matched['mas']['test']['CER_percent'] - matched['uniform']['test']['CER_percent'],
        rule='Latest completed evaluation by the common elapsed deadline, never best CER. Elapsed time runs from the original rank0 process start to the evaluation UTC; includes model startup, Agent waits, failed execution and offline recovery gaps. Preparation and time queued before GPU launch are excluded. Each branch uses two L20 GPUs.',
        source_sha256={str(Path(__file__).resolve()): sha256_file(Path(__file__))})
    filename = f"elapsed_budget_snapshot_step{records['mas']['latest_step']:03d}.json" if snapshot else 'elapsed_budget_final.json'
    output = root / 'analysis' / filename
    write_immutable_json(output, result)
    print(json.dumps(dict(output=str(output), common_minutes=budget / 60,
        matched={b: dict(step=e['step'], test_CER=e['test']['CER_percent']) for b, e in matched.items()},
        interim_snapshot=snapshot), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--snapshot', action='store_true')
    args = parser.parse_args()
    compare(args.root, args.snapshot)
