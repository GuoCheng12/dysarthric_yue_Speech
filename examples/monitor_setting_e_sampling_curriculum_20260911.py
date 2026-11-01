"""Compact read-only monitor for both real-data sampling branches."""

import argparse
import json
import os
from pathlib import Path
import time


def events(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A writer may currently be appending the final line.
    return result


def execution_status(directory):
    if (directory / 'complete.json').exists():
        return 'COMPLETED'
    supervisor = directory / 'supervisor.json'
    if not supervisor.exists():
        return 'NOT STARTED'
    pid = json.loads(supervisor.read_text())['pid']
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return 'STOPPED: supervisor exited without completion'
    return 'RUNNING'


def show(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    smoke = any(json.loads(p.read_text()).get('smoke', False) for p in root.glob('*/supervisor.json'))
    total = 4 if smoke else protocol['total_optimizer_steps_limit']
    print("Setting E: real recordings only | uniform vs MAS sampling")
    print("MAS feedback: independent dev; test never enters Agent decisions")
    print(f"{'SMOKE' if smoke else 'FORMAL'} budget: {total} updates per branch | allocation/evaluation interval: {2 if smoke else protocol['steps_per_stage']} steps")
    if protocol.get('material_curriculum'):
        if protocol['material_curriculum'].get('sampling_mode') == 'group_then_material':
            print('MAS: diagnosis -> recording selection -> six group budgets -> within-group selection -> dev reflection')
        else:
            print('MAS: diagnosis -> recording selection -> selected/ordinary mixture -> dev reflection')
        print('Displayed candidates per focus:', protocol['material_curriculum']['candidate_limit'])
    print("root:", root)
    if (root / 'variant_launcher_failed.json').exists():
        print('VARIANT LAUNCHER FAILED:', (root / 'variant_launcher_failed.json').read_text())
    group_queue = root.parent / 'queue.json'
    if group_queue.exists() and protocol.get('material_curriculum', {}).get('sampling_mode') == 'group_then_material':
        queue = json.loads(group_queue.read_text())
        if (root.parent / 'queue_cancelled.json').exists():
            cancelled = json.loads((root.parent / 'queue_cancelled.json').read_text())
            print('QUEUE REPLACED BEFORE GPU LAUNCH:', cancelled['replacement_root'])
        elif (root.parent / 'queue_failed.json').exists():
            print('QUEUE FAILED:', (root.parent / 'queue_failed.json').read_text())
        else:
            try:
                command = Path(f"/proc/{queue['pid']}/cmdline").read_bytes()
                if b'launch_group_material_exploration.py' in command:
                    if not (root.parent / 'queue_released.json').exists():
                        print(f"QUEUED: waiting for MAS PID {queue['predecessor_pid']} to finish; GPUs {','.join(queue['gpus'])}.")
                    elif not (root / 'mas/supervisor.json').exists():
                        print('GPU smoke/audit before formal training:', root.parent / 'smoke/mas')
            except FileNotFoundError:
                if not (root / 'mas/complete.json').exists():
                    print('Queue process has exited; inspect', root.parent / 'queue.log')
    for branch in ("uniform", "mas"):
        directory = Path(protocol.get('uniform_comparison_path', root / branch)) if branch == 'uniform' else root / branch
        records = events(directory / "events.jsonl")
        last = records[-1] if records else {}
        losses = [r for r in records if r["stage"] == "loss"]
        evaluations = [r for r in records if r["stage"] == "evaluation_complete"]
        status = execution_status(directory)
        print("\n" + branch.upper(), "|", status)
        if directory != root / branch:
            print('Existing ordinary control (reused after completion):', directory)
        if branch == 'mas' and status == 'NOT STARTED' and (root / 'queue.json').exists():
            queue = json.loads((root / 'queue.json').read_text())
            try:
                command = Path(f"/proc/{queue['pid']}/cmdline").read_bytes()
                if b'launch_material_candidate_variant.py' in command:
                    print('Queued: waiting for the ordinary control to release GPUs 0,1.')
            except FileNotFoundError:
                print('Queue process has exited; inspect queue.log.')
        print('Last recorded event:', last.get("stage", "not started"), "|", last.get("utc", ""))
        allocations = [r for r in records if r['stage'] == 'stage_started' and 'selected_probability' in r]
        if allocations:
            allocation = allocations[-1]
            label = 'Within-group selected probability' if allocation.get('selected_probability_scope') == 'within_group' else 'Selected material probability'
            print(f"{label}: {allocation['selected_probability']:.1%} | unique recordings: {allocation['selected_recordings']}")
        if status.startswith('STOPPED'):
            print('No training is running. Error log:', directory / 'rank-0.log')
        if last.get('stage') == 'agent_started':
            print('Last Agent request:', last['phase'], '| round', last['round_index'], '| step', last['step'])
            stage_dir = directory / f"stage-{last['round_index']:03d}"
            attempts = sorted(stage_dir.rglob('*.started.json'), key=lambda p: p.stat().st_mtime)
            if attempts:
                attempt = json.loads(attempts[-1].read_text())
                result = attempts[-1].with_name(attempts[-1].name.replace('.started', ''))
                state = 'finished' if result.exists() else 'waiting'
                elapsed = max(0, time.time()-attempt['started_unix'])
                print(f"API attempt {attempt['attempt']}/3 | {state} | {elapsed:.0f}s / {attempt['deadline_seconds']}s deadline")
        if losses:
            loss = losses[-1]
            timing = loss.get("timing", {})
            print(f"step {loss['step']}/{total} | loss {loss['loss']:.5f} | LR {loss['learning_rate']:.2g}",
                  f"| elapsed {timing.get('wall_seconds', 0):.1f}s")
        for row in evaluations[-3:]:
            metric = row['test']
            dev = row['feedback']
            print(f"dev CER {dev['CER_percent']:.3f}% | dev patient equal {dev['patient_equal_CER_percent']:.3f}%")
            print(f"eval step {row['step']:4d} | test CER {metric['CER_percent']:.3f}%",
                  f"| patient equal {metric['patient_equal_CER_percent']:.3f}%",
                  f"| elapsed {row['timing']['wall_seconds']:.0f}s",
                  "| in budget" if row["within_wall_budget"] else "| outside budget")
    print("\nFixed-step and fixed-time comparisons are reported separately.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    while True:
        if args.loop:
            print("\033[2J\033[H", end="", flush=True)
        show(args.root)
        if not args.loop:
            break
        time.sleep(3)
