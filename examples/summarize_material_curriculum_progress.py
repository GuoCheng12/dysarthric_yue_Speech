"""Read-only comparison of a running material curriculum with retained controls."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

BASELINE = Path('/data/qwen3-asr/records/therapist_mas_setting_e_v1/preserved_methods/real_sampling_mas_seed42_590_v1')


def read(path):
    return json.loads(path.read_text())


def events(path):
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A live writer can leave its last line incomplete during this read.
    return result


def summarize(root):
    result = dict(utc=datetime.now(timezone.utc).isoformat(), branches={}, matched_steps=[], rounds=[])
    protocol = read(root / 'formal/protocol.json')
    retained = {row['step']: row for row in read(BASELINE / 'results/comparison.json')['same_steps']}
    evaluated = {}
    for name in ('uniform', 'mas'):
        branch = root / 'formal' / name
        if name == 'uniform' and protocol.get('uniform_comparison_path'):
            branch = Path(protocol['uniform_comparison_path'])
        records = events(branch / 'events.jsonl')
        evaluated[name] = {r['step']: r for r in records if r['stage'] == 'evaluation_complete'}
        losses = [r for r in records if r['stage'] == 'loss']
        supervisor = read(branch / 'supervisor.json')
        status_path = Path(f"/proc/{supervisor['pid']}/status")
        process_state = None
        try:
            command = status_path.with_name('cmdline').read_bytes()
            if str(branch.parent / 'protocol.json').encode() in command and name.encode() in command.split(b'\0'):
                process_state = next(line.split(':', 1)[1].strip() for line in status_path.read_text().splitlines() if line.startswith('State:'))
        except FileNotFoundError:
            pass
        if (branch / 'initialization.json').exists():
            assert read(branch / 'initialization.json') == read(BASELINE / 'provenance/v9/mas/initialization.json')
        result['branches'][name] = dict(pid=supervisor['pid'], process_state=process_state, source_path=str(branch),
            complete=(branch / 'complete.json').exists(), failed=(branch / 'failure.json').exists(),
            latest_optimizer_step=losses[-1]['step'] if losses else 0,
            latest_evaluation_step=max(evaluated[name], default=None), last_event=records[-1]['stage'])
    for step in sorted(set(evaluated['uniform']) & set(evaluated['mas']) & set(retained)):
        row = dict(step=step)
        for label, measurement in [('new_uniform', evaluated['uniform'][step]), ('new_material_mas', evaluated['mas'][step]),
                                   ('retained_uniform', retained[step]['uniform']), ('retained_mas', retained[step]['mas'])]:
            for role in ('feedback', 'test'):
                for key in ('CER_percent', 'patient_equal_CER_percent'):
                    row[f'{label}_{role}_{key}'] = measurement[role][key]
        row['test_delta_vs_retained_mas_pp'] = row['new_material_mas_test_CER_percent'] - row['retained_mas_test_CER_percent']
        result['matched_steps'].append(row)
    branch = root / 'formal/mas'
    for stage in sorted(branch.glob('stage-*')):
        schedule = stage / 'decision/schedule/decision.json'
        selected = stage / 'decision/selected_materials.json'
        if not schedule.exists() or not selected.exists():
            continue
        materials, decision = read(selected), read(schedule)
        row = dict(round=int(stage.name.split('-')[1]), selected_probability=decision['selected_probability'],
                   unique_recordings=materials['unique_recordings'], patients=materials['patients'], texts=materials['texts'],
                   difficulty_counts=dict(Counter(r['initial_difficulty'] for r in materials['recordings'])),
                   purposes=read(stage / 'decision/diagnosis/decision.json')['focuses'])
        exposure = stage / 'actual_exposure.json'
        if exposure.exists():
            delivery = read(exposure)
            row.update(actual_channels=dict(Counter(delivery['channels'])), unique_exposed=delivery['unique_audio'])
        reflection = stage / 'reflection/reflection/decision.json'
        if reflection.exists():
            row['reflection'] = read(reflection)
        result['rounds'].append(row)
    pending = []
    for path in branch.glob('stage-*/**/*.started.json'):
        if not path.with_name(path.name.replace('.started', '')).exists():
            pending.append(dict(path=str(path), **read(path)))
    result['pending_api_attempts'] = pending
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    target = output / 'live_progress.json'
    temporary = target.with_suffix('.pending')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    if result['matched_steps']:
        with (output / 'live_comparison.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result['matched_steps'][0]))
            writer.writeheader()
            writer.writerows(result['matched_steps'])
    print(json.dumps(dict(utc=result['utc'], branches=result['branches'],
        last_matched=result['matched_steps'][-1] if result['matched_steps'] else None,
        completed_allocations=len(result['rounds']), pending_api_attempts=pending), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    summarize(parser.parse_args().root)
