"""Compare fixed dev practice cohorts with actual matched-step ordinary predictions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from therapist_harness.material_sampling_mas import measure_focus
from therapist_harness.material_selection_offline import PracticeFocus, digest, matches
from therapist_harness.sampling_curriculum_data import _join_predictions
from therapist_harness.store import sha256_file, write_immutable_json


def audit(root, through_round):
    sources = {}

    def read(path):
        sources[str(path)] = sha256_file(path)
        return json.loads(path.read_text())

    protocol = read(root / 'formal/protocol.json')
    control = Path(protocol['uniform_comparison_path'])
    manifest = Path(protocol['feedback_manifest'])
    sources[str(manifest)] = sha256_file(manifest)
    feedback = [json.loads(line) for line in manifest.read_text().splitlines()]
    index = read(Path(protocol['material_curriculum']['index_path']))
    predictions = {}

    def joined(branch, step):
        key = str(branch), step
        if key not in predictions:
            path = branch / f'evaluations/step-{step}/feedback/predictions.json'
            predictions[key] = _join_predictions(feedback, read(path))
        return predictions[key]

    rows = []
    for number in range(1, through_round + 1):
        branch = root / 'formal/mas'
        stage = branch / f'stage-{number:03d}'
        outcome = read(stage / 'reflection/outcome.json')
        assert outcome['feedback_role'] == 'dev'
        before, after = outcome['before_checkpoint_step'], outcome['checkpoint_step']
        for item in outcome['focus_outcomes']:
            focus = PracticeFocus.model_validate(item['focus'])
            ids = sorted(r['utt_id'] for r in feedback
                         if matches(index['profiles'][digest(r['clean_gt'])[:16]], focus))
            measurements = {}
            for name, directory in [('mas', branch), ('uniform', control)]:
                measurements[name] = {}
                for when, step in [('previous', before), ('current', after)]:
                    measurement = measure_focus(joined(directory, step), focus, index)
                    measurements[name][when] = measurement
                    assert measurement['cohort']['audio'] == len(ids)
                    if name == 'mas':
                        assert measurement == item[when], (number, focus.focus_id, when)
            cohorts = [measurements[b][t]['cohort'] for b in measurements for t in ('previous', 'current')]
            assert len({(c['audio'], c.get('patients'), c.get('chars')) for c in cohorts}) == 1
            changes = {b: (measurements[b]['current']['cohort']['errors'] -
                           measurements[b]['previous']['cohort']['errors']) if ids else None
                       for b in measurements}
            rows.append(dict(round=number, before_step=before, after_step=after,
                focus=focus.model_dump(), dev_ids=ids,
                matching_selected_ids_exposures=item['matching_selected_ids_exposures'],
                measurements=measurements, error_changes=changes,
                mas_minus_uniform_error_change=(changes['mas'] - changes['uniform']) if ids else None))
    summary = {}
    for level in sorted({r['focus']['level'] for r in rows}):
        related = [r for r in rows if r['focus']['level'] == level]
        counts = Counter('no_dev_cohort' if r['mas_minus_uniform_error_change'] is None else
                         'larger_decline' if r['mas_minus_uniform_error_change'] < 0 else
                         'smaller_decline' if r['mas_minus_uniform_error_change'] > 0 else
                         'equal_change' for r in related)
        summary[level] = dict(focus_round_records=len(related), **counts)
    sources[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    for filename in ('material_sampling_mas.py', 'material_selection_offline.py',
                     'sampling_curriculum_data.py', 'sampling_phonology.py', 'phonology.py', 'text.py'):
        path = REPO / 'src/therapist_harness' / filename
        sources[str(path)] = sha256_file(path)
    result = dict(passed=True, through_round=through_round, dev_only=True,
        stored_mas_outcomes_recomputed_exactly=True, same_dev_id_cohorts=True,
        control_source=str(control), rows=rows, summary=summary, source_sha256=sources,
        interpretation='Descriptive matched-step comparison, not causal attribution. Targets were selected from MAS errors; cohorts overlap across targets and rounds. Counts are focus-round records, not independent samples. This audit is not fed to the running Agents.')
    target = root / 'analysis' / f'fixed_focus_control_through_round{through_round:03d}.json'
    write_immutable_json(target, result)
    print(json.dumps(dict(output=str(target), passed=True, records=len(rows), summary=summary), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--through-round', required=True, type=int)
    args = parser.parse_args()
    audit(args.root, args.through_round)
