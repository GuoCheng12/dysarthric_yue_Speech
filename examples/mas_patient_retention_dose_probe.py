#!/usr/bin/env python3
"""A predeclared dose check of the patient-protection direction selected by MAS."""

import argparse
from pathlib import Path

from mas_patient_retention_probe import TRAINER, cpu_smoke, run
from mas_selection_dpo_probe import read, rows, save
from therapist_harness.store import sha256_file


def prepare(base, root):
    previous = base / 'patient_retention_probe_v1'
    cfg = read(previous / 'protocol.json')
    assert read(previous / 'results.json')['status'] == 'complete'
    assert read(previous / 'verified.json')['status'] == 'passed'
    assert all(sha256_file(Path(f)) == h for f, h in cfg['source_sha256'].items())
    root.mkdir(exist_ok=False)
    a = dict(cfg['arguments'], output_dir=str(root / 'training'))
    real, synthetic = rows(a['real_train_file']), rows(a['synthetic_train_file'])
    allocation = read(a['synthetic_allocation_file'])
    prior = read(previous / 'real_loss_factors.json')
    focus, alpha = set(prior['focus_ids']), .5
    assert len(real) == 96 and len(focus) == 2 and prior['focus_share'] == .25
    factors = {x['audio_id']: alpha * 96 / len(focus) if x['audio_id'] in focus
               else (1 - alpha) * 96 / (96 - len(focus)) for x in real}
    save(root / 'real_loss_factors.json', dict(prior, factors=factors, focus_share=alpha))
    save(root / 'cpu_smoke.json', cpu_smoke(real, synthetic, allocation, factors, focus, alpha))
    save(root / 'design.json', dict(
        actual_new_MAS_decision=False,
        original_actual_MAS_decision=str(previous / 'mas_decision.json'),
        dose_choice='Predeclared engineering sensitivity check: focus share .25 to .50.',
        previous_outcome='MAS share .25 reduced 14 total edits versus uniform KL10, including 13 in the focus patient; still 3 edits worse overall and 19 worse in hard than R-only. Primary target-position recovery remained25.',
        hypothesis='More real-retention weight on the same underrepresented patient may further reduce its hard-speech regression while retaining the synthetic correction benefit.',
        unchanged='96 real recordings,192 real exposures;15 synthetic pairs,96 synthetic exposures; exact order;12steps;lr1e-4;KL10;beta.1;seed42;initial R-only parent/reference.',
        acceptance='Original raw pooled/macro/hard gate on all511 development recordings. No patient exclusion or postprocessing.'))
    files = [Path(__file__), TRAINER, previous / 'protocol.json', previous / 'results.json',
             previous / 'verified.json', previous / 'mas_decision.json', previous / 'target_response.json',
             previous / 'training/training_exposures.jsonl', previous / 'real_loss_factors.json',
             root / 'real_loss_factors.json', root / 'cpu_smoke.json', root / 'design.json']
    hashes = dict(cfg['source_sha256'], **{str(f): sha256_file(f) for f in files})
    save(root / 'protocol.json', dict(cfg, source_sha256=hashes, arguments=a,
        dose_control=str(previous), actual_new_MAS_decision=False,
        intervention='Same existing MAS-selected focus patient; increase normalized real KL share .25 to .50. All audio and exposures unchanged.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('base', type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    root = base / 'patient_retention_dose_probe_v1'
    if args.stage == 'prepare':
        prepare(base, root)
    else:
        run(base, root)
