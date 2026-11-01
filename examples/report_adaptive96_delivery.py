#!/usr/bin/env python3
"""Report fresh reference-policy delivery and actual matched training exposure."""

import argparse
from collections import Counter
from pathlib import Path

from mas_selection_dpo_probe import read, rows, save
from therapist_harness.round_schema import ClusterDescriptor
from therapist_harness.store import sha256_file


def report(base, stage, root_name):
    root = base / root_name
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(p)) == h for p, h in cfg['source_sha256'].items())
    sources = [Path(__file__), root / 'protocol.json', root / 'data_receipt.json']
    arms = {}
    generations = {}
    for arm, directory in cfg['source_data_roots'].items():
        directory = Path(directory)
        sidecar = read(directory / 'materialization_sidecar.json')
        roles = {x['sample_id']: x['role'] for x in sidecar['samples']}
        generation = read(directory / 'generation/generation_summary.json')
        outputs = {x['sample_id']: x for x in generation['outputs']}
        generations[arm] = outputs
        audit = read(directory / 'sampling/preference_audit.json')
        paired = {x['audio_id']: x for x in audit['audio_results']}
        selected = rows(root / arm / 'preferences.jsonl')
        selected_ids = {x['audio_id'] for x in selected}
        exposures_path = (root / arm / 'training_receipts/training_exposures.jsonl'
                          if stage == 'trained' else root / arm / 'expected_exposures.json')
        exposures = rows(exposures_path) if stage == 'trained' else read(exposures_path)
        occurrences = {(x['step'], x['source'], x['occurrence'], x['audio_id']) for x in exposures}
        role_counts = {}
        for role in sorted(set(roles.values())):
            ids = {k for k, value in roles.items() if value == role}
            available = ids & paired.keys()
            role_counts[role] = dict(
                requested_audio=len(ids), admitted_audio=sum(outputs[k]['admitted'] for k in ids),
                all_correct_audio=sum(paired[k]['pair_count'] == 0 for k in available),
                full_pair_audio=sum(paired[k]['pair_count'] > 0 for k in available),
                full_pair_rows=sum(paired[k]['pair_count'] for k in available),
                common_pair_audio=len(ids & selected_ids),
                trained_pair_rows=sum(x['audio_id'] in ids for x in selected),
                audio_exposures=sum(x[1] == 'synthetic' and x[3] in ids for x in occurrences),
                paired_error_origin=dict(Counter(paired[k]['error_origin'] for k in available
                                                  if paired[k]['pair_count'] > 0)))
        arms[arm] = dict(
            roles=role_counts, requested_audio=len(roles), admitted_audio=generation['admitted_count'],
            pending_audio=generation['pending_count'], excluded_audio=generation['excluded_count'],
            technically_unresolved_audio=generation.get('technical_unresolved_count', 0),
            generation_attempts=generation['generation_attempt_count'],
            all_correct_audio=audit['all_correct_audio_count'], full_pair_audio=sum(x['pair_count'] > 0 for x in paired.values()),
            full_pair_rows=sum(x['pair_count'] for x in paired.values()), common_pair_audio=len(selected_ids),
            selected_pair_rows=len(selected), pair_exposure_rows=len(exposures),
            real_audio_exposures=sum(x[1] == 'real' for x in occurrences),
            synthetic_audio_exposures=sum(x[1] == 'synthetic' for x in occurrences),
            elapsed_generation_seconds=generation['elapsed_seconds'],
            seconds_per_requested_audio=generation['elapsed_seconds'] / len(roles),
            timing=generation['timing'])
        sources += [directory / p for p in ['materialization_sidecar.json', 'generation/generation_summary.json',
                                            'sampling/preference_audit.json']]
        sources += [root / arm / 'preferences.jsonl', exposures_path]

    mas, control = generations['MAS_reference'], generations['random_reference']
    assert mas.keys() == control.keys()
    assert all(mas[k]['target_text'] == control[k]['target_text'] and
               mas[k]['assignment_seed'] == control[k]['assignment_seed'] for k in mas)
    changes = dict(
        same_target_text_and_assignment_seed=True,
        reference_recording_changed=sum(mas[k]['reference_wav_sha256'] != control[k]['reference_wav_sha256'] for k in mas),
        effective_recipe_changed=sum((mas[k]['generation_mode'], mas[k]['effective_instruction_id']) !=
                                     (control[k]['generation_mode'], control[k]['effective_instruction_id']) for k in mas),
        byte_identical_final_waveforms=sum(mas[k]['output_wav_sha256'] == control[k]['output_wav_sha256'] for k in mas))

    def descriptors(directory):
        evidence, proposal = read(directory / 'evidence.json'), read(directory / 'proposal.json')
        ids = {x['subject_evidence_id'] for x in proposal['expected_outcomes']}
        return [ClusterDescriptor(**{k: x[k] for k in ClusterDescriptor.model_fields}).model_dump(mode='json')
                for x in evidence['error_clusters'] if x['evidence_id'] in ids and x['operation'] == 'substitution']

    current = base / 'adaptive_mas96_data_v1/round_001'
    legacy = base / 'lexical96_mas_v1/round_001'
    assert descriptors(current) == descriptors(legacy)
    sources += [p / n for p in [current, legacy] for n in ['evidence.json', 'proposal.json']]
    result = dict(stage=stage, actual_exposures=stage == 'trained', arms=arms, paired_generation=changes,
                  target_report_descriptor_check='Current proposal and existing report helper have exactly equal monitored substitution descriptors.',
                  limitation='Full generation yield and common-opportunity training subset are separate estimates; coarse ASR-transcript review does not establish word-level audio truth.',
                  source_sha256={str(p): sha256_file(p) for p in sources})
    save(root / f'delivery_{stage}.json', result)
    print({arm: {k: v for k, v in data.items() if k not in ['roles', 'timing']} for arm, data in arms.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepared', 'trained'])
    parser.add_argument('base', type=Path)
    parser.add_argument('--root-name', default='adaptive96_matched_DPO_recovery_v2')
    args = parser.parse_args()
    report(args.base.resolve(), args.stage, args.root_name)
