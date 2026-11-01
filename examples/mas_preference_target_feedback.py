#!/usr/bin/env python3
"""Describe actual preference errors using the existing EvidencePack attribution."""

import argparse
from pathlib import Path

from mas_selection_dpo_probe import REPO, read, rows, save
from therapist_harness.evidence import _cer_supported_phone_events
from therapist_harness.phonology import align_characters, analyze_phonological_errors
from therapist_harness.text import cer_normalization_projection
from therapist_harness.store import sha256_file


def actual_events(chosen, rejected):
    reference = cer_normalization_projection(chosen)
    hypothesis = cer_normalization_projection(rejected)
    phonology = analyze_phonological_errors(reference.surface, hypothesis.surface)
    alignment = align_characters(reference.normalized, hypothesis.normalized)
    events, _, _ = _cer_supported_phone_events(phonology, reference, hypothesis,
        {(x.reference_index, x.hypothesis_index) for x in alignment.forced_substitutions})
    return events


def describe(round_dir, preference_path):
    evidence = read(round_dir / 'evidence.json')
    proposal = read(round_dir / 'proposal.json')
    sidecar = {x['sample_id']: x for x in read(round_dir / 'materialization_sidecar.json')['samples']}
    intended = {x['subject_evidence_id'] for x in proposal['expected_outcomes']}
    families = {x['evidence_id']: (x['level'], x['reference_unit'], x['hypothesis_unit'])
        for x in evidence['error_clusters'] if x['evidence_id'] in intended and x['operation'] == 'substitution'}
    results = []
    data = rows(preference_path)
    for row in data:
        matches = [dict(level=e.level, intended_unit=e.reference_unit, observed_unit=e.hypothesis_unit,
            intended_character=e.reference_character, observed_character=e.hypothesis_character,
            context_sensitive=e.context_sensitive)
            for e in actual_events(row['chosen'], row['rejected'])
            if (e.level, e.reference_unit, e.hypothesis_unit) in families.values()]
        assigned = [m for m in matches if (m['level'], m['intended_unit'], m['observed_unit']) in
            [families[k] for k in sidecar[row['audio_id']]['evidence_ids'] if k in families]]
        if matches:
            results.append(dict(pair_id=row['pair_id'], audio_id=row['audio_id'],
                role=sidecar[row['audio_id']]['role'], chosen=row['chosen'], rejected=row['rejected'],
                matches=matches, assigned_matches=assigned))
    summary = {}
    for evidence_id, family in families.items():
        matching = [x for x in results if any((m['level'], m['intended_unit'], m['observed_unit']) == family for m in x['matches'])]
        assigned = [x for x in matching if any((m['level'], m['intended_unit'], m['observed_unit']) == family for m in x['assigned_matches'])]
        summary[evidence_id] = dict(family=family, all_pairs=len(matching), all_audio=len({x['audio_id'] for x in matching}),
            assigned_pairs=len(assigned), assigned_audio=len({x['audio_id'] for x in assigned}))
    files = [Path(__file__), REPO / 'src/therapist_harness/evidence.py',
        REPO / 'src/therapist_harness/phonology.py', REPO / 'src/therapist_harness/text.py',
        preference_path, round_dir / 'evidence.json', round_dir / 'proposal.json', round_dir / 'materialization_sidecar.json']
    return dict(source_sha256={str(f): sha256_file(f) for f in files}, total_pairs=len(data),
        summary=summary, matches=results,
        attribution='Exactly the existing EvidencePack raw-surface G2P plus projection onto forced CER substitutions. No raw-punctuation alignment or pronunciation-changing CER text as G2P input.',
        scope='Post-sampling training-opportunity feedback only. No audio pronunciation proof, new review process, admission rule or automatic data filtering.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round_dir', type=Path)
    parser.add_argument('preferences', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = describe(args.round_dir.resolve(), args.preferences.resolve())
    save(args.output.resolve(), result)
    print(result['summary'], flush=True)
