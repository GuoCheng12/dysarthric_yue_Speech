#!/usr/bin/env python3
"""Bind current content proposal and declared targets to the external feedback."""

import argparse
import json
from pathlib import Path

from mas_selection_dpo_probe import read, save
from therapist_harness.store import sha256_file


def main(base):
    old = base / 'patient_content_feedback_export_v1'
    root = base / 'patient_content_feedback_export_v2'; root.mkdir(exist_ok=False)
    data = base / 'patient_content_probe_v1'; experiment = base / 'patient_content_DPO_v1'
    payload = read(old / 'proposal_context.json')
    payload['schema_version'] = 'external_patient_content_feedback_v1'
    payload['ancestral_aggregate_proposal'] = payload.pop('source_proposal')
    payload['source_content_proposal'] = read(data / 'MAS_content/proposal.json')
    request = read(data / 'MAS_content/request.json')
    actual_input = json.loads(next(x['content'] for x in request['input'] if x['role'] == 'user'))
    payload['selected_patient_real_error_feedback_at_proposal_time'] = actual_input['real_error_feedback']
    payload['declared_content_target_observations'] = read(experiment / 'patient_content_target_observations.json')
    result = read(experiment / 'results.json')
    payload['matched_comparison'].update(MAS_new_minus_previous_content_pp=result['MAS_minus_original_CER_pp'],
        selected_patient_and_others=result['selected_patient_and_others'])
    payload['semantics']['target_response'] = 'Monitored cluster changes inherited from the ancestral aggregate proposal; these are not the new lexical target mechanism.'
    payload['semantics']['current_content_hypothesis'] = 'Use source_content_proposal and declared_content_target_observations for the actual new content intervention. Named target words remained uncorrected despite30 fewer overall errors versus previous content.'
    body = json.dumps(payload, ensure_ascii=False); assert '/data/' not in body and 'vlink' not in body
    files = [Path(__file__), old / 'proposal_context.json', old / 'provenance.json',
        data / 'MAS_content/proposal.json', data / 'MAS_content/request.json',
        experiment / 'results.json', experiment / 'patient_content_target_observations.json']
    save(root / 'proposal_context.json', payload)
    save(root / 'provenance.json', dict(source_sha256={str(f):sha256_file(f) for f in files},
        source_export_preserved=True, current_proposal_and_target_observations_correctly_bound=True,
        actual_new_MAS_call=False, original_ledgers_unchanged=True))
    save(old / 'usage_note.json', dict(status='ancestral_context_only', current_feedback=str(root / 'proposal_context.json'),
        reason='The generic exporter kept the ancestral aggregate source_proposal. Use the v2 packet binding the actual new content proposal and its target observations; no previous packet was sent to MAS.'))
    print(dict(status='complete', payload=str(root / 'proposal_context.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
