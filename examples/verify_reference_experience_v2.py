#!/usr/bin/env python3
"""Verify observed-reference counts and export corrected future-MAS context."""

import argparse
from collections import Counter
from pathlib import Path

from mas_selection_dpo_probe import read, rows, save
from therapist_harness.store import sha256_file


def main(base):
    source = base / 'reference_experience_v1'; receipt = read(source / 'receipt.json')
    assert all(sha256_file(Path(f)) == h for f, h in receipt['source_sha256'].items())
    assert all(sha256_file(Path(f)) == h for f, h in receipt['output_sha256'].items())
    observations = rows(source / 'observations.jsonl'); context = read(source / 'MAS_context.json')
    counts = Counter(x['reference_id'] for x in observations)
    assert len(observations) == context['unique_observed_audio'] == sum(counts.values())
    assert all(x['observed_unique_audio'] == counts[x['reference_id']] for x in context['cards'])
    root = base / 'reference_experience_v2'; root.mkdir(exist_ok=False)
    save(root / 'MAS_context.json', dict(context, schema_version='reference_experience_cards_v2',
        references_with_observations=len(counts), references_without_observations=len(context['cards']) - len(counts)))
    save(root / 'receipt.json', dict(status='complete', actual_MAS_request_executed=False,
        unique_observed_audio=len(observations), references_with_observations=len(counts),
        original_observations=str(source / 'observations.jsonl'),
        correction='The v1 receipt counted all120 rendered cards after defaultdict lookup;77 have observations. Individual cards and255audio observations were correct. Original files and code remain unchanged.',
        source_sha256={str(p): sha256_file(p) for p in [Path(__file__), source / 'receipt.json', source / 'MAS_context.json', source / 'observations.jsonl', source / 'receipt_count_correction.json']},
        output_sha256={str(root / 'MAS_context.json'): sha256_file(root / 'MAS_context.json')}))
    print(dict(audio=len(observations), observed_references=len(counts)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
