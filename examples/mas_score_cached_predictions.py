#!/usr/bin/env python3
"""Score cached predictions canonically, preserving the original export."""

import argparse
from pathlib import Path

from mas_selection_dpo_probe import REPO, rows, save, jsonl
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.store import sha256_file
from therapist_harness.text import edit_distance, text_normalize


def main(args):
    root = args.output_dir.resolve()
    root.mkdir(exist_ok=False)
    source = args.input.resolve()
    original = rows(source)
    corrected = []
    changed_critical = 0
    for row in original:
        reference = text_normalize(row['clean_gt'])
        hypothesis = text_normalize(row['prediction'])
        edits = edit_distance(reference, hypothesis)
        cer = edits / len(reference)
        critical = not hypothesis or cer >= .5
        changed_critical += row.get('critical') != critical
        corrected.append(dict(row, edit_count=edits, cer=cer, critical=critical,
                              exact=edits == 0, normalized_reference_characters=len(reference)))
    metrics = evaluate_snapshot(corrected, prediction_column='prediction', critical_cer_threshold=.5)
    jsonl(root / 'predictions.jsonl', corrected)
    files = [source, Path(__file__), REPO / 'src/therapist_harness/evidence.py',
             REPO / 'src/therapist_harness/text.py']
    save(root / 'summary.json', dict(metrics=metrics.model_dump(mode='json'),
        source_sha256={str(p): sha256_file(p) for p in files}, prediction_texts_unchanged=True,
        critical_flag_corrections=changed_critical, critical_semantics='empty hypothesis or CER >= 0.5',
        actual_new_asr_decoding=False, actual_new_llm_calls=False,
        scope='Canonical scoring of existing outputs; original exports remain untouched.'))
    print(metrics.overall.pooled_cer, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    main(parser.parse_args())
