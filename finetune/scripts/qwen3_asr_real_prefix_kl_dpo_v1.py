#!/usr/bin/env python3
"""Bind the real KL conditioning history without changing the KL/DPO math."""

import argparse
import json
from pathlib import Path
import sys

import qwen3_asr_edit_masked_kl_dpo as masked


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--real_prefix_kind', choices=['ground_truth', 'reference_greedy'], required=True)
    args, remaining = parser.parse_known_args()
    real_path = Path(remaining[remaining.index('--real_train_file') + 1])
    data = [json.loads(x) for x in real_path.read_text().splitlines() if x.strip()]
    if args.real_prefix_kind == 'reference_greedy':
        assert all(x['metadata']['real_prefix_kind'] == 'actual_frozen_R_only_greedy_transcript_for_KL_only' for x in data)
    else:
        assert all('real_prefix_kind' not in x.get('metadata', {}) for x in data)
    description = ('Original real-training ground-truth transcript prefixes' if args.real_prefix_kind == 'ground_truth'
                   else 'Actual frozen initial R-only greedy transcript prefixes on the same real training audio')
    description += '; full-vocabulary symmetric KL on transcript+EOS; no real CE or development labels'
    engine = masked.engine; original = engine.write_json
    source = Path(__file__); digest = engine.sha256(source)

    def write_record(path, record):
        record = dict(record)
        name = Path(path).name
        if name == 'run_protocol.json':
            record.update(real_label=description, real_prefix_kind=args.real_prefix_kind,
                source_sha256={**record['source_sha256'], str(source): digest})
        if 'real_history' in record: record['real_history'] = description
        if isinstance(record.get('residual_execution'), dict):
            record['residual_execution'] = dict(record['residual_execution'], real_history=description,
                                                real_prefix_kind=args.real_prefix_kind)
        if name == 'real_teacher_cache.json': record['conditional_history'] = description
        if name in {'trainer_state.json', 'residual_composition.json', 'training_result.json', 'residual_execution.json'}:
            record['real_prefix_kind'] = args.real_prefix_kind
        original(path, record)

    engine.write_json = write_record
    sys.argv = [sys.argv[0]] + remaining
    masked.main()
    assert engine.sha256(source) == digest


if __name__ == '__main__':
    main()
