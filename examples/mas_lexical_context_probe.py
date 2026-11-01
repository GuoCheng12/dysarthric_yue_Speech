#!/usr/bin/env python3
"""Test a fixed feedback-derived vocabulary context at ASR inference time."""

import argparse
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

from mas_selection_dpo_probe import REPO, read, save
from mas_evaluation_report import compare
from therapist_harness.store import sha256_file, sha256_directory
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.text import edit_distance, text_normalize

sys.path.insert(0, str(REPO / 'finetune/scripts'))
import evaluate_asr_checkpoint as evaluator


def main(args):
    base = args.base.resolve()
    root = base / 'lexical_asr_context_probe_v1' / args.condition
    root.mkdir(parents=True, exist_ok=False)
    p = read(base / 'lexical96_mas_v1/protocol.json')
    feedback_path = base / 'lexical_feedback_v1.json'
    counts = Counter()
    for item in read(feedback_path)['observed_surface_confusions']:
        intended = item['key'][3]
        # Maximum component count avoids counting the known joint i/m error twice.
        counts[intended] = max(counts[intended], item['count'])
    words = [word for word, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if count >= 3]
    context = '關鍵詞：' + '、'.join(words) if args.condition == 'vocabulary' else ''
    config = SimpleNamespace(base_model_path=Path(p['base_model_dir']),
        adapter_path=Path(p['initial_adapter']['path']), batch_size=8, max_new_tokens=256)
    split = Path(p['feedback_split']['path'])
    files = [Path(__file__), Path(evaluator.__file__), feedback_path, split,
        base / 'lexical96_mas_v1/protocol.json', Path(p['initial_adapter_dev_predictions']['path'])]
    hashes = {str(f): sha256_file(f) for f in files}
    assert sha256_directory(config.adapter_path) == p['initial_adapter']['sha256']
    save(root / 'protocol.json', dict(source_sha256=hashes, parent=p['initial_adapter'],
        context=context, context_selection='All intended characters whose maximum previously recorded component count is at least 3; fixed across all audio. No full sentence or item label in ASR input.',
        selected_characters=words, condition=args.condition, language='Cantonese', batch_size=8,
        max_new_tokens=256, actual_training=False, actual_llm_calls=False,
        scope='Exploratory inference-context probe, separate from TTS generation/review and DPO. Repeated development feedback; not a sealed test.'))
    rows = evaluator.read_split(split)
    model = evaluator.load_model(config)
    predictions = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        outputs = model.transcribe(audio=[x['audio'] for x in batch], context=context,
                                   language=['Cantonese'] * len(batch))
        assert len(outputs) == len(batch)
        for row, output in zip(batch, outputs, strict=True):
            reference, hypothesis = text_normalize(row['clean_gt']), text_normalize(output.text)
            edits = edit_distance(reference, hypothesis)
            cer = edits / len(reference)
            predictions.append(dict(row, prediction=output.text, detected_language=output.language,
                normalized_reference_characters=len(reference), edit_count=edits, cer=cer,
                critical=not hypothesis or cer >= .5, exact=edits == 0))
        print(dict(decoded=len(predictions), total=len(rows), condition=args.condition), flush=True)
    metrics = evaluate_snapshot(predictions, prediction_column='prediction', critical_cer_threshold=.5)
    evaluator.write_outputs(root / 'evaluation', predictions, dict(metrics=metrics.model_dump(mode='json'),
        context=context, protocol_sha256=sha256_file(root / 'protocol.json'), actual_new_asr_decoding=True))
    assert all(sha256_file(Path(f)) == digest for f, digest in hashes.items())
    assert sha256_directory(config.adapter_path) == p['initial_adapter']['sha256']
    save(root / 'paired_parent.json', compare(Path(p['initial_adapter_dev_predictions']['path']),
                                              root / 'evaluation/predictions.jsonl'))
    save(root / 'results.json', dict(status='complete', metrics=metrics.model_dump(mode='json'),
        automatic_promotion=False, actual_training=False, actual_new_asr_decoding=True))
    print(metrics.overall.model_dump(mode='json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('--condition', required=True, choices=['blank', 'vocabulary'])
    main(parser.parse_args())
