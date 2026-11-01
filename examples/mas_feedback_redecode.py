#!/usr/bin/env python3
"""Actual two-pass ASR using a fixed error-feedback trigger and vocabulary."""

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import time

from mas_lexical_context_probe import evaluator, REPO, read, save, compare
from mas_selection_dpo_probe import jsonl
from therapist_harness.store import sha256_file, sha256_directory
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.text import text_normalize, edit_distance


def main(base):
    started = time.monotonic()
    root = base / 'feedback_redecode_actual_v1'
    root.mkdir(exist_ok=False)
    protocol_path = base / 'lexical96_mas_v1/protocol.json'
    p = read(protocol_path)
    feedback_path = base / 'lexical_feedback_v1.json'
    feedback = read(feedback_path)['observed_surface_confusions']
    counts = Counter()
    for item in feedback:
        counts[item['key'][3]] = max(counts[item['key'][3]], item['count'])
    words = [w for w, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if n >= 3]
    triggers = sorted({x['key'][4] for x in feedback if x['count'] >= 3})
    context = '關鍵詞：' + '、'.join(words)
    args = SimpleNamespace(base_model_path=Path(p['base_model_dir']), adapter_path=Path(p['initial_adapter']['path']),
        batch_size=8, max_new_tokens=256)
    split = Path(p['feedback_split']['path'])
    files = [Path(__file__), Path(evaluator.__file__), Path(__file__).with_name('mas_lexical_context_probe.py'),
        protocol_path, feedback_path, split, base / 'lexical_asr_context_probe_v1/gated/protocol.json']
    hashes = {str(f): sha256_file(f) for f in files}
    assert sha256_directory(args.adapter_path) == p['initial_adapter']['sha256']
    save(root / 'protocol.json', dict(source_sha256=hashes, parent=p['initial_adapter'], trigger_characters=triggers,
        second_pass_context=context, language='Cantonese', max_batch_size=8, max_new_tokens=256,
        rule='Decode with blank context. If the first hypothesis contains a recorded frequent observed-error character, decode that audio again with the fixed vocabulary context and use the second result.',
        decision_inputs='First-pass ASR text and frozen aggregate feedback only. No item label, CER, patient identity or per-item target phrase.',
        actual_new_asr_decoding=True, actual_training=False, actual_llm_calls=False,
        scope='Actual two-pass inference execution; separate from SG review and DPO. Repeated development feedback, not an independent test.'))
    rows = evaluator.read_split(split)
    audio_hashes = {x['utt_id']: sha256_file(Path(x['audio'])) for x in rows}
    t0 = time.monotonic()
    model = evaluator.load_model(args)
    timing = dict(load_seconds=time.monotonic() - t0, first_pass_seconds=0., second_pass_seconds=0.)
    predictions, records = [], []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        t0 = time.monotonic()
        first = model.transcribe(audio=[x['audio'] for x in batch], context='', language=['Cantonese'] * len(batch))
        timing['first_pass_seconds'] += time.monotonic() - t0
        assert len(first) == len(batch)
        matches = [[c for c in triggers if c in text_normalize(x.text)] for x in first]
        indices = [i for i, values in enumerate(matches) if values]
        second = {}
        if indices:
            t0 = time.monotonic()
            outputs = model.transcribe(audio=[batch[i]['audio'] for i in indices], context=context,
                                       language=['Cantonese'] * len(indices))
            timing['second_pass_seconds'] += time.monotonic() - t0
            assert len(outputs) == len(indices)
            second = dict(zip(indices, outputs, strict=True))
        # Item labels are used only after both actual decoding decisions are complete.
        for i, row in enumerate(batch):
            result = second.get(i, first[i])
            records.append(dict(utt_id=row['utt_id'], audio=row['audio'], audio_sha256=audio_hashes[row['utt_id']],
                first_pass=first[i].text, first_context='', trigger_matches=matches[i],
                second_pass=None if i not in second else second[i].text,
                second_context=None if i not in second else context, selected_pass=2 if i in second else 1))
            reference, hypothesis = text_normalize(row['clean_gt']), text_normalize(result.text)
            edits = edit_distance(reference, hypothesis)
            cer = edits / len(reference)
            predictions.append(dict(row, prediction=result.text, detected_language=result.language,
                normalized_reference_characters=len(reference), edit_count=edits, cer=cer,
                critical=not hypothesis or cer >= .5, exact=edits == 0))
        print(dict(decoded=len(predictions), total=len(rows), second_pass_audio=sum(x['selected_pass'] == 2 for x in records)), flush=True)
    timing['elapsed_seconds'] = time.monotonic() - started
    metrics = evaluate_snapshot(predictions, prediction_column='prediction', critical_cer_threshold=.5)
    evaluator.write_outputs(root / 'evaluation', predictions, dict(metrics=metrics.model_dump(mode='json'),
        timing=timing, second_pass_audio=sum(x['selected_pass'] == 2 for x in records), actual_new_asr_decoding=True))
    jsonl(root / 'actual_decoding_records.jsonl', records)
    assert all(sha256_file(Path(f)) == digest for f, digest in hashes.items())
    assert sha256_directory(args.adapter_path) == p['initial_adapter']['sha256']
    save(root / 'paired_parent.json', compare(Path(p['initial_adapter_dev_predictions']['path']), root / 'evaluation/predictions.jsonl'))
    save(root / 'results.json', dict(status='complete', metrics=metrics.model_dump(mode='json'), timing=timing,
        automatic_asr_checkpoint_promotion=False, actual_training=False, actual_new_asr_decoding=True))
    print(metrics.overall.model_dump(mode='json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    main(parser.parse_args().base.resolve())
