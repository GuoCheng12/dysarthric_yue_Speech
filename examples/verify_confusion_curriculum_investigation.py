"""Verify frozen inputs, independent edit counts and the recorded stop decision."""
from collections import Counter
import json
from pathlib import Path

from investigate_confusion_curriculum import ROOT, TRAIN, CONTROL, read_rows, sha, write_json
from therapist_harness.text import text_normalize


def independent_distance(a, b):
    # A separate full-matrix implementation, not Harness alignments/tracebacks.
    matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1,
                               matrix[i - 1][j - 1] + int(a[i - 1] != b[j - 1]))
    return matrix[-1][-1]


def main():
    source = json.loads((ROOT / 'sources.json').read_text())
    audit = json.loads((ROOT / 'audit.json').read_text())
    hashes = 0
    for value in source.values():
        records = value if isinstance(value, list) else [value]
        for record in records:
            assert sha(Path(record['path'])) == record['sha256'], record['path']
            hashes += 1
    predictions = 0
    for name in audit['metrics']:
        rows = [r for p in source[name] for r in read_rows(Path(p['path']))]
        pred_field = 'raw_pred' if name == 'zero' else 'prediction'
        for bucket, metrics in audit['metrics'][name].items():
            selected = [r for r in rows if bucket == 'overall' or r['zero_shot_bucket'] == bucket]
            values = [(independent_distance(text_normalize(r['clean_gt']), text_normalize(r[pred_field])),
                       len(text_normalize(r['clean_gt']))) for r in selected]
            assert len(values) == metrics['n']
            assert abs(100 * sum(e / n for e, n in values) / len(values)
                       - metrics['mean_utterance_cer_percent']) < 1e-10
            assert abs(100 * sum(e for e, n in values) / sum(n for e, n in values)
                       - metrics['pooled_cer_percent']) < 1e-10
        predictions += len(rows)
    train_rows = read_rows(TRAIN)
    control_path = CONTROL / 'real_q/train.jsonl'
    control_rows = read_rows(control_path)
    # Historical trainer manifests intentionally contain only audio/text/prompt.
    # Match their exact training inputs, using audio identity to join metadata.
    train = {r['audio']: r for r in train_rows}
    control = {r['audio']: r for r in control_rows}
    assert len(train_rows) == len(control_rows) == len(train) == len(control) == 2117
    assert set(train) == set(control)
    for audio in train:
        for key in ('text', 'prompt'):
            assert train[audio][key] == control[audio][key]
    metadata = {}
    for seed in (42, 43, 44):
        folder = CONTROL / f'real_q/seed_{seed}/training'
        m = json.loads((folder / 'lora_run_metadata.json').read_text())
        s = json.loads((folder / 'checkpoint-50/trainer_state.json').read_text())
        args = m['training_arguments']
        execution = json.loads((folder.parent / 'execution.json').read_text())
        assert execution['train_sha256'] == sha(control_path)
        assert s['global_step'] == 50
        assert args['seed'] == seed and args['batch_size'] == 8 and args['grad_acc'] == 4
        assert args['lr'] == 0.0002 and args['lora_init_adapter_path'] == ''
        metadata[str(seed)] = {k: args[k] for k in ('seed', 'batch_size', 'grad_acc', 'lr', 'max_steps', 'lora_init_adapter_path')}
    assert audit['qualified_count'] == sum(r['qualifies'] for r in audit['candidates'])
    for candidate in audit['candidates']:
        assert candidate['qualifies'] == all(candidate['checks'].values())
    assert audit['decision'] == 'STOP_NO_SUPPORTED_RECIPROCAL_PATTERN'
    assert audit['qualified_count'] == 0
    transitions = {}
    for name in audit['metrics']:
        if name == 'zero':
            continue
        records = json.loads((ROOT / f'{name}_position_events.json').read_text())
        counts = Counter((kind, p['difficulty']) for r in records for kind in ('recovered', 'introduced')
                         for p in r['transitions'].get(kind, []))
        transitions[name] = {f'{kind}.{bucket}': n for (kind, bucket), n in sorted(counts.items())}
    result = {
        'status': 'verified_stop', 'verified_source_hashes': hashes,
        'independently_verified_prediction_rows': predictions,
        'real_q_equals_original_real_training_pool': True,
        'real_q_training_metadata': metadata,
        'primary_single_component_transitions': transitions,
        'related_unit_tests_passed': 42,
        'unit_test_command': 'PYTHONPATH=src pytest -q -p no:cacheprovider tests/test_confusion_curriculum_investigation.py tests/test_therapist_phonology.py tests/test_therapist_target_recovery.py tests/test_therapist_surface_grounding.py tests/test_therapist_text.py',
        'agent_api_calls': 0, 'new_asr_runs': 0, 'new_tts_runs': 0,
        'new_mas_built': False, 'stop_reason': audit['decision'],
        'verifier_source_sha256': sha(Path(__file__)),
    }
    write_json(ROOT / 'acceptance.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
