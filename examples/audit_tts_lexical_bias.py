"""Describe ASR error characters newly represented by the MAS training text."""
from collections import Counter
import json
from pathlib import Path
import statistics

from therapist_harness.text import text_normalize, edit_distance

B = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1')
ROOT = B / 'tts_utility_diagnosis_v1'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def error_characters(ref, hyp):
    """Deterministic alignment: diagonal, then deletion, then insertion on ties."""
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i, a in enumerate(ref, 1):
        for j, b in enumerate(hyp, 1):
            d[i][j] = min(d[i-1][j-1] + (a != b), d[i-1][j] + 1, d[i][j-1] + 1)
    assert d[-1][-1] == edit_distance(ref, hyp)
    i, j = len(ref), len(hyp)
    wrong = Counter()
    correct = Counter()
    edits = 0
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            if ref[i-1] != hyp[j-1]:
                wrong[hyp[j-1]] += 1
                edits += 1
            else:
                correct[hyp[j-1]] += 1
            i -= 1
            j -= 1
        elif i and d[i][j] == d[i-1][j] + 1:
            i -= 1
            edits += 1
        else:
            wrong[hyp[j-1]] += 1
            j -= 1
            edits += 1
    assert edits == d[-1][-1]
    return wrong, correct


def run():
    real = rows(B / 'tts_increment_positive_control_v2/real_q/train.jsonl')
    mas = rows(B / 'real2117_mas576_hardref_v1/mas_hard_reference/train.jsonl')
    def vocabulary(data):
        return Counter(''.join(text_normalize(r['text'].split('<asr_text>', 1)[1]) for r in data))
    rv, sv = vocabulary(real), vocabulary(mas)
    novel = set(sv) - set(rv)
    output = {'novel_characters': sorted(novel),
        'definition': 'Characters present in MAS576 training labels but absent from all2117 real training labels; this does NOT mean unseen by the pretrained model.',
        'limits': 'Post-hoc descriptive alignment of model errors; no causal intervention. Counts are edit events, not CER percentage points or clinical impairment labels.',
        'results': {}, 'means': {}}
    arms = ['real_w0', 'mas_hard_w025', 'consistency_best_w025', 'mas_labels_clean_w025', 'strict_qc_clean_w005', 'strict_qc_room_w005']
    identity = None
    for arm in arms:
        output['results'][arm] = {}
        for seed in [42, 43, 44]:
            path = ROOT / 'additive' / arm / f'seed_{seed}/evaluation/merged_step_50/predictions.jsonl'
            data = rows(path)
            ids = [(r['utt_id'], r['clean_gt']) for r in data]
            if identity is None:
                identity = ids
            assert ids == identity and len(data) == 511
            counts = Counter()
            correct_counts = Counter()
            for row in data:
                wrong, correct = error_characters(text_normalize(row['clean_gt']), text_normalize(row['prediction']))
                counts.update(wrong)
                correct_counts.update(correct)
            output['results'][arm][seed] = {
                'all_wrong_output_chars': sum(counts.values()),
                'mas_only_wrong_output_chars': sum(counts[c] for c in novel),
                'mas_only_correct_output_chars': sum(correct_counts[c] for c in novel),
                'mas_only_by_char': {c: counts[c] for c in sorted(novel) if counts[c]},
            }
        output['means'][arm] = {key: statistics.mean(r[key] for r in output['results'][arm].values())
            for key in ['all_wrong_output_chars', 'mas_only_wrong_output_chars', 'mas_only_correct_output_chars']}
    output['mas_only_training_frequencies'] = {c: sv[c] for c in sorted(novel)}
    (ROOT / 'content_transfer/lexical_bias.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'novel_characters': sorted(novel), 'means': output['means']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
