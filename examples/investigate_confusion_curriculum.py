"""Paired, source-bound audit of reciprocal ASR correction/regression.

No model training, synthesis, or API calls. The continuation decision follows
docs/confusion_curriculum_investigation_v1.zh.md, not a fitted utility score.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

from therapist_harness.capabilities import _state
from therapist_harness.evidence import _read_snapshot, _strictly_pair
from therapist_harness.phonology import map_text_to_syllables
from therapist_harness.text import tts_phonology_text


BASE = Path('/data/qwen3-asr')
REPO = BASE / 'repo/dysarthric_yue_Speech'
ROOT = BASE / 'records/therapist_mas_setting_d_v1/confusion_curriculum_investigation_v1'
M0 = BASE / 'records/therapist_mas_setting_d_v1/real_only_m0_b8g4_utterance_mean_v1'
CONTROL = BASE / 'records/therapist_mas_setting_d_v1/tts_increment_positive_control_v2'
ZERO = BASE / 'inference/outputs/setting_d_seen_patient_unseen_content_v1_qc_frozen'
TRAIN = BASE / 'finetune/data_setting_d_v1/setting_d_train.jsonl'
LEVELS = ('onset', 'nucleus', 'coda', 'tone')


def read_rows(path):
    with path.open(encoding='utf-8') as handle:
        if path.suffix == '.csv':
            return list(csv.DictReader(handle))
        return [json.loads(line) for line in handle]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')


def signature_map(row, *, single_component):
    """Only normalized-CER-supported, context-stable phonological events."""
    groups = defaultdict(list)
    for event in row.phone_events:
        groups[event.reference_index].append(event)
    result = {}
    for index, events in groups.items():
        if single_component and len(events) != 1:
            continue
        eligible = [e for e in events if e.context_sensitive is False]
        if len(eligible) != len(events):
            continue
        result[index] = [
            (event.level, event.reference_unit or '∅', event.hypothesis_unit or '∅')
            for event in events
        ]
    return result


def paired_events(before, after, *, single_component=True):
    """Keep actual correct/error transitions; migrations remain migrations."""
    _strictly_pair(after, before, role='before')
    after_by_id = {row.utt_id: row for row in after}
    events = defaultdict(lambda: defaultdict(list))
    coverage = Counter()
    carrier_transitions = defaultdict(Counter)
    for old in before:
        new = after_by_id[old.utt_id]
        old_map = signature_map(old, single_component=single_component)
        new_map = signature_map(new, single_component=single_component)
        for position in old.phonology.reference.positions:
            coverage['raw_gt_positions'] += 1
            if not position.resolved or position.context_sensitive is not False:
                coverage['gt_g2p_unresolved_or_sensitive'] += 1
                continue
            a, b = _state(old, position.index), _state(new, position.index)
            coverage[f'{a}__{b}'] += 1
            for level in LEVELS:
                key = (level, getattr(position, level) or '∅', old.difficulty)
                carrier_transitions[key][f'{a}__{b}'] += 1
            detail = {
                'utt_id': old.utt_id, 'speaker_id': old.speaker_id,
                'content_id': old.content_id, 'difficulty': old.difficulty,
                'gt_position': position.index, 'gt_character': position.character,
                'gt_jyutping': position.jyutping, 'clean_gt': old.clean_gt,
                'before': old.phonology.hypothesis.text,
                'after': new.phonology.hypothesis.text,
                'before_state': a, 'after_state': b,
            }
            for key in old_map.get(position.index, []):
                transition = (
                    'recovered' if b == 'correct'
                    else 'after_unresolved' if b == 'unresolved'
                    else 'persisted' if key in new_map.get(position.index, [])
                    else 'migrated_to_other_error'
                )
                events[key][transition].append(detail)
            for key in new_map.get(position.index, []):
                if key in old_map.get(position.index, []):
                    continue
                transition = (
                    'introduced' if a == 'correct'
                    else 'before_unresolved' if a == 'unresolved'
                    else 'arrived_from_other_error'
                )
                events[key][transition].append(detail)
    return dict(events), dict(coverage), carrier_transitions


def select(events, key, kind, groups):
    return [r for r in events.get(key, {}).get(kind, []) if r['difficulty'] in groups]


def support(rows):
    contents = Counter(r['content_id'] for r in rows)
    return {
        'positions': len(rows), 'patients': len({r['speaker_id'] for r in rows}),
        'contents': len(contents), 'gt_characters': len({r['gt_character'] for r in rows}),
        'by_content': dict(contents),
        'by_character': dict(Counter(r['gt_character'] for r in rows)),
    }


def enough(value):
    return (value['positions'] >= 5 and value['patients'] >= 3
            and value['contents'] >= 2 and value['gt_characters'] >= 2)


def pair_summary(events, key):
    level, a, b = key
    reverse = (level, b, a)
    hard = {'hard'}
    protected = {'easy', 'medium'}
    repair = select(events, key, 'recovered', hard)
    repair_undo = select(events, key, 'introduced', hard)
    harm = select(events, reverse, 'introduced', protected)
    harm_undo = select(events, reverse, 'recovered', protected)
    return {
        'hard_recovered': support(repair), 'hard_introduced': support(repair_undo),
        'easy_medium_reverse_introduced': support(harm),
        'easy_medium_reverse_recovered': support(harm_undo),
        'hard_net_recovery': len(repair) - len(repair_undo),
        'easy_medium_reverse_net_harm': len(harm) - len(harm_undo),
        'hard_old_error_migrations': len(select(events, key, 'migrated_to_other_error', hard)),
        'easy_medium_reverse_old_error_migrations': len(select(events, reverse, 'migrated_to_other_error', protected)),
        'hard_recovered_examples': repair,
        'easy_medium_reverse_introduced_examples': harm,
    }


def training_pool(rows):
    mappings = {text: map_text_to_syllables(tts_phonology_text(text))
                for text in {r['clean_gt'] for r in rows}}
    pool = defaultdict(list)
    for row in rows:
        units = {(level, getattr(p, level) or '∅')
                 for p in mappings[row['clean_gt']].positions
                 if p.resolved and p.context_sensitive is False for level in LEVELS}
        for key in units:
            pool[key].append(row)
    return pool


def pool_support(rows):
    return {'recordings': len(rows), 'patients': len({r['speaker_id'] for r in rows}),
            'contents': len({r['prompt_id'] for r in rows})}


def pool_enough(counts):
    return counts['recordings'] >= 10 and counts['patients'] >= 3 and counts['contents'] >= 2


def decide_candidate(key, primary, replications, pool):
    summary = pair_summary(primary, key)
    validation = {name: pair_summary(events, key) for name, events in replications.items()}
    replicated = [name for name, row in validation.items()
                  if row['hard_net_recovery'] > 0 and row['easy_medium_reverse_net_harm'] > 0
                  and row['hard_recovered']['positions'] >= 3
                  and row['easy_medium_reverse_introduced']['positions'] >= 3]
    a_pool = pool_support(pool.get((key[0], key[1]), []))
    b_pool = pool_support(pool.get((key[0], key[2]), []))
    checks = {
        'primary_hard_support_5_3_2_2': enough(summary['hard_recovered']),
        'primary_reverse_harm_support_5_3_2_2': enough(summary['easy_medium_reverse_introduced']),
        'primary_two_positive_net_transitions': summary['hard_net_recovery'] > 0 and summary['easy_medium_reverse_net_harm'] > 0,
        'replicated_at_least_two_seeds': len(replicated) >= 2,
        'train_pool_both_sides': pool_enough(a_pool) and pool_enough(b_pool),
    }
    return {
        'key': list(key), 'checks': checks, 'qualifies': all(checks.values()),
        'primary': summary, 'replications': validation, 'replicated_seeds': replicated,
        'train_A': a_pool, 'train_B': b_pool,
    }


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    paths = {
        'zero': [ZERO / f'setting_d_{split}.csv' for split in ('dev', 'test')],
        'm0': [M0 / 'evaluation/dev_step_50/predictions.jsonl',
               M0 / 'evaluation/test_selected_m0/predictions.jsonl'],
        **{f'real_q_seed_{seed}': [CONTROL / f'real_q/seed_{seed}/evaluation/merged_step_50/predictions.jsonl']
           for seed in (42, 43, 44)},
    }
    sources = {name: [{'path': str(p), 'sha256': sha(p)} for p in files]
               for name, files in paths.items()}
    plan = REPO / 'docs/confusion_curriculum_investigation_v1.zh.md'
    sources['plan'] = {'path': str(plan), 'sha256': sha(plan)}
    sources['script'] = {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__))}
    sources['train'] = {'path': str(TRAIN), 'sha256': sha(TRAIN)}
    write_json(ROOT / 'sources.json', sources)
    snapshots = {}
    metric_summary = {}
    raw_rows = {}
    for name, files in paths.items():
        rows = [r for p in files for r in read_rows(p)]
        assert len(rows) == 511 and len({r['utt_id'] for r in rows}) == 511
        column = 'raw_pred' if name == 'zero' else 'prediction'
        snapshots[name] = _read_snapshot(rows, role=name, prediction_column=column,
                                         critical_cer_threshold=0.5)
        raw_rows[name] = rows
        metric_summary[name] = {}
        for group in ('overall', 'easy', 'medium', 'hard'):
            subset = [r for r in snapshots[name] if group == 'overall' or r.difficulty == group]
            metric_summary[name][group] = {
                'n': len(subset),
                'mean_utterance_cer_percent': 100 * sum(r.edit_count / len(r.reference) for r in subset) / len(subset),
                'pooled_cer_percent': 100 * sum(r.edit_count for r in subset) / sum(len(r.reference) for r in subset),
            }
        print('Loaded', name, metric_summary[name], flush=True)
    train = read_rows(TRAIN)
    assert len(train) == 2117
    assert not {r['utt_id'] for r in train} & {r.utt_id for r in snapshots['zero']}
    assert not {r['prompt_id'] for r in train} & {r.content_id for r in snapshots['zero']}
    assert len({r['prompt_id'] for r in train}) == 48
    assert len({r.content_id for r in snapshots['zero']}) == 12
    pool = training_pool(train)
    strict = {}
    broad = {}
    coverages = {}
    carriers = {}
    for name, snapshot in snapshots.items():
        if name == 'zero':
            continue
        strict[name], coverages[name], raw_carriers = paired_events(snapshots['zero'], snapshot)
        broad[name], _, _ = paired_events(snapshots['zero'], snapshot, single_component=False)
        carriers[name] = [{'level': k[0], 'unit': k[1], 'difficulty': k[2], 'transitions': dict(v)}
                          for k, v in sorted(raw_carriers.items())]
    replications = {name: events for name, events in strict.items() if name != 'm0'}
    candidates = [decide_candidate(key, strict['m0'], replications, pool)
                  for key in sorted(strict['m0'])]
    candidates.sort(key=lambda r: (r['qualifies'], sum(r['checks'].values()),
                                   min(r['primary']['hard_recovered']['positions'],
                                       r['primary']['easy_medium_reverse_introduced']['positions'])), reverse=True)
    broad_candidates = [pair_summary(broad['m0'], key) | {'key': list(key)} for key in sorted(broad['m0'])]
    broad_candidates.sort(key=lambda r: min(r['hard_recovered']['positions'], r['easy_medium_reverse_introduced']['positions']), reverse=True)
    output = {
        'decision': 'CONTINUE_TO_MAS_SMOKE' if any(r['qualifies'] for r in candidates) else 'STOP_NO_SUPPORTED_RECIPROCAL_PATTERN',
        'scope': 'observational transcript-confusion evidence; not causal or clinical diagnosis',
        'metrics': metric_summary, 'coverage': coverages, 'carrier_transitions': carriers,
        'candidate_count': len(candidates), 'qualified_count': sum(r['qualifies'] for r in candidates),
        'candidates': candidates, 'exploratory_multicomponent_candidates': broad_candidates,
        'protocol': {'evaluation_rows': 511, 'evaluation_content': 12, 'train_rows': 2117,
                     'train_content': 48, 'train_eval_overlap': 0, 'agent_calls': 0, 'new_training_runs': 0},
    }
    write_json(ROOT / 'audit.json', output)
    for name, events in strict.items():
        write_json(ROOT / f'{name}_position_events.json', [
            {'key': list(key), 'transitions': value} for key, value in sorted(events.items())])
    print(json.dumps({'decision': output['decision'], 'candidate_count': len(candidates),
                      'qualified_count': output['qualified_count'], 'top_candidates': [
                          {'key': r['key'], 'checks': r['checks'],
                           'repaired': r['primary']['hard_recovered'],
                           'harmed': r['primary']['easy_medium_reverse_introduced'],
                           'replicated': r['replicated_seeds']} for r in candidates[:8]]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
