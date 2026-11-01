#!/usr/bin/env python3
"""Report actual matched reference delivery without equating errors with utility."""
from collections import Counter
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mas_literal_matched_sg_probe_v1 as run
from therapist_harness.evidence import _stable_align
from therapist_harness.preference_data import build_preference_pairs


def count_event(target, prediction, metadata):
    target = run.text_normalize(target)
    prediction = run.text_normalize(prediction)
    cost, events, ambiguous = _stable_align(target, prediction)
    a, b = metadata['intended_reference'], metadata['intended_hypothesis']
    carrier = a if metadata['role'] == 'target' else b
    rival = b if metadata['role'] == 'target' else a
    positions = {i for i, char in enumerate(target) if char == carrier}
    wrong = {event.reference_index for event in events if event.reference_index in positions}
    return dict(edit_count=cost, carrier_count=len(positions), carrier_errors=len(wrong),
        carrier_correct=len(positions - wrong), ambiguous_alignment=ambiguous,
        intended_or_reverse_substitution=sum(e.operation == 'substitution' and e.reference_unit == carrier
                                             and e.hypothesis_unit == rival for e in events))


def main():
    p = run.verify()
    results, sources = {}, [Path(__file__), run.ROOT / 'proposal.json', run.ROOT / 'plan.json']
    identity, case_rows, listening = {}, {}, []
    for arm in ['A', 'B']:
        out = run.ROOT / arm
        inputs = run.rows(out / 'sampling/validated_inputs.jsonl')
        candidates = run.rows(out / 'sampling/candidates.jsonl')
        greedy = run.rows(out / 'sampling/greedy.jsonl')
        receipt = run.read(out / 'sampling/sampling_receipt.json')
        pairs = run.rows(out / 'sampling/preferences.jsonl')
        audit = run.read(out / 'sampling/preference_audit.json')
        assert receipt['status'] == 'complete' and receipt['current_asr_sha256'] == p.initial_adapter.sha256
        for key, filename in [('candidates_sha256','candidates.jsonl'),('validated_inputs_sha256','validated_inputs.jsonl'),('greedy_sha256','greedy.jsonl')]:
            assert receipt[key] == run.sha256_file(out / 'sampling' / filename)
        rebuilt, _ = build_preference_pairs(inputs, candidates, p.initial_adapter.sha256, expected_samples=4, greedy_candidates=greedy)
        assert pairs == rebuilt
        assert all(run.sha256_file(Path(x['audio'])) == x['audio_sha256'] for x in inputs)
        generation = run.read(out / 'generation/generation_summary.json')
        metadata = {x['sample_id']:x for x in run.read(out / 'role_sidecar.json')['samples']}
        manifest = {x['sample_id']:x for x in run.rows(out / 'manifest.jsonl')}
        choices = {x['sample_id']:x for x in run.read(out / 'generation_plan.json')['choices']}
        by_id = {x['audio_id']:x for x in inputs}
        assert len(manifest) == len(generation['outputs']) == 12
        observed = []
        for output in generation['outputs']:
            sid = output['sample_id']; meta = metadata[sid]
            attempts = output['attempt_records']
            first = attempts[0]
            assert output['generation_mode'] == 'instruct2' and output['effective_instruction_id'] == 'cantonese_zh'
            assert output['assignment_seed'] == manifest[sid]['seed']
            assert run.sha256_file(Path(output['output_wav'])) == output['output_wav_sha256']
            assert (sid in by_id) == output['admitted']
            draws = [x for x in candidates if x['audio_id'] == sid]
            normal = [x for x in greedy if x['audio_id'] == sid]
            pair_rows = [x for x in pairs if x['audio_id'] == sid]
            assert len(draws) == (4 if output['admitted'] else 0)
            item = dict(sample_id=sid, metadata=meta, target_text=output['target_text'],
                reference_text=output['reference_text'], audio=output['output_wav'], audio_sha256=output['output_wav_sha256'],
                first_attempt_review=first.get('review'), first_attempt_technical_failure=first.get('technical_failure'),
                final_review=output['review'], admitted=output['admitted'], attempts=len(attempts),
                actual_distinct_pairs=len(pair_rows), candidates=[dict(prediction=x['prediction'], sample_index=x['sample_index'],
                    seed=x['seed'], ended_with_eos=x['ended_with_eos'], reached_max_new_tokens=x['reached_max_new_tokens'],
                    counts=count_event(output['target_text'],x['prediction'],meta)) for x in draws],
                greedy=[dict(prediction=x['prediction'],counts=count_event(output['target_text'],x['prediction'],meta)) for x in normal])
            item['candidate_draws_with_intended_or_reverse_error'] = sum(x['counts']['intended_or_reverse_substitution'] > 0 for x in item['candidates'])
            item['distinct_pairs_with_intended_or_reverse_error'] = sum(count_event(x['chosen'],x['rejected'],meta)['intended_or_reverse_substitution'] > 0 for x in pair_rows)
            observed.append(item)
            e = html.escape
            listening.append(f'<article><h3>{arm}: {e(sid)}</h3><p>{e(output["target_text"])}</p><audio controls src="{e(output["output_wav"])}"></audio><p>Reference: {e(output["reference_text"])}</p><p>审查转写：{e(output["review"]["asr_transcription"])}</p><p>{e(output["review"]["verdict"])}：{e(output["review"]["reason"])}</p><p>实际候选：{e(" / ".join(x["prediction"] for x in draws))}</p></article>')
        aggregate = {}
        for role in ['target','counter']:
            selected = [x for x in observed if x['metadata']['role'] == role]
            counts = Counter()
            for x in selected:
                for d in x['candidates']:
                    counts.update({k:v for k,v in d['counts'].items() if k != 'ambiguous_alignment'})
            aggregate[role] = dict(requested_audio=len(selected), admitted_audio=sum(x['admitted'] for x in selected),
                paired_audio=sum(x['actual_distinct_pairs'] > 0 for x in selected),
                distinct_pairs=sum(x['actual_distinct_pairs'] for x in selected),
                audio_with_intended_or_reverse_error=sum(x['candidate_draws_with_intended_or_reverse_error'] > 0 for x in selected),
                candidate_counts=counts)
        identity[arm] = dict(receipt=receipt, manifest=manifest, metadata=metadata, choices=choices, candidates=candidates)
        case_rows[arm] = {x['sample_id']:x for x in observed}
        results[arm] = dict(aggregate=aggregate, cases=observed, preference_audit=audit,
            first_attempt_verdicts=Counter(x['first_attempt_review']['verdict'] if x['first_attempt_review'] else 'technical_failure' for x in observed),
            final_verdicts=Counter(x['final_review']['verdict'] for x in observed),
            truncations=sum(x['reached_max_new_tokens'] or not x['ended_with_eos'] for x in candidates),
            generation_seconds=generation['elapsed_seconds'], generation_timing=generation['timing'])
        sources += [out / 'generation/generation_summary.json', out / 'manifest.jsonl', out / 'role_sidecar.json', out / 'generation_plan.json']
        sources += [out / 'sampling' / name for name in ['validated_inputs.jsonl','candidates.jsonl','greedy.jsonl','sampling_receipt.json','preferences.jsonl','preference_audit.json']]
    a, b = identity['A'], identity['B']
    assert a['manifest'].keys() == b['manifest'].keys()
    for sid in a['manifest']:
        for field in ['sample_id','target_text','seed']:
            assert a['manifest'][sid][field] == b['manifest'][sid][field]
        assert a['metadata'][sid]['speaker_id'] == b['metadata'][sid]['speaker_id']
        assert a['manifest'][sid]['reference_wav_sha256'] != b['manifest'][sid]['reference_wav_sha256']
    # Preserve independent receipts; compare the actual decoding configuration.
    config_keys = ['current_asr_sha256','samples_per_audio','seed','temperature','top_p','max_new_tokens','language','prompt']
    for key in config_keys:
        assert a['receipt'][key] == b['receipt'][key], key
    draws_a = {(x['audio_id'], x['sample_index']):x for x in a['candidates']}
    draws_b = {(x['audio_id'], x['sample_index']):x for x in b['candidates']}
    for key in draws_a.keys() & draws_b.keys():
        assert draws_a[key]['seed'] == draws_b[key]['seed']
    common_pairs = [sid for sid in a['manifest'] if case_rows['A'][sid]['actual_distinct_pairs'] and case_rows['B'][sid]['actual_distinct_pairs']]
    target_hits = [x for arm in ['A','B'] for x in results[arm]['cases']
                   if x['metadata']['role'] == 'target' and x['candidate_draws_with_intended_or_reverse_error'] > 0]
    hit_patients = sorted({x['metadata']['speaker_id'] for x in target_hits})
    hit_audio = {x['audio_sha256'] for x in target_hits}
    supports_delivery = len(hit_audio) >= 2 and len(hit_patients) == 2
    report = dict(status='complete_matched_literal_SG_delivery', arms=results, common_paired_case_ids=common_pairs,
        MAS_preregistered_delivery_test=dict(target_audio_with_actual_substitution=len(hit_audio),
            patients_with_actual_substitution=hit_patients, supported=supports_delivery,
            action='May propose matched training allocation' if supports_delivery else 'Close this proposal before training',
            criterion='Actual MAS proposal.json: intended substitution in >=2targetaudio units spanning both selectedpatients; no alteration of SG admission.'),
        paired_identity_and_sampling_verified=True, actual_new_training=False,
        interpretation='A was preferred by MAS before generation; B is same-patient comparator. Target-error delivery and counter-carrier correctness describe actual ASR text, not verified audio pronunciation or proven training utility. Separate first attempts from rulebook finals. Coarse review and chosen labels unchanged; no CER filter.4draws per admitted audio, not independent patients.',
        sources={str(f):run.sha256_file(f) for f in sources})
    run.save(run.ROOT / 'delivery_report.json', report)
    lines=['MAS新24音频：实际内容与reference交付','', '| 条件 | 目标准入 | 目标有pair | 有目标替换错误的音频 | 反向句准入 | 反向句有pair |', '|---|---:|---:|---:|---:|---:|']
    for arm in ['A','B']:
        t,c=results[arm]['aggregate']['target'],results[arm]['aggregate']['counter']
        lines.append(f'|{arm}|{t["admitted_audio"]}/6|{t["paired_audio"]}|{t["audio_with_intended_or_reverse_error"]}|{c["admitted_audio"]}/6|{c["paired_audio"]}|')
    lines += ['',f'两侧都有实际偏好对的配对case：{len(common_pairs)}/12。尚未训练。',
        f'MAS预先声明的跨患者目标重现：{supports_delivery}。{len(hit_audio)}条目标音频、{len(hit_patients)}名患者出现指定替换。', '',report['interpretation']]
    run.write_immutable_bytes(run.ROOT / 'delivery_report.zh.md', ('\n'.join(lines)+'\n').encode())
    run.write_immutable_bytes(run.ROOT / 'listening.html', ('<!doctype html><meta charset="utf-8"><title>MAS matched24</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}article{border:1px solid #ccc;padding:12px}</style><h1>新内容 × 同患者reference</h1><p>三文本粗审；不代表逐字或逐音素保真。</p><main>'+''.join(listening)+'</main>').encode())
    print(json.dumps(dict(aggregate={arm:results[arm]['aggregate'] for arm in ['A','B']},common_paired_cases=len(common_pairs),
        MAS_preregistered_delivery_test=report['MAS_preregistered_delivery_test']),ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
