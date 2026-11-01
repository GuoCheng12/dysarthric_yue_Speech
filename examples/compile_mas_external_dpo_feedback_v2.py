#!/usr/bin/env python3
"""Compile actual external training results into the next MAS proposal context.

This retains the original data-stage ledger. It is a feedback export, not a
retroactive core ReflectionPack or a claim that a new MAS reflection was called.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from mas_selection_dpo_probe import read, rows, save
from mas_explicit_feedback_correction import explicit_feedback
from therapist_harness.phonology import align_characters
from therapist_harness.store import sha256_file
from therapist_harness.text import text_normalize


def substitutions(data):
    result = Counter()
    for row in data:
        reference = text_normalize(row['clean_gt'])
        hypothesis = text_normalize(row['prediction'])
        alignment = align_characters(reference, hypothesis)
        for event in alignment.forced_substitutions:
            result[(reference[event.reference_index], hypothesis[event.hypothesis_index])] += 1
    return result


def compile_feedback(base, experiment, output):
    cfg = read(experiment / 'protocol.json')
    assert all(sha256_file(Path(p)) == h for p, h in cfg['source_sha256'].items())
    results = read(experiment / 'results.json')
    assert results['status'] == 'complete'
    source = base / 'adaptive_mas96_data_v1/round_001'
    proposal = read(source / 'proposal.json'); evidence = read(source / 'evidence.json')
    sidecar = {x['sample_id']: x for x in read(source / 'materialization_sidecar.json')['samples']}
    parent_path = base / 'parent_eval_check/predictions.jsonl'
    before = rows(parent_path); before_by_id = {x['utt_id']: x for x in before}
    speakers = sorted({x['speaker_id'] for x in before})
    anonymous = {s: f'feedback_patient_{i:03d}' for i, s in enumerate(speakers)}
    catalog = read(base / 'reference_catalog.json')
    voice_by_speaker = {x['speaker_id']: x['voice_group'] for x in catalog['cards']}
    before_errors = substitutions(before)
    files = [Path(__file__), experiment / 'protocol.json', experiment / 'results.json',
        parent_path, base / 'lexical_feedback_v1.json', base / 'reference_catalog.json']
    files += [source / n for n in ['proposal.json', 'evidence.json', 'materialization_sidecar.json',
                                  'generation/generation_summary.json']]
    arms = {}
    for arm, summary in results['arms'].items():
        out = experiment / arm
        arm_source = Path(cfg.get('source_data_roots', {}).get(arm, str(source)))
        generation_path = arm_source / 'generation/generation_summary.json'
        generation = {x['sample_id']: x for x in read(generation_path)['outputs']}
        files.append(generation_path)
        assert read(out / 'verified.json')['status'] == 'passed'
        after_path = out / 'evaluation_run/evaluation/predictions.jsonl'
        after = rows(after_path); after_by_id = {x['utt_id']: x for x in after}
        assert before_by_id.keys() == after_by_id.keys()
        patient_stats = defaultdict(lambda: dict(audio=0, characters=0, before_errors=0, after_errors=0))
        for key, old in before_by_id.items():
            new = after_by_id[key]
            assert old['speaker_id'] == new['speaker_id'] and old['clean_gt'] == new['clean_gt']
            value = patient_stats[old['speaker_id']]
            value['audio'] += 1; value['characters'] += len(text_normalize(old['clean_gt']))
            value['before_errors'] += old['edit_count']; value['after_errors'] += new['edit_count']
        patients = [dict(patient_key=anonymous[s], reference_voice_group=voice_by_speaker.get(s),
            **patient_stats[s], error_delta=patient_stats[s]['after_errors'] - patient_stats[s]['before_errors'])
            for s in speakers]
        after_errors = substitutions(after)
        changes = [dict(intended_character=k[0], observed_asr_character=k[1],
            before_count=before_errors[k], after_count=after_errors[k], delta=after_errors[k] - before_errors[k])
            for k in sorted(before_errors.keys() | after_errors.keys(),
                key=lambda k: (-(before_errors[k] + after_errors[k]), k))[:40]]
        response = read(out / 'target_response.json')
        monitored = {x['subject_evidence_id'] for x in proposal['expected_outcomes']}
        descriptors = {x['evidence_id']: (x['level'], x['reference_unit'], x['hypothesis_unit'])
            for x in evidence['error_clusters'] if x['evidence_id'] in monitored and x['operation'] == 'substitution'}
        actual = {x['descriptor']['evidence_id']: (x['descriptor']['level'],
            x['descriptor']['reference_unit'], x['descriptor']['hypothesis_unit'])
            for x in response['monitored_cluster_changes']}
        assert descriptors == actual
        exposure_path = out / 'training_receipts/training_exposures.jsonl'
        exposures = rows(exposure_path)
        occurrences = {(x['step'], x['source'], x['occurrence'], x['audio_id']) for x in exposures}
        counts = Counter(x[3] for x in occurrences if x[1] == 'synthetic')
        a = cfg['arguments'][arm]
        pref_path = Path(a['synthetic_train_file']) if a.get('synthetic_train_file') else None
        preferences = rows(pref_path) if pref_path else []
        grouped = defaultdict(list)
        for pair in preferences: grouped[pair['audio_id']].append(pair)
        deliveries = []
        for audio_id, pairs in sorted(grouped.items()):
            item = generation[audio_id]; role = sidecar[audio_id]['role']
            assert sha256_file(Path(item['output_wav'])) == pairs[0]['audio_sha256']
            deliveries.append(dict(case_key=audio_id, role=role, target=pairs[0]['chosen'],
                actual_rejected_texts=[x['rejected'] for x in pairs], pair_count=len(pairs),
                actual_audio_exposures=counts[audio_id],
                reference_id=item['generation_choice']['reference_id'],
                generation_mode=item['generation_mode'], instruction_id=item['effective_instruction_id']))
        clean_summary = dict(summary)
        clean_summary['dominant_patient_error_delta'] = clean_summary.pop('vlink247_edit_delta')
        clean_summary['dominant_patient_CER_percent'] = clean_summary.pop('vlink247_CER_percent')
        training_protocol = read(out / 'training_receipts/run_protocol.json')
        arms[arm] = dict(metrics=clean_summary, patients=patients,
            actual_training_objective=training_protocol['objective'],
            actual_training_arguments={k: a[k] for k in ['max_steps', 'real_per_step', 'synthetic_per_step',
                'synthetic_weight', 'lr', 'beta', 'seed', 'preference_token_mask'] if k in a},
            real_loss_kind='symmetric_KL' if 'kl_weight' in a else 'ground_truth_CE',
            real_audio_exposures=sum(x[1] == 'real' for x in occurrences),
            synthetic_audio_exposures=sum(x[1] == 'synthetic' for x in occurrences),
            concrete_surface_confusion_changes=changes,
            monitored_target_changes=response['monitored_cluster_changes'],
            utterance_transitions=response['transitions'], patient_response=response['patient_response'],
            actual_synthetic_delivery=deliveries)
        files += [after_path, exposure_path, out / 'verified.json', out / 'target_response.json',
                  out / 'training_receipts/run_protocol.json']
        if pref_path: files.append(pref_path)
    payload = dict(schema_version='external_dpo_feedback_v2', experiment=experiment.name,
        source_proposal=proposal, original_concrete_error_feedback=explicit_feedback(read(base / 'lexical_feedback_v1.json')),
        arms=arms, parent_promoted=results['parent_promoted'],
        matched_comparison={k: results[k] for k in ['MAS_increment_pp', 'MAS_minus_random_CER_pp',
            'greedy_minus_stochastic_CER_pp', 'bootstrap95_pp'] if k in results},
        semantics=dict(real_evaluation='Repeated511 development feedback; original2117 training pool is separate.',
            surface_counts='Forced unambiguous character substitutions only; not all CER errors or an acoustic diagnosis.',
            target_response='Track recovered original positions, new errors and insertions as well as signature counts.',
            admission='Original three-text coarse review unchanged; no word-fidelity claim or new threshold.',
            delivery='Actual training occurrences, not generated copy counts; all-correct skipped audio gets no invented negative.',
            attribution='Whole-branch outcomes cannot identify the causal effect of one reference recording.',
            continuation='A rejected endpoint does not become the next training parent.',
            provenance='External probe feedback; no retroactive rewrite of the original protocol/ledger and no LLM reflection call claimed.'))
    serialized = __import__('json').dumps(payload, ensure_ascii=False)
    assert '/data/' not in serialized and 'vlink' not in serialized
    output.mkdir(exist_ok=False)
    save(output / 'proposal_context.json', payload)
    save(output / 'provenance.json', dict(source_sha256={str(p): sha256_file(p) for p in files},
        payload_sha256=sha256_file(output / 'proposal_context.json'),
        local_patient_crosswalk=anonymous, source_ledger_mutated=False, new_API_calls=0))
    print(dict(status='complete', arms=list(arms), payload=str(output / 'proposal_context.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path); parser.add_argument('experiment', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    compile_feedback(args.base.resolve(), args.experiment.resolve(), args.output.resolve())
