#!/usr/bin/env python3
"""Derive simple reference experience cards from explicitly listed real runs."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from mas_selection_dpo_probe import read, rows, save, jsonl
from therapist_harness.store import sha256_file

RUNS = ['lexical96_mas_v1/round_001', 'adaptive_mas96_data_v1/round_001',
    'patient_assignment_probe_v1', 'patient_content_probe_v1', 'exploratory_round2_v1/adaptive_data']


def compile_cards(base):
    catalog = read(base / 'reference_catalog.json')['cards']
    reference_ids = {x['reference_wav_sha256']: x['reference_id'] for x in catalog}
    assert len(reference_ids) == len(catalog)
    sources = [Path(__file__), base / 'reference_catalog.json']
    observations, seen, protocols = [], {}, {}
    for run_index, relative in enumerate(RUNS):
        root = base / relative
        generation = read(root / 'generation/generation_summary.json')
        sample = root / 'sampling'; receipt = read(sample / 'sampling_receipt.json')
        assert generation['status'] == receipt['status'] == 'complete'
        sg_hash = generation['runtime']['speech_generator_sha256']
        assert sg_hash == '5764178a90d234997b06dcc5fd74d530419c282b6959c089f4d09b0f3e6eaf0b'
        assert sha256_file(sample / 'validated_inputs.jsonl') == receipt['validated_inputs_sha256']
        assert sha256_file(sample / 'candidates.jsonl') == receipt['candidates_sha256']
        config = {k: receipt[k] for k in ['current_asr_sha256', 'base_model_sha256', 'samples_per_audio',
            'temperature', 'top_p', 'top_k', 'max_new_tokens', 'seed', 'language', 'prompt']}
        protocol_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        protocols[protocol_id] = config
        audit = {x['audio_id']: x for x in read(sample / 'preference_audit.json')['audio_results']}
        sources += [root / 'generation/generation_summary.json', sample / 'sampling_receipt.json',
            sample / 'validated_inputs.jsonl', sample / 'candidates.jsonl', sample / 'preference_audit.json']
        for output in generation['outputs']:
            wave = output['output_wav_sha256']; reference = output['reference_wav_sha256']
            assert reference in reference_ids and sha256_file(Path(output['output_wav'])) == wave
            review = output['review']; assert review['candidate_audio_sha256'] == wave
            key = (wave, reference, output['target_text'])
            source_record = dict(run=relative, sample_id=output['sample_id'])
            if key in seen:
                seen[key]['source_records'].append(source_record)
                continue
            case = audit.get(output['sample_id'])
            observation = dict(observation_id=f'observation.{len(observations)+1:04d}', source_records=[source_record],
                source_run_index=run_index, reference_id=reference_ids[reference], audio_sha256=wave,
                speech_generator_sha256=sg_hash, target_text=output['target_text'],
                mode=output['generation_mode'], instruction_id=output['effective_instruction_id'],
                effective_prompt=output['effective_prompt_text'], admitted=output['admitted'],
                attempts=len(output['attempt_records']), verdict=review['verdict'], reason=review['reason'],
                review_asr_transcription=review['asr_transcription'],
                review_asr_sha256=review['transcription_adapter_sha256'],
                sampling_protocol_id=protocol_id,
                candidate_status=case['status'] if case else 'not_in_sampling_audit',
                actual_pair_count=case['pair_count'] if case else 0,
                actual_correct_draws=case['correct_samples'] if case else None,
                actual_sample_count=case['samples'] if case else 0)
            seen[key] = observation; observations.append(observation)
    by_reference = defaultdict(list)
    for row in observations: by_reference[row['reference_id']].append(row)
    cards = []
    for card in catalog:
        observed = by_reference[card['reference_id']]
        configs = defaultdict(list)
        for x in observed: configs[x['sampling_protocol_id']].append(x)
        # Show recent examples with different text/recipe where available. No
        # quality score, CER cutoff, or reference ranking is derived here.
        examples = []; examples_seen = set()
        for x in reversed(observed):
            key = (x['target_text'], x['mode'], x['instruction_id'])
            if key in examples_seen: continue
            examples_seen.add(key)
            examples.append({k: x[k] for k in ['observation_id', 'target_text', 'mode', 'instruction_id',
                'verdict', 'reason', 'review_asr_transcription', 'attempts', 'sampling_protocol_id',
                'candidate_status', 'actual_pair_count', 'actual_correct_draws', 'actual_sample_count']})
            if len(examples) == 3: break
        cards.append(dict(reference_id=card['reference_id'], voice_group=card['voice_group'],
            observed_unique_audio=len(observed), admitted_audio=sum(x['admitted'] for x in observed),
            review_verdicts=dict(Counter(x['verdict'] for x in observed)),
            required_retry_audio=sum(x['attempts'] > 1 for x in observed),
            actual_recipes=dict(Counter(x['mode'] + ':' + x['instruction_id'] for x in observed)),
            sampling_by_protocol={k: dict(audio=len(xs), statuses=dict(Counter(x['candidate_status'] for x in xs)),
                pair_rows=sum(x['actual_pair_count'] for x in xs)) for k, xs in configs.items()},
            examples=examples))
    root = base / 'reference_experience_v1'; root.mkdir(exist_ok=False)
    jsonl(root / 'observations.jsonl', observations)
    save(root / 'MAS_context.json', dict(schema_version='reference_experience_cards_v1', cards=cards,
        sampling_protocols=protocols, unique_observed_audio=len(observations),
        semantics='Observed generation/review and preference opportunities only. More errors/pairs are not proven training utility, coarse pass is not word fidelity, and an ASR transcript is not an acoustic diagnosis. Different sampling parents/configs remain separate. Unknown references have zero observations, not a bad-quality label.',
        scope='Five explicitly listed completed runs, not an exhaustive or randomly sampled reference benchmark. Current running DPO is unchanged; these cards are available to a future actual MAS request. No per-reference causal CER gain or listening claim.'))
    save(root / 'receipt.json', dict(status='complete', actual_MAS_request_executed=False, runs=RUNS,
        unique_observed_audio=len(observations), references_with_observations=len(by_reference),
        source_sha256={str(p): sha256_file(p) for p in sources},
        output_sha256={str(p): sha256_file(p) for p in [root / 'observations.jsonl', root / 'MAS_context.json']}))
    print(dict(status='complete', audio=len(observations), observed_references=len(by_reference)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('base', type=Path)
    compile_cards(parser.parse_args().base.resolve())
