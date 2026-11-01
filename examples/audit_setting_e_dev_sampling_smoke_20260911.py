"""Reconstruct MAS inputs from dev receipts and verify the isolated GPU smoke."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from therapist_harness.sampling_curriculum_data import build_sampling_evidence, build_sampling_outcome, make_sampling_groups
from therapist_harness.real_sampling_mas import payload_hash
from therapist_harness.store import sha256_file, write_immutable_json


def read(path):
    return json.loads(path.read_text())


def audit(root):
    protocol = read(root / 'protocol.json')
    canonical = [[json.loads(line) for line in Path(protocol[key]).read_text().splitlines() if line.strip()]
                 for key in ('train_manifest', 'feedback_manifest', 'test_manifest')]
    assert protocol['setting'] == 'E' and protocol['feedback_role'] == 'dev'
    assert tuple(map(len, canonical)) == (1862, 394, 372)
    initial = []
    source_bindings = []
    summaries = {}
    for branch in ('uniform', 'mas'):
        path = root / branch
        complete = read(path / 'complete.json')
        assert complete['passed'] and complete['smoke'] and complete['steps'] == 4
        assert complete['actual_audio_exposures'] == 128
        gradient = read(path / 'smoke_passed.json')
        assert gradient['passed'] and all(row['passed'] for row in gradient['gradient_checks'])
        assert all(row['audio_lora_updated'] and row['language_lora_frozen'] and
                   row['base_frozen'] and row['post_smoke_restored_exactly'] for row in gradient['rank_receipts'])
        prep = read(path / 'preparation.json')
        assert all(sha256_file(Path(p)) == h for key in ('input_sha256', 'source_sha256') for p, h in prep[key].items())
        source_bindings.append(prep['source_sha256'])
        subset = read(path / 'smoke_subset.json')
        parts = [[r for r in part if r['utt_id'] in subset[key]]
                 for part, key in zip(canonical, ('train_ids', 'dev_ids', 'test_ids'))]
        assert all(len(part) == 16 for part in parts)
        assert {r['clean_gt'] for r in parts[0]} == {r['clean_gt'] for r in parts[1]} == {r['clean_gt'] for r in parts[2]}
        assert len(subset['shared_texts']) == 4
        patient_sets = [{r['speaker_id'] for r in part} for part in parts]
        assert all(not patient_sets[i] & patient_sets[j] for i, j in ((0, 1), (0, 2), (1, 2)))
        initial.append(read(path / 'initialization.json'))
        selected = read(path / 'selected_checkpoint.json')
        assert selected['selection_metric'] == 'dev_patient_equal_CER_percent'
        dev_metrics = [read(path / f'evaluations/step-{step}/feedback/summary.json') for step in (0, 2, 4)]
        assert selected['step'] == 2 * min(range(3), key=lambda s: dev_metrics[s]['patient_equal_CER_percent'])
        summaries[branch] = dict(steps=4, exposures=128, subset_audio=list(map(len, parts)),
                                 subset_patients=list(map(len, patient_sets)),
                                 subset_shared_texts=len(set.intersection(*[{r['clean_gt'] for r in part} for part in parts])),
                                 timing=complete['timing'])
        if branch != 'mas':
            continue
        # Preserve actual sampler order when reconstructing the frozen groups.
        train_by_id = {r['utt_id']: r for r in parts[0]}
        train = [train_by_id[uid] for uid in subset['train_ids']]
        dev_by_id = {r['utt_id']: r for r in parts[1]}
        dev = [dev_by_id[uid] for uid in subset['dev_ids']]
        groups = make_sampling_groups(train)
        predictions = [read(path / f'evaluations/step-{step}/feedback/predictions.json') for step in (0, 2, 4)]
        exposures = [None] + [read(path / f'stage-{stage:03d}/actual_exposure.json')['per_utterance'] for stage in (1, 2)]
        for stage in (1, 2):
            directory = path / f'stage-{stage:03d}'
            expected = build_sampling_evidence(train, predictions[stage-1], groups, feedback_rows=dev,
                previous_feedback=predictions[stage-2] if stage > 1 else None,
                round_index=stage, checkpoint_step=2*(stage-1), exposures=exposures[stage-1])
            context = json.loads(read(directory / 'decision/request.json')['input'][-1]['content'])
            assert payload_hash(context['evidence']) == payload_hash(expected)
            assert bool(context['history']) == (stage > 1)
            if stage > 1:
                assert context['history'][-1]['experience'] == read(path / 'stage-001/reflection/episode.json')['experience']
            decision = read(directory / 'decision/decision.json')
            outcome = build_sampling_outcome(train, predictions[stage-1], predictions[stage], groups,
                feedback_rows=dev, decision=decision, round_index=stage, checkpoint_step=2*stage,
                before_checkpoint_step=2*(stage-1), exposures=exposures[stage])
            reflected = json.loads(read(directory / 'reflection/reflection/request.json')['input'][-1]['content'])
            assert payload_hash(reflected['outcome']) == payload_hash(outcome)
    assert initial[0] == initial[1] and source_bindings[0] == source_bindings[1]
    receipts = [read(p) for p in (root / 'mas').glob('stage-*/**/receipt.json')]
    assert len(receipts) == 6
    assert all(r['validated'] and r['request_model'] == r['response_model'] == 'gpt-5.6-sol' for r in receipts)
    result = dict(passed=True, setting='E', train_audio=1862, dev_audio=394, test_audio=372,
        feedback_role='dev', same_fresh_initialization=True, source_hashes_verified=True,
        reconstructed_dev_only_requests=True, second_proposal_received_previous_experience=True,
        actual_agent_calls=6, branches=summaries, formal_training_started=False,
        interpretation='Small smoke preserves patient disjointness and four shared texts. It verifies execution and provenance, not curriculum benefit. Formal inputs retain all 60 shared texts.')
    write_immutable_json(root / 'smoke_summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root)
