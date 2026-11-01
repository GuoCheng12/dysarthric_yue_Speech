"""Prepare, audit and compare the first concrete-material MAS curriculum."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from therapist_harness.store import sha256_file, write_immutable_json
from therapist_harness.real_sampling_mas import payload_hash

RECORDS = Path('/data/qwen3-asr/records/therapist_mas_setting_e_v1')
PRESERVED = RECORDS / 'preserved_methods/real_sampling_mas_seed42_590_v1'
INDEX = RECORDS / 'material_selection_offline_20260911_v1/index.json'
RUNNER = REPO / 'examples/run_setting_e_sampling_curriculum_20260911.py'


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    protocol = read(PRESERVED / 'configs/fresh_replication_protocol.template.json')
    for key in ('validation_sha256', 'validation_api_result', 'validation_smoke_root'):
        protocol.pop(key, None)
    protocol.update(version='setting_e_concrete_material_sampling_v1',
        requested_change='Diagnosis, concrete real-recording selection, selected-versus-ordinary scheduling; same SFT objective and update budget.',
        material_curriculum=dict(index_path=str(INDEX), candidate_limit=24),
        only_agent_action='Selected real recording IDs and selected-versus-ordinary sampling probability.',
        mas_sampling='Each exposure chooses selected unique recordings uniformly with replacement, or a continuous ordinary shuffle.',
        comparison_baseline=str(PRESERVED),
        acceptance='At step590, test pooled CER below the preserved MAS endpoint, and the dev-selected checkpoint test pooled CER below the preserved dev-selected MAS. Report other metrics and wall time; no minimum tuned improvement threshold or interim-test checkpoint selection.')
    protocol['input_sha256'][str(INDEX)] = sha256_file(INDEX)
    for kind in ('smoke', 'formal'):
        version = dict(protocol, root=str(root / kind))
        if kind == 'formal':
            version.update(validation_smoke_root=str(root / 'smoke'), validation_api_result=str(root / 'smoke/api_smoke/result.json'))
        path = root / kind / 'protocol.json'
        write_immutable_json(path, version)
        for branch in ('uniform', 'mas'):
            subprocess.run([sys.executable, str(RUNNER), 'prepare', '--protocol', str(path), '--branch', branch], cwd=REPO, check=True)
    subprocess.run([sys.executable, str(RUNNER), 'cpu-smoke', '--protocol', str(root / 'smoke/protocol.json'), '--branch', 'mas'], cwd=REPO, check=True)
    write_immutable_json(root / 'exploration_contract.json', dict(setting='E', version='concrete_material_sampling_v1',
        train_dev_test=[1862, 394, 372], seed=42, interval_steps=29, total_steps=590,
        previous_method=str(PRESERVED), acceptance=protocol['acceptance'],
        fresh_start=True, no_manual_CER_thresholds=True, old_artifacts_modified=False))


def audit_smoke(root):
    import torch
    smoke = root / 'smoke'
    protocol = read(smoke / 'protocol.json')
    grouped = protocol.get('material_curriculum', {}).get('sampling_mode') == 'group_then_material'
    groups = read(Path(protocol['groups_path'])) if grouped else None
    membership = {u: key for key, group in groups.items() for u in group['train_ids']} if grouped else None
    initializations = []
    for branch in (('mas',) if grouped else ('uniform', 'mas')):
        run = smoke / branch
        complete = read(run / 'complete.json')
        assert complete['passed'] and complete['smoke'] and complete['steps'] == 4
        assert complete['actual_audio_exposures'] == 128
        assert complete['base_frozen'] and complete['optimizer_scheduler_continuous']
        checks = read(run / 'smoke_passed.json')
        assert checks['passed'] and len(checks['gradient_checks']) == 2
        assert all(r['post_smoke_restored_exactly'] and r['audio_lora_updated'] and r['base_frozen'] for r in checks['rank_receipts'])
        initializations.append(read(run / 'initialization.json'))
    assert all(item == initializations[0] for item in initializations)
    assert initializations[0] == read(PRESERVED / 'provenance/v9/mas/initialization.json')
    receipts = list((smoke / 'mas').glob('stage-*/**/receipt.json'))
    assert len(receipts) == 10
    for path in receipts:
        item = read(path)
        assert item['validated'] and item['request_model'] == item['response_model'] == 'gpt-5.6-sol'
        assert item['request_sha256'] == sha256_file(path.parent / 'request.json')
        assert item['decision_sha256'] == sha256_file(path.parent / 'decision.json')
    selected_positions = 0
    observed_counts = Counter()
    for n in (1, 2):
        stage = smoke / 'mas' / f'stage-{n:03d}'
        plan, actual = read(stage / 'sampling_plan.json'), read(stage / 'actual_exposure.json')
        assert plan['utterance_ids'] == actual['utterance_ids']
        assert plan['channels'] == actual['channels'] and len(plan['channels']) == 64
        assert actual['sampling_plan_sha256'] == sha256_file(stage / 'sampling_plan.json')
        assert all(u in plan['selected_ids'] for u, c in zip(plan['utterance_ids'], plan['channels']) if c == 'selected')
        selected_positions += sum(c == 'selected' for c in plan['channels'])
        observed_counts.update(actual['utterance_ids'])
        outcome = read(stage / 'reflection/outcome.json')
        assert outcome['feedback_role'] == outcome['evidence']['feedback_role'] == 'dev'
        assert len(outcome['focus_outcomes']) == len(outcome['decision']['diagnosis']['focuses'])
        diagnosis_context = json.loads(read(stage / 'decision/diagnosis/request.json')['input'][-1]['content'])
        diagnosis = read(stage / 'decision/diagnosis/decision.json')
        assert diagnosis['source_evidence_sha256'] == payload_hash(diagnosis_context['evidence'])
        assert diagnosis_context['evidence']['feedback_role'] == 'dev'
        selection_input = json.loads(read(stage / 'decision/selection/request.json')['input'][-1]['content'])['context']
        assert set(selection_input) == {'diagnosis', 'index_sha256', 'training_pool_recordings', 'exposure_sha256', 'exposure_available', 'candidate_limit', 'focuses'}
        assert all('eligible_ids' not in f for f in selection_input['focuses'])
        if grouped:
            decision = read(stage / 'decision/decision.json')
            assert plan['sampler'] == 'group_then_material_with_replacement'
            assert plan['selected_probability_scope'] == 'within_group'
            assert set(actual['channels']) <= {'selected', 'group_pool'}
            assert plan['group_weights'] == decision['group_weights']
            assert set(plan['group_weights']) == set(groups)
            assert outcome['delivery']['group_counts'] == dict(Counter(membership[u] for u in actual['utterance_ids']))
            assert outcome['delivery']['selected_group_counts'] == dict(Counter(membership[u] for u, c in zip(actual['utterance_ids'], actual['channels']) if c == 'selected'))
    checkpoint = smoke / 'mas/stage-002/checkpoint'
    proof = read(checkpoint / 'provenance.json')
    assert proof['training_state_sha256'] == sha256_file(checkpoint / 'training_state.pt')
    state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
    assert state['step'] == state['scheduler']['last_epoch'] == 4
    assert all(int(s['step']) == 4 for s in state['optimizer']['state'].values())
    assert state['sampler']['exposures'] == 128
    assert Counter(state['sampler']['recording_counts']) == observed_counts
    assert not state['pending_stage_ids'] and len(state['rng_per_rank']) == 2
    result = dict(passed=True, actual_model='gpt-5.6-sol', successful_real_api_calls=10,
                  actual_selected_positions=selected_positions, full_setting_e_train_index=True,
                  dev_only_feedback=True, full_train_dev_test=[1862,394,372], optimizer_steps=4,
                  matching_fresh_initialization=True, gradient_equivalence=True,
                  actual_cumulative_exposure_and_optimizer_state_verified=True,
                  sampling_mode='group_then_material' if grouped else 'selected_and_continuous_shuffle',
                  training_branches_checked=['mas'] if grouped else ['uniform','mas'],
                  group_and_material_exposure_verified=True if grouped else None,
                  ordinary_and_selected_exposure_verified=True if not grouped else None)
    write_immutable_json(smoke / 'api_smoke/result.json', result)
    write_immutable_json(root / 'smoke_audit.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def report(root):
    formal = root / 'formal'
    protocol = read(formal / 'protocol.json')
    source = {'new_mas': formal / 'mas',
              'new_uniform': Path(protocol.get('uniform_comparison_path', formal / 'uniform'))}
    if protocol.get('uniform_comparison_path'):
        reuse = read(formal / 'uniform_reuse.json')
        assert reuse['source_path'] == str(source['new_uniform'])
        assert all(sha256_file(Path(p)) == h for p, h in reuse['source_sha256'].items())
    results = {}
    for branch, path in source.items():
        complete = read(path / 'complete.json')
        assert complete['passed'] and complete['steps'] == 590 and complete['actual_audio_exposures'] == 18880
        chosen = read(path / 'selected_checkpoint.json')
        results[branch] = dict(endpoint={'dev': complete['feedback'], 'test': complete['test']},
            dev_selected_step=chosen['step'], dev_selected_test=read(path / f"evaluations/step-{chosen['step']}/test/summary.json"),
            wall_seconds=complete['timing']['wall_seconds'], initialization=read(path / 'initialization.json'),
            source_path=str(path), complete_sha256=sha256_file(path / 'complete.json'))
        if (path.parent / 'recovery_wall_accounting.json').exists():
            accounting = read(path.parent / 'recovery_wall_accounting.json')
            results[branch]['offline_repair_gap_seconds'] = accounting['offline_repair_gap_seconds']
            results[branch]['wall_seconds_including_repair_gap'] = complete['timing']['wall_seconds'] + accounting['offline_repair_gap_seconds']
    prior = read(PRESERVED / 'results/mas/complete.json')
    prior_selected = read(PRESERVED / 'results/mas/evaluations/step-580/test/summary.json')
    assert results['new_mas']['initialization'] == results['new_uniform']['initialization'] == read(PRESERVED / 'provenance/v9/mas/initialization.json')
    improved_endpoint = results['new_mas']['endpoint']['test']['CER_percent'] < prior['test']['CER_percent']
    improved_selected = results['new_mas']['dev_selected_test']['CER_percent'] < prior_selected['CER_percent']
    fresh_control_comparison = dict(
        endpoint_delta_pp=results['new_mas']['endpoint']['test']['CER_percent'] - results['new_uniform']['endpoint']['test']['CER_percent'],
        dev_selected_delta_pp=results['new_mas']['dev_selected_test']['CER_percent'] - results['new_uniform']['dev_selected_test']['CER_percent'],
        endpoint_patient_equal_delta_pp=results['new_mas']['endpoint']['test']['patient_equal_CER_percent'] - results['new_uniform']['endpoint']['test']['patient_equal_CER_percent'],
        interpretation='Negative favors material MAS. Report this fresh control alongside the predeclared retained-MAS acceptance; ordinary replay itself may differ from the historical ordinary run.')
    write_immutable_json(root / 'final_comparison.json', dict(results=results,
        preserved_mas_endpoint=prior['test'], preserved_mas_dev_selected=prior_selected,
        fresh_control_comparison=fresh_control_comparison,
        improves_preserved_endpoint=improved_endpoint, improves_preserved_dev_selected=improved_selected,
        acceptance_passed=improved_endpoint and improved_selected))
    print(json.dumps(dict(improves_preserved_endpoint=improved_endpoint,
        improves_preserved_dev_selected=improved_selected), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'audit-smoke', 'report'))
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    {'prepare': prepare, 'audit-smoke': audit_smoke, 'report': report}[args.mode](args.root)
