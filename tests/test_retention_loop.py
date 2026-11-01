"""Explicit interface fixtures for mixed-objective bookkeeping, not ASR results."""

from collections import Counter
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from therapist_harness import agent
from therapist_harness.checkpoint_promotion import accepted_hashes
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.materialize import build_materialization_sidecar
from therapist_harness.memory import compile_intervention_summary, validate_summary_digest
from therapist_harness.outcome import compile_evaluation_plan, compile_round_result, compile_reflection_pack
from therapist_harness.preference_allocation_agent import ExposureDecision, compile_exposure_allocation
from therapist_harness.preference_contract import PreferenceLoopConfig, PreferenceDelivery, PreferenceSampleDelivery
from therapist_harness.retention_contract import RealKLRetention
from therapist_harness.retention_loop import training_pairs, adapt_scope
from therapist_harness.round_schema import SynthesisReceipt, SynthesizedSample, TrainingReceipt
from therapist_harness.schema import RunProtocol, canonical_sha256
from therapist_harness.store import sha256_file

from test_therapist_outcome import _protocol, _proposal, _assignments, _prediction_rows, SHA_A, SHA_B, SHA_C
from test_therapist_reflection import _reflection, SKILL_TEXT, SYSTEM_PROMPT


def config():
    return PreferenceLoopConfig(comparison='synthetic_only', execution_version='persistent_kl_dpo_v3',
        real_per_step=0, synthetic_per_step=1, synthetic_weight=1., review_mode='asr_transcript',
        review_asr_adapter='fixture-only', review_asr_adapter_sha256=SHA_A, review_asr_base_model_sha256=SHA_B,
        audio_review_model='fixture-not-called', audio_review_base_url='https://example.invalid',
        include_greedy=True, generation_design_path='fixture-not-read', generation_design_sha256=SHA_A,
        retention_design_path='fixture-not-read', retention_design_sha256=SHA_A,
        training_allocation='mas_audio_budget_v1', checkpoint_acceptance='real_dev_initial_parent_v2')


def protocol_fixture(tmp_path):
    protocol = _protocol(tmp_path)
    initial = tmp_path / 'initial_complete.jsonl'
    initial.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in _prediction_rows(after=False)))
    references = json.loads(Path(protocol.frozen_reference_manifest.path).read_text())['references']
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps(dict(cards=[dict(row, reference_id=f'voice{i}.ref0', voice_group=f'voice{i}',
        seconds=1., origin='original', acoustic_traits='counterfactual_fixture') for i, row in enumerate(references)])))
    design = tmp_path / 'design.json'
    design.write_text(json.dumps(dict(catalog_path=str(catalog), catalog_sha256=sha256_file(catalog))))
    options = config().model_copy(update=dict(generation_design_path=str(design), generation_design_sha256=sha256_file(design)))
    return RunProtocol.model_validate(dict(protocol.model_dump(mode='python'), real_replay_count_per_round=0,
        initial_adapter_dev_predictions=dict(path=str(initial), sha256=sha256_file(initial), expected_rows=6), preference_loop=options))


def test_actual_audio_counts_do_not_multiply_with_distinct_rejections():
    pairs = [dict(audio_id='a'), dict(audio_id='a'), dict(audio_id='b')]
    decision = ExposureDecision(hypothesis='fixture', expected_observation='fixture',
        assignments=[dict(sample_id='a', exposures=3, reason='fixture'), dict(sample_id='b', exposures=1, reason='fixture')])
    out = compile_exposure_allocation(decision, pairs, {'a': 'target', 'b': 'maintenance'}, steps=2, per_step=2, seed=42)
    assert Counter(key for step in out['steps'] for key in step) == {'a': 3, 'b': 1}
    assert out['role_exposures'] == {'target': 3, 'maintenance': 1}
    wrong = ExposureDecision.model_validate(dict(decision.model_dump(), assignments=[
        dict(sample_id='a', exposures=3, reason='fixture'), dict(sample_id='all-correct', exposures=1, reason='fixture')]))
    with pytest.raises(ValueError, match='pairless'):
        compile_exposure_allocation(wrong, pairs, {'a': 'target'}, steps=2, per_step=2, seed=42)
    with pytest.raises(ValueError, match='no actual preference'):
        compile_exposure_allocation(decision, [], {}, steps=2, per_step=2, seed=42)


def test_label_transform_keeps_original_target_and_actual_rejection():
    pair = dict(audio_id='a', chosen='麻煩你關門。', rejected='麻煩你開門', metadata={'source': 'fixture'})
    transformed = training_pairs([pair])[0]
    assert transformed['chosen'] == '麻煩你關門'
    assert transformed['rejected'] == pair['rejected']
    assert transformed['metadata']['original_target_text'] == pair['chosen']
    assert pair['chosen'].endswith('。')


def test_real_KL_stays_separate_from_real_preference_pairs_through_reflection_and_memory(tmp_path):
    protocol = protocol_fixture(tmp_path)
    evidence, proposal, target_id = _proposal(protocol)
    evidence = evidence.model_copy(update={'artifact_mode': 'counterfactual_interface_smoke'})
    proposal = proposal.model_copy(update={'evidence_sha256': canonical_sha256(evidence)})
    plan = compile_evaluation_plan(protocol, evidence, proposal, before_prediction_column='before', after_prediction_column='after')
    assignments = _assignments(protocol, proposal, evidence)
    cards = json.loads((tmp_path / 'catalog.json').read_text())['cards']
    plan_path = tmp_path / 'generation_plan.json'
    plan_path.write_text(json.dumps(dict(choices=[dict(sample_id=x['sample_id'], mode='zero_shot', instruction_id='cantonese_en',
        reference_id=next(c['reference_id'] for c in cards if c['reference_wav_sha256'] == x['reference_wav_sha256']),
        reference_wav_sha256=x['reference_wav_sha256'], target_text_sha256=hashlib.sha256(x['target_text'].encode()).hexdigest()) for x in assignments])))
    sidecar = build_materialization_sidecar(proposal, evidence, protocol, assignments, generation_plan_path=plan_path)
    synthesis = SynthesisReceipt(schema_version='1', run_id=protocol.run_id, round_index=1,
        artifact_mode='counterfactual_interface_smoke', status='complete', assignments_canonical_sha256=sidecar.assignments_canonical_sha256,
        requested=2, generated=2, failed=0, samples=[SynthesizedSample(sample_id=x.sample_id,
            audio_file_sha256=SHA_B, duration_seconds=1.) for x in sidecar.samples])
    retention = RealKLRetention(teacher_asr_sha256=protocol.initial_adapter.sha256,
        manifest_file_sha256=SHA_A, loss_factors_file_sha256=SHA_B, teacher_cache_metadata_sha256=SHA_C,
        audio_count=6, unique_audio_exposed=3, training_exposures=4, total_loss_weight=20.)
    delivery = PreferenceDelivery(current_asr_sha256=protocol.initial_adapter.sha256,
        frozen_reference_asr_sha256=protocol.initial_adapter.sha256, reviews_sha256=SHA_A,
        candidates_sha256=SHA_B, pair_manifest_sha256=SHA_C, exposure_log_sha256=SHA_A, samples_per_audio=4,
        real_audio_considered=0, real_pair_audio_count=0, real_pair_count=0, real_training_exposures=0,
        synthetic_samples=[PreferenceSampleDelivery(sample_id=x.sample_id, review_verdict='pass',
            pair_count=1 if i == 0 else 0, training_exposures=2 if i == 0 else 0) for i, x in enumerate(sidecar.samples)],
        preference_token_mask='edit_eos', chosen_text_transform='strip_terminal_sentence_punctuation_v2', real_kl_retention=retention)
    training = TrainingReceipt(schema_version='1', run_id=protocol.run_id, round_index=1,
        artifact_mode='counterfactual_interface_smoke', status='complete', parent_checkpoint_directory_sha256=protocol.initial_adapter.sha256,
        new_checkpoint_directory_sha256=SHA_C, train_manifest_file_sha256=SHA_C, training_index_file_sha256=SHA_B,
        trainer_state_file_sha256=SHA_C, real_rows=0, synthetic_rows=1, total_rows=1,
        expected_optimizer_steps=2, actual_optimizer_steps=2, preference_delivery=delivery)
    before = _prediction_rows(after=False); after = _prediction_rows(after=True)
    next(x for x in after if x['utt_id'] == 'u2')['after'] = '阿聽日錯錯'
    result = compile_round_result(protocol, evidence, proposal, plan, sidecar, synthesis, training,
        before_rows=before, after_rows=after, before_predictions_file_sha256=protocol.initial_adapter_dev_predictions.sha256,
        after_predictions_file_sha256=SHA_B)
    pack = compile_reflection_pack(protocol, evidence, proposal, plan, sidecar, result,
        prescription_prompt_file_sha256=SHA_A, reflection_prompt_file_sha256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        skill_version='v000', skill_file_sha256=hashlib.sha256(SKILL_TEXT.encode()).hexdigest())
    item = SimpleNamespace(protocol=protocol, evidence=evidence, proposal=proposal, target_id=target_id,
        plan=plan, sidecar=sidecar, synthesis=synthesis, training=training, result=result, pack=pack)
    summary = compile_intervention_summary(protocol, evidence, proposal, pack, _reflection(item))
    validate_summary_digest(summary)
    retained = summary.episode_memory.preference_delivery
    assert retained.real_pair_count == retained.real_training_exposures == 0
    assert retained.real_kl_retention == retention
    assert retained.preference_token_mask == 'edit_eos'
    assert summary.accepted_checkpoint_sha256 == protocol.initial_adapter.sha256


def test_initial_anchor_can_reject_an_apparent_current_parent_improvement(tmp_path):
    protocol = protocol_fixture(tmp_path)
    metrics = evaluate_snapshot(_prediction_rows(after=False), prediction_column='before', critical_cer_threshold=.5)
    pack = SimpleNamespace(provenance=SimpleNamespace(new_checkpoint_directory_sha256=SHA_C,
        after_predictions_file_sha256=SHA_C, parent_checkpoint_directory_sha256=SHA_A, before_predictions_file_sha256=SHA_B),
        response=SimpleNamespace(after_metrics=metrics,
            paired_delta=SimpleNamespace(overall_pooled_cer=-.01, patient_macro_cer=0., hard_pooled_cer=0.)))
    assert accepted_hashes(pack) == (SHA_C, SHA_C)
    assert accepted_hashes(pack, protocol=protocol) == (SHA_A, SHA_B)


def test_mas_stops_reading_after_a_real_terminal_event(monkeypatch):
    response = SimpleNamespace(status='completed')
    class Stream:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self):
            yield SimpleNamespace(type='response.completed', response=response)
            raise AssertionError('terminal response was already received; do not wait for a lingering stream')
    monkeypatch.setenv('TEST_RETENTION_KEY', 'fixture-only')
    monkeypatch.setattr(agent, 'OpenAI', lambda **kw: SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: Stream())))
    protocol = SimpleNamespace(agent=SimpleNamespace(api_key_env='TEST_RETENTION_KEY', base_url='https://example.invalid',
        max_retries=0, timeout_seconds=30))
    assert agent.execute_therapist_request(protocol, {'stream': True}) is response


def test_final_phase_does_not_mix_commentary_into_schema_json():
    response = SimpleNamespace(output_text='commentary{"ok":true}', output=[
        SimpleNamespace(type='message', phase='commentary', content=[SimpleNamespace(type='output_text', text='commentary')]),
        SimpleNamespace(type='message', phase='final_answer', content=[SimpleNamespace(type='output_text', text='{"ok":true}')])])
    assert json.loads(agent.terminal_output_text(response)) == {'ok': True}
    assert 'auxiliary' in adapt_scope('Optimize ordinary mean synthetic-audio DPO loss with weight 1.')
