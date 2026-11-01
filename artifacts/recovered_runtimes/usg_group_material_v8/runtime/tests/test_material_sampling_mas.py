import copy
import json

import pytest

from therapist_harness.material_selection_offline import (
    DiagnosticBrief, MaterialSelection, PracticeFocus, build_index,
)
from therapist_harness.material_sampling_mas import (
    MaterialExposureSampler, MaterialSchedule, GroupMaterialSchedule, choose_material_sampling,
    measure_focus, validate_diagnosis, validate_schedule,
)
from therapist_harness.real_sampling_mas import payload_hash
from therapist_harness.sampling_curriculum_runtime import ExposureSampler
from therapist_harness.schema import AgentConfig


def test_zero_selected_is_exact_continuous_uniform_control():
    base = ExposureSampler(['a', 'b', 'c'], {'all': ['a', 'b', 'c']})
    sampler = MaterialExposureSampler(['a', 'b', 'c'], {'all': ['a', 'b', 'c']})
    for n in (2, 6, 3):
        ids, channels = sampler.draw_materials(n, [], 0)
        assert ids == base.draw(n)
        assert channels == ['ordinary'] * n
    assert sampler.exposures == sum(sampler.recording_counts.values()) == 11


def test_mixture_resume_keeps_channel_identity_exposure_and_ordinary_sweeps():
    ids = ['a', 'b', 'c', 'd']
    sampler = MaterialExposureSampler(ids, {'all': ids})
    first, channels = sampler.draw_materials(100, ['a', 'b'], .4)
    assert all(u in {'a', 'b'} for u, c in zip(first, channels) if c == 'selected')
    ordinary = [u for u, c in zip(first, channels) if c == 'ordinary']
    for start in range(0, len(ordinary)-3, 4):
        assert set(ordinary[start:start+4]) == set(ids)
    state = copy.deepcopy(sampler.state_dict())
    expected = sampler.draw_materials(128, ['b', 'd'], .2)
    restored = MaterialExposureSampler(ids, {'all': ids})
    restored.load_state_dict(state)
    assert restored.draw_materials(128, ['b', 'd'], .2) == expected
    assert restored.state_dict() == sampler.state_dict()
    assert sum(restored.recording_counts.values()) == restored.exposures == 228
    for selected, probability in [(['a', 'a'], .3), (['heldout'], .3), ([], .1), ([], float('nan'))]:
        with pytest.raises(ValueError):
            restored.draw_materials(1, selected, probability)


def test_empty_material_selection_has_no_implicit_training_replacement():
    context = dict(round_index=1, selection_summary={'unique_recordings': 0})
    output = MaterialSchedule(round_index=1, context_sha256=payload_hash(context),
        selected_probability=0, reason='无匹配材料', expected_change='观察普通训练', history_used='无')
    validate_schedule(output, context)
    with pytest.raises(ValueError, match='empty selection'):
        validate_schedule(output.model_copy(update={'selected_probability': .1}), context)
    with pytest.raises(ValueError, match='context mismatch'):
        validate_schedule(output, dict(context, round_index=2))


def test_targeted_full_syllable_tracking_deduplicates_component_errors():
    train = [dict(utt_id='train', audio='/train.wav', speaker_id='train-p',
                  clean_gt='木', duration=2, split='train')]
    index = build_index(train, [], {})
    focus = PracticeFocus(focus_id='one', purpose='target', level='syllable', target_unit='muk6',
                          reason='音节识别', evidence_ids=['step0:phonology:1'])
    row = dict(clean_gt='木', prediction='北', speaker_id='dev-p', errors=1, chars=1, substitutions=1)
    result = measure_focus([row], focus, index)
    assert result['target_occurrences'] == 1
    assert result['cohort']['CER_percent'] == 100
    assert sum(result['component_differences'].values()) == 1
    before = measure_focus([dict(row, prediction='木', errors=0, substitutions=0)], focus, index)
    assert before['component_differences'] == {}
    assert before['target_occurrences'] == result['target_occurrences']


def test_three_real_roles_have_distinct_inputs_and_selector_never_receives_dev_transcript(monkeypatch, tmp_path):
    import therapist_harness.material_sampling_mas as module
    train = [dict(utt_id='train-one', audio='/train.wav', speaker_id='train-p',
                  clean_gt='木', duration=2, split='train', zero_shot_bucket='medium')]
    index = build_index(train, [], {})
    evidence = dict(feedback_role='dev', round_index=1, checkpoint_step=0,
        observation_ids=['step0:overall'], phonology={},
        feedback={'transcription': 'ONLY_DEV_TRANSCRIPT_SENTINEL'})
    calls = []

    def fake_call(config, *, phase, context, directory, validate, request, output_type, role):
        calls.append((phase, role, request))
        if phase == 'diagnose':
            output = DiagnosticBrief(producer='diagnosis_agent', round_index=1,
                source_evidence_sha256=context['evidence_sha256'], summary='检查音节',
                focuses=[dict(focus_id='one', purpose='target', level='syllable', target_unit='muk6',
                              reason='检查目标音节', evidence_ids=['step0:overall'])])
        elif phase == 'select':
            supplied = json.loads(request['input'][-1]['content'])
            output = MaterialSelection(context_sha256=supplied['context_sha256'],
                choices=[dict(focus_id='one', recording_ids=['train-one'], reason='目标音节匹配')])
        else:
            output = GroupMaterialSchedule(round_index=1, context_sha256=context['context_sha256'],
                allocations=[dict(group_id='medium_short', probability=1, reason='该训练组有匹配材料')],
                selected_probability=.2, reason='兼顾全池', expected_change='目标音节改善', history_used='首轮')
        validate(output)
        return output, {}, 1.

    monkeypatch.setattr(module, '_call', fake_call)
    config = AgentConfig(model='gpt-5.6-sol', base_url='https://api.laicaishare.xyz', provider='OpenAI',
        reasoning_effort='xhigh', store=False, max_retries=0, timeout_seconds=600,
        max_output_tokens=8192, api_key_env='OPENAI_API_KEY')
    result = choose_material_sampling(config,
        evidence, index=index, exposures={}, stage_exposures=928,
        groups={'medium_short': {'train_ids': ['train-one']}}, directory=tmp_path)
    assert [c[0] for c in calls] == ['diagnose', 'select', 'schedule']
    assert 'ONLY_DEV_TRANSCRIPT_SENTINEL' in json.dumps(calls[0][2])
    assert 'ONLY_DEV_TRANSCRIPT_SENTINEL' not in json.dumps(calls[1][2])
    assert result['selected_ids'] == ['train-one'] and result['selected_probability'] == .2
    assert result['decision']['selection_summary']['unique_recordings'] == 1
    brief = DiagnosticBrief.model_validate(result['decision']['diagnosis'])
    with pytest.raises(ValueError, match='current-round'):
        validate_diagnosis(brief.model_copy(update={'producer': 'offline_fixture'}), evidence)
