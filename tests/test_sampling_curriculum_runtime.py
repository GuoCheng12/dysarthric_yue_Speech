import copy

import pytest

from therapist_harness.sampling_curriculum_runtime import ExposureSampler


def make_sampler():
    return ExposureSampler(['a', 'b', 'c', 'd'], {'first': ['a', 'b'], 'last': ['c', 'd']})


def test_shuffle_preserves_every_record_once_per_sweep_across_stage_boundaries():
    sampler = make_sampler()
    ids = sampler.draw(3) + sampler.draw(5) + sampler.draw(4)
    assert all(set(ids[start:start+4]) == set('abcd') for start in range(0, 12, 4))
    assert sampler.exposures == 12


def test_weighted_sampler_changes_exposure_and_resume_is_exact():
    sampler = make_sampler()
    weights = {'first': 1.0, 'last': 0.0}
    first = sampler.draw(11, weights)
    assert set(first) <= set('ab') and len(first) == 11
    state = copy.deepcopy(sampler.state_dict())
    expected = sampler.draw(30, {'first': .2, 'last': .8})
    restored = make_sampler()
    restored.load_state_dict(state)
    assert restored.draw(30, {'first': .2, 'last': .8}) == expected
    assert restored.exposures == 41


def test_bad_plan_and_foreign_audio_are_rejected():
    with pytest.raises(AssertionError):
        ExposureSampler(['a'], {'test_recording': ['b']})
    for bad in ({'first': 1}, {'first': .2, 'last': .7}, {'first': float('nan'), 'last': 0}):
        with pytest.raises(AssertionError):
            make_sampler().draw(8, bad)


def test_stage_boundary_does_not_change_unweighted_draws():
    whole, staged = make_sampler(), make_sampler()
    assert whole.draw(128) == staged.draw(32) + staged.draw(32) + staged.draw(64)
