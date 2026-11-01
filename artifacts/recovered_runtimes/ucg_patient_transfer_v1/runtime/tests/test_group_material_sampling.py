from collections import Counter
import copy

import pytest

from therapist_harness.material_sampling_mas import MaterialExposureSampler
from therapist_harness.sampling_curriculum_runtime import ExposureSampler


def test_no_extra_selection_exactly_reproduces_original_group_sampler():
    groups = {'easy': ['a', 'b', 'c'], 'hard': ['d', 'e']}
    original = ExposureSampler(list('abcde'), groups, seed=42)
    material = MaterialExposureSampler(list('abcde'), groups, seed=42)
    for n, weights in [(53, {'easy': .3, 'hard': .7}), (928, {'easy': .6, 'hard': .4})]:
        observed, channels = material.draw_group_materials(n, ['a', 'd'], 0, weights)
        assert observed == original.draw(n, weights)
        assert set(channels) == {'group_pool'}
    assert material.exposures == sum(material.recording_counts.values()) == 981


def test_group_without_selected_material_keeps_its_budget():
    sampler = MaterialExposureSampler(list('abcd'), {'easy': ['a', 'b'], 'hard': ['c', 'd']})
    observed, channels = sampler.draw_group_materials(100, ['a'], 1, {'easy': 0, 'hard': 1})
    assert set(observed) == {'c', 'd'}
    assert set(channels) == {'group_pool'}


def test_selected_material_stays_in_group_and_state_restores_exactly():
    groups = {'easy': ['a', 'b', 'c'], 'hard': ['d', 'e', 'f']}
    sampler = MaterialExposureSampler(list('abcdef'), groups)
    first, channels = sampler.draw_group_materials(928, ['a', 'd'], .4, {'easy': .3, 'hard': .7})
    assert all(u in {'a', 'd'} for u, c in zip(first, channels) if c == 'selected')
    saved = copy.deepcopy(sampler.state_dict())
    expected = sampler.draw_group_materials(928, ['b', 'e'], .2, {'easy': .6, 'hard': .4})
    restored = MaterialExposureSampler(list('abcdef'), groups)
    restored.load_state_dict(saved)
    assert restored.draw_group_materials(928, ['b', 'e'], .2, {'easy': .6, 'hard': .4}) == expected
    assert restored.recording_counts == Counter(first + expected[0])
    assert restored.state_dict() == sampler.state_dict()


@pytest.mark.parametrize('selection,probability,weights', [
    (['a', 'a'], .1, {'easy': .5, 'hard': .5}),
    (['dev-recording'], .1, {'easy': .5, 'hard': .5}),
    ([], .1, {'easy': .5, 'hard': .5}),
    (['a'], .1, {'easy': .5, 'hard': .4}),
    (['a'], float('nan'), {'easy': .5, 'hard': .5}),
    (['a'], .1, {'easy': 1}),
])
def test_invalid_material_or_group_budget_is_rejected(selection, probability, weights):
    sampler = MaterialExposureSampler(['a', 'b'], {'easy': ['a'], 'hard': ['b']})
    with pytest.raises(ValueError):
        sampler.draw_group_materials(32, selection, probability, weights)


def test_partition_overlap_is_not_silently_double_counted():
    sampler = MaterialExposureSampler(['a', 'b'], {'easy': ['a', 'b'], 'hard': ['b']})
    with pytest.raises(ValueError, match='partition'):
        sampler.draw_group_materials(32, ['a'], .1, {'easy': .5, 'hard': .5})
