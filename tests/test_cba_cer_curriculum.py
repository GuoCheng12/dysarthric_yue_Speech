import importlib.util
from pathlib import Path
import random
import pytest

path = Path(__file__).resolve().parents[1] / 'examples/run_cba_cer_curriculum_20260914.py'
spec = importlib.util.spec_from_file_location('cba_curriculum', path)
cba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cba)


def predictions(values):
    return [dict(utt_id=uid, errors=errors, chars=100) for uid, errors in values.items()]


def test_filter_boundary_and_improved_or_regressed_examples():
    before = predictions({'persistent':80, 'improved':80, 'regressed':20, 'boundary':50, 'insertions':160})
    after = predictions({'persistent':70, 'improved':30, 'regressed':80, 'boundary':90, 'insertions':110})
    ids = [p['utt_id'] for p in before]
    assert cba.select_recordings(ids, before, after) == ['improved', 'regressed', 'boundary']
    # Previously excluded samples are eligible again when the current model improves.
    next_scores = predictions({'persistent':40, 'improved':30, 'regressed':80, 'boundary':90, 'insertions':40})
    assert cba.select_recordings(ids, before, next_scores) == ids


def test_selection_rejects_foreign_or_missing_ids():
    with pytest.raises(AssertionError):
        cba.select_recordings(['train'], predictions({'train':60}), predictions({'dev':20}))
    with pytest.raises(AssertionError):
        cba.select_recordings(['train'], [], predictions({'train':20}))


def test_sampling_obeys_selected_pool_and_exact_budget():
    selected = ['a','b','c','d','e']
    values = cba.draw_selected(selected, 29*32, random.Random(42))
    assert len(values) == 29*32 and set(values) == set(selected)
    assert all(set(values[start:start+5]) == set(selected) for start in range(0,925,5))
    assert values == cba.draw_selected(selected,29*32,random.Random(42))
