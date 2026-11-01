import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('comparison', Path(__file__).parents[1] / 'examples/voxcpm2_sft_comparison.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_mcd_identity():
    x = np.random.default_rng(42).normal(size=(8, 24))
    assert comparison.mcd_dtw(x, x) == 0


def test_mcd_known_one_frame_distance():
    a, b = np.zeros((1, 24)), np.ones((1, 24))
    expected = 10 * math.sqrt(2) / math.log(10) * math.sqrt(24)
    assert comparison.mcd_dtw(a, b) == pytest.approx(expected)
    assert comparison.mcd_dtw(b, a) == pytest.approx(expected)


def test_dtw_allows_duration_stretch_but_duration_metric_detects_it():
    x = np.random.default_rng(42).normal(size=(8, 24))
    assert comparison.mcd_dtw(x, np.repeat(x, 2, axis=0)) == 0
    assert comparison.duration_error(16, 8) == 100
    assert comparison.duration_error(8, 8) == 0


def test_fixed_panel_has_real_and_both_model_families():
    assert comparison.VARIANTS == ('real', 'cosy_sg', 'vox_0', 'vox_132', 'vox_330', 'vox_396')


def test_mel_cepstrum_excluding_c0_is_gain_invariant():
    import pysptk
    spectrum = np.exp(np.random.default_rng(42).normal(size=(8, 513)))
    a = pysptk.sp2mc(spectrum, order=24, alpha=.42)[:, 1:]
    b = pysptk.sp2mc(spectrum * 4, order=24, alpha=.42)[:, 1:]
    assert comparison.mcd_dtw(a, b) < 1e-10
