import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune/scripts"))
from train_setting_d_m0 import is_sustained_regression  # noqa: E402


def rows(values):
    return [{"metrics": {"overall": {"mean_utterance_cer": value}}} for value in values]


def test_overfitting_requires_three_large_regressions():
    assert is_sustained_regression(rows([.20, .23, .24, .225]))
    assert not is_sustained_regression(rows([.20, .23, .24]))
    assert not is_sustained_regression(rows([.20, .21, .21, .21]))
    assert not is_sustained_regression(rows([.20, .23, .24, .19]))
