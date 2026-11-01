import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from train_real_mas576 import AugmentationMonitor, Monitor, command, training  # noqa: E402


def test_fresh_initialization_and_baseline_hyperparameters(tmp_path):
    args = training.parse_args(command(tmp_path)[2:])
    assert args.lora_init_adapter_path == ""
    assert args.resume == 0 and args.resume_from == ""
    assert (args.batch_size, args.grad_acc, args.lr) == (8, 4, 2e-4)
    assert (args.lora_rank, args.lora_alpha, args.lora_dropout) == (16, 4, .05)
    assert (args.lr_scheduler_type, args.warmup_ratio, args.max_steps) == ("linear", .02, 140)


def test_retains_step50_comparison_before_overfit_stop(tmp_path, monkeypatch):
    monitor = AugmentationMonitor(tmp_path, {})
    monkeypatch.setattr(Monitor, "on_save", lambda *args, **kwargs: None)
    monitor.history = [{"step": s, "metrics": {"overall": {"mean_utterance_cer": cer}}}
        for s, cer in [(10, .20), (20, .24), (30, .24), (40, .24)]]
    control = SimpleNamespace(should_training_stop=False)
    monitor.on_save(None, SimpleNamespace(global_step=40), control)
    assert not control.should_training_stop
    monitor.history.append({"step": 50, "metrics": {"overall": {"mean_utterance_cer": .24}}})
    monitor.on_save(None, SimpleNamespace(global_step=50), control)
    assert control.should_training_stop
    assert (tmp_path / "step50_comparison.json").exists()


def test_no_stop_for_small_cer_fluctuations(tmp_path, monkeypatch):
    monitor = AugmentationMonitor(tmp_path, {})
    monkeypatch.setattr(Monitor, "on_save", lambda *args, **kwargs: None)
    monitor.history = [{"step": s, "metrics": {"overall": {"mean_utterance_cer": cer}}}
        for s, cer in [(30, .20), (40, .205), (50, .21)]]
    control = SimpleNamespace(should_training_stop=False)
    monitor.on_save(None, SimpleNamespace(global_step=50), control)
    assert not control.should_training_stop
