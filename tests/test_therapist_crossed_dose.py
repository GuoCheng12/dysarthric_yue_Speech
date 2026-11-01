"""Crossed allocation and repeat-pass contracts, including legacy serialization."""

from types import SimpleNamespace

import pytest

from therapist_harness.materialize import _allocate_speakers
from therapist_harness.schema import RunProtocol, SpeakerProfileRule
from test_therapist_outcome import _protocol


def test_historical_profile_serialization_unchanged():
    payload = {"difficulty_groups": ["easy", "medium", "hard"]}
    assert SpeakerProfileRule.model_validate(payload).model_dump(mode="json") == payload


def test_repeat_pass_budget(tmp_path):
    payload = _protocol(tmp_path).model_dump(mode="json")
    payload["lora"]["training_passes_per_round"] = 8
    payload["lora"]["optimizer_steps_per_round"] *= 8
    protocol = RunProtocol.model_validate(payload)
    assert protocol.lora.optimizer_steps_per_round == 16
    payload["lora"]["optimizer_steps_per_round"] = 2
    with pytest.raises(ValueError, match="declared passes"):
        RunProtocol.model_validate(payload)


def test_crossed_patients_same_for_all_six_texts_and_rotate():
    profile = SpeakerProfileRule(difficulty_groups=["easy", "medium", "hard"], allocation="crossed_16")
    protocol = SimpleNamespace(speaker_profiles={"all": profile}, lora=SimpleNamespace(seed=42))
    profiles = {"all": {group: tuple(f"{group}-{i:02}" for i in range(19))
                        for group in profile.difficulty_groups}}
    stimuli = [SimpleNamespace(stimulus_id=f"s{i}", speaker_profile_id="all", copies=16,
                               role="target" if i < 3 else "maintenance") for i in range(6)]
    selections = []
    for number in (1, 2, 3):
        proposal = SimpleNamespace(stimuli=stimuli, round_index=number)
        allocated = _allocate_speakers(proposal, protocol, profiles)
        selected = allocated["s0"]
        assert len(set(selected)) == 16
        assert all(value == selected for value in allocated.values())
        assert sorted(sum(s.startswith(g) for s in selected) for g in profile.difficulty_groups) == [5, 5, 6]
        assert allocated == _allocate_speakers(proposal, protocol, profiles)
        selections.append(set(selected))
    assert len({tuple(sorted(s)) for s in selections}) == 3


def test_fixed_fit_uses_eval_and_preserves_rng(tmp_path):
    import json
    import random
    import sys
    from pathlib import Path

    import numpy as np
    import torch

    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo / "examples"), str(repo / "finetune/scripts")]
    from therapist_dose_monitor import DoseMonitor

    model = torch.nn.Sequential(torch.nn.Dropout(.8))
    model.train()
    monitor = object.__new__(DoseMonitor)
    monitor.root, monitor.state, monitor.history, monitor.fit_history = tmp_path, {}, [], []
    monitor.indices = [{"row": i, "source": "real" if i < 2 else "synthetic"} for i in range(4)]
    def collate(rows):
        random.random()
        np.random.rand()
        return {"labels": torch.tensor(rows).reshape(-1, 1).float()}
    def loss(model, batch):
        assert not model.training
        return model(batch["labels"]).mean()
    from contextlib import nullcontext
    monitor.trainer = SimpleNamespace(train_dataset=[1, 3, 4, 8], data_collator=collate,
        _prepare_inputs=lambda x: x, compute_loss_context_manager=nullcontext, compute_loss=loss)
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    monitor.measure_fit(model, 0)
    monitor.measure_fit(model, 8)
    assert model.training
    assert random.getstate() == rng[0]
    assert np.array_equal(np.random.get_state()[1], rng[1][1])
    assert torch.equal(torch.get_rng_state(), rng[2])
    result = json.loads((tmp_path / "fit/step_0.json").read_text())
    assert result["groups"]["real"]["mean_utterance_loss"] == 2
    assert result["groups"]["synthetic"]["mean_utterance_loss"] == 6
    assert monitor.fit_history[0]["groups"] == monitor.fit_history[1]["groups"]
