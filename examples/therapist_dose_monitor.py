"""Fixed-training-row fit measurements and decoded real-speech dose observations."""

import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from train_m0_small_dose_pair import Monitor
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.evidence_pack import load_prediction_rows
from therapist_harness.store import write_immutable_json


class DoseMonitor(Monitor):
    def __init__(self, root, parent_predictions):
        super().__init__(root)
        self.end_step = 64
        self.state["endpoint_step"] = 64
        self.fit_history = []
        self.indices = json.loads((root / "training_index.json").read_text())
        parent = evaluate_snapshot(load_prediction_rows(parent_predictions),
            prediction_column="prediction", critical_cer_threshold=.5)
        self.parent_cer = parent.overall.mean_utterance_cer

    def measure_fit(self, model, step):
        """Use the training loss implementation, without dropout or RNG side effects."""
        trainer = self.trainer
        was_training = model.training
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        result = {"step": step, "groups": {}}
        self.state.update(status="measuring_fixed_train_fit", step=step)
        self.publish()
        try:
            with torch.random.fork_rng(), torch.no_grad():
                model.eval()
                for source in sorted({r["source"] for r in self.indices}):
                    rows = [r["row"] for r in self.indices if r["source"] == source]
                    loader = DataLoader(Subset(trainer.train_dataset, rows), batch_size=8,
                        collate_fn=trainer.data_collator, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(0))
                    total, count = 0., 0
                    for batch in loader:
                        batch = trainer._prepare_inputs(batch)
                        size = batch["labels"].shape[0]
                        with trainer.compute_loss_context_manager():
                            loss = trainer.compute_loss(model, batch)
                        value = float(loss.detach())
                        if not np.isfinite(value):
                            raise ValueError("nonfinite fixed-sample training loss")
                        total += value * size
                        count += size
                    assert count == len(rows)
                    result["groups"][source] = {"rows": count, "mean_utterance_loss": total/count}
        finally:
            model.train(was_training)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        self.fit_history.append(result)
        write_immutable_json(self.root / f"fit/step_{step}.json", result)
        print("[fixed-fit] " + json.dumps(result), flush=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.measure_fit(model, 0)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in (8, 16, 32, 64):
            control.should_save = True
        return control

    def on_save(self, args, state, control, model=None, **kwargs):
        self.measure_fit(model, state.global_step)
        super().on_save(args, state, control, model=model, **kwargs)
        if len(self.history) >= 2:
            recent = self.history[-2:]
            regressed = all(r["metrics"]["overall"]["mean_utterance_cer"] >= self.parent_cer + .05 for r in recent)
            def mean_loss(record):
                groups = record["groups"].values()
                return sum(g["rows"] * g["mean_utterance_loss"] for g in groups) / 256
            fitting = mean_loss(self.fit_history[-1]) < mean_loss(self.fit_history[0])
            if regressed and fitting:
                receipt = {"status": "stopped_severe_regression", "step": state.global_step,
                    "parent_cer": self.parent_cer, "history": self.history, "fit_history": self.fit_history,
                    "policy": "two successive overall CER >= parent + .05 while fixed train loss falls"}
                write_immutable_json(self.root / "stopped.json", receipt)
                self.state.update(status="stopped_severe_regression")
                self.publish()
                raise RuntimeError("Severe real-speech regression: stopped without endpoint substitution")
        return control
