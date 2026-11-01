"""Supervise a frozen MAS round, retaining its fixed endpoint (no best-step gate)."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from transformers import TrainerCallback

import qwen3_asr_lora_sft as training
from train_setting_d_m0 import decode

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from therapist_harness.round_training import build_round_training_command  # noqa: E402
from therapist_harness.schema import ArtifactRef  # noqa: E402
from therapist_harness.store import RunStore, sha256_file, write_immutable_json  # noqa: E402


class RoundMonitor(TrainerCallback):
    def __init__(self, root, end_step):
        self.root = root
        self.end_step = end_step
        self.history = []
        self.state = {"status": "initializing", "step": 0, "endpoint_step": end_step}
        (root / "evaluation").mkdir()

    def publish(self):
        (self.root / "monitor.json").write_text(json.dumps({**self.state, "history": self.history}, indent=2) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.state.update(status="training", step=state.global_step, last_log=logs)
        self.publish()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0 or state.global_step == self.end_step:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        self.state.update(status="evaluating_dev", step=step)
        self.publish()
        torch.cuda.empty_cache()
        start = time.monotonic()
        metrics = decode(Path(args.output_dir) / f"checkpoint-{step}", "dev",
                         self.root / f"evaluation/dev_step_{step}", self.root / f"evaluation/dev_step_{step}.log")
        self.history.append({"step": step, "metrics": metrics, "decode_seconds": time.monotonic() - start})
        self.state.update(status="training")
        self.publish()
        print("[round-monitor] " + json.dumps({"step": step, "overall": metrics["overall"]}), flush=True)
        return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    cli = parser.parse_args()
    root = cli.run.resolve()
    store = RunStore(root)
    store.verify_integrity()
    assert store.current().phase == "synthesized"
    protocol = store.protocol()
    command = build_round_training_command(protocol, protocol.initial_adapter,
        ArtifactRef(path=str(root / "train.jsonl"), sha256=sha256_file(root / "train.jsonl"),
                    expected_rows=protocol.real_replay_count_per_round + protocol.synthetic_count_per_round),
        root / "training", sys.executable)
    # Retain all diagnostic checkpoints; only the frozen endpoint continues the loop.
    command[command.index("--save_total_limit") + 1] = "0"
    command[command.index("--save_final_checkpoint") + 1] = "0"
    write_immutable_json(root / "training_execution.json", {
        "command": command, "diagnostic_save_every_steps": 10, "select_best": False,
        "test_evaluation": False, "source_sha256": {str(p): sha256_file(p) for p in
            (Path(__file__), Path(training.__file__), REPO / "finetune/official/qwen3_asr_sft.py")}})
    monitor = RoundMonitor(root, protocol.lora.optimizer_steps_per_round)
    monitor.publish()

    class SupervisedTrainer(training.QwenASRLoRATrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(monitor)

    training.QwenASRLoRATrainer = SupervisedTrainer
    sys.argv = command[1:]
    try:
        training.main()
        assert monitor.history[-1]["step"] == protocol.lora.optimizer_steps_per_round
        monitor.state.update(status="training_and_dev_complete")
        monitor.publish()
    except Exception as error:
        monitor.state.update(status="failed", error=f"{type(error).__name__}: {error}")
        monitor.publish()
        raise


if __name__ == "__main__":
    main()
