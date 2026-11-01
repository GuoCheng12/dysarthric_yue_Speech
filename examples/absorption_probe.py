"""Fixed-content M0 absorption factorial and three-round synthetic-only trajectory."""

# ruff: noqa: E402 -- standalone runner adds the repository import roots.

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "examples"), str(REPO / "finetune/scripts")]

import qwen3_asr_lora_sft as training
from torch.utils.data import SequentialSampler
from therapist_dose_monitor import DoseMonitor
from train_m0_small_dose_pair import Monitor, stratified_sample, stats
from therapist_harness.evidence import evaluate_snapshot
from therapist_harness.phonology import align_characters
from therapist_harness.store import sha256_directory, sha256_file, write_immutable_json, write_immutable_bytes
from therapist_harness.text import text_normalize

OLD = Path("/data/qwen3-asr/records/therapist_mas_setting_d_v1/therapist_crossed96_dose64_loop3_v1")
ROOT = OLD.parent / "m0_absorption_factorial_v1"
ARMS = [(96, 0), (96, 96), (96, 288), (0, 96), (0, 288), (32, 0), (32, 96), (32, 288)]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save_rows(path, records):
    write_immutable_bytes(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode())


def arm_name(s, r):
    return f"s{s:03d}_r{r:03d}"


def interleave(real, synthetic):
    """Spread synthetic evenly; no partial effective batches or epoch resampling."""
    total = len(real) + len(synthetic)
    assert total and total % 32 == 0
    a, b = iter(real), iter(synthetic)
    return [next(b) if (i + 1) * len(synthetic) // total > i * len(synthetic) // total
            else next(a) for i in range(total)]


def small_synthetic(large):
    """Each of the same 16 patients contributes two of six sentences: 32 nested rows."""
    by_patient = defaultdict(list)
    for row in large:
        by_patient[row["reference_wav"]].append(row)
    assert len(by_patient) == 16
    result = []
    for i, patient in enumerate(sorted(by_patient)):
        choices = sorted(by_patient[patient], key=lambda x: x["target_text"])
        assert len(choices) == 6
        result.extend(choices[(2 * i + offset) % 6] for offset in (0, 1))
    assert len(result) == 32
    return result


def char_states(reference, hypothesis):
    alignment = align_characters(reference, hypothesis)
    return ["unresolved" if len(pos) != 1 else
            "correct" if pos[0] is not None and reference[i] == hypothesis[pos[0]] else "error"
            for i, pos in enumerate(alignment.reference_outcomes)]


def retention(before, after):
    old = {r["utt_id"]: r for r in before}
    result = {k: Counter() for k in ("overall", "easy", "medium", "hard")}
    for row in after:
        ref = text_normalize(row["clean_gt"])
        a = char_states(ref, text_normalize(old[row["utt_id"]]["prediction"]))
        b = char_states(ref, text_normalize(row["prediction"]))
        counts = Counter(f"{x}_to_{y}" for x, y in zip(a, b))
        for k in ("overall", row["zero_shot_bucket"]):
            result[k].update(counts)
    return {k: dict(v) for k, v in result.items()}


def make_arm(root, name, real, synthetic, parent, epochs):
    r = [{"audio": x["audio"], "text": x["text"], "prompt": x["prompt"],
          "source": "real", "id": x["utt_id"]} for x in real]
    s = [{"audio": x["output_wav"], "text": "language Cantonese<asr_text>" + x["target_text"],
          "prompt": "", "source": "synthetic", "id": x["sample_id"]} for x in synthetic]
    rng = random.Random(42)
    rng.shuffle(r)
    rng.shuffle(s)
    combined = interleave(r, s)
    directory = root / name
    save_rows(directory / "train.jsonl", [{k: x[k] for k in ("audio", "text", "prompt")} for x in combined])
    write_immutable_json(directory / "training_index.json", [
        {"row": i, "source": x["source"], "id": x["id"], "audio": x["audio"]}
        for i, x in enumerate(combined)])
    command = read(OLD / "round_001/pair_config.json")["commands"]["mas"].copy()
    steps = [len(combined) // 32 * epoch for epoch in epochs]
    changes = {"--train_file": str(directory / "train.jsonl"), "--output_dir": str(directory / "training"),
               "--lora_init_adapter_path": str(parent), "--max_steps": str(max(steps)),
               "--save_steps": "1000000", "--save_total_limit": "0", "--save_final_checkpoint": "0"}
    for key, value in changes.items():
        command[command.index(key) + 1] = value
    cfg = {"real_count": len(r), "synthetic_count": len(s), "rows": len(combined),
           "epochs": epochs, "observe_steps": steps, "parent": str(parent),
           "parent_sha256": sha256_directory(parent), "command": command,
           "save_only_model": True, "real_stats": stats(real) if real else None,
           "train_sha256": sha256_file(directory / "train.jsonl"),
           "audio_sha256": {x["audio"]: sha256_file(Path(x["audio"])) for x in combined}}
    write_immutable_json(directory / "config.json", cfg)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    protocol = read(OLD / "protocol.json")
    parent = Path(protocol["initial_adapter"]["path"])
    assert sha256_directory(parent) == protocol["initial_adapter"]["sha256"]
    real = rows(Path(protocol["real_train"]["path"]))
    real288 = stratified_sample(real, 288, random.Random(42))
    real96 = stratified_sample(real288, 96, random.Random(43))
    synth96 = read(OLD / "round_001/tts_outputs.json")["outputs"]
    # Synthesis outputs carry their reference paths, preserving the crossed grid.
    synth32 = small_synthetic(synth96)
    write_immutable_json(root / "selection.json", {"real288": real288, "real96": real96,
                                                  "synthetic32_ids": [x["sample_id"] for x in synth32]})
    for s, r in ARMS:
        make_arm(root, arm_name(s, r), {0: [], 96: real96, 288: real288}[r],
                 {0: [], 32: synth32, 96: synth96}[s], parent, [1, 2, 4])
    before_path = Path(protocol["initial_adapter_dev_predictions"]["path"])
    before = rows(before_path)
    assert len(before) == 511
    write_immutable_json(root / "protocol.json", {
        "M0": protocol["initial_adapter"], "m0_predictions": str(before_path),
        "m0_metrics": evaluate_snapshot(before, prediction_column="prediction", critical_cer_threshold=.5).model_dump(mode="json"),
        "factorial_arms": [arm_name(s, r) for s, r in ARMS], "epochs": [1, 2, 4],
        "batch": 8, "grad_acc": 4, "lr": 2e-5, "seed": 42,
        "synthetic_source": str(OLD), "agent_calls": 0,
        "sequential_synthetic_only": {"rounds": 3, "synthetic_per_round": 96, "real": 0, "epochs": 2,
                                      "round1": "s096_r000/training/checkpoint-6"},
        "selection": "Overall and hard CER improve vs M0, easy/medium do not increase; report raw results if none qualify.",
        "source_sha256": {str(p): sha256_file(p) for p in [Path(__file__), Path(training.__file__),
            REPO / "examples/therapist_dose_monitor.py", REPO / "finetune/scripts/train_m0_small_dose_pair.py",
            REPO / "finetune/scripts/train_setting_d_m0.py", REPO / "finetune/official/qwen3_asr_sft.py"]}})


class AbsorptionMonitor(Monitor):
    measure_fit = DoseMonitor.measure_fit

    def __init__(self, directory, config, before):
        super().__init__(directory)
        self.config, self.before = config, before
        self.end_step = max(config["observe_steps"])
        self.state["endpoint_step"] = self.end_step
        self.indices = read(directory / "training_index.json")
        self.fit_history = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.measure_fit(model, 0)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        control.should_save = state.global_step in self.config["observe_steps"]
        return control

    def on_save(self, args, state, control, model=None, **kwargs):
        self.measure_fit(model, state.global_step)
        super().on_save(args, state, control, model=model, **kwargs)
        after = rows(self.root / f"evaluation/merged_step_{state.global_step}/predictions.jsonl")
        self.history[-1].update(epochs=state.epoch, retention_vs_m0=retention(self.before, after))
        self.publish()
        return control


def train(root, name):
    config = read(root / name / "config.json")
    protocol = read(root / "protocol.json")
    for path, digest in protocol["source_sha256"].items():
        assert sha256_file(Path(path)) == digest, path
    assert sha256_directory(Path(config["parent"])) == config["parent_sha256"]
    assert sha256_file(root / name / "train.jsonl") == config["train_sha256"]
    for path, digest in config["audio_sha256"].items():
        assert sha256_file(Path(path)) == digest
    monitor = AbsorptionMonitor(root / name, config, rows(Path(protocol["m0_predictions"])))

    class ProbeTrainer(training.QwenASRLoRATrainer):
        def __init__(self, *args, **kwargs):
            kwargs["args"].save_only_model = True
            super().__init__(*args, **kwargs)
            monitor.trainer = self
            self.add_callback(monitor)

        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

    training.QwenASRLoRATrainer = ProbeTrainer
    monitor.publish()
    sys.argv = config["command"][1:]
    training.main()
    assert [h["step"] for h in monitor.history] == config["observe_steps"]
    assert [h["epochs"] for h in monitor.history] == config["epochs"]
    write_immutable_json(root / name / "result.json", {"status": "complete", "config": config,
        "history": monitor.history, "fit_history": monitor.fit_history})
    monitor.state["status"] = "complete"
    monitor.publish()


def run(root):
    def execute(name):
        print(f"[absorption] start {name}", flush=True)
        with (root / f"{name}.log").open("x") as log:
            subprocess.run([sys.executable, str(Path(__file__)), "--root", str(root), "--train", name],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"[absorption] complete {name}", flush=True)

    for s, r in ARMS:
        execute(arm_name(s, r))
    parent = root / "s096_r000/training/checkpoint-6"
    for n in (2, 3):
        name = f"synthetic_only_round{n}"
        make_arm(root, name, [], read(OLD / f"round_{n:03d}/tts_outputs.json")["outputs"], parent, [2])
        execute(name)
        parent = root / name / "training/checkpoint-6"
    results = {p.parent.name: read(p) for p in root.glob("*/result.json")}
    assert len(results) == 10
    write_immutable_json(root / "results.json", results)
    print("[absorption] all factorial arms and pure-synthetic continuation complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root)
    elif args.train:
        train(args.root, args.train)
    else:
        run(args.root)
