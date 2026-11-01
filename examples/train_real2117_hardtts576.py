"""Fresh ASR LoRA on all real2117 plus the frozen hard-reference TTS576."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import tts_reference_policy as reference
from tts_increment_control import (
    ROOT as SOURCE, SEEDS, TEST, REPO, read, rows, jsonl, sha256_file,
    training_command, write_immutable_json as save, write_immutable_bytes,
)

ROOT = SOURCE.parent / "real2117_hardtts576_v1"
ARM = "real_plus_hard_tts"
GROUPS = ("overall", "easy", "medium", "hard", "vlink247", "other_patients")


def status(root, value, **extra):
    temporary = root / f".status_{os.getpid()}.json"
    temporary.write_text(json.dumps({"status": value, "pid": os.getpid(),
        "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}) + "\n")
    temporary.replace(root / "run_status.json")


def prepare(root):
    reference.verify(reference.ROOT)
    assert not root.exists(), root
    real_path = SOURCE / "real_q/train.jsonl"
    tts_path = reference.ROOT / "hard_multi/train.jsonl"
    real, old_mix = rows(real_path), rows(tts_path)
    selection = read(SOURCE / "selection.json")
    project = lambda r: {k: r[k] for k in ("audio", "text", "prompt")}
    assert real == [project(r) for r in selection["R"] + selection["Q"]]
    assert old_mix[:1541] == real[:1541]
    synthetic = old_mix[1541:]
    assert (len(real), len(synthetic)) == (2117, 576)
    original_real = rows(Path("/data/qwen3-asr/finetune/data_setting_d_v1/setting_d_train.jsonl"))
    key = lambda r: tuple(r[k] for k in ("audio", "text", "prompt"))
    assert Counter(map(key, real)) == Counter(map(key, original_real))
    assert [r["text"] for r in synthetic] == [r["text"] for r in real[1541:]]
    inputs = {real_path, tts_path, SOURCE / "selection.json", TEST,
              reference.ROOT / "protocol.json", reference.ROOT / "assembly.json"}
    assembly = read(reference.ROOT / "assembly.json")
    assert sha256_file(tts_path) == assembly["train_sha256"]["hard_multi"]
    for pair, row in zip(read(reference.ROOT / "pairs.json"), synthetic):
        receipt = reference.ROOT / "selected" / f"{pair['sample_id']}.json"
        assert sha256_file(receipt) == assembly["selected_sha256"][str(receipt)]
        item = read(receipt)
        reference.validate_selected(pair, item)
        record = item["records"]["hard_multi"]
        assert row["audio"] == record["output_wav"]
        assert sha256_file(Path(row["audio"])) == record["output_wav_sha256"]
        inputs.update([receipt, Path(row["audio"])])
    scope = rows(TEST)
    for field in ("utt_id", "prompt_id", *reference.HASH_FIELDS):
        assert not {r[field] for r in original_real} & {r[field] for r in scope}
    assert len({r["audio"] for r in real + synthetic}) == 2693
    root.mkdir(parents=True)
    jsonl(root / ARM / "train.jsonl", real + synthetic)
    commands = {str(seed): training_command(root, ARM, seed) for seed in SEEDS}
    for seed, cmd in commands.items():
        prior_path = SOURCE / "real_q" / f"seed_{seed}/execution.json"
        prior = read(prior_path)["command"]
        before, after = dict(zip(prior[2::2], prior[3::2])), dict(zip(cmd[2::2], cmd[3::2]))
        assert cmd[:2] == prior[:2]
        assert {k for k in after if before[k] != after[k]} == {"--train_file", "--output_dir"}
    old_cfg = read(SOURCE / "protocol.json")
    source_paths = [Path(__file__), REPO / "examples/tts_reference_policy.py",
        REPO / "examples/tts_increment_r_only.py", REPO / "examples/report_tts_increment_control.py",
        REPO / "src/therapist_harness/evidence.py", REPO / "src/therapist_harness/text.py"]
    for location, arm in locations(root).values():
        if location == root:
            continue
        for seed in SEEDS:
            inputs.update(location / arm / f"seed_{seed}" / file for file in
                ("result.json", "evaluation/merged_step_50/predictions.jsonl"))
    save(root / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "real_rows": 2117,
        "synthetic_rows": 576, "total_rows": 2693, "seeds": list(SEEDS),
        "initialization": "original_Qwen3_ASR_1.7B_new_LoRA_per_seed",
        "real_order": "exact existing real_q train manifest: R1541 then Q576",
        "tts_source": "exact frozen hard_multi576; unchanged audio, reference and labels",
        "text_relation": "Q576 real recordings and their synthetic counterparts are both included",
        "actual_stop_step": 50, "scheduler_horizon": 140, "warmup_ratio": .02,
        "batch_size": 8, "gradient_accumulation": 4, "lr": 2e-4,
        "lora": old_cfg["lora"], "commands": commands,
        "selection": "fixed step50, all three seeds, no best-checkpoint selection",
        "budget_note": "same optimizer updates; real exposure not held equal",
        "evaluation": "existing real511, including separate vlink247106 and other405; exploratory",
        "source_sha256": {**old_cfg["source_sha256"], **{str(p): sha256_file(p) for p in source_paths}},
        "input_sha256": {str(p): sha256_file(p) for p in sorted(inputs)},
        "train_sha256": sha256_file(root / ARM / "train.jsonl")})
    status(root, "prepared", completed_seeds=[])
    print("Prepared: 2117 real + unchanged 576 hard-reference TTS = 2693 rows; matched commands verified.", flush=True)


def verify(root):
    reference.verify_sources()
    cfg = read(root / "protocol.json")
    for name in ("source_sha256", "input_sha256"):
        for path, digest in cfg[name].items():
            assert sha256_file(Path(path)) == digest, path
    assert sha256_file(root / ARM / "train.jsonl") == cfg["train_sha256"]
    assert len(rows(root / ARM / "train.jsonl")) == 2693
    return cfg


def train(root, seed):
    cfg = verify(root)
    import qwen3_asr_lora_sft as training
    from train_m0_small_dose_pair import Monitor
    destination = root / ARM / f"seed_{seed}"
    destination.mkdir(exist_ok=False)
    command = cfg["commands"][str(seed)]
    assert command == training_command(root, ARM, seed)
    args = training.parse_args(command[2:])
    assert not args.lora_init_adapter_path and not args.resume and not args.resume_from
    save(destination / "execution.json", {"command": command, "actual_stop_step": 50,
        "train_sha256": cfg["train_sha256"], "initialization": "base_new_adapter"})

    class FixedMonitor(Monitor):
        def on_step_end(self, args, state, control, **kwargs):
            control.should_save = state.global_step == 50
            control.should_training_stop = state.global_step >= 50
            return control

    monitor = FixedMonitor(destination)
    monitor.end_step = 50
    monitor.state["endpoint_step"] = 50
    monitor.publish()

    class FreshTrainer(training.QwenASRLoRATrainer):
        def __init__(self, *args, **kwargs):
            kwargs["args"].save_only_model = True
            super().__init__(*args, **kwargs)
            self.add_callback(monitor)

    training.QwenASRLoRATrainer = FreshTrainer
    sys.argv = command[1:]
    try:
        training.main()
        assert [h["step"] for h in monitor.history] == [50]
        state = read(destination / "training/checkpoint-50/trainer_state.json")
        assert state["global_step"] == 50
        prior = read(SOURCE / "real_q" / f"seed_{seed}/result.json")
        lr = lambda log: [(r["step"], r["learning_rate"]) for r in log if "learning_rate" in r]
        assert lr(state["log_history"]) == lr(prior["train_log"])
        save(destination / "result.json", {"arm": ARM, "seed": seed, "step": 50,
            "metrics": monitor.history[0]["metrics"], "train_log": state["log_history"]})
        monitor.state["status"] = "complete"
        monitor.publish()
    except Exception as error:
        monitor.state.update(status="failed", error=f"{type(error).__name__}: {error}")
        monitor.publish()
        raise


def locations(root):
    return {"r_only1541": (reference.R_ONLY, "r_only"),
            "real2117": (SOURCE, "real_q"),
            "r1541_hardtts576": (reference.ROOT, "hard_multi"),
            "real2117_hardtts576": (root, ARM)}


def report(root):
    verify(root)
    from report_tts_increment_control import edit_distance, text_normalize
    from therapist_harness.evidence import evaluate_snapshot
    scope = rows(TEST)
    results = {}
    for label, (location, arm) in locations(root).items():
        results[label] = []
        for seed in SEEDS:
            destination = location / arm / f"seed_{seed}"
            saved = read(destination / "result.json")
            assert (saved["seed"], saved["step"]) == (seed, 50)
            predicted = rows(destination / "evaluation/merged_step_50/predictions.jsonl")
            assert len(predicted) == len(scope) == 511
            for p, expected in zip(predicted, scope):
                assert all(p[k] == expected[k] for k in ("utt_id", "speaker_id", "audio", "clean_gt", "zero_shot_bucket"))
            subsets = {"overall": predicted,
                **{g: [r for r in predicted if r["zero_shot_bucket"] == g] for g in ("easy", "medium", "hard")},
                "vlink247": [r for r in predicted if r["speaker_id"] == "vlink247"],
                "other_patients": [r for r in predicted if r["speaker_id"] != "vlink247"]}
            assert (len(subsets["vlink247"]), len(subsets["other_patients"])) == (106, 405)
            metrics = {}
            for group, subset in subsets.items():
                m = evaluate_snapshot(subset, prediction_column="prediction", critical_cer_threshold=.5).model_dump(mode="json")["overall"]
                ref = [text_normalize(r["clean_gt"]) for r in subset]
                edits = [edit_distance(t, text_normalize(r["prediction"])) for t, r in zip(ref, subset)]
                assert sum(edits) == m["total_edits"]
                assert math.isclose(statistics.mean(e / len(t) for e, t in zip(edits, ref)), m["mean_utterance_cer"], abs_tol=1e-12)
                if group in ("overall", "easy", "medium", "hard"):
                    for key in ("mean_utterance_cer", "pooled_cer", "patient_macro_cer", "total_edits"):
                        assert math.isclose(m[key], saved["metrics"][group][key], abs_tol=1e-12)
                metrics[group] = m
            results[label].append({"seed": seed, "step": 50, "metrics": metrics})
    means = {a: {g: statistics.mean(r["metrics"][g]["mean_utterance_cer"] for r in runs) for g in GROUPS} for a, runs in results.items()}
    new_arm = "real2117_hardtts576"
    comparisons = {}
    for baseline in ("real2117", "r_only1541", "r1541_hardtts576"):
        comparisons[baseline] = {}
        for group in GROUPS:
            deltas = [100 * (a["metrics"][group]["mean_utterance_cer"] - b["metrics"][group]["mean_utterance_cer"])
                      for a, b in zip(results[new_arm], results[baseline])]
            comparisons[baseline][group] = {"seed_deltas_pp": deltas, "mean_delta_pp": statistics.mean(deltas), "improved_seeds": sum(d < 0 for d in deltas)}
    save(root / "results.json", {"results": results, "means": means, "new_arm_minus_baseline": comparisons})
    lines = ["# 全部2117真实 + 576 hard-reference TTS", "", "固定step50，seed42/43/44；逐句平均CER（%），越低越好。", "",
        "| 训练组 | Overall | vlink247（106条） | 其他患者（405条） | Easy | Medium | Hard |", "|---|---:|---:|---:|---:|---:|---:|"]
    order = ("overall", "vlink247", "other_patients", "easy", "medium", "hard")
    for arm in means:
        lines.append(f"| {arm} | " + " | ".join(f"{100*means[arm][g]:.4f}" for g in order) + " |")
    lines += ["", "## 新实验逐seed", "", "| Seed | Overall | vlink247 | 其他患者 | Easy | Medium | Hard |", "|---|---:|---:|---:|---:|---:|---:|"]
    for result in results[new_arm]:
        lines.append(f"| {result['seed']} | " + " | ".join(f"{100*result['metrics'][g]['mean_utterance_cer']:.4f}" for g in order) + " |")
    for baseline, comparison in comparisons.items():
        lines += ["", f"新实验 − {baseline}，负值为改善（百分点）：", ""]
        lines.extend(f"- {g}: {comparison[g]['mean_delta_pp']:+.4f}，{comparison[g]['improved_seeds']}/3 seeds改善。" for g in order)
    lines += ["", "全部真实数据按既有real_q顺序加入，576条TTS复用原hard_multi音频，未重新生成或过滤；Q真实语音与对应TTS均在训练中。", "更新数和学习率序列匹配；未固定真实样本曝光量。使用既有合并dev+test反馈集，属于探索性结果；三个ASR seed不等于三套reference。", ""]
    write_immutable_bytes(root / "report.zh.md", "\n".join(lines).encode())
    save(root / "acceptance.json", {"status": "verified", "new_runs": 3, "reused_runs": 9,
        "rows_per_evaluation": 511, "independent_verification": "12 runs x 6 groups",
        "results_sha256": sha256_file(root / "results.json")})
    print("\n".join(lines), flush=True)


def run(root):
    verify(root)
    completed = []
    for seed in SEEDS:
        status(root, "running", stage="train", seed=seed, completed_seeds=completed)
        with (root / f"seed_{seed}.log").open("x") as log:
            subprocess.run([sys.executable, "-u", str(Path(__file__)), "--root", str(root),
                "--stage", "train", "--seed", str(seed)], stdout=log, stderr=subprocess.STDOUT, check=True)
        completed.append(seed)
    status(root, "running", stage="report", completed_seeds=completed)
    report(root)
    status(root, "complete", stage="report", completed_seeds=completed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage", required=True, choices=("prepare", "train", "report", "run"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.stage == "train" and args.seed is None:
        parser.error("train requires --seed")
    try:
        if args.stage == "train":
            train(args.root, args.seed)
        else:
            globals()[args.stage](args.root)
    except Exception as error:
        if args.root.exists():
            status(args.root, "failed", stage=args.stage, seed=args.seed, error=f"{type(error).__name__}: {error}")
        raise
