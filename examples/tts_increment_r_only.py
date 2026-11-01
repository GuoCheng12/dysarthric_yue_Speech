"""Add the missing fixed-step R-only arm to the frozen TTS increment control."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import math
from pathlib import Path
import statistics
import sys

from tts_increment_control import (
    ARMS, ROOT as SOURCE, SEEDS, TEST, REPO, jsonl, read, rows,
    sha256_file, stats, training_command, write_immutable_bytes, write_immutable_json,
)

ROOT = SOURCE.parent / "tts_increment_r_only1541_v1"
GROUPS = ("overall", "easy", "medium", "hard")


def verify_sources():
    original = read(SOURCE / "protocol.json")
    for path, digest in original["source_sha256"].items():
        assert sha256_file(Path(path)) == digest, path
    assert sha256_file(TEST) == original["test_sha256"]
    old = read(SOURCE / "real_q/seed_42/training/lora_run_metadata.json")
    for package, expected in old["runtime_versions"].items():
        actual = sys.version if package == "python" else version(package)
        assert actual == expected, (package, actual, expected)
    return original


def prepare():
    original = verify_sources()
    selection = read(SOURCE / "selection.json")
    retained = selection["R"]
    projected = [{k: r[k] for k in ("audio", "text", "prompt")} for r in retained]
    assert len(retained) == len({r["utt_id"] for r in retained}) == 1541
    assembly = read(SOURCE / "assembly.json")
    for arm in ARMS:
        path = SOURCE / arm / "train.jsonl"
        assert sha256_file(path) == assembly["train_sha256"][arm]
        assert projected == rows(path)[:1541]
    evaluation = rows(TEST)
    for field in ("utt_id", "prompt_id", "prompt_normalized_sha16", "prompt_jyutping_sha16"):
        assert not {r[field] for r in retained} & {r[field] for r in evaluation}
    assert all(Path(r["audio"]).is_file() for r in retained)
    ROOT.mkdir(exist_ok=False)
    jsonl(ROOT / "r_only/train.jsonl", projected)
    commands = {str(s): training_command(ROOT, "r_only", s) for s in SEEDS}
    for seed, command in commands.items():
        previous = read(SOURCE / "real_q" / f"seed_{seed}" / "execution.json")["command"]
        current_options = dict(zip(command[2::2], command[3::2]))
        previous_options = dict(zip(previous[2::2], previous[3::2]))
        assert command[:2] == previous[:2]
        assert {k for k in current_options if current_options[k] != previous_options[k]} == {
            "--train_file", "--output_dir"}
    write_immutable_json(ROOT / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment": str(SOURCE),
        "source_protocol_sha256": sha256_file(SOURCE / "protocol.json"),
        "source_selection_sha256": sha256_file(SOURCE / "selection.json"),
        "source_sha256": original["source_sha256"],
        "runner_sha256": sha256_file(Path(__file__)),
        "R": stats(retained), "synthetic_rows": 0, "seeds": list(SEEDS),
        "commands": commands, "actual_stop_step": 50,
        "scheduler": "unchanged linear schedule over 140 steps, warmup_ratio=0.02",
        "selection": "none; evaluate fixed step50 only",
        "train_sha256": sha256_file(ROOT / "r_only/train.jsonl"),
        "test_path": str(TEST), "test_sha256": sha256_file(TEST),
        "primary_comparison": "R+patient_tts576 minus R-only1541, paired by ASR seed",
        "secondary_comparisons": ["replay", "real_q", "base_tts"],
        "budget_note": "Same optimizer updates and nominal batch32, not same epochs; "
            "R-only crosses an epoch boundary and retains the original trainer's partial-batch handling.",
        "evaluation_note": "Existing 511 merged dev+test items; exploratory, not a new untouched test set.",
    })
    print("Prepared R-only: 1541 exact shared rows; three matched commands verified.", flush=True)


def train(seed):
    verify_sources()
    import qwen3_asr_lora_sft as training
    from train_m0_small_dose_pair import Monitor

    cfg = read(ROOT / "protocol.json")
    assert sha256_file(Path(__file__)) == cfg["runner_sha256"]
    assert sha256_file(ROOT / "r_only/train.jsonl") == cfg["train_sha256"]
    destination = ROOT / "r_only" / f"seed_{seed}"
    destination.mkdir(exist_ok=False)
    command = cfg["commands"][str(seed)]
    write_immutable_json(destination / "execution.json", {
        "command": command, "actual_stop_step": 50,
        "train_sha256": cfg["train_sha256"], "initialization": "base_new_adapter",
    })

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
        prior = read(SOURCE / "real_q" / f"seed_{seed}" / "result.json")
        old_lr = [(r["step"], r["learning_rate"]) for r in prior["train_log"] if "learning_rate" in r]
        new_lr = [(r["step"], r["learning_rate"]) for r in state["log_history"] if "learning_rate" in r]
        assert old_lr == new_lr
        write_immutable_json(destination / "result.json", {
            "arm": "r_only", "seed": seed, "step": 50,
            "metrics": monitor.history[0]["metrics"], "train_log": state["log_history"],
        })
        monitor.state["status"] = "complete"
        monitor.publish()
    except Exception as error:
        monitor.state.update(status="failed", error=f"{type(error).__name__}: {error}")
        monitor.publish()
        raise


def report():
    verify_sources()
    from report_tts_increment_control import edit_distance, text_normalize

    locations = {"r_only": ROOT, **{a: SOURCE for a in ARMS}}
    results = {}
    ids = [r["utt_id"] for r in rows(TEST)]
    for arm, root in locations.items():
        results[arm] = []
        for seed in SEEDS:
            d = root / arm / f"seed_{seed}"
            run = read(d / "result.json")
            assert run["seed"] == seed and run["step"] == 50
            predictions = rows(d / "evaluation/merged_step_50/predictions.jsonl")
            assert [r["utt_id"] for r in predictions] == ids
            for group in GROUPS:
                subset = [r for r in predictions if group == "overall" or r["zero_shot_bucket"] == group]
                lengths = [len(text_normalize(r["clean_gt"])) for r in subset]
                edits = [edit_distance(text_normalize(r["clean_gt"]), text_normalize(r["prediction"])) for r in subset]
                metrics = run["metrics"][group]
                assert sum(edits) == metrics["total_edits"]
                assert sum(lengths) == metrics["total_reference_characters"]
                assert math.isclose(statistics.mean(e/n for e, n in zip(edits, lengths)),
                                    metrics["mean_utterance_cer"], abs_tol=1e-12)
            results[arm].append(run)
    means = {arm: {g: statistics.mean(r["metrics"][g]["mean_utterance_cer"] for r in runs)
                   for g in GROUPS} for arm, runs in results.items()}
    comparisons = {}
    for arm in ARMS:
        comparisons[arm] = {}
        for group in GROUPS:
            deltas = [100 * (a["metrics"][group]["mean_utterance_cer"] - b["metrics"][group]["mean_utterance_cer"])
                      for a, b in zip(results[arm], results["r_only"])]
            comparisons[arm][group] = {"seed_deltas_pp": deltas,
                "mean_delta_pp": statistics.mean(deltas), "improved_seeds": sum(d < 0 for d in deltas)}
    write_immutable_json(ROOT / "results.json", {"results": results,
        "mean_utterance_cer_seed_mean": means, "comparisons_vs_r_only": comparisons})
    lines = ["# R-only 1541 补充实验", "", "固定 step50，seed42/43/44；CER 为逐句平均，单位 %。", "",
        "| 训练组 | Overall | Easy | Medium | Hard |", "|---|---:|---:|---:|---:|"]
    for arm in locations:
        lines.append(f"| {arm} | " + " | ".join(f"{100*means[arm][g]:.4f}" for g in GROUPS) + " |")
    lines += ["", "## 相对 R-only 的配对差值", "", "负值为改善，单位百分点。", ""]
    for arm in ARMS:
        lines.append(f"- {arm}：" + "；".join(
            f"{g} {comparisons[arm][g]['mean_delta_pp']:+.4f}（{comparisons[arm][g]['improved_seeds']}/3 seeds 改善）"
            for g in GROUPS))
    lines += ["", "## 每个 seed 的 Overall CER", "", "| Seed | R-only | Replay | Real Q | Patient TTS | Base TTS |",
              "|---|---:|---:|---:|---:|---:|"]
    for i, seed in enumerate(SEEDS):
        lines.append(f"| {seed} | " + " | ".join(
            f"{100*results[a][i]['metrics']['overall']['mean_utterance_cer']:.4f}" for a in locations) + " |")
    lines += ["", "## 比较口径", "",
        "- R-only 使用原 selection.json 中完全相同顺序的1541条R；其余四组复用旧实验结果。",
        "- 模型、训练源码、软件版本、逐步学习率与固定50步终点均核对；只改变训练集与输出路径。",
        "- 训练预算按optimizer更新数保持一致，不是相同epoch或完全相同样本曝光量。",
        "- 评测仍是既有511条合并dev+test，属于探索性结果；三seed方向一致不等于统计显著。",
        "- 此实验未更换reference、未重生成音频；结果只针对已有合成策略。"]
    write_immutable_bytes(ROOT / "report.zh.md", ("\n".join(lines) + "\n").encode())
    write_immutable_json(ROOT / "acceptance.json", {"status": "verified", "new_runs": 3,
        "reused_runs": 12, "rows_per_evaluation": len(ids),
        "independent_edit_distance_verification": "all 15 runs x 4 groups",
        "result_sha256": sha256_file(ROOT / "results.json")})
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--seed", type=int, choices=SEEDS)
    action.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.seed is not None:
        train(args.seed)
    else:
        report()
