"""Verify and render the fixed-endpoint real-increment experiment."""
from pathlib import Path
import statistics
import math

from tts_increment_control import ARMS, ROOT, SEEDS, read, rows, sha256_file, write_immutable_bytes, write_immutable_json
from therapist_harness.text import text_normalize


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def report(root):
    protocol = read(root / "protocol.json")
    assembly = read(root / "assembly.json")
    result = read(root / "results.json")
    for path, digest in protocol["source_sha256"].items():
        assert sha256_file(Path(path)) == digest
    pairs = read(root / "pairs.json")
    for pair in pairs:
        item = read(root / "selected" / f"{pair['sample_id']}.json")
        assert item["pair"] == pair
        for record in item["records"].values():
            assert record["target_text"] == pair["target_text"]
            assert record["seed"] == pair["seed"] + 10000 * item["attempt"]
            assert sha256_file(Path(record["output_wav"])) == record["output_wav_sha256"]
    ids = None
    for arm in ARMS:
        assert sha256_file(root / arm / "train.jsonl") == assembly["train_sha256"][arm]
        for seed in SEEDS:
            d = root / arm / f"seed_{seed}"
            assert read(d / "training/checkpoint-50/trainer_state.json")["global_step"] == 50
            current = rows(d / "evaluation/merged_step_50/predictions.jsonl")
            current_ids = [r["utt_id"] for r in current]
            assert len(current_ids) == len(set(current_ids)) == 511
            if ids is None:
                ids = current_ids
            assert ids == current_ids
            metrics = read(d / "result.json")["metrics"]
            for group in ("overall", "easy", "medium", "hard"):
                selected = [r for r in current if group == "overall" or r["zero_shot_bucket"] == group]
                distances = [edit_distance(text_normalize(r["clean_gt"]), text_normalize(r["prediction"])) for r in selected]
                lengths = [len(text_normalize(r["clean_gt"])) for r in selected]
                assert sum(distances) == metrics[group]["total_edits"]
                assert sum(lengths) == metrics[group]["total_reference_characters"]
                assert math.isclose(statistics.mean(e/n for e, n in zip(distances, lengths)),
                                    metrics[group]["mean_utterance_cer"], abs_tol=1e-12)
                assert math.isclose(sum(distances)/sum(lengths), metrics[group]["pooled_cer"], abs_tol=1e-12)
    groups = ("overall", "easy", "medium", "hard")
    lines = ["# 真实增量正对照结果", "", "所有模型从原始 Qwen3-ASR 新建 LoRA，固定 step 50。CER 单位为 %，差值为百分点。", "",
        "| 组别 | Seed | Overall | Easy | Medium | Hard | Patient macro | Pooled |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        for run in result["results"][arm]:
            m = run["metrics"]
            values = [m[g]["mean_utterance_cer"] for g in groups]
            values += [m["overall"]["patient_macro_cer"], m["overall"]["pooled_cer"]]
            lines.append(f"| {arm} | {run['seed']} | " + " | ".join(f"{100*x:.4f}" for x in values) + " |")
    lines += ["", "## 三 seed 平均", "", "| 组别 | Overall | Easy | Medium | Hard |", "| --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        lines.append(f"| {arm} | " + " | ".join(f"{100*result['mean_utterance_cer_seed_mean'][arm][g]:.4f}" for g in groups) + " |")
    comparisons = {}
    lines += ["", "## 配对差值（前者减后者；负值表示前者更好）", ""]
    for treatment, control in (("real_q", "replay"), ("patient_tts", "replay"), ("patient_tts", "base_tts"), ("patient_tts", "real_q")):
        key = f"{treatment}_minus_{control}"
        comparisons[key] = {}
        for group in groups:
            deltas = [100*(a["metrics"][group]["mean_utterance_cer"]-b["metrics"][group]["mean_utterance_cer"])
                      for a, b in zip(result["results"][treatment], result["results"][control])]
            comparisons[key][group] = {"seed_deltas_pp": deltas, "mean_delta_pp": statistics.mean(deltas), "improved_seeds": sum(x < 0 for x in deltas)}
        lines.append(f"- {treatment} − {control}：" + "；".join(f"{g} {comparisons[key][g]['mean_delta_pp']:+.4f} pp（{comparisons[key][g]['improved_seeds']}/3 seeds 改善）" for g in groups))
    write_immutable_json(root / "acceptance.json", {"status": "verified", "runs": 12, "evaluation_rows_each": 511,
        "pairs": len(pairs), "independent_edit_distance_verification": "two-row Levenshtein: all 12 runs x 4 groups",
        "comparisons": comparisons})
    write_immutable_bytes(root / "report.zh.md", ("\n".join(lines) + "\n").encode())


if __name__ == "__main__":
    report(ROOT)
