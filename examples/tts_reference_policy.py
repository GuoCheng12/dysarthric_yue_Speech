"""Matched same-patient random/hard reference policies for R + Patient TTS."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import tts_increment_control as control
from tts_increment_control import REPO, SEEDS, TEST, jsonl, read, rows, sha256_file
from tts_increment_control import write_immutable_bytes, write_immutable_json as save
from tts_increment_r_only import ROOT as R_ONLY, verify_sources
from tts_reconstruction_probe import select48

SOURCE = control.ROOT
ROOT = SOURCE.parent / "tts_reference_policy_v1"
ARMS = ("random_multi", "hard_multi")
GROUPS = ("overall", "easy", "medium", "hard")
REF_SEED = 20260907
ADAPTER = R_ONLY / "r_only/seed_42/training/checkpoint-50"
HASH_FIELDS = ("prompt_normalized_sha16", "prompt_jyutping_sha16")


def same_text(a, b):
    return any(a[k] == b[k] for k in HASH_FIELDS)


def duration_bin(seconds):
    assert 3 <= seconds <= 12
    return min(2, int((seconds - 3) // 3))


def audio_features(path):
    """Technical checks and relative-energy pause proxy; no ASR quality gate."""
    import numpy as np
    import soundfile as sf
    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    finite = bool(np.isfinite(wave).all())
    info = {"seconds": len(wave) / sr, "channels": wave.shape[1], "sample_rate": sr}
    if not len(wave) or not finite:
        return {**info, "rejection": "empty_or_nonfinite"}
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(wave)))
    dc = float(abs(np.mean(wave, dtype=np.float64)))
    clipped = float(np.mean(np.abs(wave) >= .999))
    reason = ("non_mono" if wave.shape[1] != 1 else
              "near_silent" if min(rms, peak) < 1e-4 else
              "excessive_dc" if dc > .2 else "clipped" if clipped > .02 else None)
    mono = wave.mean(axis=1)
    hop = max(1, round(sr * .01))
    frames = mono[:len(mono) // hop * hop].reshape(-1, hop)
    if not len(frames):
        return {**info, "rejection": "too_short"}
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    quiet = frame_rms < max(float(np.quantile(frame_rms, .95)) * 10 ** (-35 / 20), 1e-7)
    return {**info, "rms": rms, "peak_abs": peak, "dc_abs": dc,
            "clipping_fraction": clipped, "quiet_fraction": float(quiet.mean()),
            "quiet_seconds": float(quiet.sum() * hop / sr), "rejection": reason}


def choose_references(candidates, targets, seed=REF_SEED):
    """Match reference slots by duration bin, then rotate shared slots over Q."""
    rng = random.Random(seed)
    by_patient = defaultdict(list)
    for candidate in candidates:
        by_patient[candidate["speaker_id"]].append(candidate)
    mapping, audit = {}, {}
    for speaker in sorted({r["speaker_id"] for r in targets}):
        pool = sorted(by_patient[speaker], key=lambda r: r["utt_id"])
        rng.shuffle(pool)
        distinct = []
        for r in pool:
            if not any(same_text(r, previous) for previous in distinct):
                distinct.append(r)
        if len(distinct) < 2:
            raise ValueError(f"Need two distinct qualified R references: {speaker}")
        random_refs = rng.sample(distinct, min(3, len(distinct)))
        hard_refs = []
        for bucket, count in sorted(Counter(r["duration_bin"] for r in random_refs).items()):
            ranked = sorted((r for r in distinct if r["duration_bin"] == bucket),
                            key=lambda r: (-float(r["zero_shot_cer"]), r["utt_id"]))
            hard_refs.extend(ranked[:count])
        # Keep shared references in the same slot. This also prevents crossed
        # exclusions from making a two-reference patient's target unassignable.
        paired, remaining = [], {r["utt_id"]: r for r in hard_refs}
        random_ids = {r["utt_id"] for r in random_refs}
        for r in random_refs:
            if r["utt_id"] in remaining:
                other = remaining.pop(r["utt_id"])
            else:
                options = [v for key, v in remaining.items()
                           if key not in random_ids and v["duration_bin"] == r["duration_bin"]]
                other = min(options, key=lambda v: (abs(v["seconds"] - r["seconds"]), v["utt_id"]))
                remaining.pop(other["utt_id"])
            paired.append((r, other))
        counts = [0] * len(paired)
        selected_targets = sorted((q for q in targets if q["speaker_id"] == speaker), key=lambda q: q["utt_id"])
        rng.shuffle(selected_targets)
        for q in selected_targets:
            eligible = [i for i, pair in enumerate(paired) if not any(same_text(q, r) for r in pair)]
            if not eligible:
                raise ValueError(f"No distinct-text paired reference for {q['utt_id']}")
            least = min(counts[i] for i in eligible)
            slot = rng.choice([i for i in eligible if counts[i] == least])
            counts[slot] += 1
            mapping[q["utt_id"]] = dict(zip(ARMS, paired[slot]))
        audit[speaker] = {"candidate_rows": len(pool), "distinct_text_candidates": len(distinct),
            "slots": [{"random_utt_id": a["utt_id"], "hard_utt_id": b["utt_id"],
                       "duration_bin": a["duration_bin"], "uses": n}
                      for (a, b), n in zip(paired, counts)],
            "different_reference_targets": sum(n for (a, b), n in zip(paired, counts) if a["utt_id"] != b["utt_id"])}
    return mapping, audit


def reference(row):
    return {"utt_id": row["utt_id"], "speaker_id": row["speaker_id"],
            **{k: row[k] for k in HASH_FIELDS}, "reference_text": row["clean_gt"],
            "reference_wav": row["audio"], "reference_wav_sha256": sha256_file(Path(row["audio"])),
            "zero_shot_cer": float(row["zero_shot_cer"])}


def verify(root):
    verify_sources()
    cfg = read(root / "protocol.json")
    for key in ("source_sha256", "input_sha256", "artifact_sha256"):
        for path, digest in cfg[key].items():
            assert sha256_file(Path(path)) == digest, path
    return cfg


def prepare(root):
    original = verify_sources()
    assert not (root / "protocol.json").exists()
    selection = read(SOURCE / "selection.json")
    retained, targets = selection["R"], selection["Q"]
    assert (len(retained), len(targets)) == (1541, 576)
    for key in ("utt_id", "prompt_id", *HASH_FIELDS):
        assert not {r[key] for r in retained + targets} & {r[key] for r in rows(TEST)}
    assert not {r["utt_id"] for r in retained} & {r["utt_id"] for r in targets}
    projected = [{k: r[k] for k in ("audio", "text", "prompt")} for r in retained]
    assert projected == rows(R_ONLY / "r_only/train.jsonl")
    assert projected == rows(SOURCE / "patient_tts/train.jsonl")[:1541]
    speakers = {r["speaker_id"] for r in targets}
    candidates, rejected = [], []
    for r in retained:
        if r["speaker_id"] not in speakers:
            continue
        info = audio_features(r["audio"])
        reason = info["rejection"] or (None if 3 <= info["seconds"] <= 12 else "duration")
        if reason:
            rejected.append({"utt_id": r["utt_id"], "reason": reason, "audio": info})
        else:
            candidates.append({**r, **info, "duration_bin": duration_bin(info["seconds"])})
    mapping, audit = choose_references(candidates, targets)
    old = {p["target_utt_id"]: p for p in read(SOURCE / "pairs.json")}
    retained_by_wav = {r["audio"]: r for r in retained}
    probe_ids = {r["utt_id"] for r in select48(targets)}
    pairs = []
    for q in targets:
        p = old[q["utt_id"]]
        refs = {arm: reference(mapping[q["utt_id"]][arm]) for arm in ARMS}
        refs["old_fixed"] = reference(retained_by_wav[p["reference_wav"]])
        assert all(r["speaker_id"] == q["speaker_id"] for r in refs.values())
        assert all(not same_text(q, refs[a]) for a in ARMS)
        pairs.append({"sample_id": p["sample_id"], "target": q, "seed": p["seed"],
                      "probe": q["utt_id"] in probe_ids, "references": refs})
    root.mkdir(parents=True, exist_ok=True)
    save(root / "pairs.json", pairs)
    save(root / "reference_audit.json", {"patients": audit, "rejected": rejected, "candidates": candidates})
    commands = {a: {str(s): control.training_command(root, a, s) for s in SEEDS} for a in ARMS}
    for a in ARMS:
        for s in SEEDS:
            command = commands[a][str(s)]
            old_command = read(SOURCE / "patient_tts" / f"seed_{s}/execution.json")["command"]
            assert command[:2] == old_command[:2]
            now, prior = dict(zip(command[2::2], command[3::2])), dict(zip(old_command[2::2], old_command[3::2]))
            assert {k for k in now if now[k] != prior[k]} == {"--train_file", "--output_dir"}
    inputs = {SOURCE / n for n in ("selection.json", "pairs.json", "protocol.json", "assembly.json")}
    inputs.update({TEST, R_ONLY / "protocol.json", R_ONLY / "r_only/train.jsonl", SOURCE / "patient_tts/train.jsonl"})
    for location, arm in ((R_ONLY, "r_only"), (SOURCE, "patient_tts")):
        for seed in SEEDS:
            inputs.update(location / arm / f"seed_{seed}" / n for n in
                          ("result.json", "evaluation/merged_step_50/predictions.jsonl"))
    inputs.update(p for p in ADAPTER.iterdir() if p.is_file() and p.name.startswith("adapter"))
    inputs.update(Path(r["reference_wav"]) for p in pairs for r in p["references"].values())
    inputs.update(Path(p["target"]["audio"]) for p in pairs if p["probe"])
    source_paths = [Path(__file__), REPO / "examples/tts_increment_r_only.py",
        REPO / "examples/tts_reconstruction_probe.py", REPO / "examples/report_tts_increment_control.py",
        REPO / "src/therapist_harness/evidence.py", REPO / "src/therapist_harness/text.py"]
    save(root / "protocol.json", {"schema_version": 1, "arms": ARMS, "seeds": SEEDS,
        "R_rows": 1541, "synthetic_rows_each": 576, "probe_targets": 48,
        "reference_selection_seed": REF_SEED, "probe_selection_seed": 20260906,
        "reference_policy": "random distinct-text pool, k=min(3,pool size); hard top-k CER within matching 3-second duration bins; shared slots rotated least-used first with target-text exclusions",
        "duration_bins_seconds": [[3, 6], [6, 9], [9, 12]],
        "bin_edges": "left inclusive/right exclusive except final 12 inclusive",
        "quality_gate": "mono, finite, nonempty, peak/rms>=1e-4, abs DC<=.2, clipping<=.02; no CER filter",
        "retry_policy": "seed=original_pair_seed+10000*attempt, max10; both new arms retried together; 48 probe targets additionally require fresh old_fixed output at same seed",
        "commands": commands, "actual_stop_step": 50, "scheduler_horizon": 140,
        "selection": "fixed step50; no checkpoint or reference selection using downstream results",
        "probe_adapter": str(ADAPTER), "probe_use": "diagnostic only; do not select an arm or filter by CER",
        "limits": ["fixed reference manifest; ASR seeds are not reference replicates",
                   "old versus multi compares complete policies, not reference count alone",
                   "duration matched in bins, not exact seconds; CER not clinical severity",
                   "existing merged511 exploratory evaluation; generator already saw full2117 including Q"],
        "source_sha256": {**original["source_sha256"], **{str(p): sha256_file(p) for p in source_paths}},
        "input_sha256": {str(p): sha256_file(p) for p in sorted(inputs)},
        "artifact_sha256": {str(root / n): sha256_file(root / n) for n in ("pairs.json", "reference_audit.json")}})
    summarize_preparation(root)


def summarize_preparation(root):
    pairs = read(root / "pairs.json")
    candidate_map = {r["utt_id"]: r for r in read(root / "reference_audit.json")["candidates"]}
    summary = {"status": "prepared", "targets": len(pairs), "probe_targets": sum(p["probe"] for p in pairs),
               "patients": len({p["target"]["speaker_id"] for p in pairs}), "new_training_started": False}
    for a in ARMS:
        refs = [p["references"][a] for p in pairs]
        summary[a] = {"used_references": len({r["utt_id"] for r in refs}),
            "target_weighted_reference_cer": statistics.mean(r["zero_shot_cer"] for r in refs),
            "target_weighted_reference_seconds": statistics.mean(candidate_map[r["utt_id"]]["seconds"] for r in refs)}
    summary["different_reference_targets"] = sum(p["references"][ARMS[0]]["utt_id"] != p["references"][ARMS[1]]["utt_id"] for p in pairs)
    save(root / "preparation.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def validate_selected(pair, item):
    assert item["pair"] == pair
    expected = set(ARMS) | ({"old_fixed"} if pair["probe"] else set())
    assert set(item["records"]) == expected
    assert 0 <= item["attempt"] < 10
    for arm, record in item["records"].items():
        assert record["sample_id"] == pair["sample_id"]
        assert record["target_text"] == pair["target"]["clean_gt"]
        assert record["seed"] == pair["seed"] + 10000 * item["attempt"]
        for key in ("reference_text", "reference_wav", "reference_wav_sha256"):
            assert record[key] == pair["references"][arm][key]
        assert sha256_file(Path(record["output_wav"])) == record["output_wav_sha256"]


def synthesize_pair(root, pair, model, tts):
    """Resume whole recorded attempts; never promote an unpaired success."""
    destination = root / "selected" / f"{pair['sample_id']}.json"
    if destination.exists():
        validate_selected(pair, read(destination))
        return
    variants = (*ARMS, "old_fixed") if pair["probe"] else ARMS
    for attempt in range(10):
        folder = root / "synthesis" / pair["sample_id"] / f"attempt_{attempt}"
        receipt = folder / "receipt.json"
        if receipt.exists():
            outcome = read(receipt)
        else:
            folder.mkdir(parents=True, exist_ok=True)
            records, errors = {}, {}
            for arm in variants:
                r = pair["references"][arm]
                sample = tts.ManifestSample(sample_id=pair["sample_id"], target_text=pair["target"]["clean_gt"],
                    reference_text=r["reference_text"], reference_wav=Path(r["reference_wav"]),
                    reference_wav_sha256=r["reference_wav_sha256"], output_wav=folder / f"{arm}.wav",
                    seed=pair["seed"] + 10000 * attempt)
                # An interrupted attempt without a receipt is not silently overwritten.
                if sample.output_wav.exists():
                    raise FileExistsError(f"Incomplete attempt needs inspection: {folder}")
                try:
                    record, temporary = tts._stage_audio_once(sample, model)
                    try:
                        os.link(temporary, sample.output_wav)
                    finally:
                        temporary.unlink()
                    records[arm] = record
                except RuntimeError as error:
                    if not str(error).startswith("Speech Generator returned "):
                        raise
                    errors[arm] = str(error)
            outcome = {"attempt": attempt, "records": records, "errors": errors}
            save(receipt, outcome)
        if not outcome["errors"]:
            item = {"pair": pair, "attempt": attempt, "records": outcome["records"]}
            validate_selected(pair, item)
            save(destination, item)
            return
    raise RuntimeError(f"Ten paired technical attempts exhausted: {pair['sample_id']}")


def synthesize(root, probe_only=False):
    verify(root)
    pairs = [p for p in read(root / "pairs.json") if not probe_only or p["probe"]]
    pending = []
    for p in pairs:
        path = root / "selected" / f"{p['sample_id']}.json"
        if path.exists():
            validate_selected(p, read(path))
        else:
            pending.append(p)
    if not pending:
        return
    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts
    model, meta = tts.load_fixed_runtime()
    save(root / f"runtime_{time.time_ns()}.json", meta)
    for i, pair in enumerate(pending, 1):
        synthesize_pair(root, pair, model, tts)
        print(f"[synthesis] {i}/{len(pending)} {pair['sample_id']}", flush=True)


def diagnose(root):
    verify(root)
    from report_tts_increment_control import edit_distance, text_normalize
    probe = root / "probe"
    decoded = []
    pairs = [p for p in read(root / "pairs.json") if p["probe"]]
    for p in pairs:
        item = read(root / "selected" / f"{p['sample_id']}.json")
        validate_selected(p, item)
        audio = {"real": p["target"]["audio"], **{a: r["output_wav"] for a, r in item["records"].items()}}
        for variant, path in audio.items():
            decoded.append({**p["target"], "source_utt_id": p["target"]["utt_id"],
                "utt_id": p["target"]["utt_id"] + "::" + variant,
                "variant": variant, "audio": path, "audio_sha256": sha256_file(Path(path))})
    manifest = probe / "decode.jsonl"
    if manifest.exists():
        assert rows(manifest) == decoded
    else:
        jsonl(manifest, decoded)
    if not (probe / "asr/predictions.jsonl").exists():
        command = [sys.executable, str(REPO / "finetune/scripts/evaluate_asr_checkpoint.py"),
            "--base-model-path", "/data/qwen3-asr/models/Qwen3-ASR-1.7B", "--adapter-path", str(ADAPTER),
            "--split-jsonl", str(manifest), "--out-dir", str(probe / "asr"),
            "--batch-size", "8", "--critical-cer-threshold", ".5", "--max-new-tokens", "256"]
        with (probe / "decode.log").open("a") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    predictions = rows(probe / "asr/predictions.jsonl")
    assert [r["utt_id"] for r in predictions] == [r["utt_id"] for r in decoded]
    details = defaultdict(dict)
    for p, source in zip(predictions, decoded):
        assert all(p[k] == v for k, v in source.items())
        assert sha256_file(Path(p["audio"])) == p["audio_sha256"]
        ref, hyp = text_normalize(p["clean_gt"]), text_normalize(p["prediction"])
        features = audio_features(p["audio"])
        features["characters_per_second"] = len(ref) / features["seconds"]
        details[p["source_utt_id"]][p["variant"]] = {"audio": p["audio"], "prediction": p["prediction"],
            "cer": edit_distance(ref, hyp) / len(ref), "acoustics": features}
    lines = ["# Reference 配对试听与诊断", "", "48 个目标；固定 R-only seed42 ASR。CER/能量停顿仅作诊断，不用于过滤或选择训练组。", ""]
    for p in pairs:
        q = p["target"]
        lines += [f"## {q['utt_id']} ({q['zero_shot_bucket']})", "", f"GT：{q['clean_gt']}", ""]
        for arm in ("real", "old_fixed", *ARMS):
            x = details[q["utt_id"]][arm]
            lines.append(f"- [{arm}]({x['audio']})：{x['prediction']}；CER {100*x['cer']:.2f}%；时长 {x['acoustics']['seconds']:.2f}s")
            if arm != "real":
                r = p["references"][arm]
                lines.append(f"  reference：[录音]({r['reference_wav']})；{r['reference_text']}；原 CER {100*r['zero_shot_cer']:.2f}%")
        lines.append("")
    save(probe / "diagnostics.json", {"status": "complete", "paired": dict(details),
        "mean_diagnostic_cer": {a: statistics.mean(x[a]["cer"] for x in details.values()) for a in ("real", "old_fixed", *ARMS)},
        "prediction_sha256": sha256_file(probe / "asr/predictions.jsonl"),
        "interpretation": "No automatic impairment or content-correctness conclusion; inspect paired speech."})
    write_immutable_bytes(probe / "listening.zh.md", ("\n".join(lines) + "\n").encode())


def assemble(root):
    verify(root)
    pairs = read(root / "pairs.json")
    selected = [read(root / "selected" / f"{p['sample_id']}.json") for p in pairs]
    for p, item in zip(pairs, selected):
        validate_selected(p, item)
    retained = read(SOURCE / "selection.json")["R"]
    for arm in ARMS:
        data = [{k: r[k] for k in ("audio", "text", "prompt")} for r in retained]
        data += [{"audio": item["records"][arm]["output_wav"], "text": p["target"]["text"],
                  "prompt": p["target"]["prompt"]} for p, item in zip(pairs, selected)]
        assert len(data) == 2117
        assert [r["text"] for r in data] == [r["text"] for r in rows(SOURCE / "patient_tts/train.jsonl")]
        jsonl(root / arm / "train.jsonl", data)
    save(root / "assembly.json", {"rows_each": 2117, "shared_real_rows": 1541,
        "train_sha256": {a: sha256_file(root / a / "train.jsonl") for a in ARMS},
        "selected_sha256": {str(root / "selected" / f"{p['sample_id']}.json"): sha256_file(root / "selected" / f"{p['sample_id']}.json") for p in pairs},
        "attempt_histogram": dict(Counter(x["attempt"] for x in selected)),
        "synthetic_seconds": {a: sum(x["records"][a]["duration_seconds"] for x in selected) for a in ARMS}})


def train(root, arm, seed):
    cfg = verify(root)
    assert (root / "probe/diagnostics.json").is_file(), "Run the paired probe before ASR training"
    assembly = read(root / "assembly.json")
    assert sha256_file(root / arm / "train.jsonl") == assembly["train_sha256"][arm]
    for path, digest in assembly["selected_sha256"].items():
        assert sha256_file(Path(path)) == digest
        item = read(Path(path))
        validate_selected(item["pair"], item)
    assert control.training_command(root, arm, seed) == cfg["commands"][arm][str(seed)]
    control.train(root, arm, seed)
    new = read(root / arm / f"seed_{seed}/result.json")
    prior = read(SOURCE / "patient_tts" / f"seed_{seed}/result.json")
    lr = lambda r: [(x["step"], x["learning_rate"]) for x in r["train_log"] if "learning_rate" in x]
    assert lr(new) == lr(prior)


def report(root):
    verify(root)
    from report_tts_increment_control import edit_distance, text_normalize
    from therapist_harness.evidence import evaluate_snapshot
    locations = {"r_only": (R_ONLY, "r_only"), "old_fixed": (SOURCE, "patient_tts"),
                 **{a: (root, a) for a in ARMS}}
    scope = rows(TEST)
    results, patients = {}, {}
    for label, (location, arm) in locations.items():
        results[label], patients[label] = [], []
        for seed in SEEDS:
            d = location / arm / f"seed_{seed}"
            result = read(d / "result.json")
            assert (result["arm"], result["seed"], result["step"]) == (arm, seed, 50)
            predicted = rows(d / "evaluation/merged_step_50/predictions.jsonl")
            assert len(predicted) == len(scope) == 511
            per_patient = defaultdict(list)
            independent = defaultdict(list)
            for p, truth in zip(predicted, scope):
                assert all(p[k] == truth[k] for k in ("utt_id", "speaker_id", "clean_gt", "zero_shot_bucket"))
                ref = text_normalize(p["clean_gt"])
                edits = edit_distance(ref, text_normalize(p["prediction"]))
                per_patient[p["speaker_id"]].append(edits / len(ref))
                for g in ("overall", p["zero_shot_bucket"]):
                    independent[g].append((edits, len(ref)))
            calculated = evaluate_snapshot(predicted, prediction_column="prediction", critical_cer_threshold=.5).model_dump(mode="json")
            for group in GROUPS:
                values = independent[group]
                assert math.isclose(statistics.mean(e / n for e, n in values),
                                    result["metrics"][group]["mean_utterance_cer"], abs_tol=1e-12)
                assert sum(e for e, n in values) == result["metrics"][group]["total_edits"]
                assert sum(n for e, n in values) == result["metrics"][group]["total_reference_characters"]
                for metric in ("mean_utterance_cer", "pooled_cer", "patient_macro_cer"):
                    assert math.isclose(calculated[group][metric], result["metrics"][group][metric], abs_tol=1e-12)
            assert math.isclose(statistics.mean(v for values in per_patient.values() for v in values),
                                calculated["overall"]["mean_utterance_cer"], abs_tol=1e-12)
            results[label].append(result)
            patients[label].append({k: statistics.mean(v) for k, v in per_patient.items()})
            if label in ARMS:
                execution = read(d / "execution.json")
                assert execution["actual_stop_step"] == 50
                assert sha256_file(location / arm / "train.jsonl") == execution["train_sha256"]
                prior = read(SOURCE / "patient_tts" / f"seed_{seed}/result.json")
                lr = lambda r: [(x["step"], x["learning_rate"]) for x in r["train_log"] if "learning_rate" in x]
                assert lr(result) == lr(prior)
    means = {a: {g: statistics.mean(r["metrics"][g]["mean_utterance_cer"] for r in rs) for g in GROUPS} for a, rs in results.items()}
    comparisons = {}
    for a, b in [(a, b) for a in ARMS for b in ("r_only", "old_fixed")] + [("hard_multi", "random_multi")]:
        comparison = {}
        for g in GROUPS:
            deltas = [100 * (x["metrics"][g]["mean_utterance_cer"] - y["metrics"][g]["mean_utterance_cer"]) for x, y in zip(results[a], results[b])]
            comparison[g] = {"seed_deltas_pp": deltas, "mean_delta_pp": statistics.mean(deltas), "improved_seeds": sum(d < 0 for d in deltas)}
        comparison["per_patient_seed_deltas_pp"] = {p: [100 * (x[p] - y[p]) for x, y in zip(patients[a], patients[b])] for p in patients[a][0]}
        comparisons[f"{a}_minus_{b}"] = comparison
    save(root / "results.json", {"results": results, "means": means, "comparisons": comparisons})
    lines = ["# Reference 策略的 ASR 效用", "", "固定 step50，三个 ASR seed；固定 reference 清单。既有 merged511 探索性评测。CER 单位 %。", "",
        "| 组别 | Overall | Easy | Medium | Hard |", "|---|---:|---:|---:|---:|"]
    for a in means:
        lines.append(f"| {a} | " + " | ".join(f"{100*means[a][g]:.4f}" for g in GROUPS) + " |")
    lines += ["", "差值为百分点，负值为改善；先比较 R-only，再比较旧合成与随机 reference。", ""]
    for name, comparison in comparisons.items():
        lines.append(f"- {name}：" + "；".join(f"{g} {comparison[g]['mean_delta_pp']:+.4f}（{comparison[g]['improved_seeds']}/3 改善）" for g in GROUPS))
    lines += ["", "逐 seed 原始指标、patient macro、pooled CER 与逐患者配对差值见 results.json。三 seed 不是三套 reference，也不单独表示统计显著。"]
    write_immutable_bytes(root / "report.zh.md", ("\n".join(lines) + "\n").encode())
    save(root / "acceptance.json", {"status": "verified", "new_runs": 6, "reused_runs": 6,
        "rows_per_evaluation": 511, "independent_edit_distance_verification": "12 runs x 4 groups",
        "results_sha256": sha256_file(root / "results.json")})
    print("\n".join(lines), flush=True)


def child(root, stage, *extra):
    command = [sys.executable, "-u", str(Path(__file__)), "--root", str(root), "--stage", stage, *extra]
    log_path = root / f"{stage}_{time.time_ns()}.log"
    print(f"[stage] {stage} {' '.join(extra)}; log={log_path}", flush=True)
    with log_path.open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def probe(root):
    child(root, "synthesize", "--probe-only")
    if not (root / "probe/diagnostics.json").exists():
        child(root, "diagnose")


def run(root):
    verify(root)
    probe(root)
    child(root, "synthesize")
    if not (root / "assembly.json").exists():
        child(root, "assemble")
    for seed in SEEDS:
        for arm in ARMS:
            if not (root / arm / f"seed_{seed}/result.json").exists():
                child(root, "train", "--arm", arm, "--seed", str(seed))
    if not (root / "acceptance.json").exists():
        child(root, "report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage", required=True, choices=("prepare", "probe", "synthesize", "diagnose", "assemble", "train", "report", "run"))
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.stage == "train" and (args.arm is None or args.seed is None):
        parser.error("train requires --arm and --seed")
    if args.probe_only and args.stage != "synthesize":
        parser.error("--probe-only is only valid for synthesize")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    def status(value, **extra):
        temporary = root / f".status_{os.getpid()}.json"
        temporary.write_text(json.dumps({"status": value, "stage": args.stage,
            "arm": args.arm, "seed": args.seed, "pid": os.getpid(),
            "time_unix": time.time(), **extra}, ensure_ascii=False) + "\n")
        temporary.replace(root / "run_status.json")
    status("running")
    try:
        if args.stage == "synthesize":
            synthesize(root, args.probe_only)
        elif args.stage == "train":
            train(root, args.arm, args.seed)
        else:
            globals()[args.stage](root)
        status("prepared" if args.stage == "prepare" else "complete")
    except Exception as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
