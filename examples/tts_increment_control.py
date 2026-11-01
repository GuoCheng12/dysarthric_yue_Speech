"""Matched real-increment positive control for patient-conditioned TTS utility."""
# ruff: noqa: E402
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "finetune/scripts")]
from therapist_harness.store import sha256_file, write_immutable_json, write_immutable_bytes

ROOT = Path("/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_increment_positive_control_v2")
REAL = Path("/data/qwen3-asr/finetune/data_setting_d_v1/setting_d_train.jsonl")
TEST = Path("/data/qwen3-asr/finetune/data_setting_d_merged_test_v1/setting_d_merged_test.jsonl")
REFS = ROOT.parent / "therapist_synthetic96_loop6_v1_references.json"
ARMS = ("replay", "real_q", "patient_tts", "base_tts")
SEEDS = (42, 43, 44)


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def jsonl(p, records):
    write_immutable_bytes(p, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode())


def select_split(real, refs, count=576):
    rng = random.Random(42)
    reserved = {r["reference_wav"] for r in refs}
    speakers = {r["speaker_id"] for r in refs}
    groups = defaultdict(list)
    for r in real:
        groups[(r["speaker_id"], r["zero_shot_bucket"])].append(r)
    # Retain a replay candidate in every source stratum, as well as all references.
    for values in groups.values():
        reserved.add(sorted(values, key=lambda r: r["utt_id"])[0]["audio"])
    eligible = {k: [r for r in v if r["audio"] not in reserved]
                for k, v in groups.items() if k[0] in speakers}
    eligible = {k: v for k, v in eligible.items() if v}
    total = sum(map(len, eligible.values()))
    assert total >= count
    allocation = {k: len(v) * count // total for k, v in eligible.items()}
    ranking = sorted(eligible, key=lambda k: (-(len(eligible[k]) * count % total), k))
    for k in ranking[:count - sum(allocation.values())]:
        allocation[k] += 1
    q = []
    for k in sorted(eligible):
        values = sorted(eligible[k], key=lambda r: r["utt_id"])
        rng.shuffle(values)
        q.extend(values[:allocation[k]])
    q.sort(key=lambda r: r["utt_id"])
    ids = {r["utt_id"] for r in q}
    retained = [r for r in real if r["utt_id"] not in ids]
    pools = defaultdict(list)
    for r in retained:
        pools[(r["speaker_id"], r["zero_shot_bucket"])].append(r)
    replay = [rng.choice(pools[(r["speaker_id"], r["zero_shot_bucket"])]) for r in q]
    assert len(q) == len(ids) == count
    assert not reserved.intersection(r["audio"] for r in q)
    return retained, q, replay


def stats(data):
    return {"rows": len(data), "patients": len({r["speaker_id"] for r in data}),
        "contents": len({r["prompt_id"] for r in data}),
        "unique_gt": len({r["clean_gt"] for r in data}),
        "difficulty": dict(Counter(r["zero_shot_bucket"] for r in data)),
        "seconds": sum(float(r["duration"]) for r in data)}


def prepare(root):
    assert not root.exists()
    real, test, refs = rows(REAL), rows(TEST), read(REFS)["references"]
    assert len(real) == 2117 and len(test) == 511 and len(refs) == 57
    r, q, replay = select_split(real, refs)
    refmap = {x["speaker_id"]: x for x in refs}
    for x in refs:
        assert sha256_file(Path(x["reference_wav"])) == x["reference_wav_sha256"]
    for key in ("utt_id", "prompt_id", "prompt_normalized_sha16", "prompt_jyutping_sha16"):
        assert not {x[key] for x in real}.intersection(x[key] for x in test)
    root.mkdir(parents=True)
    save = write_immutable_json
    save(root / "selection.json", {"R": r, "Q": q, "replay": replay, "references": refs})
    pairs = [{"index": i, "sample_id": f"q{i:04d}", "target_utt_id": x["utt_id"],
        "speaker_id": x["speaker_id"], "target_text": x["clean_gt"],
        "reference_text": refmap[x["speaker_id"]]["reference_text"],
        "reference_wav": refmap[x["speaker_id"]]["reference_wav"],
        "reference_wav_sha256": refmap[x["speaker_id"]]["reference_wav_sha256"],
        "seed": 42 + i} for i, x in enumerate(q)]
    save(root / "pairs.json", pairs)
    source_paths = [Path(__file__), *sorted((REPO / "finetune/scripts").glob("*.py")),
        REPO / "finetune/official/qwen3_asr_sft.py",
        REPO / "synthesis/cosyvoice3_patient_sft/synthesize_manifest.py"]
    save(root / "protocol.json", {"R": stats(r), "Q": stats(q), "replay": stats(replay),
        "arms": list(ARMS), "seeds": list(SEEDS), "initialization": "original_Qwen3_ASR_new_LoRA",
        "batch_size": 8, "gradient_accumulation": 4, "lr": 2e-4,
        "scheduler": "linear_140_step_horizon", "warmup_ratio": .02,
        "actual_steps": 50, "evaluation_steps": [50], "lora": {"rank": 16, "alpha": 4, "dropout": .05},
        "selection": "none; fixed step50 for all arms and all seeds",
        "tts_pairing": "same GT, same reference from R, same successful seed in both arms",
        "technical_rejection": "at most 10 paired attempts, seed=base+10000*attempt; both regenerated after either fails numerical/audio-shape validity; no ASR filtering",
        "sg_training_scope": "Existing generator trained on full2117 including Q; teacher-informed transfer, not unseen-Q TTS",
        "feedback_scope": "unchanged511 merged real, no synthetic evaluation items",
        "real_sha256": sha256_file(REAL), "test_sha256": sha256_file(TEST),
        "reference_sha256": sha256_file(REFS), "source_sha256": {str(p): sha256_file(p) for p in source_paths}})
    for arm, extra in (("replay", replay), ("real_q", q)):
        jsonl(root / arm / "train.jsonl", [{k: x[k] for k in ("audio", "text", "prompt")} for x in r + extra])
    print(json.dumps({"R": stats(r), "Q": stats(q), "replay": stats(replay)}, ensure_ascii=False), flush=True)


def synthesize(root, worker):
    import os
    import torch
    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts
    patient, patient_meta = tts.load_fixed_runtime()
    base, base_meta = tts.load_fixed_runtime()
    from peft.tuners.tuners_utils import BaseTunerLayer
    module = base.model.llm.llm.model
    layers = [(name, layer) for name, layer in module.named_modules() if isinstance(layer, BaseTunerLayer)]
    for name, layer in layers:
        parent, _, child = name.rpartition(".")
        setattr(module.get_submodule(parent), child, layer.get_base_layer())
    assert len(layers) == 96
    assert not any("lora_" in k for k in base.model.llm.state_dict())
    official = torch.load(tts.DEFAULT_MODEL_DIR / "llm.pt", map_location="cpu", weights_only=True)
    current = base.model.llm.state_dict()
    assert set(official) == set(current)
    assert all(torch.equal(current[k].detach().cpu(), v) for k, v in official.items())
    del official, current
    base_meta.update(loaded_lora_keys=0, lora=None, speech_generator_checkpoint=None,
        speech_generator_resolved_checkpoint=None, speech_generator_sha256=None,
        official_base_state_verified=True, removed_adapter_layers=len(layers))
    write_immutable_json(root / f"tts_worker_{worker}_runtime.json", {"patient_tts": patient_meta, "base_tts": base_meta})
    for pair in read(root / "pairs.json")[worker::2]:
        selected = None
        for attempt in range(10):
            folder = root / "synthesis" / pair["sample_id"] / f"attempt_{attempt}"
            folder.mkdir(parents=True)
            records, errors = {}, {}
            for arm, model in (("patient_tts", patient), ("base_tts", base)):
                sample = tts.ManifestSample(sample_id=pair["sample_id"], target_text=pair["target_text"],
                    reference_text=pair["reference_text"], reference_wav=Path(pair["reference_wav"]),
                    reference_wav_sha256=pair["reference_wav_sha256"], output_wav=folder / f"{arm}.wav",
                    seed=pair["seed"] + 10000 * attempt)
                try:
                    record, temporary = tts._stage_audio_once(sample, model)
                    os.link(temporary, sample.output_wav)
                    temporary.unlink()
                    records[arm] = record
                except RuntimeError as error:
                    if not str(error).startswith("Speech Generator returned "):
                        raise
                    errors[arm] = str(error)
            write_immutable_json(folder / "receipt.json", {"attempt": attempt,
                "seed": pair["seed"] + 10000 * attempt, "records": records, "errors": errors})
            if not errors:
                selected = {"pair": pair, "attempt": attempt, "records": records}
                break
        if selected is None:
            raise RuntimeError(f"Pair exhausted 10 matched seeds: {pair['sample_id']}")
        write_immutable_json(root / "selected" / f"{pair['sample_id']}.json", selected)
        print(f"[pair] worker={worker} index={pair['index']} complete attempt={selected['attempt']}", flush=True)


def assemble(root):
    selected = [read(p) for p in sorted((root / "selected").glob("*.json"))]
    assert len(selected) == 576
    retained = read(root / "selection.json")["R"]
    for arm in ("patient_tts", "base_tts"):
        extra = []
        for x in selected:
            s = x["records"][arm]
            assert s["seed"] == x["records"]["base_tts"]["seed"]
            assert sha256_file(Path(s["output_wav"])) == s["output_wav_sha256"]
            extra.append({"audio": s["output_wav"], "text": "language Cantonese<asr_text>" + s["target_text"], "prompt": ""})
        jsonl(root / arm / "train.jsonl", [{k: x[k] for k in ("audio", "text", "prompt")} for x in retained] + extra)
    all_rows = {arm: rows(root / arm / "train.jsonl") for arm in ARMS}
    assert all(len(v) == 2117 for v in all_rows.values())
    assert all(v[:1541] == all_rows["real_q"][:1541] for v in all_rows.values())
    assert [x["text"] for x in all_rows["patient_tts"]] == [x["text"] for x in all_rows["base_tts"]]
    # Preserve exact original Q labels, not a decoded/normalized substitute.
    assert [x["text"] for x in all_rows["patient_tts"]] == [x["text"] for x in all_rows["real_q"]]
    write_immutable_json(root / "assembly.json", {"rows_each": 2117, "shared_real_rows": 1541,
        "pairs": 576, "attempt_histogram": dict(Counter(x["attempt"] for x in selected)),
        "synthetic_seconds": {a: sum(x["records"][a]["duration_seconds"] for x in selected)
                              for a in ("patient_tts", "base_tts")},
        "train_sha256": {a: sha256_file(root / a / "train.jsonl") for a in ARMS}})


def training_command(root, arm, seed):
    d = root / arm / f"seed_{seed}"
    return [sys.executable, str(REPO / "finetune/scripts/qwen3_asr_lora_sft.py"),
        "--model_path", "/data/qwen3-asr/models/Qwen3-ASR-1.7B",
        "--train_file", str(root / arm / "train.jsonl"), "--eval_file", "",
        "--output_dir", str(d / "training"), "--batch_size", "8", "--grad_acc", "4",
        "--lr", "2e-4", "--lr_scheduler_type", "linear", "--warmup_ratio", "0.02",
        "--lora_rank", "16", "--lora_alpha", "4", "--lora_dropout", "0.05",
        "--lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--seed", str(seed), "--max_steps", "140", "--log_steps", "1", "--save_steps", "50",
        "--save_total_limit", "0", "--save_final_checkpoint", "0", "--report_to", "none"]


def train(root, arm, seed):
    import qwen3_asr_lora_sft as training
    from train_m0_small_dose_pair import Monitor
    cfg = read(root / "protocol.json")
    for path, digest in cfg["source_sha256"].items():
        assert sha256_file(Path(path)) == digest, path
    assert len(rows(root / arm / "train.jsonl")) == 2117
    d = root / arm / f"seed_{seed}"
    d.mkdir()
    cmd = training_command(root, arm, seed)
    parsed = training.parse_args(cmd[2:])
    assert not parsed.lora_init_adapter_path and not parsed.resume
    write_immutable_json(d / "execution.json", {"command": cmd, "actual_stop_step": 50,
        "train_sha256": sha256_file(root / arm / "train.jsonl"), "initialization": "base_new_adapter"})

    class FixedMonitor(Monitor):
        def on_step_end(self, args, state, control, **kwargs):
            control.should_save = state.global_step == 50
            control.should_training_stop = state.global_step >= 50
            return control

    monitor = FixedMonitor(d)
    monitor.end_step = 50
    monitor.state["endpoint_step"] = 50
    monitor.publish()

    class FreshTrainer(training.QwenASRLoRATrainer):
        def __init__(self, *args, **kwargs):
            kwargs["args"].save_only_model = True
            super().__init__(*args, **kwargs)
            self.add_callback(monitor)

    training.QwenASRLoRATrainer = FreshTrainer
    sys.argv = cmd[1:]
    training.main()
    assert [h["step"] for h in monitor.history] == [50]
    state = read(d / "training/checkpoint-50/trainer_state.json")
    assert state["global_step"] == 50
    write_immutable_json(d / "result.json", {"arm": arm, "seed": seed, "step": 50,
        "metrics": monitor.history[0]["metrics"], "train_log": state["log_history"]})
    monitor.state["status"] = "complete"
    monitor.publish()


def summarize(root):
    results = {arm: [read(root / arm / f"seed_{s}/result.json") for s in SEEDS] for arm in ARMS}
    means = {a: {g: sum(r["metrics"][g]["mean_utterance_cer"] for r in rs) / 3
                 for g in ("overall", "easy", "medium", "hard")} for a, rs in results.items()}
    deltas = {a: {g: means[a][g] - means["replay"][g] for g in means[a]} for a in ARMS}
    write_immutable_json(root / "results.json", {"status": "complete", "results": results,
        "mean_utterance_cer_seed_mean": means, "delta_vs_replay": deltas,
        "fixed_step": 50, "seeds": list(SEEDS)})
    print(json.dumps({"means": means, "deltas": deltas}, ensure_ascii=False), flush=True)


def run(root):
    stage = "synthesize"
    try:
        children = []
        handles = []
        for worker in (0, 1):
            log = (root / f"tts_worker_{worker}.log").open("x")
            handles.append(log)
            children.append(subprocess.Popen([sys.executable, "-u", str(Path(__file__)),
                "--root", str(root), "--stage", "synthesize", "--worker", str(worker)], stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in children]
        for handle in handles:
            handle.close()
        assert codes == [0, 0], codes
        stage = "assemble"
        assemble(root)
        for seed in SEEDS:
            for arm in ARMS:
                stage = f"train_{arm}_{seed}"
                print(f"[experiment] {stage}", flush=True)
                with (root / f"{stage}.log").open("x") as log:
                    subprocess.run([sys.executable, "-u", str(Path(__file__)), "--root", str(root),
                        "--stage", "train", "--arm", arm, "--seed", str(seed)], stdout=log,
                        stderr=subprocess.STDOUT, check=True)
        summarize(root)
    except Exception:
        write_immutable_json(root / "failure.json", {"stage": stage, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("prepare", "run", "synthesize", "assemble", "train", "summarize"), default="run")
    parser.add_argument("--worker", type=int, choices=(0, 1))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    args = parser.parse_args()
    if args.stage == "synthesize":
        synthesize(args.root, args.worker)
    elif args.stage == "train":
        train(args.root, args.arm, args.seed)
    else:
        globals()[args.stage](args.root)
