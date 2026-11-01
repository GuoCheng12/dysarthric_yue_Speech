"""48 paired utterances: localize TTS differences without ASR training."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from therapist_harness.store import sha256_file, write_immutable_json, write_immutable_bytes
from therapist_harness.text import edit_distance, text_normalize

SOURCE = Path("/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_increment_positive_control_v2")
ROOT = SOURCE.parent / "tts_reconstruction_probe48_v1"
ADAPTER = SOURCE / "replay/seed_42/training/checkpoint-50"
VARIANTS = ("real", "patient_tts", "base_tts", "token_reconstruction", "vocoder_reconstruction")


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def select48(q):
    rng = random.Random(20260906)
    selected = []
    for bucket in ("easy", "medium", "hard"):
        groups = defaultdict(list)
        for row in q:
            if row["zero_shot_bucket"] == bucket:
                groups[row["speaker_id"]].append(row)
        speakers = sorted(groups)
        rng.shuffle(speakers)
        for values in groups.values():
            rng.shuffle(values)
        chosen = []
        while len(chosen) < 16:
            for speaker in speakers:
                if groups[speaker] and len(chosen) < 16:
                    chosen.append(groups[speaker].pop())
        selected.extend(chosen)
    assert len({x["utt_id"] for x in selected}) == 48
    return selected


def prepare():
    assert not ROOT.exists()
    source = read(SOURCE / "selection.json")
    selected = select48(source["Q"])
    pairs = {p["target_utt_id"]: p for p in read(SOURCE / "pairs.json")}
    retained = {x["audio"] for x in source["R"]}
    manifest = []
    for row in selected:
        pair = pairs[row["utt_id"]]
        assert pair["reference_wav"] in retained
        record = read(SOURCE / "selected" / f"{pair['sample_id']}.json")
        audio = {"real": row["audio"], **{a: record["records"][a]["output_wav"] for a in ("patient_tts", "base_tts")}}
        manifest.append({"row": row, "pair": pair, "seed": record["records"]["patient_tts"]["seed"],
                         "audio": audio, "input_sha256": {a: sha256_file(Path(p)) for a, p in audio.items()}})
    write_immutable_json(ROOT / "selection.json", manifest)
    write_immutable_json(ROOT / "protocol.json", {
        "purpose": "post-hoc reconstruction diagnosis; no ASR training, filtering or utility gate",
        "rows": 48, "difficulty": dict(Counter(x["row"]["zero_shot_bucket"] for x in manifest)),
        "patients": len({x["row"]["speaker_id"] for x in manifest}),
        "selection_seed": 20260906, "asr_adapter": str(ADAPTER), "variants": VARIANTS,
        "conditioning": "same aligned frontend_zero_shot reference; inject Q source tokens, forbid LLM inference",
        "vocoder": "full real mel from official frontend, no flow; no waveform trimming",
        "acoustics": "10 ms relative-energy silence proxy; WORLD voiced F0 summaries and normalized-time coarse contour, not syllable-level tone diagnosis",
        "source_sha256": sha256_file(Path(__file__)), "source_selection_sha256": sha256_file(SOURCE / "selection.json")})
    print(json.dumps(read(ROOT / "protocol.json"), ensure_ascii=False), flush=True)


def synthesize():
    import numpy as np
    import soundfile as sf
    import torch
    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts
    model, meta = tts.load_fixed_runtime()
    def forbidden(*args, **kwargs):
        raise AssertionError("LLM must not run in either reconstruction condition")
    model.model.llm.inference = forbidden
    model.model.llm.inference_bistream = forbidden
    write_immutable_json(ROOT / "runtime.json", meta)
    for index, item in enumerate(read(ROOT / "selection.json")):
        pair = item["pair"]
        for a, path in item["audio"].items():
            assert sha256_file(Path(path)) == item["input_sha256"][a]
        assert sha256_file(Path(pair["reference_wav"])) == pair["reference_wav_sha256"]
        folder = ROOT / "audio" / pair["sample_id"]
        folder.mkdir(parents=True)
        with torch.inference_mode():
            inputs = model.frontend.frontend_zero_shot(pair["target_text"],
                tts.PROMPT_INSTRUCTION + pair["reference_text"], pair["reference_wav"], model.sample_rate, "")
            token, token_len = model.frontend._extract_speech_token(item["audio"]["real"])
            inputs.update(source_speech_token=token, source_speech_token_len=token_len)
            tts._set_seed(item["seed"])
            chunks = [x["tts_speech"].detach().cpu() for x in model.model.tts(**inputs, stream=False)]
            token_wave = torch.cat(chunks, dim=-1).flatten().float().numpy()
            mel, mel_len = model.frontend._extract_speech_feat(item["audio"]["real"])
            tts._set_seed(item["seed"])
            vocoder_wave, _ = model.model.hift.inference(speech_feat=mel.transpose(1, 2), finalize=True)
            vocoder_wave = vocoder_wave.detach().cpu().flatten().float().numpy()
        generated = {}
        for name, wave in (("token_reconstruction", token_wave), ("vocoder_reconstruction", vocoder_wave)):
            assert len(wave) > 0 and np.isfinite(wave).all()
            path = folder / f"{name}.wav"
            assert not path.exists()
            sf.write(path, wave, model.sample_rate, subtype="FLOAT")
            generated[name] = {"audio": str(path), "sha256": sha256_file(path),
                "seconds": len(wave) / model.sample_rate, "peak": float(np.max(np.abs(wave))),
                "clipping_fraction": float(np.mean(np.abs(wave) >= .999))}
        write_immutable_json(folder / "receipt.json", {"generated": generated, "seed": item["seed"],
            "source_token_count": int(token_len.item()), "real_mel_frames": int(mel_len.item()),
            "reference_token_count": int(inputs["flow_prompt_speech_token_len"].item()),
            "reference_mel_frames": int(inputs["prompt_speech_feat_len"].item()), "llm_bypassed": True})
        print(f"[reconstruction] {index+1}/48 {pair['sample_id']}", flush=True)


def assemble():
    outputs = []
    for item in read(ROOT / "selection.json"):
        receipt = read(ROOT / "audio" / item["pair"]["sample_id"] / "receipt.json")
        audio = dict(item["audio"])
        audio.update({a: x["audio"] for a, x in receipt["generated"].items()})
        for variant in VARIANTS:
            outputs.append({**item["row"], "source_utt_id": item["row"]["utt_id"],
                "utt_id": item["row"]["utt_id"] + "::" + variant, "variant": variant,
                "audio": audio[variant], "audio_sha256": sha256_file(Path(audio[variant]))})
    write_immutable_bytes(ROOT / "decode.jsonl", "".join(json.dumps(x, ensure_ascii=False)+"\n" for x in outputs).encode())


def acoustics():
    import numpy as np
    import pyworld
    import librosa
    output = []
    inputs = rows(ROOT / "decode.jsonl")
    for index, row in enumerate(inputs):
        y, sr = librosa.load(row["audio"], sr=16000, mono=True)
        rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160, center=False)[0]
        threshold = max(float(np.quantile(rms, .95)) * 10**(-35/20), 1e-7)
        quiet = rms < threshold
        f0, t = pyworld.dio(y.astype(np.float64), sr, f0_floor=60, f0_ceil=500, frame_period=10)
        f0 = pyworld.stonemask(y.astype(np.float64), f0, t, sr)
        voiced = f0 > 0
        values = f0[voiced]
        bins = []
        for segment in np.array_split(f0, 100):
            v = segment[segment > 0]
            bins.append(float(np.median(v)) if len(v) else None)
        output.append({"utt_id": row["utt_id"], "source_utt_id": row["source_utt_id"], "variant": row["variant"],
            "seconds": len(y)/sr, "quiet_fraction": float(np.mean(quiet)), "voiced_fraction": float(np.mean(voiced)),
            "f0_median_hz": float(np.median(values)) if len(values) else None,
            "f0_iqr_semitones": float(12*np.log2(np.quantile(values,.75)/np.quantile(values,.25))) if len(values) else None,
            "f0_time_bins": bins})
        if (index+1) % 20 == 0:
            print(f"[acoustics] {index+1}/{len(inputs)}", flush=True)
    write_immutable_json(ROOT / "acoustics.json", output)


def summarize():
    preds = rows(ROOT / "asr/predictions.jsonl")
    acoustic = {x["utt_id"]: x for x in read(ROOT / "acoustics.json")}
    paired = defaultdict(dict)
    for row in preds:
        assert sha256_file(Path(row["audio"])) == row["audio_sha256"]
        ref, hyp = text_normalize(row["clean_gt"]), text_normalize(row["prediction"])
        assert row["edit_count"] == edit_distance(ref, hyp)
        paired[row["source_utt_id"]][row["variant"]] = row
    assert len(paired) == 48 and all(set(v) == set(VARIANTS) for v in paired.values())
    metrics = {}
    for variant in VARIANTS:
        metrics[variant] = {}
        for bucket in ("overall", "easy", "medium", "hard"):
            subset = [v[variant] for v in paired.values() if bucket == "overall" or v["real"]["zero_shot_bucket"] == bucket]
            metrics[variant][bucket] = {"n": len(subset), "cer": statistics.mean(x["cer"] for x in subset),
                "pooled_cer": sum(x["edit_count"] for x in subset)/sum(x["normalized_reference_characters"] for x in subset)}
    comparisons = {}
    for variant in VARIANTS[1:]:
        states = Counter()
        features = defaultdict(list)
        for item in paired.values():
            real, other = item["real"], item[variant]
            states[f"real_{'correct' if real['edit_count']==0 else 'wrong'}__other_{'correct' if other['edit_count']==0 else 'wrong'}"] += 1
            a, b = acoustic[real["utt_id"]], acoustic[other["utt_id"]]
            features["duration_ratio"].append(b["seconds"]/a["seconds"])
            features["abs_quiet_fraction_delta"].append(abs(b["quiet_fraction"]-a["quiet_fraction"]))
            if a["f0_median_hz"] and b["f0_median_hz"]:
                import math
                features["abs_f0_median_semitone_delta"].append(abs(12*math.log2(b["f0_median_hz"]/a["f0_median_hz"])))
            a_hyp, b_hyp = text_normalize(real["prediction"]), text_normalize(other["prediction"])
            features["prediction_distance_per_gt_char"].append(edit_distance(a_hyp,b_hyp)/real["normalized_reference_characters"])
        comparisons[variant] = {"recognition_states": dict(states), "median_features": {k: statistics.median(v) for k,v in features.items()}}
    write_immutable_json(ROOT / "results.json", {"metrics": metrics, "comparisons": comparisons, "paired_predictions": dict(paired)})
    lines = ["# 48 条配对重建诊断", "", "固定 replay seed42 ASR；没有新增训练。CER 是这 48 条诊断录音上的识别误差，不是 augmentation 后的效用。", "",
        "| 版本 | Overall CER | Easy | Medium | Hard |", "| --- | ---: | ---: | ---: | ---: |"]
    for v in VARIANTS:
        lines.append(f"| {v} | " + " | ".join(f"{metrics[v][g]['cer']*100:.2f}%" for g in ("overall","easy","medium","hard")) + " |")
    lines += ["", "## 全部配对转录", ""]
    for uid,item in paired.items():
        lines += [f"### {uid} ({item['real']['zero_shot_bucket']})", "", f"GT：{item['real']['clean_gt']}", ""]
        for v in VARIANTS:
            lines.append(f"- {v}：{item[v]['prediction']}（CER {item[v]['cer']*100:.1f}%）")
        lines.append("")
    write_immutable_bytes(ROOT / "report.zh.md", ("\n".join(lines)+"\n").encode())
    print(json.dumps({"metrics":metrics,"comparisons":comparisons}, ensure_ascii=False),flush=True)


def run():
    prepare()
    for stage in ("synthesize", "assemble", "decode", "acoustics", "summarize"):
        print(f"[stage] {stage}", flush=True)
        if stage == "decode":
            cmd = [sys.executable, str(REPO / "finetune/scripts/evaluate_asr_checkpoint.py"),
                "--base-model-path", "/data/qwen3-asr/models/Qwen3-ASR-1.7B", "--adapter-path", str(ADAPTER),
                "--split-jsonl", str(ROOT / "decode.jsonl"), "--out-dir", str(ROOT / "asr"),
                "--batch-size", "8", "--critical-cer-threshold", "0.5", "--max-new-tokens", "256"]
        else:
            cmd = [sys.executable, "-u", str(Path(__file__)), "--stage", stage]
        with (ROOT / f"{stage}.log").open("x") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("run","synthesize","assemble","acoustics","summarize"), default="run")
    globals()[p.parse_args().stage]()
