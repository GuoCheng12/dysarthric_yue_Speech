"""ASR and acoustic diagnostics plus a full-waveform listening report."""
import argparse
from collections import defaultdict
from html import escape
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

from tts_model_comparison import ROOT, PREVIOUS, REPO, SPECS, read, rows, sha256_file, write_immutable_json
from therapist_harness.store import write_immutable_bytes
from therapist_harness.text import text_normalize, edit_distance
from therapist_harness.evidence import _stable_align

OLD = ("real", "patient_tts", "base_tts", "token_reconstruction", "vocoder_reconstruction")
NEW = tuple(f"{m}_{c}" for m in SPECS for c in ("tts", "reconstruction"))
VARIANTS = OLD + NEW
LABEL = {"real": "患者原声", "patient_tts": "当前 Speech Generator（已微调）", "base_tts": "原始 CosyVoice3",
         "token_reconstruction": "CosyVoice3 token＋Flow 重建", "vocoder_reconstruction": "CosyVoice3 mel＋vocoder 重建"}
LABEL.update({f"{m}_{c}": f"{m} · {'零样本克隆' if c=='tts' else 'codec/VAE 重建'}"
              for m in SPECS for c in ("tts", "reconstruction")})


def error_retention(pairs, variants):
    """Same unambiguous rows and real-error denominator for every compared model."""
    counts = {v: {"retained": 0, "new": 0} for v in variants}
    selected = []
    denominator = 0
    for uid, pair in pairs.items():
        gt = text_normalize(pair["real"]["clean_gt"])
        aligned = {v: _stable_align(gt, text_normalize(pair[v]["prediction"])) for v in ("real", *variants)}
        if any(a[2] for a in aligned.values()):
            continue
        selected.append(uid)
        signatures = {v: {(e.operation, e.reference_index, e.reference_unit, e.hypothesis_unit)
                          for e in a[1] if e.reference_index is not None} for v, a in aligned.items()}
        denominator += len(signatures["real"])
        for v in variants:
            counts[v]["retained"] += len(signatures[v] & signatures["real"])
            counts[v]["new"] += len(signatures[v] - signatures["real"])
    return {"rows": len(selected), "utt_ids": selected, "real_SD_errors": denominator, "counts": counts}


def model_rows(model):
    output = []
    for item in read(ROOT / "selection.json"):
        for condition in ("tts", "reconstruction"):
            receipt = read(ROOT / "audio" / model / item["pair"]["sample_id"] / f"{condition}.json")
            assert sha256_file(Path(receipt["audio"])) == receipt["sha256"]
            assert receipt["input_reference_sha256"] == item["pair"]["reference_wav_sha256"]
            variant = f"{model}_{condition}"
            output.append({**item["row"], "source_utt_id": item["row"]["utt_id"],
                "utt_id": item["row"]["utt_id"]+"::"+variant, "variant": variant,
                "audio": receipt["audio"], "audio_sha256": receipt["sha256"]})
    assert len(output) == 48
    return output


def assemble_model(model):
    output = model_rows(model)
    write_immutable_bytes(ROOT / f"{model}_decode.jsonl", "".join(json.dumps(x, ensure_ascii=False)+"\n" for x in output).encode())


def assemble():
    output = [row for model in SPECS for row in model_rows(model)]
    assert len(output) == 144
    write_immutable_bytes(ROOT / "decode.jsonl", "".join(json.dumps(x, ensure_ascii=False)+"\n" for x in output).encode())


def acoustics():
    # Identical diagnostic implementation to the previous experiment, in this process only.
    import tts_reconstruction_probe as previous
    previous.ROOT = ROOT
    previous.acoustics()


def summarize():
    import librosa
    import numpy as np
    import onnxruntime as ort
    import torch
    import torchaudio.compliance.kaldi as kaldi
    torch.set_num_threads(4)
    selection = read(ROOT / "selection.json")
    selected_uids = {x["row"]["utt_id"] for x in selection}
    from tts_reconstruction_probe import SOURCE
    replay_audio = {x["audio"] for x in rows(SOURCE / "replay/train.jsonl")}
    assert not replay_audio & {x["audio"]["real"] for x in selection}
    predictions = [r for r in rows(PREVIOUS / "asr/predictions.jsonl") if r["source_utt_id"] in selected_uids]
    for model in SPECS:
        decoded = rows(ROOT / f"asr_{model}/predictions.jsonl")
        assert len(decoded) == 48
        assert {x["utt_id"]:x["audio_sha256"] for x in decoded} == {x["utt_id"]:x["audio_sha256"] for x in model_rows(model)}
        predictions += decoded
    acoustic = {x["utt_id"]: x for x in read(PREVIOUS / "acoustics.json") + read(ROOT / "acoustics.json")}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    camp_path = Path("/data/qwen3-asr/models/tts/Fun-CosyVoice3-0.5B-2512/campplus.onnx")
    camp = ort.InferenceSession(str(camp_path), sess_options=options, providers=["CPUExecutionProvider"])
    embeddings = {}

    def embedding(path):
        if path not in embeddings:
            y, _ = librosa.load(path, sr=16000, mono=True)
            feat = kaldi.fbank(torch.from_numpy(y)[None], num_mel_bins=80, dither=0, sample_frequency=16000)
            feat -= feat.mean(dim=0, keepdim=True)
            value = camp.run(None, {camp.get_inputs()[0].name: feat[None].numpy()})[0].flatten()
            embeddings[path] = value / np.linalg.norm(value)
        return embeddings[path]

    pairs = defaultdict(dict)
    for i, row in enumerate(predictions):
        assert sha256_file(Path(row["audio"])) == row["audio_sha256"]
        ref, hyp = text_normalize(row["clean_gt"]), text_normalize(row["prediction"])
        assert edit_distance(ref, hyp) == row["edit_count"]
        pairs[row["source_utt_id"]][row["variant"]] = row
        embedding(row["audio"])
        if (i+1) % 24 == 0:
            print(f"[analysis] {i+1}/{len(predictions)}", flush=True)
    assert len(pairs) == 24 and all(set(p) == set(VARIANTS) for p in pairs.values())
    metrics, comparison = {}, {}
    for v in VARIANTS:
        metrics[v] = {}
        for bucket in ("overall", "easy", "medium", "hard"):
            subset = [p[v] for p in pairs.values() if bucket == "overall" or p["real"]["zero_shot_bucket"] == bucket]
            metrics[v][bucket] = {"n": len(subset), "mean_utterance_cer": statistics.mean(r["cer"] for r in subset),
                "corpus_cer": sum(r["edit_count"] for r in subset)/sum(r["normalized_reference_characters"] for r in subset),
                "exact_sentences": sum(r["edit_count"] == 0 for r in subset)}
        sim, durations, pitch = [], [], []
        for pair in pairs.values():
            a, b = pair["real"], pair[v]
            sim.append(float(np.dot(embedding(a["audio"]), embedding(b["audio"]))))
            aa, bb = acoustic[a["utt_id"]], acoustic[b["utt_id"]]
            durations.append(bb["seconds"]/aa["seconds"])
            valid = [(12*math.log2(x), 12*math.log2(y)) for x,y in zip(aa["f0_time_bins"], bb["f0_time_bins"])
                     if x is not None and y is not None]
            if len(valid) >= 20:
                x,y = zip(*valid)
                mx,my = statistics.median(x), statistics.median(y)
                pitch.append(statistics.mean(abs((u-mx)-(w-my)) for u,w in valid))
        comparison[v] = {"median_campplus_to_real": statistics.median(sim), "median_duration_ratio": statistics.median(durations),
                         "pitch_shape_n": len(pitch), "median_centered_pitch_shape_semitones": statistics.median(pitch) if pitch else None}
    receipts = [read(p) for p in sorted((ROOT / "audio").glob("*/*/*.json"))]
    write_immutable_json(ROOT / "results.json", {"metrics": metrics, "comparison": comparison,
        "error_retention": {
            "synthesis": error_retention(pairs, ("patient_tts", "voxcpm2_tts", "qwen3tts_tts", "fish_s2_tts")),
            "reconstruction": error_retention(pairs, ("token_reconstruction", "vocoder_reconstruction",
                                "voxcpm2_reconstruction", "qwen3tts_reconstruction", "fish_s2_reconstruction"))},
        "paired_predictions": pairs, "campplus_sha256": sha256_file(camp_path),
        "acceptance": {"selected_rows":24, "audio_versions":len(VARIANTS), "new_audio":len(receipts),
                       "new_decodes":144, "new_trainings":0,
                       "real_q_unseen_by_replay_asr": True,
                       "length_limit_hits":[r["audio"] for r in receipts if r.get("hit_length_limit")],
                       "free_bytes":shutil.disk_usage(ROOT).free},
        "timing": {model:{c:{"wall_seconds":sum(r["wall_seconds"] for r in receipts if f"/{model}/" in r["audio"] and r["condition"]==c),
                             "audio_seconds":sum(r["seconds"] for r in receipts if f"/{model}/" in r["audio"] and r["condition"]==c)}
                          for c in ("tts","reconstruction")} for model in SPECS}})


def report():
    result = read(ROOT / "results.json")
    metric, comparison = result["metrics"], result["comparison"]
    lines = ["# 24 条患者语音：TTS 模型配对诊断", "",
        "19位患者，easy/medium/hard各8条。固定相同Q、reference、GT与seed；新模型均为zero-shot，Speech Generator为已微调模型。没有新增训练。",
        "", "训练暴露核对：24条Q均在Speech Generator的TTS训练wav.scp中；它们均未进入本轮冻结replay ASR的训练。旧Speech Generator是诊断参照，不是与新模型同适配条件的zero-shot对手。",
        "", "CER为同一冻结ASR对诊断录音的回译，不是增强训练收益。CAMPPlus只作音色辅助，不作病理指标。",
        "", "| 版本 | 逐句CER均值 | Corpus CER | Easy | Medium | Hard | CAMPPlus中位数 | 时长比中位数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for v in VARIANTS:
        m, c = metric[v], comparison[v]
        values = [m["overall"]["mean_utterance_cer"], m["overall"]["corpus_cer"]] + [m[b]["mean_utterance_cer"] for b in ("easy","medium","hard")]
        lines.append(f"| {LABEL[v]} | " + " | ".join(f"{x*100:.2f}%" for x in values) +
                     f" | {c['median_campplus_to_real']:.3f} | {c['median_duration_ratio']:.3f} |")
    lines += ["", "所有新wav均保留完整输出；只由官方接口分离reference前缀，没有额外裁剪或后处理。",
              "每种codec/VAE代表各自完整声学重建路径，不是相同压缩率的codec竞赛；CosyVoice token重建还包含同患者reference条件的Flow。",
              "", "验收：", "```json", json.dumps(result["acceptance"],ensure_ascii=False,indent=2), "```"]
    lines += ["", "原声ASR错误保留：只在各比较组共同无歧义对齐的句子上统计同位置、同替换字的S/D错误；这不是临床病理指标。",
              "```json", json.dumps(result["error_retention"],ensure_ascii=False,indent=2), "```"]
    write_immutable_bytes(ROOT / "report.zh.md", ("\n".join(lines)+"\n").encode())
    # A listening artifact, not an application: symlink wav files instead of copying or embedding them.
    listening = ROOT / "listening"
    (listening / "audio").mkdir(parents=True, exist_ok=True)

    def player(path):
        path = Path(path)
        name = sha256_file(path) + ".wav"
        link = listening / "audio" / name
        if not link.exists():
            link.symlink_to(path)
        return f'<audio controls preload="none" src="audio/{name}"></audio>'

    page = ['<!doctype html><meta charset="utf-8"><title>TTS 患者配对比较</title>',
        '<style>body{max-width:1200px;margin:30px auto;padding:20px;font:17px/1.6 system-ui;background:#f6f7f9;color:#222}article{background:white;border-radius:12px;padding:24px;margin:24px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.cell{padding:14px;background:#f5f5f5;border-radius:8px}audio{width:100%}h3{font-size:17px}small{color:#555}</style>',
        '<h1>患者原声与 TTS 模型对照</h1><p>24条 · 19位患者 · 同reference、GT、seed。新模型为zero-shot，Speech Generator已患者微调。全部播放完整wav，没有剪辑。</p>',
        '<p>训练暴露核对：这24条Q均在Speech Generator的TTS训练集中，但未进入冻结replay ASR的训练。旧Speech Generator仅作诊断参照。</p>',
        '<p>ASR为固定replay checkpoint；CER低不等同于患者构音保留好。重建版本用于区分声学重建与文本生成。</p>']
    synthesis = ("real", "patient_tts", "base_tts", "voxcpm2_tts", "qwen3tts_tts", "fish_s2_tts")
    for item in read(ROOT / "selection.json"):
        p = item["pair"]
        pair = result["paired_predictions"][item["row"]["utt_id"]]
        page += [f'<article><h2>{escape(p["speaker_id"])} · {escape(item["row"]["zero_shot_bucket"])} · {p["sample_id"]}</h2>',
                 f'<p>目标 GT：{escape(p["target_text"])}</p><p>Reference：{escape(p["reference_text"])}</p>',player(p["reference_wav"]),'<div class="grid">']
        for v in synthesis:
            row = pair[v]
            page += [f'<div class="cell"><h3>{LABEL[v]}</h3>{player(row["audio"])}<p>ASR：{escape(row["prediction"])}</p><small>CER {row["cer"]*100:.2f}%</small></div>']
        page += ['</div><details><summary>展开声学重建对照</summary><div class="grid">']
        for v in VARIANTS:
            if v in synthesis:
                continue
            row = pair[v]
            page += [f'<div class="cell"><h3>{LABEL[v]}</h3>{player(row["audio"])}<p>ASR：{escape(row["prediction"])}</p><small>CER {row["cer"]*100:.2f}%</small></div>']
        page += ['</div></details></article>']
    write_immutable_bytes(listening / "index.html", ("\n".join(page)+"\n").encode())
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("assemble","assemble-model","acoustics","summarize","report"))
    p.add_argument("--model", choices=SPECS)
    args = p.parse_args()
    if args.stage == "assemble-model":
        assert args.model is not None
        assemble_model(args.model)
    else:
        globals()[args.stage]()
