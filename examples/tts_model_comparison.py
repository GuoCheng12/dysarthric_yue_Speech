"""Fixed paired diagnostic, no ASR training and no synthesis-based filtering."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

from tts_reconstruction_probe import ROOT as PREVIOUS, REPO, ADAPTER, read, rows
from therapist_harness.store import sha256_file, write_immutable_json

ROOT = PREVIOUS.parent / "tts_model_comparison24_v1"
MODELS = Path("/data/qwen3-asr/models/tts/comparison24_v1")
MIN_FREE = 20 * 1024**3
SPECS = {
    "voxcpm2": ("openbmb/VoxCPM2", "32279effe8c19989596f05d353d1447f51d9e915"),
    "qwen3tts": ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "fd4b254389122332181a7c3db7f27e918eec64e3"),
    "fish_s2": ("fishaudio/s2-pro", "1de9996b6be38b745688de084d87a5633f714e4e"),
}


def prepare():
    source = read(PREVIOUS / "selection.json")
    # Fixed positions in the previous randomly selected, bucket-balanced manifest.
    selected = [x for bucket in ("easy", "medium", "hard")
                for x in [v for v in source if v["row"]["zero_shot_bucket"] == bucket][::2]]
    assert len(selected) == 24 and len({x["pair"]["sample_id"] for x in selected}) == 24
    write_immutable_json(ROOT / "selection.json", selected)
    write_immutable_json(ROOT / "protocol.json", {
        "rows": 24, "patients": len({x["row"]["speaker_id"] for x in selected}),
        "buckets": dict(Counter(x["row"]["zero_shot_bucket"] for x in selected)),
        "selection": "every other item within each frozen previous bucket; no new output inspection",
        "models": SPECS, "asr_adapter": str(ADAPTER),
        "conditions": "real, existing patient_tts, three fresh zero-shot TTS and their codec/VAE reconstructions",
        "inference": "full audio, paired original seed and reference, no denoising, no text rewriting, no retry selection",
        "scope": "generation/reconstruction diagnosis only; no new ASR or TTS training; Fish outputs not used for training",
        "storage": {"minimum_free_bytes": MIN_FREE, "initial_free_bytes": shutil.disk_usage(MODELS.parent).free,
                    "no_copies_of_existing_audio": True, "no_separate_qwen_tokenizer_download": True},
        "previous_selection_sha256": sha256_file(PREVIOUS / "selection.json")})
    print(read(ROOT / "protocol.json"), flush=True)


def download(model):
    import requests
    from huggingface_hub import snapshot_download
    repo, revision = SPECS[model]
    response = requests.get(f"https://huggingface.co/api/models/{repo}/revision/{revision}",
                            params={"blobs": "true"}, timeout=60)
    response.raise_for_status()
    info = response.json()
    files = [x for x in info["siblings"] if not x["rfilename"].endswith((".png", ".gitattributes"))]
    expected = sum(x["size"] for x in files)
    dest = MODELS / model
    remaining = sum(x["size"] for x in files if not (dest / x["rfilename"]).exists())
    free = shutil.disk_usage(MODELS.parent).free
    assert free - remaining > MIN_FREE, ("storage reserve", model, free, remaining)
    print(f"download {model}: {expected/1024**3:.2f} GiB, free {free/1024**3:.2f} GiB", flush=True)
    if remaining:
        snapshot_download(repo, revision=revision, local_dir=dest,
                          allow_patterns=[x["rfilename"] for x in files], max_workers=2)
    assert all((dest / x["rfilename"]).stat().st_size == x["size"] for x in files)
    verified = []
    for x in files:
        digest = sha256_file(dest / x["rfilename"])
        if "lfs" in x:
            assert digest == x["lfs"]["sha256"], x["rfilename"]
        verified.append({"path": x["rfilename"], "bytes": x["size"], "sha256": digest})
    write_immutable_json(ROOT / f"download_{model}.json", {
        "repository": repo, "revision": revision, "bytes": expected,
        "free_bytes_after": shutil.disk_usage(MODELS.parent).free,
        "files": verified})
    print(f"download verified: {model}", flush=True)


def resume_qwen_codec():
    """Explicit mirror transfer for the observed slow codec file; byte-identical HF revision."""
    import requests
    repo, revision = SPECS["qwen3tts"]
    r = requests.get(f"https://huggingface.co/api/models/{repo}",
                     params={"blobs":"true"}, timeout=60)
    r.raise_for_status()
    assert r.json()["sha"] == revision
    entry = next(x for x in r.json()["siblings"] if x["rfilename"] == "speech_tokenizer/model.safetensors")
    digest = entry["lfs"]["sha256"]
    dest = MODELS / "qwen3tts" / entry["rfilename"]
    candidates = list((MODELS / "qwen3tts/.cache/huggingface/download/speech_tokenizer").glob(f"*.{digest}.incomplete"))
    assert len(candidates) == 1 and not dest.exists()
    partial = candidates[0]
    offset = partial.stat().st_size
    url = "https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base/resolve/master/speech_tokenizer/model.safetensors"
    with partial.open("ab") as out:
        while out.tell() < entry["size"]:
            start = out.tell()
            end = min(start + 32*1024**2, entry["size"]) - 1
            with requests.get(url, headers={"Range":f"bytes={start}-{end}"}, stream=True, timeout=60) as response:
                assert response.status_code == 206, response.status_code
                assert response.headers["Content-Range"] == f"bytes {start}-{end}/{entry['size']}"
                for chunk in response.iter_content(1024**2):
                    out.write(chunk)
            out.flush()
            print(f"Qwen codec {out.tell()}/{entry['size']} bytes", flush=True)
    assert partial.stat().st_size == entry["size"]
    assert sha256_file(partial) == digest
    partial.rename(dest)
    write_immutable_json(ROOT / "qwen_codec_mirror_receipt.json", {
        "url":url,"hf_revision":revision,"sha256":digest,"resumed_bytes":offset,"total_bytes":entry["size"]})
    print("Qwen codec mirror transfer verified", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "download", "resume-qwen-codec"))
    p.add_argument("--model", choices=SPECS)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "download":
        download(args.model)
    else:
        resume_qwen_codec()
