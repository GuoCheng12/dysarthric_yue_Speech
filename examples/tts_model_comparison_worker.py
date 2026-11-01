"""Use official inference paths; retain every first-seed result in full."""
import argparse
import importlib.metadata
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

from tts_model_comparison import ROOT, MODELS, SPECS, read, sha256_file, write_immutable_json

SOURCES = {
    "voxcpm2": Path("/data/qwen3-asr/third_party/VoxCPM2-probe"),
    "qwen3tts": Path("/data/qwen3-asr/third_party/Qwen3-TTS-probe"),
    "fish_s2": Path("/data/qwen3-asr/third_party/fish-speech-s2-probe"),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_wave(path, wave, sr, seconds, metadata):
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    assert len(wave) and np.isfinite(wave).all()
    assert not path.exists(), path
    sf.write(path, wave, sr, subtype="FLOAT")
    return {"audio": str(path), "sha256": sha256_file(path), "samples": len(wave), "sample_rate": sr,
            "seconds": len(wave)/sr, "wall_seconds": seconds, "peak": float(np.max(np.abs(wave))),
            "clipping_fraction": float(np.mean(np.abs(wave) >= .999)), **metadata}


def load_vox():
    sys.path.insert(0, str(SOURCES["voxcpm2"] / "src"))
    from voxcpm import VoxCPM
    from voxcpm.model.voxcpm2 import VoxCPM2Model
    from voxcpm.model.utils import next_and_close
    import librosa
    original_load = VoxCPM2Model.load_state_dict
    def exact_load(self, state_dict, strict=True, **kwargs):
        return original_load(self, state_dict, strict=True, **kwargs)
    VoxCPM2Model.load_state_dict = exact_load
    pipeline = VoxCPM(str(MODELS / "voxcpm2"), enable_denoiser=False, optimize=False, device="cuda")
    VoxCPM2Model.load_state_dict = original_load
    model = pipeline.tts_model
    model.eval()
    vae = model.audio_vae
    for item in read(ROOT / "selection.json"):
        for key in ("target_text", "reference_text"):
            tokens = model.text_tokenizer.tokenize(item["pair"][key])
            assert not any(len(t.replace("▁", "")) > 1 and all("\u4e00" <= c <= "\u9fff" for c in t.replace("▁", "")) for t in tokens)

    def generate(item):
        p = item["pair"]
        cache = model.build_prompt_cache(prompt_text=p["reference_text"], prompt_wav_path=p["reference_wav"],
                                         reference_wav_path=p["reference_wav"])
        wav, _, features = next_and_close(model._generate_with_prompt_cache(
            target_text=p["target_text"], prompt_cache=cache, max_len=384,
            inference_timesteps=10, cfg_value=2.0, retry_badcase=False,
            retry_badcase_ratio_threshold=1000, streaming=False, seed=item["seed"]))
        return wav.float().cpu().numpy(), model.sample_rate, {
            "generated_patches": int(features.shape[0]), "hit_length_limit": features.shape[0] >= 384}

    def reconstruct(item):
        wave, _ = librosa.load(item["audio"]["real"], sr=vae.sample_rate, mono=True)
        z = vae.encode(torch.from_numpy(wave).to("cuda")[None, None], vae.sample_rate)
        wave = vae.decode(z).float().cpu().numpy()
        return wave, vae.out_sample_rate, {"latent_shape": list(z.shape), "llm_bypassed": True}

    return generate, reconstruct, {"mode": "ref_continuation", "cfg": 2.0, "flow_steps": 10,
                                 "max_patches": 384, "denoising": False, "retry_badcase": False,
                                 "text_normalization": False, "compile": False}


def load_qwen():
    sys.path.insert(0, str(SOURCES["qwen3tts"]))
    from qwen_tts import Qwen3TTSModel
    pipeline = Qwen3TTSModel.from_pretrained(str(MODELS / "qwen3tts"),
                device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    pipeline.model.eval()
    codec = pipeline.model.speech_tokenizer
    observed = {}
    original_generate = pipeline.model.generate

    def observed_generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        observed["codes"] = int(result[0][0].shape[0])
        return result
    pipeline.model.generate = observed_generate

    def generate(item):
        p = item["pair"]
        wavs, sr = pipeline.generate_voice_clone(text=p["target_text"], language="Auto",
            ref_audio=p["reference_wav"], ref_text=p["reference_text"], x_vector_only_mode=False,
            max_new_tokens=768)
        assert len(wavs) == 1
        return wavs[0], sr, {"generated_frames": observed["codes"], "hit_length_limit": observed["codes"] >= 767}

    def reconstruct(item):
        encoded = codec.encode(item["audio"]["real"])
        wavs, sr = codec.decode(encoded)
        return wavs[0], sr, {"code_shape": list(encoded.audio_codes[0].shape), "llm_bypassed": True}

    return generate, reconstruct, {"mode": "ICL", "language": "Auto", "x_vector_only_mode": False,
                                  "max_new_tokens": 768, "attention": "sdpa",
                                  "generation_defaults": pipeline.generate_defaults}


def load_fish():
    sys.path.insert(0, str(SOURCES["fish_s2"]))
    import fish_speech.models.text2semantic.inference as inference
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(SOURCES["fish_s2"] / "fish_speech/configs/modded_dac_vq.yaml")
    codec = instantiate(cfg)
    state = torch.load(MODELS / "fish_s2/codec.pth", map_location="cpu", weights_only=True, mmap=True)
    # Official inference regenerates these six non-persistent RoPE/mask buffers.
    # Require every learned weight; do not silently tolerate any other mismatch.
    cache_keys = {f"{module}.{buffer}" for module in
                  ("encoder.block.4.block.5", "quantizer.pre_module", "quantizer.post_module")
                  for buffer in ("freqs_cis", "causal_mask")}
    loaded = codec.load_state_dict(state, strict=False, assign=True)
    assert not loaded.missing_keys
    assert set(loaded.unexpected_keys) == cache_keys
    codec.eval().to(device="cuda", dtype=torch.bfloat16)
    model, decode = inference.init_model(MODELS / "fish_s2", "cuda", torch.bfloat16, compile=False)
    with torch.device("cuda"):
        model.setup_caches(max_batch_size=1, max_seq_len=model.config.max_seq_len,
                           dtype=next(model.parameters()).dtype)
    eos = model.tokenizer.get_token_id(inference.IM_END_TOKEN)
    observed = {}
    original_generate = inference.generate

    def observed_generate(*args, **kwargs):
        value = original_generate(*args, **kwargs)
        observed["ended_with_eos"] = bool(value[0, -1].item() == eos)
        return value
    inference.generate = observed_generate

    def generate(item):
        p = item["pair"]
        reference = inference.encode_audio(p["reference_wav"], codec, "cuda")
        set_seed(item["seed"])
        chunks = [x.codes for x in inference.generate_long(model=model, device="cuda", decode_one_token=decode,
            text="<|speaker:0|>" + p["target_text"], prompt_text=[p["reference_text"]],
            prompt_tokens=[reference], num_samples=1, max_new_tokens=1323,
            temperature=1.0, top_p=.9, top_k=30, compile=False) if x.action == "sample"]
        assert len(chunks) == 1
        codes = chunks[0]
        wave = inference.decode_to_audio(codes, codec).float().cpu().numpy()
        return wave, codec.sample_rate, {"generated_frames": int(codes.shape[-1]),
                    "ended_with_eos": observed["ended_with_eos"], "hit_length_limit": not observed["ended_with_eos"]}

    def reconstruct(item):
        codes = inference.encode_audio(item["audio"]["real"], codec, "cuda")
        wave = inference.decode_to_audio(codes, codec).float().cpu().numpy()
        return wave, codec.sample_rate, {"code_shape": list(codes.shape), "llm_bypassed": True}

    return generate, reconstruct, {"mode": "reference-conditioned", "temperature": 1.0, "top_p": .9,
                                   "top_k": 30, "max_new_tokens": 1323, "compile": False,
                                   "regenerated_nonpersistent_buffers": sorted(cache_keys)}


def run(name, limit):
    torch.set_num_threads(4)
    assert (ROOT / f"download_{name}.json").exists()
    generate, reconstruct, config = {"voxcpm2": load_vox, "qwen3tts": load_qwen, "fish_s2": load_fish}[name]()
    runtime = {"model": list(SPECS[name]), "config": config,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCES[name], text=True).strip(),
        "versions": {x: importlib.metadata.version(x) for x in ("torch", "torchaudio", "transformers", "numpy")}}
    runtime_path = ROOT / f"runtime_{name}.json"
    if runtime_path.exists():
        assert read(runtime_path) == runtime
    else:
        write_immutable_json(runtime_path, runtime)
    for i, item in enumerate(read(ROOT / "selection.json")[:limit]):
        pair = item["pair"]
        folder = ROOT / "audio" / name / pair["sample_id"]
        folder.mkdir(parents=True, exist_ok=True)
        assert sha256_file(Path(pair["reference_wav"])) == pair["reference_wav_sha256"]
        assert sha256_file(Path(item["audio"]["real"])) == item["input_sha256"]["real"]
        for condition, function in (("reconstruction", reconstruct), ("tts", generate)):
            receipt_path = folder / f"{condition}.json"
            if receipt_path.exists():
                receipt = read(receipt_path)
                assert sha256_file(Path(receipt["audio"])) == receipt["sha256"]
                continue
            set_seed(item["seed"])
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode():
                wave, sr, details = function(item)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            receipt = save_wave(folder / f"{condition}.wav", wave, sr, elapsed, {
                "seed": item["seed"], "sample_id": pair["sample_id"], "condition": condition,
                "input_reference_sha256": pair["reference_wav_sha256"],
                "worker_sha256": sha256_file(Path(__file__)), **details})
            write_immutable_json(receipt_path, receipt)
            print(f"[{name}] {i+1}/{limit} {pair['sample_id']} {condition} "
                  f"audio={receipt['seconds']:.2f}s wall={elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=SPECS)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    run(args.model, args.limit)
