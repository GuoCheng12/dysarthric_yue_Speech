#!/usr/bin/env python3
"""Materialize one frozen Therapist Harness manifest with Speech Generator.

This is the maintained CosyVoice3 inference boundary.  It intentionally has
no base-model mode, retry policy, reference substitution, or refill policy.
Every validated manifest row causes exactly one ``inference_zero_shot`` call.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthesis.cosyvoice3_patient_sft.wetext_local_cache import (  # noqa: E402
    DEFAULT_WETEXT_CACHE,
    bind_local_wetext_snapshot,
    wetext_installation_evidence,
)


DEFAULT_COSYVOICE_ROOT = Path("/data/qwen3-asr/third_party/CosyVoice")
DEFAULT_TRANSFORMERS_OVERLAY = Path(
    "/data/qwen3-asr/overlays/cosyvoice-transformers451"
)
DEFAULT_MODEL_DIR = Path("/data/qwen3-asr/models/tts/Fun-CosyVoice3-0.5B-2512")
SPEECH_GENERATOR_CHECKPOINT = Path(
    "/data/qwen3-asr/records/cosyvoice3_reference_sft_setting_d_v1/speech_generator.pt"
)
SPEECH_GENERATOR_SHA256 = (
    "5764178a90d234997b06dcc5fd74d530419c282b6959c089f4d09b0f3e6eaf0b"
)
BASE_LLM_SHA256 = "69f43bd545131c30e98947fb360ea8b4dc9916d8e83dded7757c7ea4f5a24970"
MODEL_FILE_SHA256 = {
    "llm.pt": BASE_LLM_SHA256,
    "cosyvoice3.yaml": "f5a6b2c6f05139d0f18861a1fe506f751e787026b77c05f7e8fef9f8a4405965",
    "flow.pt": "a6fab32a7825e5b0bc855ddd948f8db9370b0a786fbc249caa4595e95b608e4b",
    "hift.pt": "b279d7641eb97ae55b3b540cfba4f953c26492a2df758328a89a4d007ab87a65",
    "campplus.onnx": "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73",
    "speech_tokenizer_v3.onnx": "23236a74175dbdda47afc66dbadd5bcb41303c467a57c261cb8539ad9db9208d",
    "speech_tokenizer_v3.batch.onnx": "b156b8a7bbff436585e153f4637b9a368009005ac66efa108a6c8bfb34e5ee43",
    "CosyVoice-BlankEN/config.json": "168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185",
    "CosyVoice-BlankEN/generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "CosyVoice-BlankEN/merges.txt": "ac8ff86a72bee70828fbc1119bc4398c6f3a9a6e490d7b0dbe917be025478bd0",
    "CosyVoice-BlankEN/model.safetensors": "130282af0dfa9fe5840737cc49a0d339d06075f83c5a315c3372c9a0740d0b96",
    "CosyVoice-BlankEN/tokenizer_config.json": "482bd979881423375ca5414e4e0d94cd7c5349dbb17fffd46b4d36d71e62a1bc",
    "CosyVoice-BlankEN/vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
TRANSFORMERS_OVERLAY_SHA256 = (
    "acde01791d0c393de7734a92fc3cf2f900ea089767bd0f5e2bc010f5589acb5c"
)
COSYVOICE_SOURCE_SHA256 = (
    "2396a65ff94698e85c5624c8c5bc46bfa199e8542d2706c25e58b8281269f46e"
)
WETEXT_CRITICAL_CONTENT_SHA256 = (
    "1d9a5354d54fad3d65f0f92fd79d0fa297d420522a0497f4bc22a6f91d608ee4"
)
TRANSFORMERS_VERSION = "4.51.3"
PROMPT_INSTRUCTION = "You are a helpful assistant.<|endofprompt|>"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
REQUIRED_FIELDS = (
    "sample_id",
    "target_text",
    "reference_text",
    "reference_wav",
    "reference_wav_sha256",
    "output_wav",
    "seed",
)


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    target_text: str
    reference_text: str
    reference_wav: Path
    reference_wav_sha256: str
    output_wav: Path
    seed: int


@dataclass(frozen=True)
class FrozenManifest:
    path: Path
    sha256: str
    samples: tuple[ManifestSample, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: Path, *, exclude_wav: bool = False) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and (not exclude_wav or path.suffix.lower() != ".wav")
    )
    for path in files:
        if path.is_symlink():
            raise ValueError(f"runtime source tree contains a symlink: {path}")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _absolute_path(value: Any, field: str, line_number: int) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"line {line_number}: {field} must be a non-empty string")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"line {line_number}: {field} must be an absolute, traversal-free path"
        )
    return path


def _nonempty_text(value: Any, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {field} must be non-empty text")
    return value


def _read_stable_bytes(path: Path) -> bytes:
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(before, field) != getattr(after, field) for field in identity_fields
    ):
        raise RuntimeError(f"manifest changed while it was being read: {path}")
    return content


def load_frozen_manifest(path: Path) -> FrozenManifest:
    """Validate every row before a model can be constructed or called."""

    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("manifest path must be absolute and traversal-free")
    if not path.is_file():
        raise FileNotFoundError(f"manifest is not a regular file: {path}")
    content = _read_stable_bytes(path)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("manifest must be UTF-8") from error
    lines = text.splitlines()
    if not lines:
        raise ValueError("manifest must contain at least one row")

    samples: list[ManifestSample] = []
    seen_ids: set[str] = set()
    seen_outputs: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"line {line_number}: blank rows are not allowed")
        try:
            payload = json.loads(line, object_pairs_hook=_json_object)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"line {line_number}: invalid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number}: each JSONL row must be an object")
        missing = [field for field in REQUIRED_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"line {line_number}: missing required fields: {missing}")

        sample_id = _nonempty_text(payload["sample_id"], "sample_id", line_number)
        if sample_id != sample_id.strip():
            raise ValueError(f"line {line_number}: sample_id has outer whitespace")
        if sample_id in seen_ids:
            raise ValueError(f"line {line_number}: duplicate sample_id: {sample_id}")

        reference_wav = _absolute_path(
            payload["reference_wav"], "reference_wav", line_number
        )
        if not reference_wav.is_file():
            raise FileNotFoundError(
                f"line {line_number}: reference_wav does not exist: {reference_wav}"
            )
        reference_wav_sha256 = payload["reference_wav_sha256"]
        if (
            not isinstance(reference_wav_sha256, str)
            or len(reference_wav_sha256) != 64
            or any(char not in "0123456789abcdef" for char in reference_wav_sha256)
        ):
            raise ValueError(
                f"line {line_number}: reference_wav_sha256 must be lowercase SHA-256"
            )
        if sha256_file(reference_wav) != reference_wav_sha256:
            raise ValueError(f"line {line_number}: reference_wav SHA-256 mismatch")
        output_wav = _absolute_path(payload["output_wav"], "output_wav", line_number)
        if output_wav.suffix.lower() != ".wav":
            raise ValueError(f"line {line_number}: output_wav must end in .wav")
        output_key = str(output_wav.resolve(strict=False))
        if output_key in seen_outputs:
            raise ValueError(f"line {line_number}: duplicate output_wav: {output_wav}")
        if os.path.lexists(output_wav):
            raise FileExistsError(
                f"line {line_number}: output_wav already exists: {output_wav}"
            )
        if not output_wav.parent.is_dir():
            raise FileNotFoundError(
                f"line {line_number}: output directory does not exist: "
                f"{output_wav.parent}"
            )

        seed = payload["seed"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed > torch.iinfo(torch.int64).max
        ):
            raise ValueError(
                f"line {line_number}: seed must be a non-negative signed int64"
            )

        samples.append(
            ManifestSample(
                sample_id=sample_id,
                target_text=_nonempty_text(
                    payload["target_text"], "target_text", line_number
                ),
                reference_text=_nonempty_text(
                    payload["reference_text"], "reference_text", line_number
                ),
                reference_wav=reference_wav,
                reference_wav_sha256=reference_wav_sha256,
                output_wav=output_wav,
                seed=seed,
            )
        )
        seen_ids.add(sample_id)
        seen_outputs.add(output_key)

    return FrozenManifest(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        samples=tuple(samples),
    )


def validate_summary_path(path: Path, manifest: FrozenManifest) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("summary path must be absolute and traversal-free")
    if path.suffix.lower() != ".json":
        raise ValueError("summary path must end in .json")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"summary directory does not exist: {path.parent}")
    if os.path.lexists(path):
        raise FileExistsError(f"summary already exists: {path}")
    output_paths = {
        str(row.output_wav.resolve(strict=False)) for row in manifest.samples
    }
    if str(path.resolve(strict=False)) in output_paths:
        raise ValueError("summary path must not equal an output_wav path")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_speech_generator_lora(model: Any) -> int:
    """Inject the frozen C4 adapter and require an exact LoRA key match."""

    from peft import LoraConfig, inject_adapter_in_model

    checkpoint = SPEECH_GENERATOR_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Speech Generator checkpoint is missing: {checkpoint}")
    digest = sha256_file(checkpoint)
    if digest != SPEECH_GENERATOR_SHA256:
        raise RuntimeError(
            "Speech Generator checkpoint hash mismatch: "
            f"expected {SPEECH_GENERATOR_SHA256}, found {digest}"
        )
    config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )
    patient_llm = model.model.llm
    patient_llm.llm.model = inject_adapter_in_model(config, patient_llm.llm.model)
    raw_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(raw_state, Mapping):
        raise TypeError("Speech Generator checkpoint must contain a state mapping")
    unknown_metadata = (
        set(raw_state)
        - {key for key in raw_state if isinstance(key, str) and "lora_" in key}
        - {"epoch", "step"}
    )
    if unknown_metadata:
        raise RuntimeError(
            f"unexpected non-LoRA checkpoint keys: {sorted(unknown_metadata)}"
        )
    adapter_state = {
        key: value
        for key, value in raw_state.items()
        if isinstance(key, str) and "lora_" in key
    }
    expected = {key for key in patient_llm.state_dict() if "lora_" in key}
    actual = set(adapter_state)
    if not expected or actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"LoRA key mismatch: missing={missing}, unexpected={unexpected}"
        )
    incompatible = patient_llm.load_state_dict(adapter_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected LoRA keys: {incompatible.unexpected_keys}")
    patient_llm.eval()
    return len(adapter_state)


def load_fixed_runtime() -> tuple[Any, dict[str, Any]]:
    """Load the one supported model/runtime combination."""

    for required_dir in (
        DEFAULT_TRANSFORMERS_OVERLAY,
        DEFAULT_COSYVOICE_ROOT,
        DEFAULT_MODEL_DIR,
    ):
        if not required_dir.is_dir():
            raise FileNotFoundError(
                f"required runtime directory is missing: {required_dir}"
            )
    model_file_hashes: dict[str, str] = {}
    for relative_path, expected_hash in MODEL_FILE_SHA256.items():
        path = DEFAULT_MODEL_DIR / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"required model file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"frozen model file hash mismatch: {relative_path}")
        model_file_hashes[relative_path] = actual_hash

    overlay_hash, overlay_files = source_tree_sha256(DEFAULT_TRANSFORMERS_OVERLAY)
    if overlay_hash != TRANSFORMERS_OVERLAY_SHA256:
        raise RuntimeError("Transformers overlay source hash mismatch")
    cosyvoice_hash, cosyvoice_files = source_tree_sha256(
        DEFAULT_COSYVOICE_ROOT, exclude_wav=True
    )
    if cosyvoice_hash != COSYVOICE_SOURCE_SHA256:
        raise RuntimeError("CosyVoice source hash mismatch")

    wetext_cache = bind_local_wetext_snapshot(
        DEFAULT_WETEXT_CACHE,
        expected_critical_content_sha256=WETEXT_CRITICAL_CONTENT_SHA256,
    )
    sys.path[:0] = [
        str(DEFAULT_TRANSFORMERS_OVERLAY),
        str(DEFAULT_COSYVOICE_ROOT),
        str(DEFAULT_COSYVOICE_ROOT / "third_party" / "Matcha-TTS"),
    ]
    import transformers

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected transformers {TRANSFORMERS_VERSION}, "
            f"found {transformers.__version__}"
        )
    transformers_path = Path(transformers.__file__).resolve()
    if not transformers_path.is_relative_to(DEFAULT_TRANSFORMERS_OVERLAY.resolve()):
        raise RuntimeError(
            f"Transformers was not imported from the frozen overlay: {transformers_path}"
        )
    import transformers.integrations.deepspeed as transformers_deepspeed

    transformers_deepspeed.is_deepspeed_available = lambda: False
    from cosyvoice.cli.cosyvoice import AutoModel

    model = AutoModel(
        model_dir=str(DEFAULT_MODEL_DIR),
        load_trt=False,
        load_vllm=False,
        fp16=False,
    )
    loaded_lora_keys = load_speech_generator_lora(model)
    return model, {
        "model_dir": str(DEFAULT_MODEL_DIR),
        "base_llm_sha256": BASE_LLM_SHA256,
        "model_file_sha256": model_file_hashes,
        "speech_generator_checkpoint": str(SPEECH_GENERATOR_CHECKPOINT),
        "speech_generator_resolved_checkpoint": str(
            SPEECH_GENERATOR_CHECKPOINT.resolve()
        ),
        "speech_generator_sha256": SPEECH_GENERATOR_SHA256,
        "loaded_lora_keys": loaded_lora_keys,
        "lora": {
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": list(LORA_TARGET_MODULES),
        },
        "transformers": transformers.__version__,
        "transformers_path": str(transformers_path),
        "transformers_overlay_sha256": overlay_hash,
        "transformers_overlay_files": overlay_files,
        "cosyvoice_source_sha256": cosyvoice_hash,
        "cosyvoice_source_files": cosyvoice_files,
        "peft": importlib.metadata.version("peft"),
        "wetext": wetext_installation_evidence(),
        "wetext_cache": wetext_cache,
        "inference": {
            "stream": False,
            "text_frontend": False,
            "prompt_instruction": PROMPT_INSTRUCTION,
            "calls_per_sample": 1,
            "retry": False,
            "reference_substitution": False,
        },
    }


def _stage_audio_once(
    sample: ManifestSample, model: Any
) -> tuple[dict[str, Any], Path]:
    _set_seed(sample.seed)
    prompt_text = f"{PROMPT_INSTRUCTION}{sample.reference_text}"

    # This is deliberately the only inference call in the per-sample path.
    inference = model.inference_zero_shot(
        sample.target_text,
        prompt_text,
        str(sample.reference_wav),
        stream=False,
        text_frontend=False,
    )
    chunks = [output["tts_speech"].detach().cpu().float() for output in inference]
    if not chunks:
        raise RuntimeError(f"Speech Generator returned no audio: {sample.sample_id}")
    waveform = torch.cat(chunks, dim=-1).squeeze(0).numpy()
    if waveform.ndim != 1 or waveform.size == 0:
        raise RuntimeError(
            f"Speech Generator returned invalid shape for {sample.sample_id}: "
            f"{waveform.shape}"
        )
    if not np.isfinite(waveform).all():
        raise RuntimeError(
            f"Speech Generator returned non-finite audio: {sample.sample_id}"
        )
    peak = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    dc_abs = float(abs(np.mean(waveform, dtype=np.float64)))
    clipping_fraction = float(np.mean(np.abs(waveform) >= 0.999))
    if peak < 1e-4 or rms < 1e-4:
        raise RuntimeError(
            f"Speech Generator returned near-silent audio: {sample.sample_id}"
        )
    if dc_abs > 0.2:
        raise RuntimeError(
            f"Speech Generator returned excessive DC: {sample.sample_id}"
        )
    if clipping_fraction > 0.02:
        raise RuntimeError(
            f"Speech Generator returned severely clipped audio: {sample.sample_id}"
        )
    sample_rate = model.sample_rate
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate <= 0
    ):
        raise RuntimeError(f"invalid Speech Generator sample rate: {sample_rate!r}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{sample.output_wav.stem}.",
        suffix=".tmp.wav",
        dir=sample.output_wav.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        sf.write(temporary, waveform, sample_rate, format="WAV")
        info = sf.info(temporary)
        if info.samplerate != sample_rate or info.channels != 1 or info.frames <= 0:
            raise RuntimeError(
                f"written audio failed validation for {sample.sample_id}"
            )
        if not 0.25 <= info.duration <= 30.0:
            raise RuntimeError(
                f"Speech Generator returned implausible duration for "
                f"{sample.sample_id}: {info.duration}"
            )
        digest = sha256_file(temporary)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    record = {
        "sample_id": sample.sample_id,
        "target_text": sample.target_text,
        "reference_text": sample.reference_text,
        "reference_wav": str(sample.reference_wav),
        "reference_wav_sha256": sample.reference_wav_sha256,
        "output_wav": str(sample.output_wav),
        "output_wav_sha256": digest,
        "duration_seconds": info.duration,
        "frames": info.frames,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "peak_abs": peak,
        "rms": rms,
        "dc_abs": dc_abs,
        "clipping_fraction": clipping_fraction,
        "seed": sample.seed,
        "inference_call_count": 1,
    }
    return record, temporary


def _stage_json(path: Path, payload: Mapping[str, Any]) -> Path:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp.json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def materialize_manifest(
    manifest_path: Path,
    summary_path: Path,
    *,
    runtime_factory: Callable[[], tuple[Any, dict[str, Any]]] = load_fixed_runtime,
) -> dict[str, Any]:
    """Validate first, then synthesize once per row and publish one summary."""

    manifest = load_frozen_manifest(manifest_path)
    validate_summary_path(summary_path, manifest)
    model, runtime = runtime_factory()
    started = time.time()
    staged_audio: list[tuple[ManifestSample, dict[str, Any], Path]] = []
    published: list[Path] = []
    summary_temporary: Path | None = None
    try:
        for index, sample in enumerate(manifest.samples, start=1):
            print(
                f"[{index}/{len(manifest.samples)}] {sample.sample_id}",
                flush=True,
            )
            record, temporary = _stage_audio_once(sample, model)
            staged_audio.append((sample, record, temporary))

        if sha256_file(manifest.path) != manifest.sha256:
            raise RuntimeError("manifest changed during synthesis")
        outputs = [record for _, record, _ in staged_audio]
        summary = {
            "schema_version": 1,
            "status": "complete",
            "manifest": {
                "path": str(manifest.path),
                "sha256": manifest.sha256,
                "sample_count": len(manifest.samples),
            },
            "runtime": runtime,
            "output_count": len(outputs),
            "outputs": outputs,
            "elapsed_seconds": time.time() - started,
        }
        summary_temporary = _stage_json(summary_path, summary)
        for sample, _, temporary in staged_audio:
            os.link(temporary, sample.output_wav)
            published.append(sample.output_wav)
        os.link(summary_temporary, summary_path)
        return summary
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for _, _, temporary in staged_audio:
            temporary.unlink(missing_ok=True)
        if summary_temporary is not None:
            summary_temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    summary = materialize_manifest(args.manifest, args.summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
