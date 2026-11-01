"""Resume the frozen experiment after a reproduced empty-token TTS failure."""

import json
import os
from pathlib import Path
import sys
import time

import tts_reference_policy as experiment


def install_empty_token_guard(backend):
    original = backend.token2wav

    def token2wav(*args, **kwargs):
        token, uid = kwargs["token"], kwargs["uuid"]
        if token.shape[1] == 0 and backend.llm_end_dict[uid] is True:
            # Non-streaming tts() has already joined the LLM thread here.
            # Its normal cleanup lies after token2wav(), so clean this failed
            # session before sending the existing paired retry loop an error.
            with backend.lock:
                for cache in (backend.tts_speech_token_dict, backend.llm_end_dict, backend.hift_cache_dict):
                    cache.pop(uid)
            raise RuntimeError("Speech Generator returned no speech tokens: completed LLM emitted an empty sequence")
        return original(*args, **kwargs)

    backend.token2wav = token2wav


def main():
    root = experiment.ROOT
    experiment.verify(root)
    amendment = experiment.read(root / "empty_token_recovery.json")
    assert experiment.sha256_file(Path(__file__)) == amendment["recovery_runner_sha256"]
    assert experiment.sha256_file(root / "protocol.json") == amendment["original_protocol_sha256"]
    evidence = experiment.read(root / "recovery_diagnosis/evidence.json")
    assert evidence["status"] == "reproduced"
    assert evidence["shapes"][0]["token_shape"] == [1, 0]
    assert evidence["shapes"][1]["mel_shape"] == [1, 80, 0]
    for path, digest in amendment["completed_selected_sha256"].items():
        assert experiment.sha256_file(Path(path)) == digest

    from synthesis.cosyvoice3_patient_sft import synthesize_manifest as tts
    original_loader = tts.load_fixed_runtime

    def load():
        model, metadata = original_loader()
        install_empty_token_guard(model.model)
        metadata["empty_token_guard"] = {
            "source_sha256": amendment["recovery_runner_sha256"],
            "behavior": "completed zero-token output enters unchanged paired technical retries; nonempty output untouched"}
        return model, metadata

    tts.load_fixed_runtime = load
    def status(value, **extra):
        temporary = root / ".recovery_status.tmp"
        temporary.write_text(json.dumps({"status": value, "stage": "synthesize_recovery",
            "pid": os.getpid(), "time_unix": time.time(), **extra}) + "\n")
        temporary.replace(root / "run_status.json")
    status("running")
    try:
        experiment.synthesize(root)
    except Exception as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    status("complete")
    print("[recovery] Synthesis complete; continuing frozen ASR pipeline in a fresh process", flush=True)
    # Replace this process to release TTS CUDA state and its Transformers
    # overlay before the original runner launches the ASR training processes.
    os.execv(sys.executable, [sys.executable, "-u", str(Path(experiment.__file__)),
                            "--root", str(root), "--stage", "run"])


if __name__ == "__main__":
    main()
