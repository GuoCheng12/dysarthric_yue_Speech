"""Finish the fixed diagnostic after all Fish generation receipts exist."""
import subprocess
import time

from tts_model_comparison import ROOT, REPO, ADAPTER

PYTHON = "/data/qwen3-asr/venvs/qwen3-asr/bin/python"


def run(args):
    print("RUN", " ".join(args), flush=True)
    started = time.monotonic()
    subprocess.run([PYTHON, *args], cwd=REPO, check=True)
    print(f"DONE {args[0]}: {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    run(["examples/report_tts_model_comparison.py", "assemble-model", "--model", "fish_s2"])
    run(["finetune/scripts/evaluate_asr_checkpoint.py",
         "--base-model-path", "/data/qwen3-asr/models/Qwen3-ASR-1.7B",
         "--adapter-path", str(ADAPTER),
         "--split-jsonl", str(ROOT / "fish_s2_decode.jsonl"),
         "--out-dir", str(ROOT / "asr_fish_s2"),
         "--batch-size", "8", "--critical-cer-threshold", "0.5", "--max-new-tokens", "256"])
    for stage in ("assemble", "acoustics", "summarize", "report"):
        run(["examples/report_tts_model_comparison.py", stage])
    run(["-m", "pytest", "-q", "tests/test_tts_model_comparison.py", "tests/test_tts_reconstruction_probe.py"])
    print("COMPLETE", ROOT / "listening/index.html", flush=True)
