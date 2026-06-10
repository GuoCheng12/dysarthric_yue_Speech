#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import torch
from qwen_asr import Qwen3ASRModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-ASR smoke inference")
    parser.add_argument(
        "audio",
        help="Audio path or URL accepted by qwen-asr.",
    )
    parser.add_argument(
        "--model",
        default="/data/qwen3-asr/models/Qwen3-ASR-1.7B",
        help="Local model directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help='Language hint, for example "Chinese", "English", or omit for auto.',
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Allow CPU execution. This is mainly for debugging and can be very slow.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if model_path.exists() and not model_path.is_dir():
        raise SystemExit(f"Model path is not a directory: {model_path}")

    audio_path = Path(args.audio)
    if "://" not in args.audio and not audio_path.is_file():
        raise SystemExit(f"Audio path is not readable: {audio_path}")

    has_cuda = torch.cuda.is_available()
    if not has_cuda and not args.cpu:
        raise SystemExit(
            "需要重新开启挂载 GPU 的 DevBox 后再跑 inference "
            "(torch.cuda.is_available() is False)."
        )

    dtype = torch.bfloat16 if has_cuda else torch.float32
    device_map = "cuda:0" if has_cuda else "cpu"

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    results = model.transcribe(audio=args.audio, language=args.language)

    for item in results:
        print(
            json.dumps(
                {
                    "language": getattr(item, "language", None),
                    "text": getattr(item, "text", None),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
