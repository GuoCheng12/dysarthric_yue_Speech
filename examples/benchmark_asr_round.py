"""Short disposable throughput run using the maintained LoRA training entrypoint."""

import argparse
import json
from pathlib import Path
import sys
import time

started = time.perf_counter()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--mode", choices=["mixed", "synthetic", "longest"], default="mixed")
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--right-padding", action="store_true")
parser.add_argument("--prefetch", type=int, default=2)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
import accelerate.utils.other  # noqa: E402

# This benchmark uses a single GPU and no DeepSpeed engine. The installed
# optional DeepSpeed package requires a CUDA toolkit absent on this host.
accelerate.utils.other.is_deepspeed_available = lambda: False

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "finetune/scripts"))
import qwen3_asr_lora_sft as entry  # noqa: E402

real = [json.loads(line) for line in Path(
    "/data/qwen3-asr/finetune/data_setting_d_v1/setting_d_train.jsonl"
).read_text().splitlines()]
summary = json.loads(Path(
    "/data/qwen3-asr/records/therapist_mas_setting_d_v1/"
    "prescription_synthesis_smoke_v1/synthesis_summary_attempt2.json"
).read_text())
synthetic = [{"audio": row["output_wav"], "text": "language Cantonese<asr_text>" + row["target_text"], "prompt": ""} for row in summary["outputs"]]
rows = synthetic if args.mode == "synthetic" else real + synthetic
durations = {row["audio"]: sf.info(row["audio"]).duration for row in rows}
if args.mode == "longest":
    rows = sorted(rows, key=lambda row: durations[row["audio"]], reverse=True)[:32]
manifest = args.output / "train.jsonl"
with manifest.open("x") as handle:
    for row in rows:
        handle.write(json.dumps({key: row.get(key, "") for key in ("audio", "text", "prompt")}, ensure_ascii=False) + "\n")

result = {"batch": args.batch, "grad_acc": 32 // args.batch, "mode": args.mode,
          "benchmark_only_right_padding": args.right_padding,
          "samples": len(rows), "duration_quantiles_seconds": np.quantile(list(durations.values()), [0, .5, .9, .99, 1]).tolist(),
          "steps": [], "checkpoint_saved": False}


class RightPaddingProcessor:
    def __init__(self, processor):
        self.processor = processor

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *positional, **kwargs):
        kwargs["padding_side"] = "right"
        return self.processor(*positional, **kwargs)


class Timing(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result["startup_seconds"] = time.perf_counter() - started
        result["bf16"] = args.bf16
        result["gradient_checkpointing"] = args.gradient_checkpointing
        result["optimizer"] = str(args.optim)
        result["num_workers"] = args.dataloader_num_workers
        result["pin_memory"] = args.dataloader_pin_memory
        result["persistent_workers"] = args.dataloader_persistent_workers
        result["prefetch_factor"] = args.dataloader_prefetch_factor

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        measurement = {"step": state.global_step, "seconds": time.perf_counter() - self.started,
                       "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        result["steps"].append(measurement)
        print("BENCHMARK " + json.dumps(measurement), flush=True)


class MeasuredTrainer(entry.QwenASRLoRATrainer):
    def __init__(self, *positional, **kwargs):
        super().__init__(*positional, **kwargs)
        self.add_callback(Timing())
        features = [self.train_dataset[index] for index in range(4)]
        singles = [self.data_collator([row])["labels"][0] for row in features]
        expected = [row[row != -100].tolist() for row in singles]
        original = self.data_collator(features)["labels"]
        result["original_batched_label_mismatches_of_four"] = sum(
            row[row != -100].tolist() != target for row, target in zip(original, expected)
        )
        if args.right_padding:
            self.data_collator.processor = RightPaddingProcessor(self.data_collator.processor)
            corrected = self.data_collator(features)["labels"]
            assert all(row[row != -100].tolist() == target for row, target in zip(corrected, expected))
            result["corrected_batched_labels_match_single"] = True
        result["padding_side"] = self.data_collator.processor.tokenizer.padding_side
        result["attention_implementation"] = str(self.model.config._attn_implementation)


entry.QwenASRLoRATrainer = MeasuredTrainer
sys.argv = ["benchmark", "--model_path", "/data/qwen3-asr/models/Qwen3-ASR-1.7B",
            "--train_file", str(manifest), "--output_dir", str(args.output / "training"),
            "--lora_init_adapter_path", "/data/qwen3-asr/finetune/setting_d_gradacc_fixed_v1/lora_current_default_r16_scale025_step140_seed42_v1/step_checkpoints/checkpoint-50",
            "--batch_size", str(args.batch), "--grad_acc", str(32 // args.batch),
            "--max_steps", str(args.steps), "--log_steps", "1",
            "--prefetch_factor", str(args.prefetch),
            "--save_strategy", "no", "--save_final_checkpoint", "0"]
try:
    entry.main()
    result["status"] = "complete"
except Exception as error:
    result["status"] = "out_of_memory" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed"
    result["error"] = str(error)
    raise
finally:
    result["wall_seconds"] = time.perf_counter() - started
    with (args.output / "benchmark.json").open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
