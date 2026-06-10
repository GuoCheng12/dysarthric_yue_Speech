# Experiment 2 Pooled SFT Runbook

Do not start these commands until the user confirms.

## Paths

Workspace:

`/data/qwen3-asr/finetune`

Model:

`/data/qwen3-asr/models/Qwen3-ASR-1.7B`

Official SFT script:

`/data/qwen3-asr/finetune/official/qwen3_asr_sft.py`

SFT JSONL:

- `data/e2_train.jsonl`
- `data/e2_dev.jsonl`
- `data/e2_test.jsonl`
- `data/e2_smoke_train.jsonl`
- `data/e2_smoke_dev.jsonl`

## Preflight

```bash
cd /data/qwen3-asr/finetune
source /data/qwen3-asr/venvs/qwen3-asr/bin/activate
python scripts/prepare_e2_sft_jsonl.py \
  --manifest /data/qwen3-asr/inference/outputs/experiment2/clean_manifest.csv \
  --out-dir /data/qwen3-asr/finetune/data
python -m py_compile official/qwen3_asr_sft.py scripts/prepare_e2_sft_jsonl.py scripts/evaluate_e2_checkpoint.py
```

## Smoke Fine-Tune

This is the first command to run after user confirmation.

```bash
cd /data/qwen3-asr/finetune
source /data/qwen3-asr/venvs/qwen3-asr/bin/activate

python official/qwen3_asr_sft.py \
  --model_path /data/qwen3-asr/models/Qwen3-ASR-1.7B \
  --train_file data/e2_smoke_train.jsonl \
  --eval_file data/e2_smoke_dev.jsonl \
  --output_dir /data/qwen3-asr/finetune/e2-pooled-sft-smoke \
  --batch_size 1 \
  --grad_acc 4 \
  --lr 1e-5 \
  --epochs 1 \
  --log_steps 1 \
  --save_steps 4 \
  --save_total_limit 2 \
  --num_workers 0 \
  --pin_memory 0 \
  --persistent_workers 0
```

Smoke pass criteria:

- No OOM or data-collator crash.
- Loss is finite.
- At least one checkpoint is saved.
- Checkpoint can be loaded by `Qwen3ASRModel.from_pretrained`.

## Full Pooled Fine-Tune

Start only after smoke passes.

```bash
cd /data/qwen3-asr/finetune
source /data/qwen3-asr/venvs/qwen3-asr/bin/activate

python official/qwen3_asr_sft.py \
  --model_path /data/qwen3-asr/models/Qwen3-ASR-1.7B \
  --train_file data/e2_train.jsonl \
  --eval_file data/e2_dev.jsonl \
  --output_dir /data/qwen3-asr/finetune/e2-pooled-sft \
  --batch_size 1 \
  --grad_acc 32 \
  --lr 1e-5 \
  --epochs 1 \
  --log_steps 5 \
  --save_steps 69 \
  --save_total_limit 1 \
  --num_workers 2 \
  --pin_memory 1 \
  --persistent_workers 1 \
  --prefetch_factor 2
```

The 2026-06-10 smoke run showed each full Trainer checkpoint is about 12 GiB
because `optimizer.pt` is about 8.1 GiB and `model.safetensors` is about 4.1 GiB.
With the current 50 GiB `/data` volume, keep only the final full-SFT checkpoint
unless more storage is mounted.

If full SFT OOMs, reduce memory pressure in this order:

1. Keep `batch_size=1`.
2. Set `num_workers=0`, `pin_memory=0`, `persistent_workers=0`.
3. Patch official script to enable gradient checkpointing and `use_cache=False`.
4. If still unstable, switch to PEFT/LoRA instead of full SFT.

## Evaluation

Evaluate the best dev checkpoint first, then the final selected checkpoint on test:

```bash
cd /data/qwen3-asr/finetune
source /data/qwen3-asr/venvs/qwen3-asr/bin/activate

python scripts/evaluate_e2_checkpoint.py \
  --model-path /data/qwen3-asr/finetune/e2-pooled-sft/checkpoint-STEP \
  --split-csv /data/qwen3-asr/inference/outputs/experiment2/clean_read_sentence_dev.csv \
  --out-dir /data/qwen3-asr/finetune/e2-pooled-sft/eval_dev_checkpoint_STEP \
  --language Cantonese \
  --batch-size 8
```

Primary report:

`summary_by_group.csv`

Decision criteria:

- Overall CER should decrease.
- Hard bucket CER should not regress sharply.
- Per-speaker regressions should be inspected before claiming success.

## Fixed Baseline For Later Experiments

The selected official E2 baseline is:

`E2_fullSFT_clean_pooled_epoch2`

It is the epoch-2 checkpoint from the independent 3-epoch full-SFT run:

`/data/qwen3-asr/finetune/e2-pooled-sft-3epoch/checkpoint-138`

This checkpoint was selected because it had the best dev decoding metrics among epochs 1-3.
Its canonical local artifacts are stored in:

`finetune/baselines/E2_fullSFT_clean_pooled_epoch2/`

All future methods must be reported as:

`zero_shot` vs `E2_fullSFT_clean_pooled_epoch2` vs `new_method`

Use:

`finetune/scripts/build_three_way_comparison.py`

to build the required per-utterance and group-summary comparison tables.
