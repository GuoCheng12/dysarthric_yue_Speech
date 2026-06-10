# E3 LoRA SFT Protocol

## Motivation

Experiment 2 showed that full-parameter SFT improves overall and hard test
performance, but epoch 3 started to overfit relative to epoch 2. Experiment 3
tests whether a parameter-efficient LoRA adaptation is less aggressive while
still recovering hard samples.

## Official-Code Check

The official Qwen3-ASR repository provides `finetuning/qwen3_asr_sft.py` and a
fine-tuning README for full SFT over JSONL audio-text pairs. It does not provide
a LoRA/PEFT fine-tuning entrypoint in the official finetuning folder. Therefore,
this repository adds a local LoRA wrapper around the official SFT data pipeline.

## First LoRA Setting

Use one setting first:

- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- Target modules: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Base model: `Qwen/Qwen3-ASR-1.7B` or private local equivalent
- Data: same clean train/dev/test as E2

Rank 16 is the first choice because rank 8 may under-adapt and rank 32 is closer
to a stronger adaptation. If rank 16 underfits or overfits, sweep 8 and 32 next.

## Remote Dependency

Install PEFT in the training environment before running:

```bash
source /data/qwen3-asr/venvs/qwen3-asr/bin/activate
pip install -U peft
```

## Training Command

Before training, verify LoRA module matching:

```bash
cd /data/qwen3-asr/repo/dysarthric_yue_Speech
source /data/qwen3-asr/venvs/qwen3-asr/bin/activate

python finetune/scripts/qwen3_asr_lora_sft.py \
  --model_path /data/qwen3-asr/models/Qwen3-ASR-1.7B \
  --train_file /data/qwen3-asr/finetune/data/e2_train.jsonl \
  --eval_file /data/qwen3-asr/finetune/data/e2_dev.jsonl \
  --output_dir /data/qwen3-asr/finetune/e3-lora-r16-dryrun \
  --batch_size 1 \
  --grad_acc 32 \
  --lr 2e-4 \
  --epochs 3 \
  --save_steps 69 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --dry_run_model_setup 1
```

Observed dry-run result on the current Qwen3-ASR-1.7B runtime:

- matched Linear modules: `268`
- trainable parameters: `19,791,872`
- total parameters: `2,057,844,352`
- trainable ratio: `0.9618%`

```bash
cd /data/qwen3-asr/repo/dysarthric_yue_Speech
source /data/qwen3-asr/venvs/qwen3-asr/bin/activate

python finetune/scripts/qwen3_asr_lora_sft.py \
  --model_path /data/qwen3-asr/models/Qwen3-ASR-1.7B \
  --train_file /data/qwen3-asr/finetune/data/e2_train.jsonl \
  --eval_file /data/qwen3-asr/finetune/data/e2_dev.jsonl \
  --output_dir /data/qwen3-asr/finetune/e3-lora-r16 \
  --batch_size 1 \
  --grad_acc 32 \
  --lr 2e-4 \
  --epochs 3 \
  --log_steps 5 \
  --save_steps 69 \
  --save_total_limit 3 \
  --num_workers 2 \
  --pin_memory 1 \
  --persistent_workers 1 \
  --prefetch_factor 2 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

## Dev Evaluation

Evaluate all saved checkpoints on the dev split:

```bash
python finetune/scripts/evaluate_e2_lora_checkpoint.py \
  --base-model-path /data/qwen3-asr/models/Qwen3-ASR-1.7B \
  --adapter-path /data/qwen3-asr/finetune/e3-lora-r16/checkpoint-STEP \
  --split-csv /data/qwen3-asr/inference/outputs/experiment2/clean_read_sentence_dev.csv \
  --out-dir /data/qwen3-asr/finetune/e3-lora-r16/eval_dev_checkpoint_STEP \
  --language Cantonese \
  --batch-size 8
```

Pick `LoRA SFT best-dev` by dev overall CER, then inspect hard CER and
regressions before running test.

## Test Evaluation And Comparison

After selecting best-dev:

```bash
python finetune/scripts/evaluate_e2_lora_checkpoint.py \
  --base-model-path /data/qwen3-asr/models/Qwen3-ASR-1.7B \
  --adapter-path /data/qwen3-asr/finetune/e3-lora-r16/checkpoint-BEST \
  --split-csv /data/qwen3-asr/inference/outputs/experiment2/clean_read_sentence_test.csv \
  --out-dir /data/qwen3-asr/finetune/e3-lora-r16/eval_test_checkpoint_BEST \
  --language Cantonese \
  --batch-size 8

python finetune/scripts/build_three_way_comparison.py \
  --baseline-predictions /data/qwen3-asr/finetune/baselines/E2_fullSFT_clean_pooled_epoch2/test_predictions.csv \
  --new-predictions /data/qwen3-asr/finetune/e3-lora-r16/eval_test_checkpoint_BEST/predictions.csv \
  --new-method-name E3_LoRA_r16_best_dev \
  --out-dir /data/qwen3-asr/finetune/e3-lora-r16/eval_test_three_way

python finetune/scripts/summarize_three_way_focus_metrics.py \
  --three-way /data/qwen3-asr/finetune/e3-lora-r16/eval_test_three_way/three_way_per_utterance.csv \
  --out-dir /data/qwen3-asr/finetune/e3-lora-r16/eval_test_focus_metrics
```

## Main Readout

Compare `full SFT epoch2` vs `LoRA SFT best-dev` with:

- hard CER: `zero_shot_bucket=hard`
- medium regression: rows where zero-shot is medium/non-critical but LoRA becomes critical
- per-speaker worst case: largest positive delta versus `E2_fullSFT_clean_pooled_epoch2`

The LoRA run is useful only if it maintains hard-sample gains while reducing
medium regressions or per-speaker damage.
