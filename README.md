# Dysarthric Yue Speech ASR Experiments

This repository contains the code framework used for Cantonese dysarthric speech ASR experiments with Qwen3-ASR.

The repository intentionally excludes patient audio, transcripts, raw inference outputs, model weights, fine-tuned checkpoints, caches, virtual environments, and local proxy/Codex authentication files.

## Current Baseline

The fixed adaptation baseline is:

`E2_fullSFT_clean_pooled_epoch2`

It refers to full-parameter SFT of `Qwen/Qwen3-ASR-1.7B` on automatically cleaned pooled read-sentence patient speech, using the epoch-2 checkpoint from the 3-epoch run.

Headline test-set result:

| group | sample_count | zero_shot_cer | E2_fullSFT_clean_pooled_epoch2_cer | delta_cer | zero_shot_critical_rate | E2_fullSFT_clean_pooled_epoch2_critical_rate |
|---|---:|---:|---:|---:|---:|---:|
| overall | 258 | 0.277309 | 0.139254 | -0.138055 | 0.205426 | 0.089147 |
| easy | 135 | 0.060905 | 0.022489 | -0.038416 | 0.000000 | 0.000000 |
| medium | 70 | 0.304910 | 0.116754 | -0.188156 | 0.000000 | 0.042857 |
| hard | 53 | 0.792076 | 0.466393 | -0.325683 | 1.000000 | 0.377358 |

## Repository Layout

```text
src/
  smoke_infer.py                # single-file Qwen3-ASR smoke inference
  eval_asr_dir.py               # directory-level ASR evaluation utilities
inference/scripts/
  batch_infer_vlink_data_raw.py # batch zero-shot inference
  score_vlink_results.py        # TextNorm_CER and Critical_Error scoring
  prepare_experiment2_clean_manifest.py
finetune/
  official/qwen3_asr_sft.py     # Qwen3-ASR SFT script copy
  scripts/                      # JSONL prep, checkpoint eval, three-way comparison
benchmarks/
  gpu_benchmark.py
  qwen_sft_batch_probe.py
docs/
  experiment2_context.md
  run_e2_pooled_sft.md
  evaluation_protocol.md
  e2_fullsft_clean_pooled_epoch2_baseline.md
examples/
  env.example
  dataset_layout.md
results/
  public aggregate summaries only
```

## Evaluation Rule For Future Experiments

All future methods should report the same test/dev rows with three columns:

1. `zero_shot`
2. `E2_fullSFT_clean_pooled_epoch2`
3. `new_method`

Use:

```bash
python finetune/scripts/build_three_way_comparison.py \
  --baseline-predictions path/to/E2/test_predictions.csv \
  --new-predictions path/to/new_method_test_predictions.csv \
  --new-method-name METHOD_NAME \
  --out-dir finetune/experiments/METHOD_NAME/eval_test
```

The generated `three_way_summary_by_group.csv` is the canonical comparison table.

## Data And Model Policy

This public repository is code-only. Keep the following outside git:

- patient audio and transcript files
- per-utterance predictions and manifests containing patient text
- Qwen model weights
- full fine-tuned checkpoints and optimizer states
- local VPN/proxy/Codex authentication files

See `examples/dataset_layout.md` for the expected private filesystem layout.
