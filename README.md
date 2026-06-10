# Dysarthric Yue Speech ASR Experiments

This repository is the source of truth for Cantonese dysarthric speech ASR experiments with Qwen3-ASR.

It manages code, experiment protocols, resource registries, dataset registries, aggregate results, and reproducibility documentation. Large or sensitive payloads such as patient audio, raw transcripts, per-utterance private predictions, model weights, checkpoints, caches, virtual environments, and local proxy/Codex authentication files are tracked through manifests and checksums rather than committed as raw files.

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
  e3_lora_sft_protocol.md
  evaluation_protocol.md
  e2_fullsft_clean_pooled_epoch2_baseline.md
examples/
  env.example
  dataset_layout.md
results/
  public aggregate summaries only
records/
  e2_error_movement/           # sanitized per-sample error movement records
data/
  registry/                    # private dataset manifests and version records
artifacts/
  registry/                    # checkpoint/result artifact records
```

## Repository Management Rule

All project work should be represented in this repository.

- Code changes live in `src/`, `inference/`, `finetune/`, or `benchmarks/`.
- Dataset versions live in `data/registry/`.
- Model/checkpoint/result artifact records live in `artifacts/registry/`.
- Experiment decisions and protocols live in `docs/`.
- Public aggregate summaries live in `results/`.
- Sanitized experiment records live in `records/`.

Private or large files are still kept outside Git; the repository records where they are, how they were produced, and how to verify them.

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

Keep the following payloads outside normal Git commits:

- patient audio and transcript files
- per-utterance predictions and manifests containing patient text
- Qwen model weights
- full fine-tuned checkpoints and optimizer states
- local VPN/proxy/Codex authentication files

For each such payload, add or update a registry entry with its purpose, private path or storage URI, version, checksum when available, generation command, and privacy level.

See `examples/dataset_layout.md` for the expected private filesystem layout.

## Current Records

- `records/e2_error_movement/`: test-set movement table for `zero_shot` vs `E2_fullSFT_clean_pooled_epoch2`.
  - `Rescued`: zero-shot critical, epoch2 non-critical.
  - `Still hard`: critical before and after epoch2.
  - `Regression`: non-critical in zero-shot, critical after epoch2.
  - `Stable easy`: non-critical in both.
