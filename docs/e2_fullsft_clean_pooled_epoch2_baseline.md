# E2_fullSFT_clean_pooled_epoch2

This is the canonical Experiment 2 baseline for later experiments.

## Identity

- Baseline name: `E2_fullSFT_clean_pooled_epoch2`
- Training type: full-parameter SFT, not LoRA
- Training data: cleaned real-data-only pooled `read_sentence` rows
- Training split: `outputs/experiment2/clean_read_sentence_train.csv`
- Dev split: `outputs/experiment2/clean_read_sentence_dev.csv`
- Test split: `outputs/experiment2/clean_read_sentence_test.csv`
- Source run: `finetune/e2-pooled-sft-3epoch`
- Source checkpoint: `checkpoint-138`
- Epoch: 2

## Remote References

- Remote checkpoint: `/data/qwen3-asr/finetune/e2-pooled-sft-3epoch/checkpoint-138`
- Remote dev eval: `/data/qwen3-asr/finetune/e2-pooled-sft-3epoch/eval_dev_checkpoint_138`
- Remote test eval: `/data/qwen3-asr/finetune/e2-pooled-sft-3epoch/eval_test_checkpoint_138`

## Canonical Local Artifacts

- `dev_predictions.csv`
- `dev_summary_by_group.csv`
- `test_predictions.csv`
- `test_summary_by_group.csv`
- `test_zero_shot_vs_E2_fullSFT_clean_pooled_epoch2_comparison.csv`
- `test_zero_shot_vs_E2_fullSFT_clean_pooled_epoch2_comparison.xlsx`

## Test Summary

| group | sample_count | zero_shot_cer | E2_fullSFT_clean_pooled_epoch2_cer | delta_cer | zero_shot_critical_rate | E2_fullSFT_clean_pooled_epoch2_critical_rate |
|---|---:|---:|---:|---:|---:|---:|
| overall | 258 | 0.277309 | 0.139254 | -0.138055 | 0.205426 | 0.089147 |
| easy | 135 | 0.060905 | 0.022489 | -0.038416 | 0.000000 | 0.000000 |
| medium | 70 | 0.304910 | 0.116754 | -0.188156 | 0.000000 | 0.042857 |
| hard | 53 | 0.792076 | 0.466393 | -0.325683 | 1.000000 | 0.377358 |

## Rule For Future Experiments

Every later method must report three columns on the same dev/test split:

1. `zero_shot`
2. `E2_fullSFT_clean_pooled_epoch2`
3. `new_method`

At minimum, report CER and critical error rate/count for overall, zero-shot difficulty buckets, disease tag, duration bucket, and speaker.
