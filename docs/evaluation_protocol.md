# Evaluation Protocol After E2

The fixed baseline for all later experiments is:

`E2_fullSFT_clean_pooled_epoch2`

It refers to the full-parameter SFT epoch-2 checkpoint from `finetune/e2-pooled-sft-3epoch/checkpoint-138`.

## Required Comparison

All new methods must be compared on exactly the same cleaned dev/test rows and must include:

1. `zero_shot`
2. `E2_fullSFT_clean_pooled_epoch2`
3. `new_method`

Do not report only zero-shot vs new method. The E2 epoch-2 baseline is now the adaptation baseline.

## Required Per-Utterance Fields

- `utt_id`
- `speaker_id`
- `disease_tag`
- `duration`
- `duration_bucket`
- `zero_shot_bucket`
- `task_type`
- `clean_gt`
- `zero_shot_predict`
- `E2_fullSFT_clean_pooled_epoch2_predict`
- `new_method_name`
- `new_method_predict`
- `zero_shot_cer`
- `E2_fullSFT_clean_pooled_epoch2_cer`
- `new_method_cer`
- `delta_new_vs_zero_cer`
- `delta_new_vs_E2_fullSFT_clean_pooled_epoch2_cer`
- `zero_shot_critical`
- `E2_fullSFT_clean_pooled_epoch2_critical`
- `new_method_critical`
- `audio_path`

## Required Group Summary Fields

- `group`
- `sample_count`
- `zero_shot_cer`
- `E2_fullSFT_clean_pooled_epoch2_cer`
- `new_method_cer`
- `delta_new_vs_zero_cer`
- `delta_new_vs_E2_fullSFT_clean_pooled_epoch2_cer`
- `zero_shot_critical_rate`
- `E2_fullSFT_clean_pooled_epoch2_critical_rate`
- `new_method_critical_rate`
- `delta_new_vs_zero_critical_rate`
- `delta_new_vs_E2_fullSFT_clean_pooled_epoch2_critical_rate`
- `zero_shot_critical_count`
- `E2_fullSFT_clean_pooled_epoch2_critical_count`
- `new_method_critical_count`

## Standard Builder

Use `finetune/scripts/build_three_way_comparison.py` after evaluating a new checkpoint:

```bash
python finetune/scripts/build_three_way_comparison.py \
  --baseline-predictions finetune/baselines/E2_fullSFT_clean_pooled_epoch2/test_predictions.csv \
  --new-predictions path/to/new_method_test_predictions.csv \
  --new-method-name METHOD_NAME \
  --out-dir finetune/experiments/METHOD_NAME/eval_test
```

The script writes:

- `three_way_per_utterance.csv`
- `three_way_summary_by_group.csv`
