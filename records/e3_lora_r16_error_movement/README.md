# E3 LoRA r16 Test Results And Error Movement

Method: `E3_LoRA_r16_best_dev_checkpoint_483`

This folder is separate from `records/e2_error_movement/`. It records the test-set result and error movement table for the LoRA rank-16 best-dev checkpoint only.

Selected checkpoint:

- remote checkpoint: `/data/qwen3-asr/finetune/e3-lora-r16-10epoch/checkpoint-483`
- selected epoch: `7`
- selection rule: lowest dev overall CER among all saved rank-16 LoRA checkpoints

## Test Overall Result

| group | n | zero-shot CER | E2 full-SFT CER | E3 LoRA r16 CER | E3 - E2 CER | zero critical | E2 critical | E3 critical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 258 | 0.277309 | 0.139254 | 0.042415 | -0.096840 | 53 | 23 | 7 |
| easy | 135 | 0.060905 | 0.022489 | 0.006127 | -0.016362 | 0 | 0 | 1 |
| medium | 70 | 0.304910 | 0.116754 | 0.025169 | -0.091585 | 0 | 3 | 0 |
| hard | 53 | 0.792076 | 0.466393 | 0.157622 | -0.308772 | 53 | 20 | 6 |

## Error Movement Definition

| category | definition | meaning |
|---|---|---|
| Rescued | zero-shot critical=1, E3 critical=0 | hard samples recovered by LoRA adaptation |
| Still hard | zero-shot critical=1, E3 critical=1 | remaining core difficult samples after LoRA adaptation |
| Regression | zero-shot critical=0, E3 critical=1 | new critical errors introduced by LoRA adaptation |
| Stable easy | zero-shot critical=0, E3 critical=0 | samples that stayed non-critical before and after LoRA adaptation |

## Error Movement Summary

| category | n | percent | avg zero-shot CER | avg E3 CER | avg delta CER | zero critical | E3 critical |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rescued | 47 | 0.182171 | 0.767595 | 0.054655 | -0.712939 | 47 | 0 |
| Still hard | 6 | 0.023256 | 0.983846 | 0.964193 | -0.019653 | 6 | 6 |
| Regression | 1 | 0.003876 | 0.142857 | 0.571429 | 0.428572 | 0 | 1 |
| Stable easy | 204 | 0.790698 | 0.144230 | 0.009890 | -0.134340 | 0 | 0 |

## Files

- `test_result_summary.csv`
- `test_error_movement_summary.csv`
- `test_error_movement_per_sample_sanitized.csv`
- `test_error_movement_by_zero_shot_bucket.csv`
- `test_error_movement_by_disease_tag.csv`
- `test_error_movement_by_duration_bucket.csv`
- `test_error_movement_compare_e2_vs_e3.csv`

The per-sample CSV is sanitized: it excludes transcript text, model predictions, audio paths, and raw speaker IDs.
