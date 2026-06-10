# Test Error Movement Table

Baseline: `E2_fullSFT_clean_pooled_epoch2` on the cleaned Experiment 2 test split.

This record classifies each test sample by whether it was critical before and after epoch-2 full-SFT. The per-sample CSV is sanitized: it excludes transcript text, model predictions, audio paths, and speaker IDs.

## Four Categories

| category | definition | meaning |
|---|---|---|
| Rescued | zero-shot critical=1, epoch2 critical=0 | hard samples recovered by full-SFT adaptation |
| Still hard | zero-shot critical=1, epoch2 critical=1 | remaining core difficult samples after adaptation |
| Regression | zero-shot critical=0, epoch2 critical=1 | new critical errors introduced by fine-tuning |
| Stable easy | zero-shot critical=0, epoch2 critical=0 | samples that stayed non-critical before and after adaptation |

## Summary

| category | n | percent | avg zero-shot CER | avg epoch2 CER | avg delta CER | zero critical | epoch2 critical |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rescued | 33 | 0.127907 | 0.733254 | 0.197733 | -0.535521 | 33 | 0 |
| Still hard | 20 | 0.077519 | 0.889132 | 0.909683 | 0.020551 | 20 | 20 |
| Regression | 3 | 0.011628 | 0.369841 | 0.842857 | 0.473016 | 0 | 3 |
| Stable easy | 202 | 0.782946 | 0.140873 | 0.042971 | -0.097901 | 0 | 0 |

## Files

- `test_error_movement_per_sample_sanitized.csv`
- `test_error_movement_summary.csv`
- `test_error_movement_by_zero_shot_bucket.csv`
- `test_error_movement_by_disease_tag.csv`
- `test_error_movement_by_duration_bucket.csv`
