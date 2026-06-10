# E3 LoRA r16 GT Prediction Comparison Index

This folder indexes the private test-set comparison table for `E3_LoRA_r16_best_dev_checkpoint_483`.

The full table contains patient transcript text and model predictions, so it is not committed to Git. It is stored on the private DevBox only.

Private CSV:

```text
/data/qwen3-asr/finetune/e3-lora-r16-10epoch/eval_test_gt_prediction_comparison/test_gt_vs_predictions_zero_full_lora.csv
```

Rows:

- samples: `258`
- CSV lines including header: `259`

SHA256:

```text
6febbe80daa4190ff52abce83e9dc40b36db56d2147366f6502fbc26d00f910f
```

Columns:

- `row_no`
- `utt_id`
- `speaker_id`
- `disease_tag`
- `duration`
- `duration_bucket`
- `zero_shot_bucket`
- `task_type`
- `clean_gt`
- `zero_shot_predict`
- `zero_shot_cer`
- `zero_shot_critical`
- `full_sft_method`
- `full_sft_predict`
- `full_sft_cer`
- `full_sft_critical`
- `lora_method`
- `lora_predict`
- `lora_cer`
- `lora_critical`
- `delta_lora_minus_full_sft_cer`
- `delta_lora_minus_zero_shot_cer`
- `audio_path`

Privacy note: this file includes GT text, model outputs, audio paths, and raw speaker IDs. Keep it outside public Git unless the dataset owner explicitly approves publication.
