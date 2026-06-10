# E3 LoRA r16 Best-Dev Results

This record summarizes the LoRA rank-16 SFT run trained for 10 epochs on the same clean train/dev/test split as E2.

Selection result:

- selected method: `E3_LoRA_r16_best_dev_checkpoint_483`
- selected checkpoint: `checkpoint-483`
- selected epoch: `7`
- selection criterion: lowest dev overall CER after running ASR inference for all saved LoRA checkpoints
- note: trainer dev loss was lowest at `checkpoint-69`, but ASR dev CER was best at `checkpoint-483`

Private raw prediction tables, audio paths, transcripts, and unmasked speaker IDs are not committed. Full private artifacts are on the remote DevBox under `/data/qwen3-asr/finetune/e3-lora-r16-10epoch`.

Files:

- `dev_checkpoint_selection.csv`: all LoRA checkpoints ranked by dev CER.
- `dev_focus_cer_metrics.csv`, `test_focus_cer_metrics.csv`: hard and medium CER comparison against zero-shot and E2 full-SFT epoch2.
- `dev_medium_regression_metrics.csv`, `test_medium_regression_metrics.csv`: medium-bucket critical regression counts.
- `dev_per_speaker_worst_cases_sanitized.csv`, `test_per_speaker_worst_cases_sanitized.csv`: worst speaker-level deltas with hashed speaker keys.
