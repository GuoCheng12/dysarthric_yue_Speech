# Maintained ASR utilities

## Runtime path

- `qwen3_asr_lora_sft.py`: ordinary Qwen3-ASR LoRA training, fixed seed, correct
  utterance-mean gradient accumulation (including incomplete windows), explicit
  right-padded training labels, and optional continuation from the previous adapter.
- `train_setting_d_m0.py`: real-only 8x4 M0 training from the base model, decoded
  dev monitoring every 10 steps, dev-only selection, and one final test decode.
- `evaluate_asr_checkpoint.py`: one strict JSONL decoder for base Qwen3-ASR and LoRA;
  emits canonical overall/difficulty/patient metrics and refuses partial batches.
- `project_registry.py`: verify frozen split rows/hashes and selected model artifacts.

Setting D is now a frozen external artifact. Its former one-off Setting C construction,
QC filtering, prompt-leakage, and E9/E10/E13/E17-E19 analysis entrypoints were archived
after their reusable logic was reduced to `therapist_harness.text`,
`therapist_harness.evidence`, and `therapist_harness.materialize`.

The maintained registry validator checks the exact Setting D JSONL row counts and
SHA-256 values. Static augmentation builders are historical and are not the Therapist
loop materializer.
