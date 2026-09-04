# Setting D C4 CosyVoice3 retraining

Status: completed; early-stopped on a dev-loss plateau after epoch 16.

Default checkpoint name: **speech generator**. Human matched-listening selected
epoch 8 over epoch 12. The stable checkpoint alias is
`/data/qwen3-asr/records/cosyvoice3_reference_sft_setting_d_v1/speech_generator.pt`.

This is a fresh reference-conditioned CosyVoice3 LLM LoRA run on Setting D. It
starts from the released `llm.pt`; the old Setting C C4 checkpoint is not used
for initialization.

## Frozen data

- Effective train: 2,108 utterances, 57 speakers
- Dev: 258 utterances
- Test: 253 utterances (never used for checkpoint selection)
- References: same speaker, distinct utterance, train-only reference pool
- Content overlap: zero for prompt id, cleaned text, normalized text, Jyutping,
  and `clean_gt`
- Removed patients: `vlink201`, `vlink203`, `vlink225`, `vlink243`
- Removed noisy utterance: `vlink211/7290411755587742893228`

Nine remaining speakers with only one train utterance are excluded from TTS
training because they cannot form a distinct same-speaker reference pair. None
of them appears in dev or test.

## Frozen C4 configuration

- Model: CosyVoice3 LLM only
- LoRA: rank 8, alpha 16, dropout 0.05
- Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Learning rate: `1e-4`
- Optimizer/scheduler: Adam + constant LR, 2,500 warmup steps
- Dynamic batch: at most 2,000 mel frames; gradient accumulation 2
- AMP enabled; seed 1986; maximum 50 epochs
- Dev supervision: minimum 15 epochs, patience 8, minimum improvement 0.002,
  severe relative loss rise 5%

The 50-epoch cap follows the actual C4 `run_summary.json`, which is treated as
authoritative for the executed C4 run.

## Frozen inference boundary

The maintained synthesizer also freezes the 57-entry train-only reference manifest,
the reference WAV hashes, every loaded CosyVoice model component, the patched
Transformers source tree, the CosyVoice source tree, and the local text frontend.
Each manifest row is attempted exactly once with its assigned reference and seed.

Before any final WAV is published, the full batch must pass duration, non-silence,
finite-value, DC-offset, and severe-clipping checks. A failure stops the batch; it does
not trigger a retry, alternate reference, or replacement sample. Exact paths, hashes,
and thresholds are recorded in `conf/speech_generator_epoch8.yaml`.

## Result

- Best dev loss: epoch 8, loss 3.724884, acc 0.216534
- Best dev accuracy: epoch 12, loss 3.727999, acc 0.218121
- Stop: epoch 16, after eight epochs without significant dev-loss improvement
- Severe overfitting: not observed
- Frozen best-dev checkpoint: `best_dev.pt` -> `epoch_8_whole.pt`

The original sweep runner was removed from the maintained runtime after this checkpoint
was frozen because it encoded checkpoint-search and monitoring behavior that the new
closed loop does not use. Its source is preserved in the cleanup archive documented in
`docs/cleanup_20260904.md`; the executed configuration is normalized in
`conf/speech_generator_epoch8.yaml`. The test split was not used for TTS checkpoint
selection.
