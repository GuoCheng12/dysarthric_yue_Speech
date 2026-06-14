# DSI V1 Deterministic Residual Generator

This record tracks Step C of the deterministic residual generator line.

Goal:

```text
normal Cantonese TTS features + patient_id -> dysarthric residual mel
```

Input feature dataset:

- Feature root: `/data/qwen3-asr/synthesis/dsi_v1/residual_features_v1_hubert_chinese`
- Feature manifest: `/data/qwen3-asr/synthesis/dsi_v1/residual_features_v1_hubert_chinese/feature_manifest.csv`
- Source pair data: CosyVoice3 TTS setting V1 + cleaned patient audio
- Source split: `prompt_disjoint_v1`
- Feature time grid: normal-TTS mel frames
- Target residual: `residual_mel = dys_mel_aligned - norm_mel`

Model V1:

- Type: deterministic residual-mel generator
- Inputs: `norm_mel`, `norm_ssl`, and learned `patient_id` embedding
- Output: 80-bin residual mel on the normal-TTS time grid
- Architecture: mel projection + SSL projection + patient embedding + temporal residual Conv1d blocks
- Training loss: masked L1 residual loss plus `0.1 *` masked L1 temporal-difference smoothness loss

Step C audit:

- Audit mode: deterministic sample of 32 feature files
- Manifest rows: 2707
- Audited tensor rows: 32
- Issue count: 0
- Error count: 0
- Warning count: 0
- Split counts: train/dev/test = 2159/273/275
- Zero-shot bucket counts: easy/medium/hard = 1367/764/576
- Unique patient count: 71
- Dev/test patients absent from train: none

The first full audit attempt was stopped because reading every `.pt` file
interactively from PVC was slow and left a detached process after SSH
interruption. The sampled audit validates the tensor contract used by the
smoke training run. Full audit should be rerun as a noninteractive batch job or
after adding progress logging.

Overfit smoke test:

- Remote output root: `/data/qwen3-asr/synthesis/dsi_v1/residual_generator_v1_hubert_chinese/smoke_overfit_32`
- Training/eval rows: same 32 train rows
- Device: CUDA on NVIDIA L20
- Batch size: 4
- Hidden dim: 256
- Residual Conv1d blocks: 4
- LR: `1e-3`
- Weight decay: `1e-4`
- Max steps: 400
- Completed epochs: 50
- Initial residual L1: `3.887165`
- Best residual L1: `1.185420`
- Relative residual-L1 reduction: about `69.5%`
- Best checkpoint: private `/data` artifact, not committed

Interpretation:

- This is not a generalization result because train and dev are intentionally
  the same 32 rows.
- The result verifies that Step B features, tensor padding/masking, the V1
  generator, and the residual loss form a working training loop.
- Formal Step C training should use the original train/dev split and select
  checkpoints by held-out dev residual metrics before any waveform synthesis
  or ASR-side evaluation.

See:

- `audit_sample32_summary.csv`
- `smoke_overfit_32_summary.csv`

Reproduction commands:

```bash
cd /data/qwen3-asr/repo/dysarthric_yue_Speech
source /data/qwen3-asr/env.sh
source /data/qwen3-asr/venvs/qwen3-asr/bin/activate

python synthesis/dsi_v1/scripts/audit_dsi_residual_features.py \
  --feature-manifest /data/qwen3-asr/synthesis/dsi_v1/residual_features_v1_hubert_chinese/feature_manifest.csv \
  --out-dir /data/qwen3-asr/synthesis/dsi_v1/residual_generator_v1_hubert_chinese/audit_step_c_sample32 \
  --sample-limit 32

python synthesis/dsi_v1/scripts/train_dsi_residual_generator.py \
  --feature-manifest /data/qwen3-asr/synthesis/dsi_v1/residual_features_v1_hubert_chinese/feature_manifest.csv \
  --out-dir /data/qwen3-asr/synthesis/dsi_v1/residual_generator_v1_hubert_chinese/smoke_overfit_32 \
  --overfit-n 32 \
  --batch-size 4 \
  --epochs 80 \
  --max-steps 400 \
  --eval-every 20 \
  --lr 1e-3 \
  --hidden-dim 256 \
  --num-layers 4 \
  --smooth-weight 0.1
```
