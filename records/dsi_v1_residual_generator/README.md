# DSI V1 Deterministic Residual Generator

This record tracks Step C and the first Step D diagnostic demo of the
deterministic residual generator line.

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

Formal train/dev run:

- Remote output root: `/data/qwen3-asr/synthesis/dsi_v1/residual_generator_v1_hubert_chinese/train_dev_v1_h256_l4_lr1e3_bs8_epoch20`
- Train/dev rows: 2159/273
- Device: CUDA on NVIDIA L20
- Batch size: 8
- Hidden dim: 256
- Residual Conv1d blocks: 4
- LR: `1e-3`
- Weight decay: `1e-4`
- Completed epochs: 20
- Steps: 5400
- Initial dev residual L1: `4.005409`
- Best dev residual L1: `1.685058`
- Best checkpoint: epoch 16, step 4320
- Final epoch-20 dev residual L1: `1.698058`
- Best checkpoint: private `/data` artifact, not committed

This run shows that the deterministic generator improves held-out dev residual
mel distance, not only the 32-sample overfit smoke set. The slight dev rebound
after epoch 16 means downstream demos should use `best_dev.pt`, not `last.pt`.

Step D diagnostic demo:

- Remote output root: `/data/qwen3-asr/synthesis/dsi_v1/residual_generator_v1_hubert_chinese/step_d_demo_gl_v1`
- Local copied private artifact root: `/Users/wuguocheng/Documents/Codex/2026-06-08/devbox-qwen-qwen3-asr-1-7b/artifacts/step_d_demo_gl_v1`
- Checkpoint: formal train/dev `best_dev.pt`
- Demo rows: 4, sampled from dev/test = 2/2
- Zero-shot buckets in demo: easy/medium/hard = 1/2/1
- Diagnostic inversion: `librosa.feature.inverse.mel_to_audio` with 64 Griffin-Lim iterations
- WAV sample rate: 16 kHz
- Diagnostic WAV files: 12 (`norm_gl`, `target_aligned_gl`, `pred_gl` for each row)
- Reference WAV copies: 8 (`norm_original`, `dys_original` for each row)
- Average baseline L1, `norm_mel -> target_mel`: `3.560273`
- Average predicted L1, `pred_mel -> target_mel`: `1.860956`
- Average relative L1 gain: `0.473916`

The Step D WAV files are for sanity checking only. Griffin-Lim inversion is
not a final neural vocoder, and the target mel is the DTW-aligned patient mel
on the normal-TTS time grid. Audio quality should therefore be interpreted
conservatively; the key result is that predicted mel moves substantially closer
to aligned patient mel before vocoder-quality work.

Step E0 HiFT compatibility demo:

- Remote output root: `/data/qwen3-asr/synthesis/dsi_v1/residual_generator_v1_hubert_chinese/step_e_demo_hift_global_v1`
- Local copied private artifact root: `/Users/wuguocheng/Documents/Codex/2026-06-08/devbox-qwen-qwen3-asr-1-7b/artifacts/step_e_demo_hift_global_v1`
- Vocoder: CosyVoice3 `CausalHiFTGenerator`
- Vocoder checkpoint: `/data/qwen3-asr/models/tts/Fun-CosyVoice3-0.5B-2512/hift.pt`
- Vocoder sample rate: 24 kHz
- Input predictions: Step D `step_d_demo_gl_v1/predictions`
- Demo rows: 4, matching Step D
- Calibration: global mean/std calibration from each utterance's normal-side
  Step B mel to CosyVoice3 normal-side mel
- Output WAV files: 12 (`norm_hift_global`, `target_hift_global`,
  `pred_hift_global` for each row)
- Reference WAV copies: 8 (`norm_original`, `dys_original` for each row)
- `pred_hift_global` duration range: `3.74` to `5.34` seconds
- `pred_hift_global` peak mean/max: `0.562667` / `0.845345`
- `pred_hift_global` RMS mean: `0.056658`

This is a stronger demo than Griffin-Lim because it uses a neural vocoder, but
it is still a compatibility experiment. CosyVoice3 HiFT is trained on 24 kHz
CosyVoice log-magnitude mel (`n_fft=1920`, `hop=480`, `center=False`), while
Step B residual features were built from 16 kHz torchaudio log-power mel
(`n_fft=400`, `hop=320`, `center=True`). The current script therefore uses
normal-side calibration before vocoding. The correct production version should
rebuild Step B features directly in the CosyVoice3 mel space or train a
matched vocoder.

Step E1 CosyVoice3-mel rebuild:

- Goal: rebuild the residual feature dataset in the exact mel space expected
  by CosyVoice3 HiFT, then retrain the residual generator.
- New feature root: `/data/qwen3-asr/synthesis/dsi_v1/residual_features_v2_cosyvoice3_mel`
- New generator root: `/data/qwen3-asr/synthesis/dsi_v1/residual_generator_v2_cosyvoice3_mel`
- Mel frontend: CosyVoice3/Matcha log-magnitude mel, 24 kHz, 80 bins,
  `n_fft=1920`, `win=1920`, `hop=480`, `center=False`, `fmin=0`, `fmax=None`
- SSL source: reused from Step B HuBERT features and interpolated to the new
  CosyVoice3 mel frame length
- Smoke build: 6 rows, generated 6/6, audit issues 0
- Full pipeline: started at
  `/data/qwen3-asr/synthesis/dsi_v1/residual_generator_v2_cosyvoice3_mel/e1_pipeline_20260615_144636`
- Full pipeline result: 2707/2707 features generated, sample audit issues 0,
  best dev residual L1 0.825369 at epoch 16 / step 4320.
- HiFT demo result: 12 dev/test listening triplets generated with no
  calibration, mean norm-to-target L1 2.195493 to predicted-target L1
  0.830757, mean relative L1 gain 0.611711.
- Local listening triplets:
  `/Users/wuguocheng/Documents/Codex/2026-06-08/devbox-qwen-qwen3-asr-1-7b/artifacts/step_e1_demo_hift_none_v1_listen_triplets`

Pipeline order:

1. Build full CosyVoice3-mel residual features.
2. Audit 32 sampled feature files.
3. Train residual generator with the same train/dev split and hyperparameters
   as Step C formal run.

See:

- `audit_sample32_summary.csv`
- `smoke_overfit_32_summary.csv`
- `train_dev_v1_h256_l4_lr1e3_bs8_epoch20_summary.csv`
- `step_d_demo_gl_v1_summary.csv`
- `step_e_demo_hift_global_v1_summary.csv`

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

python synthesis/dsi_v1/scripts/train_dsi_residual_generator.py \
  --feature-manifest /data/qwen3-asr/synthesis/dsi_v1/residual_features_v1_hubert_chinese/feature_manifest.csv \
  --out-dir /data/qwen3-asr/synthesis/dsi_v1/residual_generator_v1_hubert_chinese/train_dev_v1_h256_l4_lr1e3_bs8_epoch20 \
  --batch-size 8 \
  --epochs 20 \
  --lr 1e-3 \
  --hidden-dim 256 \
  --num-layers 4 \
  --smooth-weight 0.1 \
  --train-eval-limit 128 \
  --eval-every 250

python synthesis/dsi_v1/scripts/synthesize_dsi_residual_demo.py \
  --feature-manifest /data/qwen3-asr/synthesis/dsi_v1/residual_features_v1_hubert_chinese/feature_manifest.csv \
  --checkpoint /data/qwen3-asr/synthesis/dsi_v1/residual_generator_v1_hubert_chinese/train_dev_v1_h256_l4_lr1e3_bs8_epoch20/best_dev.pt \
  --out-dir /data/qwen3-asr/synthesis/dsi_v1/residual_generator_v1_hubert_chinese/step_d_demo_gl_v1 \
  --split dev \
  --split test \
  --limit-per-split 2 \
  --griffinlim-iters 64 \
  --copy-reference-wavs

export PYTHONPATH=/data/qwen3-asr/third_party/CosyVoice:/data/qwen3-asr/third_party/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH:-}
python synthesis/dsi_v1/scripts/vocode_dsi_hift_demo.py \
  --prediction-dir /data/qwen3-asr/synthesis/dsi_v1/residual_generator_v1_hubert_chinese/step_d_demo_gl_v1/predictions \
  --feature-manifest /data/qwen3-asr/synthesis/dsi_v1/residual_features_v1_hubert_chinese/feature_manifest.csv \
  --out-dir /data/qwen3-asr/synthesis/dsi_v1/residual_generator_v1_hubert_chinese/step_e_demo_hift_global_v1 \
  --calibration global \
  --copy-reference-wavs

python synthesis/dsi_v1/scripts/build_dsi_cosyvoice_mel_features.py \
  --source-feature-manifest /data/qwen3-asr/synthesis/dsi_v1/residual_features_v1_hubert_chinese/feature_manifest.csv \
  --out-dir /data/qwen3-asr/synthesis/dsi_v1/residual_features_v2_cosyvoice3_mel \
  --out-manifest /data/qwen3-asr/synthesis/dsi_v1/residual_features_v2_cosyvoice3_mel/feature_manifest.csv \
  --flush-every 25
```
