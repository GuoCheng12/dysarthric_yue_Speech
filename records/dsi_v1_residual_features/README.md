# DSI V1 Residual Feature Dataset

This record tracks Step B of the deterministic residual generator line.

Input pair data:

- Normal side: CosyVoice3 TTS setting V1
- Dysarthric side: cleaned patient audio
- Source split: `prompt_disjoint_v1`
- Pair root: `/data/qwen3-asr/synthesis/dsi_v1/pair_data_v1_cosyvoice3_tts_setting_v1`

Feature build:

- Feature root: `/data/qwen3-asr/synthesis/dsi_v1/residual_features_v1_hubert_chinese`
- Manifest: `/data/qwen3-asr/synthesis/dsi_v1/residual_features_v1_hubert_chinese/feature_manifest.csv`
- SSL model: `TencentGameMate/chinese-hubert-base`
- Local SSL model root: `/data/qwen3-asr/models/ssl/chinese-hubert-base`
- Mel: 80-bin log-mel, 16 kHz, `n_fft=400`, `win_length=400`, `hop_length=320`
- DTW metric: cosine over z-normalized mel frames
- Feature dtype: float16
- Output time grid: normal-TTS mel frames

Each private `.pt` file contains:

```text
norm_mel
dys_mel_aligned
residual_mel = dys_mel_aligned - norm_mel
norm_ssl
dys_ssl_aligned
dtw_norm_to_dys_path
```

Build result:

- Rows: 2707
- Completed: 2707
- Errors: 0
- Split counts: train/dev/test = 2159/273/275
- Zero-shot bucket counts: easy/medium/hard = 1367/764/576
- Output size: about 2.2 GB

See `residual_features_v1_hubert_chinese_public_summary.csv`.

Reproduction command:

```bash
cd /data/qwen3-asr/repo/dysarthric_yue_Speech
source /data/qwen3-asr/env.sh
source /data/qwen3-asr/venvs/qwen3-asr/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export PYTHONPATH=/data/qwen3-asr/overlays/cosyvoice-transformers451:${PYTHONPATH:-}

python synthesis/dsi_v1/scripts/build_dsi_residual_features.py \
  --generated-csv /data/qwen3-asr/synthesis/dsi_v1/pair_data_v1_cosyvoice3_tts_setting_v1/pair_manifest.generated.csv \
  --out-dir /data/qwen3-asr/synthesis/dsi_v1/residual_features_v1_hubert_chinese \
  --out-manifest /data/qwen3-asr/synthesis/dsi_v1/residual_features_v1_hubert_chinese/feature_manifest.csv \
  --ssl-model /data/qwen3-asr/models/ssl/chinese-hubert-base \
  --flush-every 25
```

The `PYTHONPATH` overlay is required in the current DevBox because the main
`transformers` version refuses to load `.bin` weights under `torch==2.5.1`.
The overlay uses `transformers==4.51.3` only for this build step.
