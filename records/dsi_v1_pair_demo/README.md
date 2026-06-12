# DSI V1 Pair-Data Demo

This record documents the first demo pair-data build for the deterministic
residual generator line.

Goal:

```text
normal Cantonese TTS audio + patient_id -> patient-style dysarthric mel
```

Demo status:

- Source split: `prompt_disjoint_v1`
- Demo size: 6 pairs
- Split coverage: train/dev/test = 2/2/2
- Real dysarthric side: existing cleaned read-sentence patient audio
- Normal TTS side: Microsoft Edge neural TTS Hong Kong Cantonese voice
- TTS backend: `edge-tts`
- TTS voice: `zh-HK-HiuGaaiNeural`
- Remote output root: `/data/qwen3-asr/synthesis/dsi_v1/pair_demo_v2_edge`
- Status: accepted as pair-data demo after ASR sanity check

CosyVoice compatibility retest:

- Remote output root: `/data/qwen3-asr/synthesis/dsi_v1/pair_demo_v3_cosyvoice_compat`
- TTS backend: `ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai`
- Code path: `ASLP-lab/WenetSpeech-Yue-TTS` Space code
- Compatibility overlay: `transformers==4.51.3`, `tokenizers==0.21.4`
- Added disk footprint: about 84 MB for the Python overlay plus sub-MB demo wavs
- ASR sanity result: 6/6 non-critical, average TTS TextNorm_CER `0.220085`
- Status: usable for controlled DSI V1 experiments, but keep the ASR readback
  caveat visible because some short prompts still show homophone/word-choice
  drift such as `幫手包` -> `雙手抱` and `啤牌` -> `pair 牌`

TTS setting V1:

- Language: Cantonese/Yue
- Speaker: fixed single high-quality neutral speaker
- Backend/model: `ASLP-lab/Cosyvoice2-Yue`
- Prompt: `F01_中立_20054.wav`
- Style/emotion: neutral, no emotion control
- Speed: `0.9`
- Pitch/energy: default
- Output sample rate: resampled to 16 kHz
- Loudness: RMS-normalized to `-23 dBFS`
- Noise/reverb: none
- Config record: `tts_setting_v1.yaml`
- Remote output root: `/data/qwen3-asr/synthesis/dsi_v1/pair_demo_v5_tts_setting_v1`
- ASR sanity result: 6/6 non-critical, average TTS TextNorm_CER `0.171652`

Rejected earlier attempts:

- `/data/qwen3-asr/synthesis/dsi_v1/pair_demo`: CosyVoice2 cached speaker demo.
  User listening check found the audio unusable.
- `/data/qwen3-asr/synthesis/dsi_v1/pair_demo_ab_test`: local TTS A/B scratch
  outputs. CosyVoice and VITS candidates were rejected because Qwen3-ASR
  readback showed severe content mismatch.
- Earlier CosyVoice attempts used an incompatible local route: latest
  `transformers/tokenizers` plus path-style `prompt_wav` for `instruct2`.
  The fixed route uses the official Space code, passes `load_wav(..., 16000)`
  as `prompt_speech_16k`, disables text frontend for already-clean Cantonese
  prompts, and prepends the compatibility overlay before importing CosyVoice.

The private generated manifest contains patient text and absolute audio paths,
so it is not committed. See `private_manifest.yaml`.

## Public Summary

See `pair_demo_public_summary.csv`.

For the CosyVoice compatibility retest, see
`cosyvoice_compat_public_summary.csv`.

For the neutral V1 TTS setting, see `tts_setting_v1_public_summary.csv`.

## Reproduction Commands

```bash
cd /data/qwen3-asr/repo/dysarthric_yue_Speech
source /data/qwen3-asr/env.sh
source /data/qwen3-asr/venvs/qwen3-asr/bin/activate
export PYTHONPATH=/data/qwen3-asr/third_party/CosyVoice:/data/qwen3-asr/third_party/CosyVoice/third_party/Matcha-TTS

python synthesis/dsi_v1/scripts/prepare_pair_demo_manifest.py \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_train.jsonl \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_dev.jsonl \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_test.jsonl \
  --per-split 2 \
  --tts-root /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v2_edge/norm_tts_wav \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v2_edge/pair_demo_manifest.csv \
  --out-jsonl /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v2_edge/pair_demo_manifest.jsonl

python synthesis/dsi_v1/scripts/generate_edge_tts_demo_pairs.py \
  --pair-manifest /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v2_edge/pair_demo_manifest.csv \
  --voice zh-HK-HiuGaaiNeural \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v2_edge/pair_demo_manifest.generated.csv
```

CosyVoice compatibility retest:

```bash
cd /data/qwen3-asr/repo/dysarthric_yue_Speech
source /data/qwen3-asr/env.sh
source /data/qwen3-asr/venvs/qwen3-asr/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

python synthesis/dsi_v1/scripts/prepare_pair_demo_manifest.py \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_train.jsonl \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_dev.jsonl \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_test.jsonl \
  --per-split 2 \
  --tts-root /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v3_cosyvoice_compat/norm_tts_wav \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v3_cosyvoice_compat/pair_demo_manifest.csv \
  --out-jsonl /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v3_cosyvoice_compat/pair_demo_manifest.jsonl

python synthesis/dsi_v1/scripts/generate_cosyvoice_demo_pairs.py \
  --pair-manifest /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v3_cosyvoice_compat/pair_demo_manifest.csv \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v3_cosyvoice_compat/pair_demo_manifest.generated.csv \
  --cosyvoice-repo /data/qwen3-asr/third_party/WenetSpeech-Yue-TTS-code-git \
  --pythonpath-prepend /data/qwen3-asr/overlays/cosyvoice-transformers451 \
  --model-dir /data/qwen3-asr/models/tts/WSYue-TTS/CosyVoice2-Yue-ZoengJyutGaai/CosyVoice2-yue-zjg \
  --mode instruct2 \
  --prompt-wav /data/qwen3-asr/third_party/WenetSpeech-Yue-TTS-code-git/asset/sg_017_090.wav \
  --instruction "用粤语说这句话" \
  --text-frontend false \
  --overwrite
```

TTS setting V1 candidate:

```bash
cd /data/qwen3-asr/repo/dysarthric_yue_Speech
source /data/qwen3-asr/env.sh
source /data/qwen3-asr/venvs/qwen3-asr/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

python synthesis/dsi_v1/scripts/prepare_pair_demo_manifest.py \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_train.jsonl \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_dev.jsonl \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_test.jsonl \
  --per-split 2 \
  --tts-root /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v5_tts_setting_v1/norm_tts_wav \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v5_tts_setting_v1/pair_demo_manifest.csv \
  --out-jsonl /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v5_tts_setting_v1/pair_demo_manifest.jsonl

python synthesis/dsi_v1/scripts/generate_cosyvoice_demo_pairs.py \
  --pair-manifest /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v5_tts_setting_v1/pair_demo_manifest.csv \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v5_tts_setting_v1/pair_demo_manifest.generated.csv \
  --cosyvoice-repo /data/qwen3-asr/third_party/WenetSpeech-Yue-TTS-code-git \
  --pythonpath-prepend /data/qwen3-asr/overlays/cosyvoice-transformers451 \
  --model-dir /data/qwen3-asr/models/tts/Cosyvoice2-Yue \
  --mode instruct2 \
  --prompt-wav /data/qwen3-asr/third_party/WenetSpeech-Yue-TTS-code-git/asset/F01_中立_20054.wav \
  --instruction "用粤语说这句话" \
  --text-frontend false \
  --speed 0.9 \
  --target-sample-rate 16000 \
  --target-rms-dbfs -23.0 \
  --overwrite
```

Full DSI V1 pair-data build uses the same setting with `--all`:

```bash
cd /data/qwen3-asr/repo/dysarthric_yue_Speech
source /data/qwen3-asr/env.sh
source /data/qwen3-asr/venvs/qwen3-asr/bin/activate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

python synthesis/dsi_v1/scripts/prepare_pair_demo_manifest.py \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_train.jsonl \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_dev.jsonl \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_test.jsonl \
  --all \
  --tts-root /data/qwen3-asr/synthesis/dsi_v1/pair_data_v1_tts_setting_v1/norm_tts_wav \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_data_v1_tts_setting_v1/pair_manifest.csv \
  --out-jsonl /data/qwen3-asr/synthesis/dsi_v1/pair_data_v1_tts_setting_v1/pair_manifest.jsonl

python synthesis/dsi_v1/scripts/generate_cosyvoice_demo_pairs.py \
  --pair-manifest /data/qwen3-asr/synthesis/dsi_v1/pair_data_v1_tts_setting_v1/pair_manifest.csv \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_data_v1_tts_setting_v1/pair_manifest.generated.csv \
  --cosyvoice-repo /data/qwen3-asr/third_party/WenetSpeech-Yue-TTS-code-git \
  --pythonpath-prepend /data/qwen3-asr/overlays/cosyvoice-transformers451 \
  --model-dir /data/qwen3-asr/models/tts/Cosyvoice2-Yue \
  --mode instruct2 \
  --prompt-wav /data/qwen3-asr/third_party/WenetSpeech-Yue-TTS-code-git/asset/F01_中立_20054.wav \
  --instruction "用粤语说这句话" \
  --text-frontend false \
  --speed 0.9 \
  --target-sample-rate 16000 \
  --target-rms-dbfs -23.0 \
  --flush-every 1
```

## Notes

This is only the pair-data demo gate. It does not yet run SSL feature
extraction, DTW alignment, residual training, or reconstruction evaluation.
