# DSI V1 Pair-Data Demo

This directory contains the first data-building step for the deterministic
residual generator line.

Goal:

```text
normal Cantonese TTS audio + patient_id -> patient-style dysarthric mel
```

The pair-data schema keeps the real dysarthric utterance as the target side and
adds a normal TTS waveform with the same cleaned prompt:

```text
utt_id
patient_id
clean_text
jyutping
dys_wav_path
norm_wav_path
split
prompt_id
```

For the demo, rows are sampled from the existing `prompt_disjoint_v1` manifests,
so the train/dev/test split and `prompt_id` audit remain intact.

## Demo Pipeline

Create a small pair manifest:

```bash
python synthesis/dsi_v1/scripts/prepare_pair_demo_manifest.py \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_train.jsonl \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_dev.jsonl \
  --input-jsonl /data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_test.jsonl \
  --per-split 2 \
  --tts-root /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v2_edge/norm_tts_wav \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v2_edge/pair_demo_manifest.csv \
  --out-jsonl /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v2_edge/pair_demo_manifest.jsonl
```

Generate normal Cantonese TTS waveforms with Microsoft Edge neural TTS Hong Kong
Cantonese voice:

```bash
python synthesis/dsi_v1/scripts/generate_edge_tts_demo_pairs.py \
  --pair-manifest /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v2_edge/pair_demo_manifest.csv \
  --voice zh-HK-HiuGaaiNeural \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v2_edge/pair_demo_manifest.generated.csv
```

The generated audio and private manifests stay outside Git. Public records can
store aggregate counts, commands, model choice, and checksum summaries.

## CosyVoice Compatibility Note

CosyVoice2-Yue needs the WenetSpeech-Yue Space code path and a compatible
`transformers/tokenizers` pair. The current Qwen-ASR environment has newer
packages that made CosyVoice2 produce long, repeated, content-mismatched audio.
The repaired route is:

```bash
python synthesis/dsi_v1/scripts/generate_cosyvoice_demo_pairs.py \
  --pair-manifest /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v3_cosyvoice_compat/pair_demo_manifest.csv \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_demo_v3_cosyvoice_compat/pair_demo_manifest.generated.csv \
  --cosyvoice-repo /data/qwen3-asr/third_party/WenetSpeech-Yue-TTS-code-git \
  --pythonpath-prepend /data/qwen3-asr/overlays/cosyvoice-transformers451 \
  --model-dir /data/qwen3-asr/models/tts/WSYue-TTS/CosyVoice2-Yue-ZoengJyutGaai/CosyVoice2-yue-zjg \
  --mode instruct2 \
  --prompt-wav /data/qwen3-asr/third_party/WenetSpeech-Yue-TTS-code-git/asset/sg_017_090.wav \
  --instruction "用粤语说这句话" \
  --text-frontend false
```

The compatibility overlay only contains Python packages, not model weights.
Keep Qwen-ASR validation results beside the private generated manifest before
using CosyVoice-normal audio for downstream residual-generator experiments.

## TTS Setting V1

The first controlled normal-TTS setting for DSI uses one fixed neutral Cantonese
speaker and removes nuisance variation before residual modeling:

- `ASLP-lab/Cosyvoice2-Yue`
- prompt wav `F01_中立_20054.wav`
- instruction `用粤语说这句话`
- text frontend disabled because `clean_text` is already normalized Cantonese
- speed `0.9`
- resample to 16 kHz
- RMS-normalize to `-23 dBFS`
- no noise, reverb, pitch, emotion, or energy augmentation

The TTS-side audio should be treated as the normal reference only after ASR
readback validation passes the non-critical gate.
