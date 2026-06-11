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
  --tts-root /data/qwen3-asr/synthesis/dsi_v1/pair_demo/norm_tts_wav \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_demo/pair_demo_manifest.csv \
  --out-jsonl /data/qwen3-asr/synthesis/dsi_v1/pair_demo/pair_demo_manifest.jsonl
```

Generate normal Cantonese TTS waveforms with a Cantonese CosyVoice model:

```bash
python synthesis/dsi_v1/scripts/generate_cosyvoice_demo_pairs.py \
  --pair-manifest /data/qwen3-asr/synthesis/dsi_v1/pair_demo/pair_demo_manifest.csv \
  --cosyvoice-repo /data/qwen3-asr/third_party/CosyVoice \
  --model-dir /data/qwen3-asr/models/tts/Cosyvoice2-Yue \
  --mode cached_zero_shot \
  --speaker-id my_zero_shot_spk \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_demo/pair_demo_manifest.generated.csv
```

The generated audio and private manifests stay outside Git. Public records can
store aggregate counts, commands, model choice, and checksum summaries.
