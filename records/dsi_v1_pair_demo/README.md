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
- Normal TTS side: Cantonese CosyVoice2 model, cached zero-shot speaker
- TTS model: `ASLP-lab/Cosyvoice2-Yue`
- CosyVoice code: `/data/qwen3-asr/third_party/CosyVoice`
- Remote output root: `/data/qwen3-asr/synthesis/dsi_v1/pair_demo`

The private generated manifest contains patient text and absolute audio paths,
so it is not committed. See `private_manifest.yaml`.

## Public Summary

See `pair_demo_public_summary.csv`.

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
  --tts-root /data/qwen3-asr/synthesis/dsi_v1/pair_demo/norm_tts_wav \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_demo/pair_demo_manifest.csv \
  --out-jsonl /data/qwen3-asr/synthesis/dsi_v1/pair_demo/pair_demo_manifest.jsonl

python synthesis/dsi_v1/scripts/generate_cosyvoice_demo_pairs.py \
  --pair-manifest /data/qwen3-asr/synthesis/dsi_v1/pair_demo/pair_demo_manifest.csv \
  --cosyvoice-repo /data/qwen3-asr/third_party/CosyVoice \
  --model-dir /data/qwen3-asr/models/tts/Cosyvoice2-Yue \
  --mode cached_zero_shot \
  --speaker-id my_zero_shot_spk \
  --out-csv /data/qwen3-asr/synthesis/dsi_v1/pair_demo/pair_demo_manifest.generated.csv
```

## Notes

This is only the pair-data demo gate. It does not yet run SSL feature
extraction, DTW alignment, residual training, or reconstruction evaluation.
