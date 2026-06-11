# E3 LoRA Target-Preset Sweep: Prompt-Disjoint V1

This record stores a one-epoch LoRA target-module sweep on the prompt-disjoint
v1 split. The sweep first tests audio-side adaptation while keeping the text
decoder and LM head frozen, then adds a `decoder_only` control to identify how
much of the default LoRA gain comes from the text decoder.

Setup:

- base model: `/data/qwen3-asr/models/Qwen3-ASR-1.7B`
- split: `prompt_disjoint_v1`
- train/dev/test samples: `2159/273/275`
- LoRA rank: `16`
- LoRA scale: `0.25` (`alpha = 4.0`)
- LoRA dropout: `0.05`
- learning rate: `2e-4`
- epochs per run: `1`
- checkpoint evaluated: `checkpoint-68`

The zero-shot reference from the same test evaluation table is:

| model | CER | critical count |
|---|---:|---:|
| zero-shot | 0.302886 | 62 |

## Test Result

| method | target preset | trainable params | CER | critical count | hard CER | medium CER | easy CER |
|---|---|---:|---:|---:|---:|---:|---:|
| default LoRA scale=0.25 step68 | `current_default` | 19.79M | 0.214282 | 43 | 0.555690 | 0.227528 | 0.049834 |
| decoder only | `decoder_only` | 17.43M | 0.226799 | 43 | 0.594879 | 0.240420 | 0.049885 |
| audio attention q/k/v | `audio_attn_qkv` | 2.36M | 0.258753 | 50 | 0.635308 | 0.281677 | 0.072571 |
| audio attention q/k/v/out | `audio_attn_qkvo` | 3.15M | 0.267855 | 58 | 0.642737 | 0.299555 | 0.077372 |
| projector proj1/proj2 | `audio_projector` | 0.08M | 0.281894 | 57 | 0.719584 | 0.305538 | 0.067221 |
| conv_out + projector | `audio_adapter_convout_proj` | 0.22M | 0.279774 | 60 | 0.704980 | 0.298985 | 0.073394 |
| audio encoder all layers | `audio_encoder_all` | 7.08M | 0.250830 | 53 | 0.635367 | 0.271368 | 0.062362 |

Interpretation: `decoder_only` recovers most of the one-epoch default LoRA gain,
including the same overall critical count. This means the text decoder is the
dominant contributor. The default LoRA still has better overall CER and hard CER
than `decoder_only`, so audio q/k/v likely adds useful hard-sample correction in
combination with decoder adaptation. All audio-side-only LoRA presets improve
over zero-shot, but none beat the default LoRA baseline. Among audio-side-only
runs, `audio_encoder_all` has the best overall CER, while `audio_attn_qkv` has
the best critical count. Adding audio `out_proj` hurts relative to q/k/v only,
and projector-only adaptation is too weak for this split.

## Public Files

- `target_preset_test_summary.csv`
- `target_preset_test_summary_with_decoder.csv`
- `run_config.json`

Full per-utterance predictions and checkpoints are private and remain under:

- `/data/qwen3-asr/finetune/prompt_disjoint_v1/e3_lora_target_preset_1epoch`
- `/data/qwen3-asr/finetune/prompt_disjoint_v1/e3_lora_decoder_only_1epoch`
