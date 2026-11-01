# Setting D LoRA Baseline (step-selected)

Setting D is the user-audited successor to Setting C. It preserves the original
content-disjoint dev/test splits, removes `vlink201`, `vlink203`, `vlink225`,
`vlink243`, and removes noisy utterance
`vlink211/7290411755587742893228` from train.

## Data

- train: 2,117 utterances, 66 speakers, 48 contents
- dev: 258 utterances, 36 speakers, 6 contents
- test: 253 utterances, 33 speakers, 6 contents
- cross-split overlap: zero for prompt id, cleaned text, normalized text, and Jyutping

## Baseline protocol

- base model: Qwen3-ASR-1.7B
- LoRA: current-default targets, rank 16, alpha 4 (scale 0.25), dropout 0.05
- optimizer schedule: effective batch 32, learning rate 2e-4, linear decay, seed 42
- training: 140 optimizer steps
- supervision: generative dev CER every 10 steps
- checkpoint selection: minimum overall dev CER; test was evaluated once after selection

## Results

| split/group | zero-shot CER | LoRA CER | delta |
|---|---:|---:|---:|
| dev overall | 26.98% | **19.00%** | -7.98 pp |
| dev easy | 7.20% | 6.62% | -0.58 pp |
| dev medium | 28.65% | 18.61% | -10.05 pp |
| dev hard | 73.89% | 50.22% | -23.66 pp |
| test overall | **24.39%** | 24.59% | +0.20 pp |
| test easy | **5.60%** | 8.04% | +2.44 pp |
| test medium | **28.89%** | 29.95% | +1.07 pp |
| test hard | 69.59% | **62.16%** | -7.43 pp |

The selected checkpoint is step 50 (about 0.75 effective epoch) from the rerun with
the corrected Trainer gradient-accumulation normalization. Dev CER reached
its minimum at step 50, then rose with continued training, demonstrating strong
small-data overfitting. The LoRA baseline improves hard test speech but does not
beat zero-shot on overall test CER because easy and medium speech regress.

Authoritative artifacts:

- run summary: `/data/qwen3-asr/finetune/setting_d_gradacc_fixed_v1/lora_current_default_r16_scale025_step140_seed42_v1/run_summary.json`
- step curve: `/data/qwen3-asr/finetune/setting_d_gradacc_fixed_v1/lora_current_default_r16_scale025_step140_seed42_v1/dev_cer_by_step.csv`
- best checkpoint: `/data/qwen3-asr/finetune/setting_d_gradacc_fixed_v1/lora_current_default_r16_scale025_step140_seed42_v1/step_checkpoints/checkpoint-50`
