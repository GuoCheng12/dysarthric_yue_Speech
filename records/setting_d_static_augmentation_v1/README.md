# Setting D static Speech Generator augmentation

This record preserves the two results that motivate the closed-loop project. The
one-off training and report builders were removed during repository cleanup; immutable
source artifacts remain at the private paths below.

## Formal dev-selected Full-1440 run

Training used 2,117 real Setting D utterances plus 1,440 Speech Generator utterances.
The Qwen3-ASR LoRA configuration matched the real-only baseline. Checkpoint 60 was
selected on the 258-utterance dev split; the 253-utterance test was not used for
selection.

| method | test overall CER | easy | medium | hard |
|---|---:|---:|---:|---:|
| zero-shot | 24.39% | 5.60% | 28.89% | 69.59% |
| real-only LoRA, step 40 | 24.93% | 8.39% | 29.68% | 63.44% |
| real + Speech Generator 1440, step 60 | 25.21% | 8.16% | 30.21% | 64.73% |

Static augmentation therefore did not beat real-only on this formal run. More synthetic
data alone is not the proposed contribution.

Source:
`/data/qwen3-asr/finetune/setting_d_highinfo1440_gradacc_fixed_v1/lora_current_default_r16_scale025_step140_seed42_v1/run_summary.json`

## Matched patient-TTS control

An exploratory A/B/C control held the 1,440 texts, references, quantities, seeds, TTS
family, and ASR configuration fixed. Its 511 real utterances merged the former dev and
test and were used for checkpoint selection, so these numbers are diagnostic rather
than an unbiased test result.

| arm | selected step | overall CER | hard CER |
|---|---:|---:|---:|
| A: real only | 50 | 22.06% | 57.23% |
| B: real + patient Speech Generator | 60 | 22.33% | 56.37% |
| C: real + official CosyVoice3 | 40 | 24.69% | 63.82% |

B was 7.45 percentage points better than C on hard CER, with a speaker-cluster
bootstrap probability of 0.9992 that B was better. B versus A was only -0.86 points on
hard CER and its 95% interval crossed zero. This supports that patient-conditioned TTS
contains ASR-usable information relative to ordinary synthesis, while also showing that
static use of that information is not yet reliably better than real-only training.

Source:
`/data/qwen3-asr/finetune/setting_d_tts_control_ablation_v1/acceptance_report.json`

Methodology warning: the matched control must never be reported as the final Setting D
test estimate.
