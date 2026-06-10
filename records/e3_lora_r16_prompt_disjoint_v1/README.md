# E3 LoRA r16: Prompt-Disjoint V1

This record stores the prompt-disjoint v1 LoRA r16 training check.

Training setup:

- base model: `/data/qwen3-asr/models/Qwen3-ASR-1.7B`
- train split: `/data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_train.jsonl`
- dev split: `/data/qwen3-asr/finetune/data_prompt_disjoint_v1/prompt_disjoint_dev.jsonl`
- epochs: `8`
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- save/eval interval: `68` steps
- output: `/data/qwen3-asr/finetune/prompt_disjoint_v1/e3_lora_r16_8epoch`

## Result

Training completed all `544` steps. However, dev CER selection shows that all
LoRA checkpoints are worse than the zero-shot baseline on prompt-disjoint dev.

Best checkpoint by dev CER:

```text
checkpoint-68
```

Dev best-checkpoint metrics:

| checkpoint | epoch | dev CER | dev critical rate | dev critical count |
|---|---:|---:|---:|---:|
| checkpoint-68 | 1 | 0.485896 | 0.553114 | 151 |

Test metrics for the best-dev checkpoint:

| model | test CER | test critical rate | test critical count |
|---|---:|---:|---:|
| zero-shot | 0.290028 | 0.203636 | 56 |
| LoRA r16 checkpoint-68 | 0.443592 | 0.410909 | 113 |

Interpretation: on the prompt-disjoint split, this LoRA r16 run does not improve
open-prompt ASR. It slightly helps the hard bucket but damages easy and medium
samples enough that overall performance is much worse.

## Public Files

- `dev_checkpoint_selection_summary.csv`
- `test_best_dev_checkpoint_68_focus_summary.csv`
- `lora_run_metadata.json`
- `private_manifest.yaml`

The full per-utterance prediction tables are private because they contain
patient transcripts, predictions, audio paths, and raw speaker identifiers.
