# E3 LoRA Scale Sweep: Prompt-Disjoint V1

This record stores a one-epoch LoRA scaling sweep on the prompt-disjoint v1
split.

Setup:

- base model: `/data/qwen3-asr/models/Qwen3-ASR-1.7B`
- split: `prompt_disjoint_v1`
- train samples: `2159`
- dev samples: `273`
- test samples: `275`
- LoRA rank: `16`
- LoRA dropout: `0.05`
- learning rate: `2e-4`
- epochs per run: `1`
- target modules: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

Here `lora_scale` means the LoRA adapter multiplier `alpha / rank`. For this
sweep, `alpha = rank * lora_scale`.

## Test Result

| lora_scale | alpha | CER | critical count | hard CER | medium CER | easy CER |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 1.6 | 0.233107 | 45 | 0.619991 | 0.247614 | 0.047046 |
| 0.25 | 4.0 | 0.214282 | 43 | 0.555690 | 0.227528 | 0.049834 |
| 0.50 | 8.0 | 0.233896 | 46 | 0.552569 | 0.264476 | 0.069874 |

The zero-shot reference from the same evaluation table is:

| model | CER | critical count |
|---|---:|---:|
| zero-shot | 0.302886 | 62 |

Interpretation: reducing LoRA scale substantially improves the one-epoch LoRA
result compared with zero-shot and avoids the severe easy/medium regression
seen in the earlier high-scale multi-epoch LoRA run. In this sweep, `0.25` is
the best overall setting. `0.5` improves hard CER slightly more than `0.25`, but
it hurts medium and easy samples, so its overall CER and critical count are
worse.

## Public Files

- `sweep_test_overall.csv`
- `test_focus_summary_scale_0p1.csv`
- `test_focus_summary_scale_0p25.csv`
- `test_focus_summary_scale_0p5.csv`
- `private_manifest.yaml`

Full per-utterance predictions and audio paths are private and are not stored
in git.
