# Prompt Leakage Check: E2/E3 Test Split

This record checks whether cleaned test prompts also appear in the cleaned training split.

Checked layers:

- raw text exact match using `raw_gt`
- normalized text exact match using the project TextNorm-style normalizer on `clean_gt`
- Jyutping sequence exact match using `pycantonese.characters_to_jyutping()`

## Result

| layer | seen-prompt test | unseen-prompt test | seen rate |
|---|---:|---:|---:|
| raw | 251 | 7 | 0.972868 |
| normalized | 251 | 7 | 0.972868 |
| Jyutping | 251 | 7 | 0.972868 |
| any layer | 251 | 7 | 0.972868 |

The three layers produce the same seen/unseen split. Seen prompts are highly repeated in train: among seen test rows, the raw train match count ranges from `20` to `82`, with mean `33.11`.

The 7 unseen-prompt test rows are all from `vlink_Kaho`; 6 are zero-shot hard and 1 is zero-shot easy.

## Seen/Unseen Test Metrics

Using the `any_layer` split:

| subset | n | zero-shot CER | E2 full-SFT CER | E3 LoRA r16 CER | zero critical | E2 critical | E3 critical |
|---|---:|---:|---:|---:|---:|---:|---:|
| seen-prompt test | 251 | 0.266110 | 0.126173 | 0.033577 | 47 | 18 | 5 |
| unseen-prompt test | 7 | 0.678906 | 0.608328 | 0.359307 | 6 | 5 | 2 |

Interpretation: the current test set is heavily prompt-seen. The headline test result should therefore be reported with separate seen-prompt and unseen-prompt rows, and future split design should make prompt identity disjoint when the goal is prompt generalization.

## Files

- `test_prompt_leakage_summary.csv`
- `test_seen_unseen_metrics_by_layer.csv`
- `test_prompt_leakage_per_sample_sanitized.csv`

The per-sample CSV is sanitized: it excludes transcript text, Jyutping strings, model predictions, audio paths, and raw speaker IDs.

## Private Detail Table

The private table includes raw/clean text, Jyutping sequences, and matched train utterance IDs:

```text
/data/qwen3-asr/finetune/prompt_leakage/e2_e3_test/test_prompt_leakage_private_matches.csv
```

SHA256:

```text
1b4d1f006def3bfd73dc2d4c41abd6c23c52b16b992eedc61353fe29f0188528
```
