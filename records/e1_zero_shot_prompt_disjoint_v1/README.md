# E1 Zero-Shot: Prompt-Disjoint V1

This record stores the aggregate zero-shot Qwen3-ASR-1.7B baseline for the
prompt-disjoint v1 split.

The model was run once on the full clean read-sentence manifest, then the test
set table was materialized by filtering `split=test` from the same prediction
cache. This keeps the full-dataset and test-set metrics tied to the same
zero-shot predictions.

## Main Tables

- `test_set_performance.csv`: aggregate performance on the prompt-disjoint test
  set only.
- `full_dataset_performance.csv`: aggregate performance on all clean
  read-sentence rows in prompt-disjoint v1.

Both tables include:

- overall row
- E1 zero-shot difficulty bucket rows
- disease-tag rows
- duration-bucket rows
- split rows

## Overall Results

| scope | samples | avg_textnorm_cer | avg_critical_error | critical_error_count |
|---|---:|---:|---:|---:|
| test_set | 275 | 0.290028 | 0.203636 | 56 |
| full_dataset | 2707 | 0.281067 | 0.208349 | 564 |

Critical error rule:

```text
inference_error_or_blank_or_textnorm_cer>=0.5
```

No inference errors occurred in this run.

## Private Outputs

The per-utterance prediction tables are private because they contain patient
transcripts, model predictions, audio paths, and raw speaker identifiers. They
are indexed in `private_manifest.yaml`.
