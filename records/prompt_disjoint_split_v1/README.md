# Prompt-Disjoint Split V1

This record defines the new prompt-disjoint clean read-sentence split for the
next zero-shot, full-SFT, and LoRA experiments.

The split unit is not a filename. Each utterance is assigned to a prompt group
using connected components over three text identity layers:

- raw `clean_gt`
- normalized `clean_gt`
- Jyutping sequence

This means that if two rows share any one of those identities, directly or
through a chain of equivalent rows, they must stay in the same split. The goal
is to prevent prompt leakage from visually different but pronunciation-equivalent
Cantonese text.

## Split Summary

| split | samples | prompt groups | speakers | tracked hard speaker rows | easy | medium | hard | HD | SCA | unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2159 | 99 | 71 | 54 | 1092 | 608 | 459 | 96 | 208 | 1855 |
| dev | 273 | 12 | 61 | 6 | 140 | 78 | 55 | 12 | 24 | 237 |
| test | 275 | 12 | 43 | 6 | 135 | 78 | 62 | 12 | 26 | 237 |

Input rows: `2708`; unique utterances after duplicate `utt_id` removal:
`2707`. One duplicate input row was skipped.

## Overlap Audit

| layer | train-dev overlap | train-test overlap | dev-test overlap |
|---|---:|---:|---:|
| prompt_id | 0 | 0 | 0 |
| raw clean text hash | 0 | 0 | 0 |
| normalized clean text hash | 0 | 0 | 0 |
| Jyutping sequence hash | 0 | 0 | 0 |

## Public Files

- `split_audit_by_split_public.csv`
- `split_speaker_counts_sanitized.csv`
- `prompt_overlap_audit.csv`
- `prompt_group_split_assignment.csv`
- `prompt_group_key_audit_sanitized.csv`
- `private_manifest.yaml`

The full manifest and SFT JSONL files are private because they contain patient
transcripts, audio paths, and raw speaker identifiers. Step 4 has not been
started from this split.
