# Repository Management Policy

This repository is the management root for code, resources, data versions, experiment protocols, and aggregate results.

## Principle

Everything important should be represented in git, but not every byte belongs in git.

The repository should contain:

- source code
- runbooks and protocols
- data-cleaning and split definitions
- dataset and artifact registry files
- dependency and environment templates
- public aggregate metrics
- scripts that reproduce private artifacts from private inputs

The repository should not contain:

- patient audio payloads
- raw private transcripts
- per-utterance private prediction tables
- model weights
- full fine-tuned checkpoints
- optimizer states
- local credentials, proxy configs, or Codex auth state

## Required Registry Entries

Whenever a new dataset, model, checkpoint, or generated result becomes part of the project, add a small registry entry under one of:

- `data/registry/`
- `artifacts/registry/`

Each entry should include:

- stable name
- version or date
- privacy level
- storage location or retrieval procedure
- checksum if available
- expected file count or size
- generation command or source script
- downstream experiments that depend on it

## Experiment Rule

Every new method should be runnable from the repository with private paths supplied by environment variables or command-line flags.

Every evaluation after Experiment 2 must compare:

1. `zero_shot`
2. `E2_fullSFT_clean_pooled_epoch2`
3. `new_method`

The canonical helper is:

```bash
python finetune/scripts/build_three_way_comparison.py \
  --baseline-predictions path/to/E2/test_predictions.csv \
  --new-predictions path/to/new_method_test_predictions.csv \
  --new-method-name METHOD_NAME \
  --out-dir finetune/experiments/METHOD_NAME/eval_test
```

## Commit Hygiene

Before committing, check:

```bash
git status --short
git ls-files | rg '\\.(wav|flac|mp3|m4a|zip|jsonl|safetensors|bin|pt|pth|ckpt)$|(^|/)auth\\.|(^|/)config\\.yaml$|checkpoint-'
git grep -n -E 'auth\\.json|auth\\.toml|password|secret|token='
```

The second command should return no tracked payload files. The third command should only match ignore rules or benign tokenizer-related code.
