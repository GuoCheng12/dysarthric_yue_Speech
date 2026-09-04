# Repository management policy

Everything important must be represented in Git, but private or large payloads stay
outside it.

Tracked material includes source code, frozen protocols, split and artifact registries,
sanitized aggregate results, and instructions that reproduce private artifacts. Patient
audio and transcripts, row-level predictions, model weights, checkpoints, optimizer
states, API keys, local proxy settings, and authentication files are never tracked.

Every promoted dataset or artifact has a registry entry with a stable ID, private
locator, version, checksum, expected size/count, provenance, and downstream role.

## Therapist-loop rules

- A run starts only from a frozen, hashed protocol.
- Setting D dev is the real-speech feedback split used during rounds.
- Setting D test remains sealed until the fixed final round.
- Each round starts from the previous adapter and uses fixed real replay.
- Agent output cannot alter training hyperparameters, references, seeds, or paths.
- Each phase cites immutable hashed artifacts.
- A failure stops the round; there is no retry, refill, silent fallback, rollback, or
  alternate method.

See `docs/evaluation_protocol.md` for the final comparison.

## Hygiene check

```bash
git status --short
git ls-files | grep -E '\.(wav|flac|mp3|m4a|zip|jsonl|safetensors|bin|pt|pth|ckpt)$|(^|/)auth\.|checkpoint-'
git grep -n -E 'password|secret|token=|sk-[A-Za-z0-9]'
```

The payload scan should return no tracked private or model artifacts. Credential scans
may match only this policy's literal search terms, never an actual credential.
