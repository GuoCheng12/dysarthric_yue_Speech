# Project registry

The registries separate stable experiment identities from private local payloads.

| scope | source of truth |
|---|---|
| dataset split counts, paths, hashes | `data/registry/asr_dataset_settings_v1.yaml` |
| selected ASR and TTS artifacts | `artifacts/registry/asr_baselines_checkpoints_v1.yaml` |
| current aggregate findings | `records/README.zh.md` |
| closed-loop protocol | `configs/therapist_setting_d_v1.yaml` |

Setting D is the active setting. Settings A, B, and C remain historical comparison
settings and must not be merged into the Setting D leaderboard.

```bash
/data/qwen3-asr/venvs/qwen3-asr/bin/python finetune/scripts/project_registry.py datasets
/data/qwen3-asr/venvs/qwen3-asr/bin/python finetune/scripts/project_registry.py baselines --setting setting_d
/data/qwen3-asr/venvs/qwen3-asr/bin/python finetune/scripts/project_registry.py validate --check-paths
```

Registry entries contain paths and hashes, not model weights, audio, transcripts,
row-level predictions, or credentials. An artifact is promoted only after its lineage,
selection split, checkpoint rule, and sanitized aggregate result are documented.
