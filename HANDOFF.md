# Current handoff

## Active objective

Build a reproducible therapist-inspired closed loop that improves dysarthric Cantonese
ASR through feedback-conditioned synthetic training prescriptions. The contribution is
the sequential intervention policy, not a new static augmentation benchmark.

## Frozen assets

- Qwen3-ASR: `/data/qwen3-asr/models/Qwen3-ASR-1.7B`
- Setting D: `2117 / 258 / 253`, with `48 / 6 / 6` content groups
- Speech Generator: `/data/qwen3-asr/records/cosyvoice3_reference_sft_setting_d_v1/speech_generator.pt`
- Speech Generator SHA-256: `5764178a90d234997b06dcc5fd74d530419c282b6959c089f4d09b0f3e6eaf0b`
- Agent target: `gpt-5.6-sol`, Responses API, `xhigh`, `store=false`, no retries

Never copy the API key into a config or record. The third-party gateway currently
requires a curl-like `User-Agent` header.

## Current evidence

- Zero-shot Setting D test CER: 24.39%; hard: 69.59%.
- Real-only LoRA test CER: 24.93%; hard: 63.44%; easy and medium regress.
- Formal Full-1440 static augmentation: 25.21% overall; 64.73% hard.
- Matched exploratory TTS control shows patient Speech Generator data is much better
  than official-base CosyVoice3 data on hard real speech, but is not reliably better
  than real-only. See `records/setting_d_static_augmentation_v1/README.md`.

## Method contract

The Harness fixes splits, hashes, TTS checkpoint, references, seeds, budgets, LoRA
configuration, training steps, and evaluation. The Therapist Agent only chooses
evidence-cited stimulus text, target/contrast/protection roles, and one permitted patient
allocation profile.

Each round starts from the previous adapter and uses fixed real replay plus only the
current round's synthetic data. A completed round is never rolled back or rejected. The
real feedback result and reflection are appended to the intervention ledger. The test
split stays sealed until one authorized final evaluation after the fixed last round.

## Implemented boundary

The strict schemas, canonical text identities, paired character evidence, proposal
materializer, one-call TTS adapter, immutable phase store, sanitized Agent request,
initial skill, and draft protocol are implemented in `src/therapist_harness/` and
`synthesis/cosyvoice3_patient_sft/`.

## Next implementation unit

1. wrap extracted evidence and the frozen action contract into a complete `EvidencePack`;
2. write the deterministic assignments and real replay into immutable round manifests;
3. connect fixed-step LoRA continuation and real-speech evaluation to run-store phases;
4. run one fake-backend end-to-end round before any GPU experiment.

Do not restore the old `inference/multi_agent`, `SkillOpt`, probe, Gate, or verifier
trees. Their source is recoverable only for historical inspection from the cleanup
archive in `docs/cleanup_20260904.md`.
