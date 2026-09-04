# Dysarthric Cantonese ASR

This repository contains the maintained code and reproducibility records for
therapist-inspired closed-loop adaptation of Qwen3-ASR on dysarthric Cantonese speech.

## Current research line

The active setting is **Setting D**: seen patients and globally unseen content, after
removing four unusable patients and one noisy utterance.

| split | utterances | patients | unique contents | easy | medium | hard |
|---|---:|---:|---:|---:|---:|---:|
| train | 2,117 | 66 | 48 | 1,096 | 608 | 413 |
| dev | 258 | 36 | 6 | 134 | 70 | 54 |
| test | 253 | 33 | 6 | 131 | 75 | 47 |

Train/dev/test have zero overlap at prompt ID, cleaned text, normalized text, and
Jyutping levels. The frozen split hashes live in
`data/registry/asr_dataset_settings_v1.yaml`.

The current real-only LoRA improves hard test CER from 69.59% to 63.44%, but overall
test CER changes from 24.39% to 24.93% because easy and medium speech regress. A formal
1,440-utterance static Speech Generator augmentation run also failed to beat real-only.
These observations motivate an adaptive data prescription rather than more static data.

```text
Qwen3-ASR on real feedback speech
  -> aggregate character-level error evidence
  -> Therapist Agent proposes one bounded prescription
  -> patient-conditioned CosyVoice3 synthesis
  -> fixed real replay + LoRA continuation
  -> reassessment on real speech
  -> reflection and versioned skill update
  -> next round
```

There is no training-time proposal Gate, utility score, rollback, or automatic
best-round selection. Phone-level evidence describes ASR output errors at GT
phonological positions; it is not a deterministic patient pronunciation rulebook.

## Maintained layout

```text
src/therapist_harness/
  schema.py                        strict experiment contracts
  text.py                          canonical CER/content identities
  evidence.py                      paired real-speech error evidence
  materialize.py                   proposal -> deterministic TTS assignments
  agent.py, store.py               Agent request and immutable run state
configs/therapist_setting_d_v1.yaml
docs/therapist_closed_loop.md      method and ownership boundary
finetune/scripts/
  qwen3_asr_lora_sft.py            ordinary LoRA + adapter continuation
  evaluate_asr_checkpoint.py       strict base/LoRA real-speech decoding
synthesis/cosyvoice3_patient_sft/
  synthesize_manifest.py           one-call, fixed-reference generation
data/registry/                     frozen dataset identifiers and hashes
artifacts/registry/                selected checkpoint identifiers
records/                            sanitized aggregate findings only
```

The current extractor is deliberately character-level. Jyutping/phone-context evidence
is a future, separately tested extension; the repository does not imply that it already
exists.

The ASR trainer intentionally contains only the ordinary LoRA path required by the
closed loop: fixed seed, correct gradient accumulation, and optional continuation from
the previous adapter. Historical CTC, hard-weight, bridge, preference, rulebook, Gate,
and verifier implementations were removed. See `records/rejected_approaches.md` and
`docs/cleanup_20260904.md`.

## Review the draft protocol

```bash
cd /data/qwen3-asr/repo/dysarthric_yue_Speech
PYTHONPATH=src /data/qwen3-asr/venvs/qwen3-asr/bin/python \
  -m therapist_harness validate configs/therapist_setting_d_v1.yaml
```

The checked-in protocol is deliberately marked `draft`. The Harness refuses to create
a run until the experimental budgets are reviewed and its status is changed to
`frozen`.

## Privacy and artifact policy

Patient audio, transcripts, row-level predictions, model weights, checkpoints, and API
credentials remain outside Git. Repository files may contain stable private locators
and hashes, but no private payload. The sealed Setting D test must not be exposed to the
Agent or used for round selection.
