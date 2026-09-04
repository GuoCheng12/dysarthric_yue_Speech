# Rejected approaches retained as findings

Executable probe trees and one-off runners for the following directions were removed on
2026-09-04. Their pre-cleanup source and records remain recoverable from the external
cleanup archive recorded in `docs/cleanup_20260904.md`.

## Flat-phone patient rulebooks

ZIPA/GT-phone alignment, patient mapping sweeps, direct-pattern probes, and residual
phone generators did not produce a stable, deterministic impairment mapping that
generalized to new sentences. The new framework may use phone-level **ASR error
evidence**, but it must not present that evidence as a patient pronunciation rulebook.

## Proposal pre-scoring and Meta-Utility Gates

Teacher-forced loss, short micro-trial decoded CER, hidden-GT acoustic recoverability,
and learned acoustic verifiers did not provide a reliable or conceptually clean oracle
for accepting a proposal before the intervention. They also shifted the story toward
validation-set proposal selection. The current loop therefore performs the frozen
intervention, measures the response on real speech, and learns from the result.

## Alternative ASR and TTS branches

CTC auxiliary loss, hard-weighted loss, gated/pre-projector temporal bridges,
multi-kernel/attention/conformer-lite bridges, preference tuning, Matcha-TTS, and the DSI
residual generator were exploratory branches, not dependencies of the current method.
Their detailed runners and probe outputs were removed; the ordinary Qwen3-ASR LoRA path
and reference-conditioned CosyVoice3 Speech Generator remain the maintained baseline.
