# Dysarthric Cantonese ASR

Research code for therapist-inspired closed-loop adaptation of Qwen3-ASR on
dysarthric Cantonese speech.

The project combines:

- real-speech ASR error assessment;
- a Therapist Agent that proposes bounded training interventions;
- patient-conditioned CosyVoice3 speech generation;
- Qwen3-ASR LoRA adaptation and reassessment.

## Repository layout

- `src/therapist_harness/`: closed-loop contracts and orchestration components.
- `finetune/scripts/`: maintained ASR training, evaluation, and registry utilities.
- `synthesis/cosyvoice3_patient_sft/`: maintained patient Speech Generator interface.
- `configs/`: experiment protocols.
- `data/registry/` and `artifacts/registry/`: dataset and model identities.
- `docs/`: method, evaluation, and repository policies.

See `docs/therapist_closed_loop.md` for the current framework design.

## Data policy

Patient audio, transcripts, row-level predictions, model weights, checkpoints, and API
credentials are not stored in Git.
