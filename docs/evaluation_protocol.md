# Setting D evaluation protocol

## Closed-loop development

Each round evaluates the current adapter on the same 258-utterance real Setting D dev
split. The Agent receives only a sanitized aggregate `EvidencePack`; it never receives
audio, paths, patient IDs, row-level transcripts, or any test information.

Every round reports the full result vector rather than a scalar utility:

- sample-weighted mean utterance CER and pooled CER;
- patient macro CER;
- exact and critical counts;
- easy, medium, and hard breakdowns;
- paired recovered, newly-wrong, preserved-correct, and persistent-wrong counts;
- per-patient improved, worsened, and tied counts;
- response of cited error clusters and off-target changes.

Training continues for the protocol's fixed number of rounds. There is no rollback,
proposal rejection, dense checkpoint search, or best-round selection.

## Final comparison

After the method, budgets, round count, and final adapter are frozen, the 253-utterance
Setting D test is evaluated once with explicit authorization. The required table is:

1. zero-shot Qwen3-ASR;
2. real-only LoRA;
3. static patient Speech Generator augmentation;
4. Therapist Agent closed-loop;
5. Therapist Agent without skill update, as the small secondary ablation.

Report overall, easy, medium, hard, and patient macro CER, plus paired patient-level
uncertainty. Settings A, B, and C are historical context and cannot be pooled into this
table.

## Interpretation boundary

The maintained extractor currently computes character-level GT-versus-ASR errors.
Jyutping onset/nucleus/coda/tone evidence is a planned extension and must be separately
validated before use. If added, it will locate recognition failures; it will not
directly measure the patient's acoustic impairment or prove that the Speech Generator
reproduced it.
