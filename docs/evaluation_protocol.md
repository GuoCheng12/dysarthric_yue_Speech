# Setting D evaluation protocol

## Current user-confirmed scope: train / merged test only

The user reconfirmed the two-way split after the first real MAS pilot:

- Train: 2117 real utterances, unchanged.
- Test: 511 utterances = original dev 258 + original test 253.
- Test manifest: `/data/qwen3-asr/finetune/data_setting_d_merged_test_v1/setting_d_merged_test.jsonl`.
- Manifest SHA-256: `189d749e7e65f14a0440d7f6707a0e1dcfa1a5a62c6c0e5f362b08c1c53e2f98`.
- 36 test patients, 12 test content groups; easy/medium/hard = 265/145/101.

Use this same merged test for the requested monitoring and reported comparisons;
there is no separate active dev set. Recompute metrics from all 511 paired prediction
rows, including patient-macro CER, rather than averaging the old split summaries.

The first pilot accidentally used the old dev-only contract below. Its immutable
protocol, EvidencePack, EvaluationPlan and Reflection describe what actually ran and
must not be relabeled as 511-row artifacts. Corrected checkpoint evaluations are in
`therapist_m0_b8g4_pilot_v1/merged_test_reassessment_v1` under the external records root.
The next Agent/Harness run needs its input and protocol updated to the two-way scope;
the old frozen pilot and main draft YAML do not implement that correction automatically.

## Historical three-way contract (not the current reporting scope)

### Closed-loop development

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

### Final comparison

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

The maintained extractor computes character-level GT-versus-ASR errors and conservative
surface-Jyutping onset/nucleus/coda/tone differences. Post-round measurements also track
the same GT carrier positions to distinguish correction from error migration. They
locate recognition failures; they do not
directly measure the patient's acoustic impairment or prove that the Speech Generator
reproduced it.
