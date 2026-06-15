# Step E1 Segment Diagnostics V1

This record summarizes a dev/test diagnostic for the E1 CosyVoice3-mel
residual generator.

Private output root:

```text
/data/qwen3-asr/synthesis/dsi_v1/residual_generator_v2_cosyvoice3_mel/segment_diagnostics_v1/devtest_jyutping_segments
```

Prediction export root:

```text
/data/qwen3-asr/synthesis/dsi_v1/residual_generator_v2_cosyvoice3_mel/segment_diagnostics_v1/prediction_export_devtest
```

Scope:

- Samples: 548 dev/test utterances
- Approximate Jyutping segments: 6658
- Patients: 61
- Unknown text rows: 0

Method:

1. Convert `clean_text` to Jyutping syllables with `pycantonese`.
2. Assign each syllable a proportional slice of the normal-TTS mel timeline.
3. Use the E1 target mel that was already DTW-aligned to the normal-TTS
   timeline.
4. Compare `norm_mel`, `target_mel`, and `pred_mel` per segment.

Important limitation:

This is not forced alignment. Segment boundaries are approximate syllable slices
on the normal-TTS timeline. The diagnostic is useful for group-level residual
patterns by initial/final/tone, but not for exact phonetic boundary claims.

Headline:

- Overall mean segment norm-to-target L1: 2.272858
- Overall mean segment pred-to-target L1: 0.807834
- Overall mean relative L1 gain: 0.595087
- Overall mean gap ratio: 0.404913
- Hard zero-shot bucket has larger residual gap than easy/medium:
  0.432409 vs 0.397085/0.398650

Interpretation:

The E1 model transfers a large portion of the residual energy, but the remaining
gap is not uniform. It is larger for several consonant-heavy and coda-sensitive
groups, consistent with the listening observation that global identity transfers
better than patient-specific articulation changes.

Repo-safe aggregate tables:

- `overall_summary.csv`
- `by_initial.csv`
- `by_final.csv`
- `by_tone.csv`
- `by_zero_shot_bucket.csv`

Private tables not committed publicly:

- `segment_metrics_private.csv`
- `sample_metrics_private.csv`
- `by_jyutping_private.csv`
- `worst_segments_private.csv`
