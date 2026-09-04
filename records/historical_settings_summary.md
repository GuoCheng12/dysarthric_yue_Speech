# Historical ASR settings

This compact record preserves only the aggregate context needed to interpret the active
Setting D experiment. Row-level predictions, per-patient tables, exploratory probe
outputs, and one-off checkpoint sweeps are intentionally excluded from Git.

| setting | isolation | train / dev / test | zero-shot test CER | selected LoRA test CER | interpretation |
|---|---|---:|---:|---:|---|
| A | seen patient, unseen content | 2159 / 273 / 275 | 27.73% | 4.24% | strong within-patient adaptation |
| B | unseen patient, unseen content | 986 / 107 / 106 | 28.41% | 24.42% | difficult cross-patient generalization |
| C | seen patient, globally unseen content | 2130 / 258 / 253 | 24.39% | 25.47% | hard speech improves while easy/medium regress |

Setting C was the precursor to Setting D. Setting D preserves its globally
content-disjoint evaluation design after the dataset QC exclusions recorded in the
frozen registry.

These numbers summarize historical runs with their original selection protocols; they
must not be pooled into the Setting D final comparison.
