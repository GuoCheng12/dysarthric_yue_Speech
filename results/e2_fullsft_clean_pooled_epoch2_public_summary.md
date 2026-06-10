# E2 Full-SFT Clean Pooled Epoch-2 Public Summary

This table excludes per-speaker and per-utterance rows.

| group | n | zero-shot CER | E2 CER | delta CER | zero-shot critical | E2 critical |
|---|---:|---:|---:|---:|---:|---:|
| overall | 258 | 0.277309 | 0.139254 | -0.138055 | 0.205426 | 0.089147 |
| zero_shot_bucket=easy | 135 | 0.060905 | 0.022489 | -0.038416 | 0 | 0 |
| zero_shot_bucket=hard | 53 | 0.792076 | 0.466393 | -0.325683 | 1 | 0.377358 |
| zero_shot_bucket=medium | 70 | 0.30491 | 0.116754 | -0.188156 | 0 | 0.042857 |
| disease_tag=HD | 12 | 0.198759 | 0.07458 | -0.124179 | 0.083333 | 0.083333 |
| disease_tag=SCA | 25 | 0.554217 | 0.307951 | -0.246266 | 0.44 | 0.24 |
| disease_tag=unknown | 221 | 0.25025 | 0.123683 | -0.126568 | 0.18552 | 0.072398 |
| duration_bucket=long | 27 | 0.460522 | 0.24019 | -0.220332 | 0.333333 | 0.148148 |
| duration_bucket=medium | 178 | 0.254884 | 0.097084 | -0.1578 | 0.185393 | 0.061798 |
| duration_bucket=short | 52 | 0.245046 | 0.141219 | -0.103827 | 0.192308 | 0.134615 |
| duration_bucket=very_short | 1 | 1.0 | 4.818182 | 3.818182 | 1 | 1 |
