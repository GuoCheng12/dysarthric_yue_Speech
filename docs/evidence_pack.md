# EvidencePack v3

`EvidencePack` is the complete state passed to the Therapist Agent before it writes
one training proposal. The row-level ASR outputs remain inside the Harness; the Agent
receives only deterministic aggregate evidence.

## Round semantics

For proposal round `r`, the evaluated model is always the model that exists before the
intervention:

| proposal round | current snapshot | previous snapshot |
|---:|---|---|
| 1 | `M0` Real-only LoRA | none |
| 2 | `M1` after intervention 1 | `M0` |
| 3 | `M2` after intervention 2 | `M1` |
| 4 | `M3` after intervention 3 | `M2` |

Two permanent anchors are included in every pack and are hash-bound in the run
protocol, including their dev-prediction files and prediction-column names:

- `m0`: the corrected-gradient Real-only LoRA checkpoint that starts the loop;
- `zero_shot`: the original Qwen3-ASR output, used only to diagnose adaptation gains
  and forgetting. It is not a training start point.

Thus, the input is not a free-form transcript dump. The Harness first decodes all 258
real Setting D dev utterances with `M(r-1)`, then reduces the paired snapshots into the
sections below.

## Agent-visible sections

### `data_scope`

Contains the Setting D dev hash, sample/speaker/content counts, critical-CER threshold,
analysis implementation IDs, per-snapshot phonology coverage, and the bounded
cluster-selection policy. `sealed_test_used` is always false.

The difficulty labels are not recomputed from the current model. They are the frozen
zero-shot character-CER buckets: easy `< 0.2`, medium `0.2 <= CER < 0.5`, and hard
`>= 0.5`.

### `snapshots`

Contains aggregate overall/easy/medium/hard metrics and cryptographic identities for:

- `current` = `M(r-1)`;
- `m0` = permanent Real-only anchor;
- `zero_shot` = permanent diagnostic anchor;
- `previous` = `M(r-2)`, absent only in round 1.

Each metric block reports mean utterance CER, pooled CER, patient-macro CER, total
edits/reference characters, exact count, and critical count. A
`closed_loop_model_index` makes the lineage explicit: M0 is index 0, round `r` sees
current index `r-1`, and its previous snapshot is index `r-2`.

### `comparisons`

Contains four explicitly named paired comparisons:

- `m0_vs_zero_shot`: the Real-only adaptation's original gains and regressions;
- `current_vs_zero_shot`: what remains better or worse than zero-shot now;
- `current_vs_m0`: cumulative closed-loop change;
- `current_vs_previous`: paired net change after the immediately preceding combined
  round of real replay, synthetic intervention, and further optimization. It is not
  an isolated causal effect of the synthetic proposal.

Every delta is `evaluated - reference`; therefore a negative CER delta is improvement.
Each comparison includes difficulty-stratified exact/wrong transitions, critical
transitions, utterance-level improved/worsened/tied counts, and anonymous patient-level
improved/worsened/tied counts.

### `error_clusters`

Clusters are selected deterministically from the union of zero-shot, M0, previous,
current, and all historical intervention targets. A corrected historical target
therefore remains visible instead of disappearing when its current count reaches zero.
All historical targets are retained; in addition, at most ten new clusters per level
are sent:

- character;
- onset;
- nucleus;
- coda;
- tone.

Each cluster records the operation, reference and hypothesis units, current/M0/
zero-shot/previous error rates, opportunity denominators,
patient/content coverage, difficulty-specific rates, and alignment uncertainty.

Phone-component clusters are marginal patterns such as `coda k -> empty`, aggregated
across onset, nucleus, and tone. They are not split into one cluster per complete
syllable. Character-level edit clusters remain character-specific.

`targetable=true` currently requires at least two current errors spanning at least two
patients and two dev contents, with acceptable alignment uncertainty. This flag does
not select a proposal; it tells the Agent which observations have minimum cross-patient
and cross-content support. Pure character insertions are not primary targets, while a
component gain such as empty-onset to `/n/` inside a forced character substitution is
not incorrectly treated as a pure character insertion. New clusters are ranked with
targetable patterns first.

`tts_actionable=true` is a separate deterministic executability check. Phone-component
clusters, resolved pronounceable character deletions, and resolved non-homophonic
character substitutions can be expressed through a GT-side TTS carrier. Character
insertions and homophonic orthographic substitutions cannot. For example, `褂 -> 掛`
and `柺 -> 枴` remain useful diagnostic observations but CosyVoice3 cannot encode which
same-pronunciation spelling the ASR should emit. Every error-cluster stimulus citation must
be both targetable and TTS-actionable.

### `capability_anchors`

Errors alone do not identify abilities worth retaining. This separate positive
evidence section includes at most 12 deterministic anchors across easy/medium cohorts
and onset/nucleus/coda/tone carriers. An anchor needs at least two currently correct
GT carrier positions spanning two patients and two contents; it does not require any
current error. It reports current correct/unresolved counts and paired retention,
acquisition, and loss relative to M0 and, when available, the previous round.

Correct means whole-character CER-equivalent recognition at a GT carrier position
with a unique alignment, not a phone matching inside an otherwise wrong character.
These are transcript-grounded ASR abilities, not acoustic impairment measurements.
`maintenance` texts cite an anchor and contain its GT carrier. Each available
easy/medium capability cohort must receive maintenance coverage. Reassessment freezes
the pre-round correct carrier cohort and counts retained and lost positions separately
from new gains, so improving other positions cannot hide forgotten abilities.

The existing `protection` role now means repair of an observed easy/medium regression
relative to zero-shot, M0, or previous. A global error increase confined to hard
patients is not sufficient evidence for this role.

### `intervention_history`

From round 2 onward, this contains compact structured records of completed
interventions: cited target descriptors, target/contrast/protection/maintenance dose,
speaker-profile dose, checkpoint and dev-prediction lineage, paired real-dev change,
patient directions, verdict, and controlled easy/medium safety labels. Its required
`episode_memory` also preserves actual Reflection rationale, provisional explanations,
lesson candidates, testable next-round guidance, Reflection hashes, and the active
Skill version/hash. Immediate feedback therefore reaches round 2 even if Skill v000
has not changed. The full prior stimulus list is not repeated in this compact history.
History must cover
exactly rounds `1..r-1`, and its doses, hashes, and parent-child checkpoint chain are
validated before the Pack can be used.

`artifact_mode` distinguishes recorded interventions from counterfactual interface
smokes; every episode must match the pack's mode. Exporting a run's history reads all
committed ledgers, not only beneficial rounds. Old synthetic data is not added to the
next round's training set by this memory mechanism.

### `action_contract`

States the fixed 60-text/480-synthetic/2117-real-replay budget, permitted roles,
content-isolation levels, and the only fields the Agent may control. Agent v1 uses one
Harness-fixed all-patient reference profile. `proposal_constraints` also freezes the
maximum number of primary targets, per-stimulus copy cap, and linked-copy/text caps for
a cluster that only just meets an evidence threshold. It also fixes sentence length,
threshold-edge whole-batch carrier-exposure caps, and the prohibition on
non-targetable/non-actionable stimulus evidence. It contains no learning rate, LoRA,
step, reference, seed, path, or checkpoint choice.

There is no per-sentence primary-carrier cap or density cutoff. Natural, meaningful
sentences may be purpose-designed concentrated practice, not only common daily
requests. The Therapist explains concentrated versus varied practice in its existing
rationale; carrier counts remain descriptive text coverage for Reflection, not
evidence of acoustic realization or an empirically optimal training dose. The separate
threshold-edge batch limits remain unchanged. Historical Packs containing
`max_primary_carriers_per_stimulus` belong to the old contract: rebuild a new Pack
from source predictions; do not rewrite archived records or silently drop the field.

## Proposal evidence references

The Proposal v3 `evidence_ids` fields use a validated reference namespace rather than
forcing every claim to cite an error cluster:

- an exact error-cluster `evidence_id` for a cluster-level claim;
- an exact capability-anchor `evidence_id` for a positive ability/maintenance claim;
- `snapshot.current`, `snapshot.m0`, `snapshot.zero_shot`, and
  `snapshot.previous` when present for an absolute metric claim;
- `comparison.m0_vs_zero_shot`, `comparison.current_vs_zero_shot`,
  `comparison.current_vs_m0`, and `comparison.current_vs_previous` when present for a
  paired-change claim;
- `intervention.<ledger_id>` for a response-to-intervention history claim.

Target, contrast, and protection stimuli cite error clusters; maintenance stimuli cite
capability anchors instead. An aggregate metric alone cannot specify training text.
Agent v1 uses exactly one primary cluster, and every
stimulus error cluster must be targetable and TTS-actionable. The executable validator checks
cited phones against the actual TTS text after removing only punctuation and
whitespace; it never uses the CER equivalence normalizer for this check. A contrast
must cite its primary target plus an observed same-level, same-reference comparator;
the text must contain the primary GT carrier and each comparator's representable
alternative carrier. The same grounding and carrier-exposure checks run again at
materialization, so a persisted or manually constructed proposal cannot bypass them.

Proposal v3 also contains structured directional `expected_outcomes` for its primary,
every comparator and protection pattern, each maintenance anchor
(`capability_lost_count`, `non_increase`), and mandatory easy/medium safety. These are
compiled into the post-intervention evaluation plan; free-form expected-response prose
is not the evaluation contract.

## Conservative phonological evidence

The maintained phonology layer pins PyCantonese 5.0.0 both in the dependency file and
at runtime, and uses whole-sentence Jyutping on the written surface after removing
only punctuation and whitespace. CER remains computed on the project-normalized
strings. A phone event is retained only where a unique surface-to-CER projection maps
it onto a substitution shared by every optimal CER alignment. Thus pronunciation-
changing CER equivalences such as `冇 -> 無` cannot manufacture tone evidence.
Projection-unresolved and non-CER-forced exclusions are counted separately in
phonology coverage. Context-sensitive dictionary readings are also reported and
excluded from Agent-facing phone clusters. Whole-character insertions and deletions
are never relabelled as four separate phone errors.

The resulting phone rate is explicitly a confirmed-substitution lower bound. It means
that GT and ASR text differ at that GT phonological position; it is not an acoustic
measurement, a direct tone-error measurement, or a deterministic patient impairment
rule. Coverage is reported separately for current, M0, zero-shot, and previous because
a phone-rate change is unsafe to interpret when attribution coverage also changes.

## Excluded from Agent input

The serialized pack contains no audio, filesystem paths, patient or speaker IDs,
utterance IDs, full GT transcripts, full ASR hypotheses, test metrics, or test content.
The private paired rows are retained only as hash-bound Harness artifacts.

## Historical round-1 artifact and rebuilding

The earlier error-only pack was generated from the corrected-gradient Real-only M0 dev predictions:

`/data/qwen3-asr/records/therapist_mas_setting_d_v1/evidence/round_001_m0_evidence_pack_v7.json`

The rebuilt v7 artifact contains 50 bounded clusters selected from 972 candidates. Forty-four meet both the
minimum support rule and TTS-actionability rule. Five high-count character clusters are
explicitly non-actionable homophonic substitutions; two prominent examples are the
single-content `褂 -> 掛` and `柺 -> 枴` lures. This is an executability screen, not a
claim that the remaining clusters are clinical impairment patterns.

Character clusters now retain original `reference_surface_units` and
`hypothesis_surface_units`, separately from CER scoring identities. The full ASR
model directory, not just its weight index, is fingerprinted by the current draft
protocol. No new ASR decoding or Agent call was used to rebuild this pack.

This v7 file predates positive capability anchors and the extended role contract.
It remains immutable historical evidence, not the current maintenance-enabled Agent
input. Rebuild the EvidencePack from the same bound prediction files before running
the new interface. A rebuilt pack can be checked without an Agent call:

```bash
PYTHONPATH=src /data/qwen3-asr/venvs/qwen3-asr/bin/python \
  -m therapist_harness verify-evidence configs/therapist_setting_d_v1.yaml \
  /absolute/path/to/rebuilt_evidence_pack.json
```

Earlier packs and smoke outputs were not overwritten. They are historical artifacts;
rebuild downstream proposals/results before reusing them under the repaired semantics.
