# Post-intervention Agent contracts

## Role decision

The next LLM call is a second mode of the existing Therapist, not a third reviewer
Agent:

```text
Therapist.prescribe(EvidencePack) -> Proposal
Harness executes one fixed intervention
OutcomeCompiler(Proposal, before, after) -> ReflectionPack
Therapist.reflect(ReflectionPack) -> Reflection
Harness commits an immutable intervention ledger
SkillOptimizer.optimize(ledger history) -> SkillDecision   # planned, not implemented
```

`OutcomeCompiler` is deterministic code.  It computes and binds measurements; it does
not score, accept, reject, roll back, or choose a checkpoint.  Reusing the Therapist
for reflection matches the intended assessment-prescription-intervention-reassessment
cycle without creating an LLM committee or reintroducing the rejected Gate design.

The first independent Agent after the Therapist is the `SkillOptimizer`.  It should not
run until at least two complete real intervention ledgers exist.

## ReflectionPack input

The Harness constructs one sanitized `ReflectionPack` after every completed round.
Incomplete synthesis is fail-fast and produces no `RoundResult` or Reflection.  The
pack embeds the exact typed `EvaluationPlan`, `MaterializationSidecar`,
`prescribed_proposal`, and `RoundResultV2`; their canonical hashes and cross-bindings
are revalidated whenever
the pack is loaded. Before a Reflection request is sent, the formal entry point also
requires the frozen `RunProtocol`, source `EvidencePack`, and source `Proposal`, then
reconstructs Proposal-to-stimulus carrier grounding. Internal hashes alone are not a
trust boundary: a self-consistent, rehashed carrier-count forgery is rejected against
those source artifacts.

### Identity and provenance

- `run_id`, `round_index`;
- hashes of the frozen protocol, EvidencePack, Proposal, EvaluationPlan, and
  RoundResult;
- Therapist instruction hash;
- active skill version and hash;
- parent/new checkpoint hashes and paired-prediction hashes.

### Prescribed intervention

- assessment and falsifiable hypothesis frozen into the pre-intervention
  `EvaluationPlan`;
- the structured expected outcomes;
- every prescribed unique text, role, evidence citation, copy count, and rationale,
  bound to the actual source Proposal rather than reconstructed from target labels;
- primary, comparator, and regression-protection cluster descriptors, plus positive
  easy/medium capability anchors for maintenance;
- planned unique-text, copy, and carrier-exposure dose by role and cluster;
- fixed real-replay count and fixed optimizer-step count;
- declared risks.

### Realized intervention

- requested/generated/failed synthetic counts by role and cluster;
- realized audio duration by role and cluster;
- requested-text carrier exposure for completed samples; this is not acoustic
  verification. `whole_batch_text_carrier_exposures` also counts uncited incidental
  carriers; shared-reference signatures overlap and must not be added as independent dose;
- actual real-replay count and optimizer steps;
- no silent replacement for failed synthesis.  The current implementation accepts
  only complete synthesis receipts for training. A failed synthesis now preserves
  completed WAVs and writes a separate failure report, then stops without retry/refill;

### Observed real-speech response

- before/after overall, easy, medium, hard, and patient-macro metrics;
- paired exact, utterance-CER, and critical-state transitions for every difficulty;
- patient direction counts;
- retention of each maintained pre-round correct carrier cohort: retained, lost to
  error, lost to unresolved alignment, and newly correct counts. New gains cannot
  cancel losses from the fixed pre-round correct cohort;
- before/after error count, opportunity count, rate, patient coverage, content
  coverage, and difficulty breakdown for every expected cluster outcome;
- paired GT carrier-position transitions: correct, signature error, other
  substitution, whole-character deletion, unresolved. Exact-signature decrease is
  diagnostic only; a cluster outcome also requires a paired recovery criterion.
  Reflection target findings copy `target_positions.response`, never frequency alone;
- selected off-target cluster changes produced by a fixed deterministic reporting
  policy;
- synthesis QC and attribution-coverage changes.

Current ReflectionPack payloads use `schema_version=3`; Reflection output uses
`schema_version=4` (the Python class is still named `ReflectionV2`). Round-result
payloads use `schema_version=3` and embed the source synthesis
receipt. The formal Reflection entry rederives metric/outcome/realization views from
their sources and frozen plan. Old immutable smoke artifacts remain historical;
rebuild their evidence and downstream payloads before reuse under this contract.

### History and citation namespace

- compact summaries of all earlier committed interventions, including harmful and
  conflicting results and their actual Reflection interpretation, explanations,
  lesson candidates, and testable next-round guidance;
- an explicit list of valid `result.*`, `proposal.*`, and `intervention.*` citation
  IDs.

The pack must not contain sealed-test information, raw audio, real per-utterance
transcripts, patient/speaker IDs, paths, TTS reference identities, or alternative
candidate proposals. Prescribed synthetic text is intentionally included: the
Therapist must see what was actually practised before interpreting its response.

## Reflection output

The Therapist returns one strict structured `ReflectionV2`. The excerpt below shows
the main fields; the runtime schema also requires identity, rationale, and citations:

```yaml
compiled_expectation_verdict: supported | mixed | refuted
hypothesis_verdict: supported | mixed | refuted | inconclusive

expectation_checks:
  - outcome_id: <Proposal outcome ID>
    verdict: met | not_met
    result_citations: [result.*]
    interpretation: <bounded evidence-based explanation>

target_findings:
  - evidence_id: <primary cluster ID>
    response: improved | unchanged | worsened | inconclusive
    result_citations: [result.*]
    interpretation: <text>

safety_findings:
  - measure: overall_pooled_cer | easy_pooled_cer | medium_pooled_cer |
      hard_pooled_cer | patient_macro_cer
    response: improved | unchanged | worsened
    result_citations: [result.metric.*]

patient_heterogeneity:
  pattern: all_improved | improvement_with_ties | bidirectional |
    worsening_with_ties | all_worsened | all_tied
  result_citations: [result.*]
  interpretation: <text>

capability_findings:
  - evidence_id: <maintained capability ID>
    response: retained | regressed | inconclusive
    interpretation: <explanation of fixed-cohort retention>
    result_citations: [result.capability.*]

competing_explanations:
  - explanation_id: <stable local explanation ID>
    explanation: <provisional explanation, not a measured causal fact>
    predicted_observation: <what a subsequent intervention should reveal>
    result_citations: [result.*]

lesson_candidates:
  - action: retain | adjust | avoid
    condition: <general response-to-intervention condition>
    guidance: <general intervention guidance>
    caveat: <scope limit>
    predicted_observation: <testable prediction>
    result_citations: [result.*]

next_round_guidance:
  primary_strategy: retain | retarget | reassess
  counterbalance_strategy: <text>
  protection_strategy: <text>
  maintenance_strategy: <text>
  primary_practice_dose_direction: increase | maintain | decrease
  rationale: <reason for the next intervention change>
  predicted_observation: <testable prediction>
  result_citations: [result.*]
```

The Agent does not recompute metrics.  The Compiler determines each outcome verdict:
`decrease` is met only for a strict decrease; `non_increase` is met for a decrease or
tie, with additional paired recovery/retention checks for cluster and capability
outcomes. It also determines `compiled_expectation_verdict`: all outcomes met is
`supported`, all primary outcomes not met is `refuted`, and every other complete
pattern is `mixed`. The Agent copies these factual labels. Its separate
`hypothesis_verdict` is an interpretation supported by citations, not a Compiler-forced
causal conclusion. Competing explanations may go beyond the predefined confound list,
but must be grounded and make a testable prediction. A validator enforces exact
outcome order, all five metric findings, capability facts, patient-pattern consistency,
required result citations, privacy, and all cross-artifact hashes.

A complete round can have an inconclusive hypothesis or carrier interpretation.
A failed or incomplete execution still stops before reflection and remains visible
in the run journal; it is not converted into a completed inconclusive round.

The Reflection cannot accept/reject a round, request rollback, select a checkpoint,
change ASR/TTS hyperparameters, inspect the sealed test, or update the Skill directly.
`primary_practice_dose_direction` changes only the combined target + contrast copy
allocation. Copies are redistributed to or from other roles within the exact fixed
synthetic count; it never requests fewer total synthetic samples, minutes, training
steps, or real replay. The next action contract is included in the ReflectionPack.

## Implemented episode memory and Skill activation

`compile_intervention_summary(...)` validates the source ReflectionPack and Reflection
and creates an immutable ledger entry. Its required `episode_memory` preserves the
actual rationale, explanations, lesson candidates, next-round guidance, Reflection
hashes, and the Skill version used in that round. Every subsequent EvidencePack must
carry all previous episodes, even when no Skill update has happened. Recorded
interventions and `counterfactual_interface_smoke` histories cannot be mixed.

`SkillMemory` stores immutable version snapshots, explicit activation records, and
per-round bindings. Both prescribe and reflect use the same captured version.
An update after round 2 becomes active in round 3; `no_change` explicitly retains the
old version. Neither action removes episode history. `RunStore` now validates and
commits these ledgers and checks the active Skill used by the following prescription.

Direct Agent calls require `--skill /absolute/path/SKILL.md` or
`--skills-dir /absolute/path/run/skills`; there is no implicit `v000` fallback.

## Planned Skill Optimizer input and output (not implemented)

After at least two ledgers, the low-frequency `SkillOptimizer` reads:

- the active Skill version, hash, and stable rule IDs;
- every completed immutable intervention summary up to the trigger round, not just
  successful rounds;
- prescribed and realized dose by cluster/role;
- RoundResult plus Reflection;
- which Skill rules were active;
- a fixed update contract and a declaration that sealed test was not used.

The proposed future interface is a `SkillDecision`, not a freely rewritten `SKILL.md`:

```yaml
decision: patch | no_change
base_skill_version: vNNN
base_skill_sha256: <sha256>
source_ledger_ids: [...]
operations:
  - operation: add | revise | retire
    rule_id: <stable ID>
    condition: <general condition>
    guidance: <general guidance>
    rationale: <text>
    supporting_ledger_ids: [...]
    contradicting_ledger_ids: [...]
    evidence_strength: single_round | repeated | mixed
no_change_reason: <required only for no_change>
```

The future Harness integration would validate and deterministically render this
decision. No LLM SkillOptimizer call, `SkillDecision` validator, or rule-patch renderer
has been implemented; the current interface only activates an explicitly supplied
version or records `no_change`. Allowing
`no_change` prevents forced learning from weak evidence.  Rules may encode general
response-to-intervention knowledge, but cannot memorize patients, stimulus sentences,
content IDs, one-off cluster IDs, or fixed impairment rules.

For a four-round main experiment, the planned Skill update is after round 2 so that it
can influence rounds 3 and 4.  A round-4 update would be a post-run artifact and cannot
support the main self-improvement claim.
