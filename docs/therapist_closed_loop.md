# Therapist-inspired closed-loop adaptation

## Research question

Can a therapist-inspired agent use the current ASR's real-speech errors to prescribe
the next fixed-budget synthetic intervention, then improve its prescription from the
observed response to intervention?

This project does **not** claim that ASR errors reveal deterministic patient
pronunciation rules. Phone-level evidence describes where an ASR hypothesis differs
from the GT phonological position. The Speech Generator is an intervention tool; its
output is not ground-truth evidence about impairment.

## Ownership boundary

The Harness owns every experimental degree of freedom that must remain comparable:

- Setting D splits and their hashes;
- Speech Generator checkpoint, patient references, and seeds;
- synthetic and real-replay budgets;
- LoRA modules, rank, scale, learning rate, batch size, steps, and scheduler;
- ASR evaluation, checkpoint lineage, artifacts, and the sealed test.

In prescribe mode the Therapist reads a sanitized `EvidencePack` and returns one
`Proposal`. In reflect mode the same role reads the completed intervention and its
real-speech response, then supplies the next-round interpretation and guidance.
It chooses the error evidence to target, stimulus texts, target/contrast/protection/maintenance
role, and bounded copy dose. Agent v1 cannot choose a patient allocation profile: the
Harness fixes all-patient balanced reference allocation. It cannot choose
hyperparameters, paths, references, seeds, or checkpoints.

## Four-round sequence

Each round follows one irreversible sequence:

```text
assess real feedback speech
  -> one Therapist proposal
  -> validate budget, citations, profiles, and content isolation
  -> synthesize every assigned utterance exactly once
  -> fixed real replay + current-round synthetic LoRA update
  -> reassess on real feedback speech
  -> one reflection
  -> append result to the intervention ledger
```

Round `t+1` starts from round `t`'s adapter. A harmful round is recorded and still forms
part of the sequence; there is no proposal Gate, automatic rejection, rollback, dense
checkpoint search, or best-round selection. Old synthetic audio is not accumulated into
the next round unless a later frozen protocol explicitly changes that rule.

The official test remains sealed throughout the loop. After the fixed number of rounds,
one explicitly authorized final evaluation compares zero-shot, real-only LoRA, static
Speech Generator augmentation, and the Therapist loop. A small secondary ablation runs
the same Therapist loop without skill updates.

## Contracts

The currently validated prescription contracts are in
`src/therapist_harness/schema.py`:

- `RunProtocol`: human-frozen experiment constants and hashes;
- `EvidencePack`: metrics, paired transitions, error clusters, positive easy/medium
  capability anchors, complete episode history, and action bounds;
- `Proposal`: evidence-cited hypothesis, exact stimulus prescription, and structured
  post-intervention expectations.

The post-intervention contracts are in `src/therapist_harness/round_schema.py`:

- `EvaluationPlan`: pre-intervention measurement registration;
- `MaterializationSidecar`: exact Proposal-to-TTS dose binding;
- `RoundResultV2`: deterministic paired real-speech response;
- `ReflectionPack` (schema 3) and `ReflectionV2` (schema 4): sanitized reflection input
  containing the actual prescribed texts, and fact-grounded Therapist interpretation.

The rejected provisional `RoundResult`, `Reflection`, and forced `SkillPatch` classes
have been removed. The immutable intervention ledger and versioned active-Skill
loading are implemented in `memory.py`/`memory_schema.py`. LLM SkillOptimizer authoring
and rule-patch rendering remain unimplemented until completed intervention histories
exist.

The exact per-round Agent input and comparison semantics are defined in
`docs/evidence_pack.md`.
The post-intervention Reflection mode and later Skill Optimizer interfaces are defined
in `docs/reflection_contract.md`.

The local run store writes one immutable event per phase. It rejects phase skipping and
artifact overwrite. Agent requests use the Responses API with structured output,
`store=false`, `max_retries=0`, and sanitized aggregate evidence only.

## Current implementation status

`configs/therapist_setting_d_v1.yaml` records verified Setting D, Speech Generator,
Real-only M0, and zero-shot dev-prediction anchors. It intentionally remains
`protocol_status: draft`: the round budget and update length are working choices, not
frozen experimental facts. `validate` accepts a draft for review; `init` refuses it
until a human changes the status to `frozen`.

```bash
PYTHONPATH=src /data/qwen3-asr/venvs/qwen3-asr/bin/python \
  -m therapist_harness validate configs/therapist_setting_d_v1.yaml
```

Paired predictions can now be reduced to canonical metrics, transitions, bounded
character plus conservative Jyutping-component error clusters, and at most 12 positive
easy/medium capability anchors. Maintenance anchors require correct support across
two patients and two contents, not an existing error. `maintenance` preserves those
abilities; `protection` repairs an observed easy/medium regression, not a global
increase confined to hard patients. The Therapist
Agent uses a fixed evidence-semantics instruction, an explicitly selected or
round-bound active Skill, Responses API strict
structured output, validated aggregate/cluster citations, TTS-actionability checks,
actual-TTS-text character/Jyutping grounding, and carrier-exposure dose limits.
Natural, meaningful concentrated practice is allowed with a stated rationale;
texts need not all be common daily requests. There is no per-sentence primary-carrier
maximum. Counts still feed the sidecar and Reflection; only the separate existing
threshold-edge batch exposure limits remain hard carrier-dose bounds.
Proposal v3 also supplies machine-readable directional outcomes for reassessment.

The deterministic Outcome Compiler and the existing Therapist's Reflection mode are
now implemented. The Compiler freezes an `EvaluationPlan` before intervention, binds
the materialized dose to the source Proposal, recomputes paired real-speech metrics and
cluster changes, and emits a sanitized `ReflectionPack`. The formal Reflection entry
revalidates that pack against the frozen Protocol plus the source EvidencePack and
Proposal before making an API request. It does not score, accept, reject, roll back, or
select an intervention.

Reflection separates Compiler-owned `compiled_expectation_verdict` from the Agent's
independent `hypothesis_verdict`. Measured outcomes and carrier retention remain fixed;
the Therapist supplies cited interpretations, provisional explanations, and testable
next-round predictions. The actual Reflection enters the next EvidencePack immediately,
without waiting for a Skill update. An explicit update after round 2 affects round 3;
`no_change` retains the current Skill but never removes episode history. Both prescribe
and reflect use the same captured Skill within a round.

For example, initialize the Skill registry and prescribe with a rebuilt EvidencePack
(paths below are placeholders for a prepared run):

```bash
PYTHONPATH=src /data/qwen3-asr/venvs/qwen3-asr/bin/python \
  -m therapist_harness init-skills /absolute/path/run/skills \
  src/therapist_harness/skills/v000/SKILL.md

PYTHONPATH=src /data/qwen3-asr/venvs/qwen3-asr/bin/python \
  -m therapist_harness propose configs/therapist_setting_d_v1.yaml \
  /absolute/path/round_001/evidence.json /absolute/path/round_001/proposal.json \
  --skills-dir /absolute/path/run/skills
```

Both `propose` and `reflect` require `--skills-dir` or an explicit
`--skill /absolute/path/SKILL.md`. There is no hardcoded v000 default.
`compile-ledger` builds a validated Reflection-derived episode, and `export-history`
exports all committed episodes for `build-evidence --history`. The registry supports
an explicit supplied Skill update or `activate-skill ... --after-round 2 --no-change`;
it does not yet call a SkillOptimizer model.

Three revised real-evidence smoke cases cover an actionable phone family,
easy-regression protection, and rejection of high-count homophonic orthographic lures.
Those historical cases test the Agent boundary, not production-scale policy quality: all are Round 1 with
6 texts/24 copies rather than the formal 60/480, and do not exercise intervention
history or skill updates. A validated proposal can be materialized into deterministic
private Speech Generator assignments after four-layer train/dev/test content-isolation
checks.

A counterfactual interface smoke has exercised the strict Reflection schema with
`gpt-5.6-sol` at `xhigh`; it proves only that the interface and validation boundary are
executable, not that an intervention is effective. The current deterministic tests
also exercise a source-bound fixture round through ledger commit, immediate Reflection
feedback to the next proposal input, and immutable Skill activation. The new Agent
smoke targets these extended interfaces; counterfactual histories are explicitly marked
and cannot advance a real execution run. No formal GPU closed-loop training has been
started for this framework. The independent Skill Optimizer follows only after at
least two complete real intervention ledgers, and draft budgets still require freezing
before a formal run.
