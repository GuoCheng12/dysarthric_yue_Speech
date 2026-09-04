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

The Therapist Agent can only read a sanitized `EvidencePack` and return one `Proposal`.
It chooses the error evidence to target, stimulus texts, target/contrast/protection role,
and a permitted patient allocation profile. It cannot choose hyperparameters, paths,
references, seeds, or checkpoints.

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

The executable Pydantic contracts are in `src/therapist_harness/schema.py`:

- `RunProtocol`: human-frozen experiment constants and hashes;
- `EvidencePack`: metrics, paired transitions, error clusters, history, and action bounds;
- `Proposal`: evidence-cited hypothesis and exact stimulus prescription;
- `RoundResult`: observed result vector without a scalar utility;
- `Reflection`: response-to-intervention interpretation;
- `SkillPatch`: cited, versioned changes to therapist knowledge.

The local run store writes one immutable event per phase. It rejects phase skipping and
artifact overwrite. Agent requests use the Responses API with structured output,
`store=false`, `max_retries=0`, and sanitized aggregate evidence only.

## Current implementation status

`configs/therapist_setting_d_v1.yaml` records verified Setting D and Speech Generator
artifacts. It intentionally remains `protocol_status: draft`: the round budget and
update length are working choices, not frozen experimental facts. `validate` accepts a
draft for review; `init` refuses it until a human changes the status to `frozen`.

```bash
PYTHONPATH=src /data/qwen3-asr/venvs/qwen3-asr/bin/python \
  -m therapist_harness validate configs/therapist_setting_d_v1.yaml
```

Paired predictions can now be reduced to canonical character-level metrics, transitions,
and anonymized error clusters. A validated proposal can be materialized into deterministic
private Speech Generator assignments after four-layer train/dev/test content-isolation
checks. The remaining integration work is to wrap those components into complete
`EvidencePack` and training manifests, connect the fixed ASR train/eval commands, and run
one fake-backend end-to-end round. No GPU closed-loop experiment should start before that
integration test passes and the draft budgets are frozen.
