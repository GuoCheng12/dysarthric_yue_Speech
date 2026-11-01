"""Persistent real-recording practice units, measured transitions and exact budgets."""

from collections import Counter, defaultdict
import hashlib
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .phonology import align_characters
from .real_curriculum import catalog, position_states
from .text import text_normalize

ENDPOINTS = (20, 40, 60, 80, 100)
COURSE_ROWS = 240


class UnitDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str = Field(pattern=r"^U[0-9]{3}$")
    patients: list[str] = Field(min_length=1)
    contents: list[str]  # Empty means all TRAIN contents for these patients.
    target: Literal["deletion", "substitution", "insertion", "regression", "maintenance"]
    evidence_ids: list[str] = Field(min_length=1)
    rationale: str
    expected_change: str


class Allocation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str = Field(pattern=r"^U[0-9]{3}$")
    exposures: int = Field(ge=1, le=240)
    reason: str


class CourseProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diagnosis: str
    hypothesis: str
    new_units: list[UnitDefinition] = Field(max_length=4)
    allocations: list[Allocation] = Field(min_length=1, max_length=4)
    risk: str


class UnitReflection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str
    observation_ids: list[str] = Field(min_length=1, description=(
        "Cite at least one ID from this unit's unit_observation_ids; optionally add "
        "IDs from the current outcome's global_observation_ids. No other-unit or invented IDs."))
    conclusion: Literal["progress", "local_fit_only", "regression", "mixed", "no_change"]
    explanation: str
    next_action: Literal["continue", "increase", "decrease", "pause", "replace"]


class CourseReflection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observed_changes: str
    units: list[UnitReflection] = Field(min_length=1, max_length=4)
    next_guidance: str


def make_coverage(ids, seed=42):
    """2240 shared exposures: one shuffled traversal before any recycling."""
    assert len(ids) == len(set(ids)) and len(ids) >= 640
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    stream = (order * ((2240 + len(order) - 1) // len(order)))[:2240]
    return {20: stream[:640], **{end: stream[640+i*400:1040+i*400]
                                for i, end in enumerate(ENDPOINTS[1:])}}


def members(unit, entries):
    return sorted(r["id"] for r in entries if r["patient"] in unit["patients"]
                  and (not unit["contents"] or r["content"] in unit["contents"]))


def register_proposal(proposal, registry, entries, backgrounds):
    """No silent repair: compile only proposals satisfying the actual experiment."""
    updated = {k: dict(v) for k, v in registry.items()}
    by_id = {r["id"]: r for r in entries}
    known_patients = {r["patient"] for r in entries}
    known_contents = {r["content"] for r in entries}
    for definition in proposal.new_units:
        u = definition.model_dump(mode="json")
        key = u["unit_id"]
        assert key not in updated, "unit IDs/definitions cannot be overwritten"
        assert len(u["patients"]) == len(set(u["patients"])) and set(u["patients"]) <= known_patients
        assert len(u["contents"]) == len(set(u["contents"])) and set(u["contents"]) <= known_contents
        pool = members(u, entries)
        assert len({by_id[i]["content"] for i in pool}) >= 2, "a practice unit needs multiple contents"
        assert set(u["evidence_ids"]) <= set(pool), "evidence must belong to the unit"
        cited = [by_id[i] for i in u["evidence_ids"]]
        target = u["target"]
        if target in ("substitution", "deletion", "insertion"):
            assert any(r["SDI"][("substitution", "deletion", "insertion").index(target)] > 0 for r in cited), "unsupported error target"
        elif target == "regression":
            assert any(r["recent_regressed_chars"] > 0 or r["regressed_chars"] > 0 for r in cited), "unsupported regression"
        else:
            assert any(r["cer"] == 0 for r in cited), "maintenance requires correct evidence"
        updated[key] = {**u, "member_ids": pool}
    allocation = {a.unit_id: a.exposures for a in proposal.allocations}
    assert len(allocation) == len(proposal.allocations), "duplicate allocation"
    assert set(allocation) <= set(updated) and sum(allocation.values()) == COURSE_ROWS
    assert {u.unit_id for u in proposal.new_units} <= set(allocation), "new units must be used"
    active_members = [i for key in allocation for i in updated[key]["member_ids"]]
    assert len(active_members) == len(set(active_members)), "active unit pools must not overlap"
    for background in backgrounds:
        counts = Counter(background)
        assert max(counts.values()) <= 2
        for key, quota in allocation.items():
            capacity = sum(2-counts[i] for i in updated[key]["member_ids"])
            assert quota <= capacity, f"insufficient two-exposure capacity for {key}"
    return updated


def allocation_actions(previous, current):
    old = {a["unit_id"]: a["exposures"] for a in previous}
    new = {a["unit_id"]: a["exposures"] for a in current}
    return {key: {"before": old.get(key, 0), "after": new.get(key, 0),
                  "action": "pause" if key not in new else "activate" if key not in old
                  else "increase" if new[key] > old[key] else "decrease" if new[key] < old[key] else "continue"}
            for key in sorted(set(old) | set(new))}


def compile_block(end, background, entries, registry=None, allocations=None, previous_manifests=()):
    assert end in ENDPOINTS
    if end == 20:
        assert len(background) == 640
        ordered = [(i, "coverage", None) for i in background]
    else:
        assert len(background) == 400
        counts = Counter(background)
        assert max(counts.values()) <= 2
        past = Counter(r["id"] for m in previous_manifests for r in m)
        by_id = {r["id"]: r for r in entries}
        chosen = []
        groups = [(a["unit_id"], a["exposures"], registry[a["unit_id"]]["member_ids"])
                  for a in allocations] if allocations is not None else [("random", 240, sorted(by_id))]
        assert sum(quota for _, quota, _ in groups) == 240
        for key, quota, pool in groups:
            seed = int.from_bytes(hashlib.sha256(f"42:{end}:{key}".encode()).digest()[:8], "big")
            tie = list(pool)
            random.Random(seed).shuffle(tie)
            ranks = {sid: i for i, sid in enumerate(tie)}
            for _ in range(quota):
                eligible = [sid for sid in pool if counts[sid] < 2]
                assert eligible, f"exhausted exposure capacity: {key}"
                # Static curricula rotate by exposure history, never by fresh CER.
                sid = min(eligible, key=lambda i: (past[i] + counts[i], ranks[i])) if allocations is not None else min(eligible, key=ranks.__getitem__)
                chosen.append((sid, "course", None if key == "random" else key))
                counts[sid] += 1
                if allocations is None:
                    ranks[sid] += len(pool)  # random without replacement, before recycling
        random.Random(500+end).shuffle(chosen)
        b, c = iter(background), iter(chosen)
        ordered = []
        for i in range(640):
            course_slot = ((i % 32)+1)*12//32 > (i % 32)*12//32
            ordered.append(next(c) if course_slot else (next(b), "coverage", None))
    rows = [{"position": (end-20)*32+n, "step": end-19+n//32, "id": sid,
             "stream": stream, "unit_id": unit} for n, (sid, stream, unit) in enumerate(ordered)]
    assert len(rows) == 640 and max(Counter(r["id"] for r in rows).values()) <= 2
    return rows


def paired_changes(gt, before, after):
    ref, old, new = map(text_normalize, (gt, before, after))
    a, b = position_states(ref, old), position_states(ref, new)
    pairs = Counter(zip(a, b))
    return {"recent_recovered_chars": pairs[("error", "correct")],
            "recent_regressed_chars": pairs[("correct", "error")],
            "persistent_error_chars": pairs[("error", "error")],
            "recent_uncertain_chars": sum(x == "unresolved" or y == "unresolved" for x, y in zip(a, b))}


def learning_catalog(current, previous, zero, manifests, previous_losses, current_losses):
    cat, mapping = catalog(current, zero)
    before = {r["utt_id"]: r for r in previous}
    assert set(before) == {r["utt_id"] for r in current}
    total = Counter(r["id"] for m in manifests for r in m)
    recent = Counter(r["id"] for r in manifests[-1]) if manifests else Counter()
    for entry in cat["recordings"]:
        sid, row = entry["id"], mapping[entry["id"]]
        old = before[row["utt_id"]]
        assert old["clean_gt"] == row["clean_gt"] and old["speaker_id"] == row["speaker_id"]
        ref = text_normalize(row["clean_gt"])
        entry.update(previous_prediction=old["prediction"],
                     previous_cer=align_characters(ref, text_normalize(old["prediction"])).edit_distance/len(ref),
                     exposures_recent=recent[sid], exposures_total=total[sid],
                     loss_previous=previous_losses.get(sid), loss_current=current_losses.get(sid),
                     **paired_changes(row["clean_gt"], old["prediction"], row["prediction"]))
    return cat, mapping


def summarize_pair(before, after, ids):
    if not ids:
        return {"rows": 0, "before": None, "after": None, "critical_transitions": None}
    assert set(ids) <= set(before) & set(after)
    def metric(rows):
        errors, lengths, patient = [], [], defaultdict(list)
        for r in rows:
            ref = text_normalize(r["clean_gt"])
            n = align_characters(ref, text_normalize(r["prediction"])).edit_distance
            errors.append(n)
            lengths.append(len(ref))
            patient[r["speaker_id"]].append((n, len(ref)))
        return {"mean_utterance_cer": sum(n/d for n, d in zip(errors, lengths))/len(errors),
                "pooled_cer": sum(errors)/sum(lengths), "total_edits": sum(errors),
                "total_reference_characters": sum(lengths),
                "patient_macro_cer": sum(sum(n for n, _ in v)/sum(d for _, d in v)
                                         for v in patient.values())/len(patient)}
    def critical(r):
        ref, hyp = text_normalize(r["clean_gt"]), text_normalize(r["prediction"])
        return not hyp or align_characters(ref, hyp).edit_distance/len(ref) >= .5
    transitions = {"remained": [], "exited": [], "entered": [], "neither": []}
    chars = Counter()
    for i in sorted(ids):
        a, b = before[i], after[i]
        assert a["clean_gt"] == b["clean_gt"] and a["speaker_id"] == b["speaker_id"]
        old, new = critical(a), critical(b)
        transitions["remained" if old and new else "exited" if old else "entered" if new else "neither"].append(i)
        chars.update(paired_changes(b["clean_gt"], a["prediction"], b["prediction"]))
    return {"rows": len(ids), "before": metric([before[i] for i in ids]),
            "after": metric([after[i] for i in ids]), "character_transitions": dict(chars),
            "critical_transitions": {k: {"count": len(v), "ids": v} for k, v in transitions.items()}}


def outcome_pack(step, proposal, registry, manifest, before, after, feedback_before,
                 feedback_after, id_map, losses_before, losses_after):
    """All IDs supplied here are anonymous Rxxxx / Fxxxx; audio never enters the pack."""
    trained = {r["id"] for r in manifest}
    observations = {}
    for bucket in ("overall", "easy", "medium", "hard"):
        ids = sorted(i for i, r in feedback_after.items() if bucket == "overall" or r["zero_shot_bucket"] == bucket)
        observations[f"S{step}_feedback_{bucket}"] = summarize_pair(feedback_before, feedback_after, ids)
    global_observations = list(observations)
    unit_observations = {}
    for allocation in proposal["allocations"]:
        key = allocation["unit_id"]
        selected = {r["id"] for r in manifest if r["unit_id"] == key}
        patients = set(registry[key]["patients"])
        practice_contents = defaultdict(set)
        for i in trained:
            practice_contents[id_map[i]["patient"]].add(id_map[i]["content"])
        unpracticed = sorted(i for i, r in id_map.items() if r["patient"] in patients
                             and i not in trained and r["content"] not in practice_contents[r["patient"]])
        heldout = sorted(i for i, r in feedback_after.items() if r["patient"] in patients)
        named_sets = {"practiced": sorted(selected), "unpracticed_other_content": unpracticed, "heldout_content": heldout}
        unit_observations[key] = []
        for name, ids in named_sets.items():
            oid = f"S{step}_{key}_{name}"
            record = summarize_pair(feedback_before if name == "heldout_content" else before,
                                    feedback_after if name == "heldout_content" else after, ids)
            if name == "practiced":
                record["unique_rows"] = len(selected)
                record["course_exposures"] = allocation["exposures"]
                record["total_exposures_on_practiced"] = sum(r["id"] in selected for r in manifest)
                record["dropout_off_loss_before"] = sum(losses_before[i] for i in ids)/len(ids)
                record["dropout_off_loss_after"] = sum(losses_after[i] for i in ids)/len(ids)
            observations[oid] = record
            unit_observations[key].append(oid)
    return {"step": step, "proposal": proposal, "unit_observation_ids": unit_observations,
            "global_observation_ids": global_observations,
            "observations": observations,
            "interpretation": "Joint intervention outcomes, not isolated causal effects of individual units. Empty subsets are unobserved, not zero CER."}


def validate_reflection(reflection, outcome):
    expected = set(outcome["unit_observation_ids"])
    assert len(reflection.units) == len(expected) and {r.unit_id for r in reflection.units} == expected
    global_ids = set(outcome["global_observation_ids"])
    for record in reflection.units:
        cited = set(record.observation_ids)
        own = set(outcome["unit_observation_ids"][record.unit_id])
        assert cited <= set(outcome["observations"]), "unknown observation"
        assert cited <= own | global_ids, "unit-mismatched observation"
        assert cited & own, "reflection must cite its own unit observation"


def compact_catalog(cat):
    columns = list(cat["recordings"][0])
    return {"contents": cat["contents"], "recording_columns": columns,
            "recordings": [[r[k] for k in columns] for r in cat["recordings"]],
            "table_encoding": "Each row follows recording_columns exactly; CER values are fractions."}


def patient_progress(cat, background_blocks):
    """Measured train-patient summaries, not LLM-inferred clinical severity."""
    groups = defaultdict(list)
    for r in cat["recordings"]:
        groups[r["patient"]].append(r)
    result = {}
    for patient, data in sorted(groups.items()):
        ids = {r["id"] for r in data}
        result[patient] = {"recordings": len(data), "contents": len({r["content"] for r in data}),
            **{k: sum(r[k] for r in data)/len(data) for k in ("cer", "previous_cer", "zero_cer")},
            **{k: sum(r[k] for r in data) for k in ("exposures_recent", "exposures_total",
                "recent_recovered_chars", "recent_regressed_chars", "persistent_error_chars")},
            "all_content_capacity_by_endpoint": {str(end): 2*len(data)-sum(i in ids for i in background)
                                                  for end, background in background_blocks.items()}}
    return result
