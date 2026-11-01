"""Real-recording curriculum contracts; independent of the legacy TTS protocol."""

from collections import Counter
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .phonology import align_characters
from .text import text_normalize


class PracticeGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["repair", "contrast", "protection", "maintenance"]
    rationale: str
    evidence_ids: list[str] = Field(min_length=1)
    sample_ids: list[str] = Field(min_length=1)
    expected_change: str


class CurriculumProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diagnosis: str
    hypothesis: str
    groups: list[PracticeGroup] = Field(min_length=1, max_length=4)
    risk: str


class CurriculumReflection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observed_changes: str
    hypothesis_status: Literal["supported", "mixed", "not_supported"]
    explanation: str
    next_selection_guidance: str


def position_states(reference, hypothesis):
    alignment = align_characters(reference, hypothesis)
    return ["unresolved" if len(p) != 1 else
            "correct" if p[0] is not None and reference[i] == hypothesis[p[0]] else "error"
            for i, p in enumerate(alignment.reference_outcomes)]


def catalog(current, zero):
    """Anonymous full train catalog, retaining surface GT separately from CER text."""
    assert len({r["utt_id"] for r in current}) == len(current)
    assert {r["utt_id"] for r in current} == set(zero)
    patients = {v: f"P{i:03d}" for i, v in enumerate(sorted({r["speaker_id"] for r in current}), 1)}
    contents = {v: f"C{i:02d}" for i, v in enumerate(sorted({r["clean_gt"] for r in current}), 1)}
    entries, mapping = [], {}
    for n, row in enumerate(sorted(current, key=lambda r: r["utt_id"]), 1):
        z = zero[row["utt_id"]]
        assert z["clean_gt"] == row["clean_gt"] and z["speaker_id"] == row["speaker_id"]
        ref, hyp, old = map(text_normalize, (row["clean_gt"], row["prediction"], z["raw_pred"]))
        a = align_characters(ref, hyp)
        old_states, new_states = position_states(ref, old), position_states(ref, hyp)
        transitions = Counter(zip(old_states, new_states))
        sid = f"R{n:04d}"
        entries.append({"id": sid, "patient": patients[row["speaker_id"]],
            "content": contents[row["clean_gt"]], "bucket": row["zero_shot_bucket"],
            "seconds": round(float(row["duration"]), 2), "prediction": row["prediction"],
            "cer": round(a.edit_distance / len(ref), 6),
            "zero_cer": round(align_characters(ref, old).edit_distance / len(ref), 6),
            "SDI": [a.substitution_count, a.deletion_count, a.insertion_count],
            "regressed_chars": transitions[("correct", "error")],
            "recovered_chars": transitions[("error", "correct")],
            "uncertain_chars": sum(x == "unresolved" or y == "unresolved" for x, y in zip(old_states, new_states))})
        mapping[sid] = row
    return {"contents": {v: k for k, v in contents.items()}, "recordings": entries}, mapping


def validate_selection(ids, entries, background):
    by_id = {r["id"]: r for r in entries}
    assert len(ids) == len(set(ids)) == 64, "select exactly 64 distinct recordings"
    assert not set(ids) & set(background), "background cannot be selected again"
    assert set(ids) <= set(by_id), "unknown recording ID"
    all_ids = list(background) + list(ids)
    assert len(all_ids) == len(set(all_ids)) == 96
    for field in ("patient", "content"):
        assert max(Counter(by_id[i][field] for i in all_ids).values()) <= 8, field
    return ids


def compile_proposal(proposal, entries, background):
    known = {r["id"] for r in entries}
    for group in proposal.groups:
        assert set(group.evidence_ids) <= known, "unknown evidence citation"
    ids = [i for group in proposal.groups for i in group.sample_ids]
    return validate_selection(ids, entries, background)


def rule_selection(entries, background, method, seed):
    by_id = {r["id"]: r for r in entries}
    patients = Counter(by_id[i]["patient"] for i in background)
    contents = Counter(by_id[i]["content"] for i in background)
    pool = [r for r in entries if r["id"] not in background]
    rng = random.Random(seed)
    rng.shuffle(pool)
    if method == "difficulty":
        pool.sort(key=lambda r: r["cer"], reverse=True)
    else:
        assert method == "random"
        # Proportional original-difficulty allocation; random within each bucket.
        counts = Counter(r["bucket"] for r in pool)
        remaining = {k: 64 * v // len(pool) for k, v in counts.items()}
        for k in sorted(counts, key=lambda k: (64 * counts[k]) % len(pool), reverse=True)[:64-sum(remaining.values())]:
            remaining[k] += 1
    chosen = []
    for r in pool:
        if patients[r["patient"]] >= 8 or contents[r["content"]] >= 8:
            continue
        if method == "random" and remaining[r["bucket"]] == 0:
            continue
        chosen.append(r["id"])
        patients[r["patient"]] += 1
        contents[r["content"]] += 1
        if method == "random":
            remaining[r["bucket"]] -= 1
        if len(chosen) == 64:
            break
    return validate_selection(chosen, entries, background)


def selection_stats(ids, entries):
    selected = [r for r in entries if r["id"] in set(ids)]
    return {"rows": len(selected), "patients": len({r["patient"] for r in selected}),
            "contents": len({r["content"] for r in selected}),
            "buckets": dict(Counter(r["bucket"] for r in selected)),
            "seconds": sum(r["seconds"] for r in selected),
            "mean_current_train_cer": sum(r["cer"] for r in selected) / len(selected),
            "regressed_chars": sum(r["regressed_chars"] for r in selected),
            "max_patient_count": max(Counter(r["patient"] for r in selected).values()),
            "max_content_count": max(Counter(r["content"] for r in selected).values())}
