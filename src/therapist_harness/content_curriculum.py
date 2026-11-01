"""Content-first real-speech lessons: evidence, exact delivery and reassessment.

This module does not train models or generate speech. Character regions refer to
ASR-normalized text, not acoustic phone boundaries or clinical impairments.
"""

from collections import Counter
import hashlib
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .curriculum_v2 import summarize_pair
from .phonology import align_characters
from .text import text_normalize

REGIONS = ("whole", "first_half", "second_half", "last_two")
SELECTORS = ("error", "deletion", "substitution", "regression", "correct", "any")
BUCKETS = ("easy", "medium", "hard")


class ContentDose(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_id: str = Field(pattern=r"^C[0-9]{2}$")
    role: Literal["target", "related", "protection"]
    region: Literal["whole", "first_half", "second_half", "last_two"]
    selection: Literal["error", "deletion", "substitution", "regression", "correct", "any"]
    buckets: list[Literal["easy", "medium", "hard"]] = Field(min_length=1, max_length=3)
    exposures: int = Field(ge=1, le=240)
    reason: str = Field(min_length=1)


class ContentLesson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lesson_id: str = Field(pattern=r"^L[0-9]{3}$")
    title: str = Field(min_length=1)
    recognition_goal: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=2)
    hypothesis: str = Field(min_length=1)
    practice: list[ContentDose] = Field(min_length=3, max_length=6)
    expected_transfer: str = Field(min_length=1)
    revision_condition: str = Field(min_length=1)


class ContentPrescription(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diagnosis: str = Field(min_length=1)
    lessons: list[ContentLesson] = Field(min_length=2, max_length=3)
    history_used: str = Field(min_length=1)
    risk: str = Field(min_length=1)


class LessonReflection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lesson_id: str
    observation_ids: list[str] = Field(min_length=1)
    delivery_interpretation: str
    conclusion: Literal["progress", "local_fit_only", "regression", "mixed", "no_change"]
    next_action: Literal["continue", "reduce", "replace", "pause"]
    explanation: str
    reusable_lesson: str


class ContentReflection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observed_changes: str
    lessons: list[LessonReflection] = Field(min_length=2, max_length=3)
    next_guidance: str


def region_bounds(text, region):
    n = len(text)
    assert n >= 2, "content regions require at least two normalized characters"
    return {"whole": (0, n), "first_half": (0, n // 2),
            "second_half": (n // 2, n), "last_two": (n - 2, n)}[region]


def character_states(reference, hypothesis):
    result = []
    for i, choices in enumerate(align_characters(reference, hypothesis).reference_outcomes):
        result.append("uncertain" if len(choices) != 1 else "deletion" if choices[0] is None
                      else "correct" if reference[i] == hypothesis[choices[0]] else "substitution")
    return result


def enrich_catalog(catalog):
    """Compute selector membership once, from this checkpoint's real predictions."""
    entries = []
    for row in catalog["recordings"]:
        ref = text_normalize(catalog["contents"][row["content"]])
        hyp, old = map(text_normalize, (row["prediction"], row["previous_prediction"]))
        now, before = character_states(ref, hyp), character_states(ref, old)
        regions = {}
        for region in REGIONS:
            a, b = region_bounds(ref, region)
            states = now[a:b]
            regions[region] = {
                "error": sum(v in ("substitution", "deletion") for v in states),
                "deletion": states.count("deletion"), "substitution": states.count("substitution"),
                "regression": sum(before[i] == "correct" and now[i] in ("substitution", "deletion") for i in range(a, b)),
                # Correct means an exactly correct whole recording, not just a
                # correct span in an otherwise erroneous/inserted transcript.
                "correct": int(ref == hyp), "any": 1, "uncertain": states.count("uncertain"),
            }
        entries.append({**row, "region_hits": regions})
    return entries


def eligible(dose, entries):
    return [r for r in entries if r["content"] == dose.content_id and r["bucket"] in dose.buckets
            and r["region_hits"][dose.region][dose.selection] > 0]


def evidence_pack(catalog, background, checkpoint_step, feedback, history=()):
    entries = enrich_catalog(catalog)
    counts = Counter(background)
    assert set(background) <= {r["id"] for r in entries}
    assert max(counts.values(), default=0) <= 2
    contents = []
    for cid, text in sorted(catalog["contents"].items()):
        data = [r for r in entries if r["content"] == cid]
        ref = text_normalize(text)
        summaries, capacities = {}, []
        for bucket in BUCKETS:
            group = [r for r in data if r["bucket"] == bucket]
            summaries[bucket] = {"rows": len(group), "patients": len({r["patient"] for r in group}),
                "cer": sum(r["cer"] for r in group) / len(group) if group else None,
                "previous_cer": sum(r["previous_cer"] for r in group) / len(group) if group else None}
        for region in REGIONS:
            for selection in SELECTORS:
                for bucket in BUCKETS:
                    group = [r for r in data if r["bucket"] == bucket and r["region_hits"][region][selection] > 0]
                    if group:
                        capacities.append([region, selection, bucket, len(group),
                            len({r["patient"] for r in group}), sum(2-counts[r["id"]] for r in group)])
        examples = sorted(data, key=lambda r: (-r["recent_regressed_chars"], -r["cer"], r["id"]))[:3]
        contents.append({"evidence_id": f"E_{cid}", "content_id": cid, "text": text,
            "normalized_characters": len(ref), "rows": len(data), "patients": len({r["patient"] for r in data}),
            "bucket_progress": summaries,
            "regions": {k: ref[slice(*region_bounds(ref, k))] for k in REGIONS},
            "selector_capacity": capacities,
            "examples": [{k: r[k] for k in ("id", "patient", "bucket", "prediction", "previous_prediction", "cer", "previous_cer")} for r in examples]})
    return {"checkpoint_step": checkpoint_step, "scope": "existing train recordings only",
        "train_rows": len(entries), "train_patients": len({r["patient"] for r in entries}),
        "content_count": len(contents), "course_exposures": 240, "coverage_exposures": len(background),
        "capacity_columns": ["region", "selection", "bucket", "rows", "patients", "exposure_capacity"],
        "contents": contents, "feedback": feedback, "history": list(history)}, entries


def validate_prescription(proposal, entries, background):
    lessons = proposal.lessons
    assert len({l.lesson_id for l in lessons}) == len(lessons), "duplicate lesson ID"
    counts = Counter(background)
    assert set(background) <= {r["id"] for r in entries}, "unknown coverage recording"
    assert max(counts.values(), default=0) <= 2, "coverage exceeds repetition cap"
    known_contents = {r["content"] for r in entries}
    doses = [d for lesson in lessons for d in lesson.practice]
    assert sum(d.exposures for d in doses) == 240, "allocate exactly 240 course exposures"
    assert len({d.content_id for d in doses}) == len(doses), "each content has one role per block"
    assert any(d.role == "protection" for d in doses), "include explicit protection practice"
    for lesson in lessons:
        targets = [d for d in lesson.practice if d.role == "target"]
        assert len(targets) >= 2, "a lesson needs at least two distinct target contents"
        assert any(d.role == "related" for d in lesson.practice), "explain related-content practice"
        assert len(lesson.evidence_ids) == len(set(lesson.evidence_ids))
        assert set(lesson.evidence_ids) <= {f"E_{d.content_id}" for d in lesson.practice}, "unrelated evidence citation"
        assert {f"E_{d.content_id}" for d in targets} <= set(lesson.evidence_ids), "cite each target content"
        for dose in lesson.practice:
            assert dose.content_id in known_contents, "unknown or held-out content"
            assert len(dose.buckets) == len(set(dose.buckets)), "duplicate bucket"
            if dose.role == "target":
                assert dose.selection in ("error", "deletion", "substitution", "regression"), "target practice must match an observed error"
            if dose.role == "protection":
                assert dose.selection in ("regression", "correct"), "protection is regression repair or exact-correct rehearsal"
                assert set(dose.buckets) <= {"easy", "medium"}, "protection covers original easy/medium recordings across patients"
            group = eligible(dose, entries)
            assert dose.exposures <= sum(2-counts[r["id"]] for r in group), f"insufficient capacity: {dose.content_id}"


def compile_lesson_block(proposal, entries, background, end_step, previous_manifests=()):
    """Explicit content quotas, balanced eligible patients, then rotating recordings."""
    validate_prescription(proposal, entries, background)
    assert len(background) == 400 and end_step >= 40 and end_step % 20 == 0
    counts, patient_counts = Counter(background), Counter()
    past = Counter(r["id"] for m in previous_manifests for r in m)
    chosen = []
    for lesson in proposal.lessons:
        for dose in lesson.practice:
            pool = eligible(dose, entries)
            order = sorted(r["id"] for r in pool)
            seed = int.from_bytes(hashlib.sha256(f"42:{end_step}:{dose.content_id}".encode()).digest()[:8], "big")
            random.Random(seed).shuffle(order)
            rank = {sid: n for n, sid in enumerate(order)}
            per_patient = Counter()
            for _ in range(dose.exposures):
                available = [r for r in pool if counts[r["id"]] < 2]
                row = min(available, key=lambda r: (per_patient[r["patient"]], patient_counts[r["patient"]],
                    past[r["id"]] + counts[r["id"]], rank[r["id"]]))
                chosen.append({"id": row["id"], "stream": "course", "unit_id": lesson.lesson_id,
                    "content": dose.content_id, "role": dose.role, "region": dose.region,
                    "selection": dose.selection})
                counts[row["id"]] += 1
                per_patient[row["patient"]] += 1
                patient_counts[row["patient"]] += 1
    random.Random(500+end_step).shuffle(chosen)
    b, c = iter(background), iter(chosen)
    manifest = []
    for n in range(640):
        course_slot = ((n % 32)+1)*12//32 > (n % 32)*12//32
        row = next(c) if course_slot else {"id": next(b), "stream": "coverage", "unit_id": None}
        manifest.append({**row, "position": (end_step-20)*32+n, "step": end_step-19+n//32})
    assert max(Counter(r["id"] for r in manifest).values()) <= 2
    return manifest


def delivery_report(proposal, entries, background, manifest):
    by_id = {r["id"]: r for r in entries}
    counts = Counter(background)
    result = {}
    assert [r["id"] for r in manifest if r["stream"] == "coverage"] == list(background)
    assert len(manifest) == 640
    assert max(Counter(r["id"] for r in manifest).values()) <= 2
    for n in range(0, 640, 32):
        assert Counter(r["stream"] for r in manifest[n:n+32]) == {"coverage": 20, "course": 12}
    for lesson in proposal.lessons:
        summary = []
        for dose in lesson.practice:
            selected = [r for r in manifest if r["unit_id"] == lesson.lesson_id and r.get("content") == dose.content_id]
            pool = eligible(dose, entries)
            pool_ids = {r["id"] for r in pool}
            assert len(selected) == dose.exposures and all(r["id"] in pool_ids for r in selected)
            assert all(r["stream"] == "course" and r["role"] == dose.role
                       and r["region"] == dose.region and r["selection"] == dose.selection for r in selected)
            actual = [by_id[r["id"]] for r in selected]
            summary.append({"content_id": dose.content_id, "role": dose.role, "region": dose.region,
                "selection": dose.selection, "planned_exposures": dose.exposures, "actual_exposures": len(actual),
                "eligible_recordings": len(pool), "eligible_patients": len({r["patient"] for r in pool}),
                "exposure_capacity": sum(2-counts[r["id"]] for r in pool),
                "unique_recordings": len({r["id"] for r in actual}),
                "patient_exposures": dict(sorted(Counter(r["patient"] for r in actual).items())),
                "bucket_exposures": dict(Counter(r["bucket"] for r in actual)),
                "selector_match_exposures": sum(r["region_hits"][dose.region][dose.selection] > 0 for r in actual),
                "recent_regression_exposures": sum(r["region_hits"][dose.region]["regression"] > 0 for r in actual),
                "exact_correct_exposures": sum(r["region_hits"]["whole"]["correct"] > 0 for r in actual),
                "audio_seconds_with_repeats": sum(r["seconds"] for r in actual)})
        result[lesson.lesson_id] = summary
    return result


def focus_changes(before, after, ids, region):
    """Measure the prescribed text region, separately from whole-recording CER."""
    old_counts, new_counts, transitions = Counter(), Counter(), Counter()
    positions = 0
    for sid in ids:
        ref = text_normalize(before[sid]["clean_gt"])
        assert ref == text_normalize(after[sid]["clean_gt"])
        a, b = region_bounds(ref, region)
        old = character_states(ref, text_normalize(before[sid]["prediction"]))[a:b]
        new = character_states(ref, text_normalize(after[sid]["prediction"]))[a:b]
        old_counts.update(old)
        new_counts.update(new)
        transitions.update(zip(old, new))
        positions += b-a
    return {"unique_recordings": len(ids), "region": region, "reference_positions": positions,
        "before": {s: old_counts[s] for s in ("correct", "deletion", "substitution", "uncertain")},
        "after": {s: new_counts[s] for s in ("correct", "deletion", "substitution", "uncertain")},
        "recovered_positions": sum(transitions[s, "correct"] for s in ("deletion", "substitution")),
        "regressed_positions": sum(transitions["correct", s] for s in ("deletion", "substitution"))}


def lesson_outcome(step, proposal, delivery, manifest, mapping, before, after,
                   feedback_before, feedback_after):
    """Call only after an actual update; fixture tests must be labelled as such."""
    trained = {r["id"] for r in manifest}
    practiced_pairs = {(mapping[i]["patient"], mapping[i]["content"]) for i in trained}
    observations, lesson_ids = {}, {}
    for bucket in ("overall", *BUCKETS):
        ids = [i for i, r in feedback_after.items() if bucket == "overall" or r["zero_shot_bucket"] == bucket]
        observations[f"S{step}_feedback_{bucket}"] = summarize_pair(feedback_before, feedback_after, ids)
    globals_ = list(observations)
    for lesson in proposal.lessons:
        selected = [r for r in manifest if r["unit_id"] == lesson.lesson_id]
        patients = {mapping[r["id"]]["patient"] for r in selected}
        contents = {d.content_id for d in lesson.practice}
        subsets = {f"practiced_{role}": sorted({r["id"] for r in selected if r["role"] == role})
                   for role in ("target", "related", "protection")}
        subsets["same_content_unpracticed_patients"] = [i for i, r in mapping.items() if r["content"] in contents
            and (r["patient"], r["content"]) not in practiced_pairs]
        subsets["same_patients_other_content"] = [i for i, r in mapping.items() if r["patient"] in patients
            and r["content"] not in contents and (r["patient"], r["content"]) not in practiced_pairs]
        subsets["heldout_content"] = [i for i, r in feedback_after.items() if r["patient"] in patients]
        lesson_ids[lesson.lesson_id] = []
        for name, ids in subsets.items():
            oid = f"S{step}_{lesson.lesson_id}_{name}"
            observations[oid] = summarize_pair(feedback_before if name == "heldout_content" else before,
                feedback_after if name == "heldout_content" else after, ids)
            lesson_ids[lesson.lesson_id].append(oid)
        for dose in lesson.practice:
            ids = sorted({r["id"] for r in selected if mapping[r["id"]]["content"] == dose.content_id})
            oid = f"S{step}_{lesson.lesson_id}_{dose.content_id}_region"
            observations[oid] = focus_changes(before, after, ids, dose.region)
            lesson_ids[lesson.lesson_id].append(oid)
    return {"step": step, "proposal": proposal.model_dump(mode="json"), "delivery": delivery,
        "global_observation_ids": globals_, "lesson_observation_ids": lesson_ids, "observations": observations,
        "interpretation": "Joint outcomes with coverage; train subsets are not globally unseen content. Empty subsets are unobserved."}


def validate_lesson_reflection(reflection, outcome):
    expected = set(outcome["lesson_observation_ids"])
    assert len(reflection.lessons) == len(expected) and {r.lesson_id for r in reflection.lessons} == expected
    for r in reflection.lessons:
        own = set(outcome["lesson_observation_ids"][r.lesson_id])
        assert set(r.observation_ids) <= own | set(outcome["global_observation_ids"])
        assert set(r.observation_ids) & own, "cite own-lesson observations"
