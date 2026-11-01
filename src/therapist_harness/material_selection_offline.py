"""Offline real-recording search for a diagnosis-driven material selector.

This module does not import the trainer, call an API, or change sampling weights.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .sampling_phonology import LEVELS, pronunciation


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PracticeFocus(Contract):
    focus_id: str = Field(min_length=1)
    purpose: Literal["target", "contrast", "maintenance"]
    level: Literal["onset", "nucleus", "coda", "tone", "syllable", "text"]
    target_unit: str
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class DiagnosticBrief(Contract):
    producer: Literal["diagnosis_agent", "offline_fixture"]
    round_index: int = Field(ge=0)
    source_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary: str = Field(min_length=1)
    focuses: list[PracticeFocus] = Field(min_length=1, max_length=6)


class MaterialChoice(Contract):
    focus_id: str
    recording_ids: list[str]
    reason: str = Field(min_length=1)


class MaterialSelection(Contract):
    context_sha256: str
    choices: list[MaterialChoice]


def build_index(train_rows, excluded_rows, sources):
    """Read held-out identities only for separation; never index their texts."""
    for key in ("utt_id", "audio", "speaker_id"):
        if {r[key] for r in train_rows} & {r[key] for r in excluded_rows}:
            raise ValueError(f"held-out overlap: {key}")
    ids = [r["utt_id"] for r in train_rows]
    if len(set(ids)) != len(ids) or any(r["split"] != "train" for r in train_rows):
        raise ValueError("index requires unique train recordings")
    profiles = {}
    for target in sorted({r["clean_gt"] for r in train_rows}):
        mapping = pronunciation(target)
        profiles[digest(target)[:16]] = {
            "target_text": target, "positions": [asdict(p) for p in mapping.positions],
            "coverage": asdict(mapping.coverage),
            "components": {level: dict(Counter(getattr(p, level) for p in mapping.positions
                                               if p.resolved)) for level in LEVELS},
        }
    recordings = [dict(recording_id=r["utt_id"], audio=r["audio"],
                       patient_id=r["speaker_id"], text_id=digest(r["clean_gt"])[:16],
                       seconds=float(r["duration"]),
                       initial_difficulty=r.get("zero_shot_bucket", "unknown"))
                  for r in train_rows]
    index = dict(version="material_selection_offline_v1", sources=sources,
                 profiles=profiles, recordings=recordings)
    index["index_sha256"] = digest(index)
    return index


def _check_index(index):
    if digest({k: v for k, v in index.items() if k != "index_sha256"}) != index["index_sha256"]:
        raise ValueError("index hash mismatch")


def matches(profile, focus):
    if focus.level == "text":
        return profile["target_text"] == focus.target_unit
    if focus.level == "syllable":
        return any(p["jyutping"] == focus.target_unit for p in profile["positions"])
    return focus.target_unit in profile["components"][focus.level]


def inventory(index):
    items = []
    for level in LEVELS:
        units = sorted({u for p in index["profiles"].values() for u in p["components"][level]})
        for unit in units:
            rows = [r for r in index["recordings"]
                    if unit in index["profiles"][r["text_id"]]["components"][level]]
            items.append(dict(level=level, unit=unit, recordings=len(rows),
                              texts=len({r["text_id"] for r in rows}),
                              patients=len({r["patient_id"] for r in rows})))
    return items


def prepare_candidates(index, brief, exposures=None, limit=24):
    """Balanced shortlist, plus exact full eligible IDs for expanding a query.

The shortlist is a retrieval convenience, not an Agent decision. Previous
exposures are supplied separately so the offline index stays immutable.
"""
    _check_index(index)
    brief = DiagnosticBrief.model_validate(brief)
    if limit <= 0 or len({f.focus_id for f in brief.focuses}) != len(brief.focuses):
        raise ValueError("positive limit and unique focus IDs required")
    exposure_available = exposures is not None
    exposures = exposures or {}
    known_ids = {r["recording_id"] for r in index["recordings"]}
    if not set(exposures) <= known_ids or any(type(n) is not int or n < 0 for n in exposures.values()):
        raise ValueError("exposure counts require known train IDs and nonnegative integers")
    packs = []
    for focus in brief.focuses:
        eligible = [r for r in index["recordings"] if matches(index["profiles"][r["text_id"]], focus)]
        remaining = list(eligible)
        picked, patients, texts = [], Counter(), Counter()
        while remaining and len(picked) < limit:
            r = min(remaining, key=lambda r: (texts[r["text_id"]], patients[r["patient_id"]],
                    exposures.get(r["recording_id"], 0), r["recording_id"]))
            remaining.remove(r)
            profile = index["profiles"][r["text_id"]]
            field = "jyutping" if focus.level == "syllable" else focus.level
            positions = [p for p in profile["positions"] if p["jyutping"] is not None and
                         (focus.level == "text" or p[field] == focus.target_unit)]
            picked.append({k: v for k, v in r.items() if k != "audio"} | dict(
                target_text=profile["target_text"], matching_positions=positions,
                prior_exposures=exposures.get(r["recording_id"], 0) if exposure_available else None))
            patients[r["patient_id"]] += 1
            texts[r["text_id"]] += 1
        packs.append(dict(focus_id=focus.focus_id, eligible_count=len(eligible),
                          fraction_of_training_pool=len(eligible) / len(index["recordings"]),
                          eligible_texts=len({r["text_id"] for r in eligible}),
                          eligible_patients=len({r["patient_id"] for r in eligible}),
                          eligible_ids=[r["recording_id"] for r in eligible], candidates=picked))
    return dict(diagnosis=brief.model_dump(), index_sha256=index["index_sha256"],
                training_pool_recordings=len(index["recordings"]),
                exposure_sha256=digest(exposures) if exposure_available else None,
                exposure_available=exposure_available, candidate_limit=limit, focuses=packs)


def selector_context(pack):
    # Full eligible identities stay local. Only displayed candidates can be chosen.
    return {k: v for k, v in pack.items() if k != "focuses"} | {
        "focuses": [{k: v for k, v in f.items() if k != "eligible_ids"} for f in pack["focuses"]]}


def build_selector_request(pack):
    from openai.lib._pydantic import to_strict_json_schema
    context = selector_context(pack)
    schema = to_strict_json_schema(MaterialSelection)
    variants = []
    for focus in context["focuses"]:
        choice = deepcopy(schema["$defs"]["MaterialChoice"])
        choice["properties"]["focus_id"]["enum"] = [focus["focus_id"]]
        ids = [row["recording_id"] for row in focus["candidates"]]
        if ids:
            choice["properties"]["recording_ids"]["items"]["enum"] = ids
        else:
            choice["properties"]["recording_ids"]["maxItems"] = 0
        variants.append(choice)
    schema["properties"]["choices"]["items"] = {"anyOf": variants}
    prompt = Path(__file__).with_name("prompts") / "material_selection_offline.md"
    return dict(model="gpt-5.6-sol", reasoning={"effort": "xhigh"}, store=False,
                stream=True, max_output_tokens=6000,
                input=[dict(role="system", content=prompt.read_text()),
                       dict(role="user", content=json.dumps(dict(context=context,
                            context_sha256=digest(context)), ensure_ascii=False))],
                text={"format": dict(type="json_schema", name="material_selection",
                                     strict=True, schema=schema)})


def validate_selection(index, pack, selection):
    _check_index(index)
    choice = MaterialSelection.model_validate(selection)
    if pack["index_sha256"] != index["index_sha256"] or choice.context_sha256 != digest(selector_context(pack)):
        raise ValueError("selection belongs to a different context or index")
    focuses = {f.focus_id: f for f in DiagnosticBrief.model_validate(pack["diagnosis"]).focuses}
    if len(choice.choices) != len(focuses) or {c.focus_id for c in choice.choices} != set(focuses):
        raise ValueError("one selection per diagnostic focus required")
    recordings = {r["recording_id"]: r for r in index["recordings"]}
    pools = {p["focus_id"]: {r["recording_id"] for r in p["candidates"]} for p in pack["focuses"]}
    selected = {}
    for c in choice.choices:
        if len(set(c.recording_ids)) != len(c.recording_ids):
            raise ValueError("duplicate recording in focus")
        for uid in c.recording_ids:
            if uid not in pools[c.focus_id] or uid not in recordings:
                raise ValueError("selection must use displayed train candidates")
            r = recordings[uid]
            if not matches(index["profiles"][r["text_id"]], focuses[c.focus_id]):
                raise ValueError("recording does not match diagnostic focus")
            selected.setdefault(uid, dict(r, target_text=index["profiles"][r["text_id"]]["target_text"],
                                         focus_ids=[]))["focus_ids"].append(c.focus_id)
    return dict(index_sha256=index["index_sha256"], context_sha256=choice.context_sha256,
                recordings=list(selected.values()), choices=choice.model_dump()["choices"],
                unique_recordings=len(selected),
                patients=len({r["patient_id"] for r in selected.values()}),
                texts=len({r["text_id"] for r in selected.values()}),
                training_connected=False)
