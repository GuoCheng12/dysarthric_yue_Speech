from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schema import ArtifactRef, RunProtocol, canonical_sha256


RoundPhase = Literal[
    "initialized",
    "assessed",
    "proposed",
    "materialized",
    "synthesized",
    "trained",
    "reassessed",
    "reflected",
    "committed",
    "completed",
]

ROUND_PHASES: tuple[RoundPhase, ...] = (
    "assessed",
    "proposed",
    "materialized",
    "synthesized",
    "trained",
    "reassessed",
    "reflected",
    "committed",
)

REQUIRED_ARTIFACTS: dict[RoundPhase, set[str]] = {
    "initialized": {"protocol"},
    "assessed": {"evidence", "therapist_input"},
    "proposed": {"agent_request", "agent_response", "proposal"},
    "materialized": {"assignments", "synthesis_manifest", "train_manifest"},
    "synthesized": {"synthesis_summary"},
    "trained": {"checkpoint"},
    "reassessed": {"result"},
    "reflected": {"reflection"},
    "committed": {"ledger_entry"},
    "completed": {"test_authorization", "final_test_result"},
}
SKILL_UPDATE_ARTIFACTS = {"skill_patch", "skill_version"}


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    sequence: int = Field(ge=0)
    phase: RoundPhase
    round_index: int = Field(ge=0)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: dict[str, str]
    artifact_paths: dict[str, str]
    artifact_kinds: dict[str, Literal["file", "directory"]]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    if not path.is_dir():
        raise NotADirectoryError(path)
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    for candidate in files:
        if candidate.is_symlink():
            raise ValueError(f"directory artifact contains a symlink: {candidate}")
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(sha256_file(candidate).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_path(path: Path, kind: Literal["file", "directory"]) -> str:
    if kind == "file":
        if not path.is_file():
            raise FileNotFoundError(path)
        return sha256_file(path)
    return sha256_directory(path)


def write_immutable_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_immutable_json(path: Path, payload: Any) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    write_immutable_bytes(path, content)


def verify_artifact(reference: ArtifactRef) -> None:
    path = Path(reference.path)
    actual = sha256_path(path, reference.kind)
    if actual != reference.sha256:
        raise ValueError(f"artifact hash mismatch for {path}: {actual}")
    if reference.kind == "file" and reference.expected_rows is not None:
        with path.open("rb") as handle:
            rows = sum(1 for line in handle if line.strip())
        if rows != reference.expected_rows:
            raise ValueError(
                f"artifact row count mismatch for {path}: {rows} != "
                f"{reference.expected_rows}"
            )


class RunStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        self.protocol_path = self.run_dir / "protocol.json"
        self.events_dir = self.run_dir / "events"

    @classmethod
    def initialize(cls, protocol: RunProtocol, run_dir: Path) -> "RunStore":
        if protocol.protocol_status != "frozen":
            raise ValueError("only a frozen protocol can initialize a run")
        if not Path(protocol.base_model_dir).is_dir():
            raise FileNotFoundError(protocol.base_model_dir)
        for reference in (
            protocol.base_model_fingerprint,
            protocol.real_train,
            protocol.feedback_split,
            protocol.sealed_test_split,
            protocol.speech_generator,
            protocol.frozen_reference_manifest,
        ):
            verify_artifact(reference)
        if protocol.initial_adapter is not None:
            verify_artifact(protocol.initial_adapter)
        run_dir = run_dir.resolve()
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        store = cls(run_dir)
        write_immutable_json(store.protocol_path, protocol.model_dump(mode="json"))
        store._write_event(
            Event(
                sequence=0,
                phase="initialized",
                round_index=0,
                protocol_sha256=canonical_sha256(protocol),
                artifact_sha256={"protocol": sha256_file(store.protocol_path)},
                artifact_paths={"protocol": str(store.protocol_path)},
                artifact_kinds={"protocol": "file"},
            )
        )
        return store

    def protocol(self) -> RunProtocol:
        return RunProtocol.model_validate_json(self.protocol_path.read_text("utf-8"))

    def events(self) -> list[Event]:
        paths = sorted(self.events_dir.glob("*.json"))
        events = [Event.model_validate_json(path.read_text("utf-8")) for path in paths]
        if [event.sequence for event in events] != list(range(len(events))):
            raise RuntimeError("event sequence is not contiguous")
        return events

    def current(self) -> Event:
        events = self.events()
        if not events:
            raise RuntimeError("run has no initialization event")
        return events[-1]

    def advance(self, phase: RoundPhase, artifacts: dict[str, Path]) -> Event:
        previous = self.current()
        self._verify_event(previous)
        protocol = self.protocol()
        expected_phase, expected_round = self._next(previous, protocol.round_count)
        if phase != expected_phase:
            raise ValueError(
                f"invalid transition: {previous.phase}/round-{previous.round_index} "
                f"requires {expected_phase}/round-{expected_round}, got {phase}"
            )
        required_artifacts = set(REQUIRED_ARTIFACTS[phase])
        if phase == "committed" and expected_round in protocol.skill_update_rounds:
            required_artifacts.update(SKILL_UPDATE_ARTIFACTS)
        missing_artifacts = required_artifacts - set(artifacts)
        if missing_artifacts:
            raise ValueError(
                f"{phase} transition is missing artifacts: {sorted(missing_artifacts)}"
            )
        hashes: dict[str, str] = {}
        paths: dict[str, str] = {}
        kinds: dict[str, Literal["file", "directory"]] = {}
        for name, path in sorted(artifacts.items()):
            if not name:
                raise ValueError("transition artifact names must be non-empty")
            resolved = path.resolve()
            if resolved.is_file():
                kind: Literal["file", "directory"] = "file"
            elif resolved.is_dir():
                kind = "directory"
            else:
                raise FileNotFoundError(f"missing transition artifact {name}: {path}")
            hashes[name] = sha256_path(resolved, kind)
            paths[name] = str(resolved)
            kinds[name] = kind
        event = Event(
            sequence=previous.sequence + 1,
            phase=phase,
            round_index=expected_round,
            protocol_sha256=canonical_sha256(protocol),
            artifact_sha256=hashes,
            artifact_paths=paths,
            artifact_kinds=kinds,
        )
        self._write_event(event)
        return event

    def verify_integrity(self) -> None:
        protocol_digest = canonical_sha256(self.protocol())
        for event in self.events():
            if event.protocol_sha256 != protocol_digest:
                raise RuntimeError(
                    f"event {event.sequence} protocol hash does not match run protocol"
                )
            self._verify_event(event)

    @staticmethod
    def _verify_event(event: Event) -> None:
        names = set(event.artifact_sha256)
        if names != set(event.artifact_paths) or names != set(event.artifact_kinds):
            raise RuntimeError(f"event {event.sequence} artifact maps differ")
        for name in sorted(names):
            path = Path(event.artifact_paths[name])
            actual = sha256_path(path, event.artifact_kinds[name])
            if actual != event.artifact_sha256[name]:
                raise RuntimeError(
                    f"event {event.sequence} artifact changed: {name} ({path})"
                )

    @staticmethod
    def _next(previous: Event, round_count: int) -> tuple[RoundPhase, int]:
        if previous.phase == "initialized":
            return "assessed", 1
        if previous.phase == "completed":
            raise ValueError("completed run cannot advance")
        if previous.phase == "committed":
            if previous.round_index == round_count:
                return "completed", previous.round_index
            return "assessed", previous.round_index + 1
        index = ROUND_PHASES.index(previous.phase)
        return ROUND_PHASES[index + 1], previous.round_index

    def _write_event(self, event: Event) -> None:
        path = self.events_dir / f"{event.sequence:04d}_{event.phase}.json"
        write_immutable_json(path, event.model_dump(mode="json"))
