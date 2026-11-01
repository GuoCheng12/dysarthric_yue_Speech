from __future__ import annotations

import csv
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
    "assessed": {"evidence"},
    "proposed": {"agent_request", "agent_response", "proposal", "evaluation_plan"},
    "materialized": {"synthesis_manifest", "materialization_sidecar"},
    "synthesized": {"synthesis_summary", "train_manifest", "training_index"},
    "trained": {"checkpoint", "training_receipt"},
    "reassessed": {"after_predictions", "round_result", "reflection_pack"},
    "reflected": {"reflection_request", "reflection_response", "reflection"},
    "committed": {"ledger_entry"},
    "completed": {"test_authorization", "final_test_result"},
}
SKILL_UPDATE_ARTIFACTS = {"skill_activation", "skill_version"}


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
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                try:
                    next(reader)
                except StopIteration:
                    rows = 0
                else:
                    rows = sum(1 for row in reader if any(cell.strip() for cell in row))
        else:
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
            protocol.initial_adapter_dev_predictions,
            protocol.zero_shot_dev_predictions,
            protocol.real_train,
            protocol.feedback_split,
            protocol.speech_generator,
            protocol.frozen_reference_manifest,
        ):
            verify_artifact(reference)
        if protocol.sealed_test_split is not None:
            verify_artifact(protocol.sealed_test_split)
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

    def intervention_history(self):
        from .memory import validate_summary_digest
        from .schema import InterventionSummary

        self.verify_integrity()
        history = [
            InterventionSummary.model_validate_json(
                Path(event.artifact_paths["ledger_entry"]).read_text("utf-8")
            )
            for event in self.events()
            if event.phase == "committed"
        ]
        for summary in history:
            validate_summary_digest(summary)
        return history

    def advance(self, phase: RoundPhase, artifacts: dict[str, Path]) -> Event:
        previous = self.current()
        self.verify_integrity()
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
        self._validate_transition(event, protocol)
        self._write_event(event)
        return event

    def _validate_transition(self, event: Event, protocol: RunProtocol) -> None:
        """Check existing domain contracts before recording a success event.

        This is not a training runner or an assertion that an external process
        executed honestly. It checks registered outputs against their sources.
        """

        # Local imports avoid the materialize -> store dependency cycle.
        from openai.types.responses.response import Response

        from .agent import (
            PRESCRIBE_TASK,
            REFLECT_TASK,
            parse_reflection_response,
            parse_therapist_response,
        )
        from .materialize import build_materialization_sidecar
        from .memory import SkillActivation, compile_intervention_summary
        from .outcome import (
            _compile_realization,
            _verify_plan,
            compile_round_result,
            validate_reflection_pack,
        )
        from .round_schema import (
            EvaluationPlan,
            MaterializationSidecar,
            ReflectionPack,
            ReflectionV2,
            RoundResultV2,
            SynthesisReceipt,
            TrainingReceipt,
        )
        from .schema import (
            EvidencePack,
            InterventionSummary,
            Proposal,
            validate_evidence_pack,
        )

        if event.phase == "completed":
            raise NotImplementedError(
                f"{event.phase} execution contract is not implemented; "
                "an artifact journal cannot declare this phase successful"
            )

        paths: dict[str, Path] = {}
        prior_events = self.events()
        for previous in [*prior_events, event]:
            if previous.round_index == event.round_index:
                if previous is event and set(paths).intersection(event.artifact_paths):
                    raise ValueError(
                        "cannot redefine previously registered round artifacts"
                    )
                paths.update(
                    {key: Path(value) for key, value in previous.artifact_paths.items()}
                )

        def read_model(name: str, model: type[BaseModel]) -> Any:
            return model.model_validate_json(paths[name].read_text("utf-8"))

        def read_rows(name: str) -> list[dict[str, Any]]:
            rows = [
                json.loads(line) for line in paths[name].read_text("utf-8").splitlines()
            ]
            if not rows or any(not isinstance(row, dict) for row in rows):
                raise ValueError(f"{name} must contain non-empty JSONL objects")
            return rows

        def validate_request(name: str, key: str, payload: BaseModel) -> None:
            request = json.loads(paths[name].read_text("utf-8"))
            if (
                request.get("model") != protocol.agent.model
                or request.get("reasoning")
                != {"effort": protocol.agent.reasoning_effort}
                or request.get("store") != protocol.agent.store
            ):
                raise ValueError(f"{name} differs from the frozen Agent configuration")
            user_message = request["input"][-1]
            expected = {
                "task": {"evidence": PRESCRIBE_TASK, "reflection_pack": REFLECT_TASK}[key],
                f"{key}_sha256": canonical_sha256(payload),
                key: payload.model_dump(mode="json"),
            }
            if (
                user_message["role"] != "user"
                or json.loads(user_message["content"]) != expected
            ):
                raise ValueError(f"{name} input differs from its registered source")

        evidence = read_model("evidence", EvidencePack)
        validate_evidence_pack(evidence, protocol)
        if evidence.artifact_mode != "recorded_execution":
            raise ValueError("counterfactual evidence cannot advance an execution run")
        if evidence.round_index != event.round_index:
            raise ValueError("EvidencePack round differs from transition")
        if event.round_index == 1:
            parent_hash = protocol.initial_adapter.sha256
            before_path = Path(protocol.initial_adapter_dev_predictions.path)
        else:
            history = self.intervention_history()
            if evidence.intervention_history != history:
                raise ValueError(
                    "EvidencePack history differs from committed reflection episodes"
                )
            parent_event = next(
                item
                for item in prior_events
                if item.round_index == event.round_index - 1 and item.phase == "trained"
            )
            evaluation_event = next(
                item
                for item in prior_events
                if item.round_index == event.round_index - 1
                and item.phase == "reassessed"
            )
            parent_hash = parent_event.artifact_sha256["checkpoint"]
            before_path = Path(evaluation_event.artifact_paths["after_predictions"])
            if protocol.preference_loop is not None and protocol.preference_loop.checkpoint_acceptance != 'always':
                decision=json.loads(Path(evaluation_event.artifact_paths['promotion_decision']).read_text())
                parent_hash=decision['accepted_checkpoint_sha256']
                before_path=Path(decision['accepted_predictions'])
                if sha256_directory(Path(decision['accepted_checkpoint']))!=parent_hash:
                    raise ValueError('accepted parent changed')
        if (
            evidence.snapshots.current.checkpoint_sha256 != parent_hash
            or evidence.snapshots.current.predictions_sha256 != sha256_file(before_path)
        ):
            raise ValueError("EvidencePack current snapshot differs from prior round")
        if event.phase == "assessed":
            return

        proposal = read_model("proposal", Proposal)
        plan = read_model("evaluation_plan", EvaluationPlan)
        _verify_plan(protocol, evidence, proposal, plan)
        if event.phase == "proposed":
            validate_request("agent_request", "evidence", evidence)
            if event.round_index > 1:
                request = json.loads(paths["agent_request"].read_text("utf-8"))
                previous_commit = next(
                    item
                    for item in prior_events
                    if item.phase == "committed"
                    and item.round_index == event.round_index - 1
                )
                if "skill_activation" in previous_commit.artifact_paths:
                    expected_skill = SkillActivation.model_validate_json(
                        Path(
                            previous_commit.artifact_paths["skill_activation"]
                        ).read_text("utf-8")
                    ).skill.file_sha256
                else:
                    expected_skill = history[-1].episode_memory.active_skill_sha256
                skill_text = request["input"][1]["content"].removeprefix(
                    "Current therapist skill:\n"
                )
                if hashlib.sha256(skill_text.encode()).hexdigest() != expected_skill:
                    raise ValueError(
                        "prescription does not use the next round's active Skill"
                    )
            parsed = parse_therapist_response(
                read_model("agent_response", Response), protocol, evidence
            )
            if parsed != proposal:
                raise ValueError("registered Proposal differs from Agent output")
            return

        sidecar = read_model("materialization_sidecar", MaterializationSidecar)
        assignments = read_rows("synthesis_manifest")
        expected_sidecar = build_materialization_sidecar(
            proposal, evidence, protocol, assignments,
            **({'generation_plan_path':paths['generation_plan']} if protocol.preference_loop is not None and protocol.preference_loop.generation_design_path else {})
        )
        if sidecar != expected_sidecar:
            raise ValueError("registered sidecar differs from synthesis assignments")
        if event.phase == "materialized":
            return

        synthesis = read_model("synthesis_summary", SynthesisReceipt)
        if synthesis.artifact_mode != "recorded_execution":
            raise ValueError("counterfactual synthesis cannot advance an execution run")
        if (synthesis.run_id, synthesis.round_index) != (
            protocol.run_id,
            event.round_index,
        ):
            raise ValueError("synthesis receipt identity differs from transition")
        _compile_realization(sidecar, synthesis)
        sample_hashes = {
            item.sample_id: item.audio_file_sha256 for item in synthesis.samples
        }
        for assignment in assignments:
            if (
                sha256_file(Path(assignment["output_wav"]))
                != sample_hashes[assignment["sample_id"]]
            ):
                raise ValueError("generated audio hash differs from synthesis receipt")
        train_rows = read_rows("train_manifest")
        if protocol.preference_loop is not None:
            from .preference_loop import validate_registered_preferences
            validate_registered_preferences(protocol, paths, train_rows, assignments, parent_hash)
        elif len(train_rows) != (
            protocol.real_replay_count_per_round + protocol.synthetic_count_per_round
        ) or any(not row.get("audio") or not row.get("text") for row in train_rows):
            raise ValueError("train manifest differs from the frozen row budget/schema")
        if event.phase == "synthesized":
            return

        training = read_model("training_receipt", TrainingReceipt)
        checkpoint = paths["checkpoint"]
        adapter_config = json.loads(
            (checkpoint / "adapter_config.json").read_text("utf-8")
        )
        if (
            not (checkpoint / "adapter_model.safetensors").is_file()
            or adapter_config.get("r") != protocol.lora.rank
            or adapter_config.get("lora_alpha") != protocol.lora.alpha
            or adapter_config.get("lora_dropout") != protocol.lora.dropout
            or set(adapter_config.get("target_modules", []))
            != set(protocol.lora.target_modules)
        ):
            raise ValueError(
                "checkpoint adapter differs from frozen LoRA configuration"
            )
        state_path = checkpoint / "trainer_state.json"
        trainer_state = json.loads(state_path.read_text("utf-8"))
        if (
            training.artifact_mode != "recorded_execution"
            or (training.run_id, training.round_index)
            != (protocol.run_id, event.round_index)
            or training.parent_checkpoint_directory_sha256 != parent_hash
            or training.new_checkpoint_directory_sha256 != sha256_directory(checkpoint)
            or training.train_manifest_file_sha256
            != sha256_file(paths["train_manifest"])
            or training.training_index_file_sha256
            != sha256_file(paths["training_index"])
            or training.trainer_state_file_sha256 != sha256_file(state_path)
            or training.total_rows != len(train_rows)
            or training.actual_optimizer_steps
            != protocol.lora.optimizer_steps_per_round
            or trainer_state.get("global_step") != training.actual_optimizer_steps
        ):
            raise ValueError(
                "training receipt differs from registered execution artifacts"
            )
        from .preference_contract import validate_preference_training
        validate_preference_training(protocol, training, sample_ids=sample_hashes)
        if protocol.preference_loop is not None:
            from .preference_loop import validate_registered_preference_training
            validate_registered_preference_training(protocol, paths, training)
        if event.phase == "trained":
            return

        result = read_model("round_result", RoundResultV2)
        from .evidence_pack import load_prediction_rows

        before_rows = load_prediction_rows(before_path)
        after_rows = load_prediction_rows(paths["after_predictions"])
        expected_result = compile_round_result(
            protocol,
            evidence,
            proposal,
            plan,
            sidecar,
            synthesis,
            training,
            before_rows=before_rows,
            after_rows=after_rows,
            before_predictions_file_sha256=sha256_file(before_path),
            after_predictions_file_sha256=sha256_file(paths["after_predictions"]),
        )
        if result != expected_result:
            raise ValueError("RoundResult differs from registered prediction files")
        pack = read_model("reflection_pack", ReflectionPack)
        validate_reflection_pack(pack, protocol, evidence, proposal)
        if pack.round_result != result or pack.materialization_sidecar != sidecar:
            raise ValueError("ReflectionPack differs from the registered round result")
        if protocol.preference_loop is not None and protocol.preference_loop.checkpoint_acceptance != 'always':
            from .checkpoint_promotion import make_decision
            decision=json.loads(paths['promotion_decision'].read_text())
            expected=make_decision(pack,parent=Path(decision['parent_checkpoint']),before=before_path,
                candidate=paths['checkpoint'],after=paths['after_predictions'],reference_sha256=protocol.initial_adapter.sha256,protocol=protocol)
            if decision!=expected:
                raise ValueError('checkpoint acceptance differs from fixed rule and measured result')
        prescription_request = json.loads(paths["agent_request"].read_text("utf-8"))
        prescribed_skill = prescription_request["input"][1]["content"].removeprefix(
            "Current therapist skill:\n"
        )
        if (
            hashlib.sha256(prescribed_skill.encode()).hexdigest()
            != pack.provenance.skill_file_sha256
        ):
            raise ValueError("ReflectionPack Skill differs from prescription Skill")
        if event.phase == "reassessed":
            return

        reflection = read_model("reflection", ReflectionV2)
        validate_request("reflection_request", "reflection_pack", pack)
        parsed = parse_reflection_response(
            read_model("reflection_response", Response), pack
        )
        if reflection != parsed:
            raise ValueError("registered Reflection differs from Agent output")
        reflection_request = json.loads(paths["reflection_request"].read_text("utf-8"))
        reflected_skill = reflection_request["input"][1]["content"].removeprefix(
            "Frozen Therapist skill used for this round "
            "(interpretation only; it does not authorize a new prescription):\n"
        )
        if (
            hashlib.sha256(reflected_skill.encode()).hexdigest()
            != pack.provenance.skill_file_sha256
        ):
            raise ValueError("Reflection request Skill differs from prescription Skill")
        if event.phase == "reflected":
            return

        ledger = read_model("ledger_entry", InterventionSummary)
        if ledger != compile_intervention_summary(
            protocol, evidence, proposal, pack, reflection
        ):
            raise ValueError(
                "ledger differs from the measured result and actual Reflection"
            )
        if event.round_index in protocol.skill_update_rounds:
            activation = read_model("skill_activation", SkillActivation)
            if (
                activation.after_round_index != event.round_index
                or activation.action == "initial"
            ):
                raise ValueError("Skill activation differs from the scheduled round")
            if sha256_file(paths["skill_version"]) != activation.skill.file_sha256:
                raise ValueError("Skill version differs from the registered activation")
            activation.skill.read_text()
            if activation.action == "no_change":
                if (
                    activation.skill.version != pack.provenance.skill_version
                    or activation.skill.file_sha256 != pack.provenance.skill_file_sha256
                ):
                    raise ValueError("no_change must retain the round's active Skill")
            elif (
                activation.skill.version
                != f"v{int(pack.provenance.skill_version[1:]) + 1:03d}"
                or activation.skill.file_sha256 == pack.provenance.skill_file_sha256
            ):
                raise ValueError("updated Skill must create the next distinct version")

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
