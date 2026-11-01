from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .agent import (
    build_reflection_request,
    build_therapist_request,
    execute_therapist_request,
    parse_reflection_response,
    parse_therapist_response,
    serialize_response,
)
from .evidence_pack import build_evidence_pack_from_files
from .memory import SkillMemory, compile_intervention_summary, validate_summary_digest
from .schema import (
    EvidencePack,
    InterventionSummary,
    Proposal,
    RunProtocol,
    canonical_sha256,
    validate_evidence_pack,
)
from .round_schema import ReflectionPack, ReflectionV2
from .store import RunStore, sha256_file, write_immutable_json


def load_protocol(path: Path) -> RunProtocol:
    payload = yaml.safe_load(path.read_text("utf-8"))
    return RunProtocol.model_validate(payload)


def parse_artifacts(values: list[str]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("artifact must use name=/absolute/path form")
        name, raw_path = value.split("=", 1)
        if not name or name in artifacts:
            raise ValueError(f"invalid or duplicate artifact name: {name}")
        path = Path(raw_path)
        if not path.is_absolute():
            raise ValueError("artifact paths must be absolute")
        artifacts[name] = path
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="therapist-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("protocol", type=Path)

    verify_evidence = subparsers.add_parser("verify-evidence")
    verify_evidence.add_argument("protocol", type=Path)
    verify_evidence.add_argument("evidence", type=Path)

    propose = subparsers.add_parser("propose")
    propose.add_argument("protocol", type=Path)
    propose.add_argument("evidence", type=Path)
    propose.add_argument("output", type=Path)
    propose_skill = propose.add_mutually_exclusive_group(required=True)
    propose_skill.add_argument("--skill", type=Path)
    propose_skill.add_argument("--skills-dir", type=Path)

    reflect = subparsers.add_parser("reflect")
    reflect.add_argument("protocol", type=Path)
    reflect.add_argument("evidence", type=Path)
    reflect.add_argument("proposal", type=Path)
    reflect.add_argument("reflection_pack", type=Path)
    reflect.add_argument("output", type=Path)
    reflect_skill = reflect.add_mutually_exclusive_group(required=True)
    reflect_skill.add_argument("--skill", type=Path)
    reflect_skill.add_argument("--skills-dir", type=Path)

    ledger = subparsers.add_parser("compile-ledger")
    for name in (
        "protocol",
        "evidence",
        "proposal",
        "reflection_pack",
        "reflection",
        "output",
    ):
        ledger.add_argument(name, type=Path)

    history = subparsers.add_parser("export-history")
    history.add_argument("run_dir", type=Path)
    history.add_argument("output", type=Path)

    init_skills = subparsers.add_parser("init-skills")
    init_skills.add_argument("skills_dir", type=Path)
    init_skills.add_argument("initial_skill", type=Path)

    activate_skill = subparsers.add_parser("activate-skill")
    activate_skill.add_argument("protocol", type=Path)
    activate_skill.add_argument("skills_dir", type=Path)
    activate_skill.add_argument("--after-round", type=int, required=True)
    skill_action = activate_skill.add_mutually_exclusive_group(required=True)
    skill_action.add_argument("--updated-skill", type=Path)
    skill_action.add_argument("--no-change", action="store_true")

    build_evidence = subparsers.add_parser("build-evidence")
    build_evidence.add_argument("protocol", type=Path)
    build_evidence.add_argument("output", type=Path)
    build_evidence.add_argument("--round-index", type=int, required=True)
    build_evidence.add_argument("--current-predictions", type=Path, required=True)
    build_evidence.add_argument("--current-column", required=True)
    build_evidence.add_argument("--current-checkpoint", type=Path, required=True)
    build_evidence.add_argument("--m0-predictions", type=Path, required=True)
    build_evidence.add_argument("--m0-column", required=True)
    build_evidence.add_argument("--zero-shot-predictions", type=Path, required=True)
    build_evidence.add_argument("--zero-shot-column", required=True)
    build_evidence.add_argument("--previous-predictions", type=Path)
    build_evidence.add_argument("--previous-column")
    build_evidence.add_argument("--previous-checkpoint", type=Path)
    build_evidence.add_argument("--history", type=Path)
    build_evidence.add_argument("--max-clusters-per-level", type=int, default=10)

    initialize = subparsers.add_parser("init")
    initialize.add_argument("protocol", type=Path)
    initialize.add_argument("run_dir", type=Path)

    status = subparsers.add_parser("status")
    status.add_argument("run_dir", type=Path)

    verify = subparsers.add_parser("verify")
    verify.add_argument("run_dir", type=Path)

    advance = subparsers.add_parser("advance")
    advance.add_argument("run_dir", type=Path)
    advance.add_argument("phase")
    advance.add_argument("--artifact", action="append", default=[])
    return parser


def _round_skill_text(args: argparse.Namespace, round_index: int) -> str:
    if args.skills_dir is not None:
        return SkillMemory(args.skills_dir).bind_round(round_index).read_text()
    return args.skill.read_text("utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "init-skills":
        memory = SkillMemory.initialize(args.skills_dir, args.initial_skill)
        print(memory.resolve(1).model_dump_json())
        return 0
    if args.command == "activate-skill":
        activation = SkillMemory(args.skills_dir).activate(
            load_protocol(args.protocol),
            after_round_index=args.after_round,
            updated_skill=args.updated_skill,
        )
        print(activation.model_dump_json())
        return 0
    if args.command == "compile-ledger":
        summary = compile_intervention_summary(
            load_protocol(args.protocol),
            EvidencePack.model_validate_json(args.evidence.read_text("utf-8")),
            Proposal.model_validate_json(args.proposal.read_text("utf-8")),
            ReflectionPack.model_validate_json(args.reflection_pack.read_text("utf-8")),
            ReflectionV2.model_validate_json(args.reflection.read_text("utf-8")),
        )
        write_immutable_json(args.output, summary.model_dump(mode="json"))
        print(json.dumps({"status": "written", "path": str(args.output.resolve())}))
        return 0
    if args.command == "export-history":
        history = RunStore(args.run_dir).intervention_history()
        write_immutable_json(
            args.output, [item.model_dump(mode="json") for item in history]
        )
        print(
            json.dumps({"episodes": len(history), "path": str(args.output.resolve())})
        )
        return 0
    if args.command == "validate":
        protocol = load_protocol(args.protocol)
        print(json.dumps({"status": "valid", "sha256": canonical_sha256(protocol)}))
        return 0
    if args.command == "verify-evidence":
        protocol = load_protocol(args.protocol)
        pack = EvidencePack.model_validate_json(args.evidence.read_text("utf-8"))
        validate_evidence_pack(pack, protocol)
        build_therapist_request(
            protocol,
            pack,
            system_prompt="EvidencePack privacy validation",
            skill_text="EvidencePack privacy validation",
        )
        print(
            json.dumps(
                {
                    "status": "valid",
                    "file_sha256": sha256_file(args.evidence),
                    "canonical_sha256": canonical_sha256(pack),
                    "error_clusters": len(pack.error_clusters),
                }
            )
        )
        return 0
    if args.command == "propose":
        protocol = load_protocol(args.protocol)
        pack = EvidencePack.model_validate_json(args.evidence.read_text("utf-8"))
        package_root = Path(__file__).resolve().parent
        request_path = args.output.with_name("agent_request.json")
        response_path = args.output.with_name("agent_response.json")
        for path in (args.output, request_path, response_path):
            if path.exists():
                raise FileExistsError(
                    f"refusing paid call with existing output: {path}"
                )
        request = build_therapist_request(
            protocol,
            pack,
            system_prompt=(package_root / "prompts" / "therapist.md").read_text(
                "utf-8"
            ),
            skill_text=_round_skill_text(args, pack.round_index),
        )
        write_immutable_json(request_path, request)
        response = execute_therapist_request(protocol, request)
        write_immutable_json(response_path, serialize_response(response))
        proposal = parse_therapist_response(response, protocol, pack)
        write_immutable_json(args.output, proposal.model_dump(mode="json"))
        print(
            json.dumps(
                {
                    "status": "written",
                    "path": str(args.output.resolve()),
                    "evidence_sha256": proposal.evidence_sha256,
                    "proposal_sha256": canonical_sha256(proposal),
                    "request_path": str(request_path.resolve()),
                    "response_path": str(response_path.resolve()),
                    "stimuli": len(proposal.stimuli),
                    "copies": sum(item.copies for item in proposal.stimuli),
                }
            )
        )
        return 0
    if args.command == "reflect":
        protocol = load_protocol(args.protocol)
        evidence = EvidencePack.model_validate_json(args.evidence.read_text("utf-8"))
        proposal = Proposal.model_validate_json(args.proposal.read_text("utf-8"))
        pack = ReflectionPack.model_validate_json(
            args.reflection_pack.read_text("utf-8")
        )
        package_root = Path(__file__).resolve().parent
        request_path = args.output.with_name("reflection_request.json")
        response_path = args.output.with_name("reflection_response.json")
        for path in (args.output, request_path, response_path):
            if path.exists():
                raise FileExistsError(
                    f"refusing paid call with existing output: {path}"
                )
        request = build_reflection_request(
            protocol,
            evidence,
            proposal,
            pack,
            system_prompt=(package_root / "prompts" / "reflection.md").read_text(
                "utf-8"
            ),
            skill_text=_round_skill_text(args, pack.round_index),
        )
        write_immutable_json(request_path, request)
        response = execute_therapist_request(protocol, request)
        write_immutable_json(response_path, serialize_response(response))
        reflection = parse_reflection_response(response, pack)
        write_immutable_json(args.output, reflection.model_dump(mode="json"))
        print(
            json.dumps(
                {
                    "status": "written",
                    "path": str(args.output.resolve()),
                    "reflection_pack_sha256": (
                        reflection.reflection_pack_canonical_sha256
                    ),
                    "verdict": reflection.hypothesis_verdict,
                    "request_path": str(request_path.resolve()),
                    "response_path": str(response_path.resolve()),
                }
            )
        )
        return 0
    if args.command == "build-evidence":
        protocol = load_protocol(args.protocol)
        history: list[InterventionSummary] = []
        if args.history is not None:
            payload = json.loads(args.history.read_text("utf-8"))
            if not isinstance(payload, list):
                raise TypeError("history must be a JSON array")
            history = [InterventionSummary.model_validate(item) for item in payload]
            for summary in history:
                validate_summary_digest(summary)
        pack = build_evidence_pack_from_files(
            protocol,
            round_index=args.round_index,
            current_predictions=args.current_predictions,
            current_prediction_column=args.current_column,
            current_checkpoint=args.current_checkpoint,
            m0_predictions=args.m0_predictions,
            m0_prediction_column=args.m0_column,
            zero_shot_predictions=args.zero_shot_predictions,
            zero_shot_prediction_column=args.zero_shot_column,
            previous_predictions=args.previous_predictions,
            previous_prediction_column=args.previous_column,
            previous_checkpoint=args.previous_checkpoint,
            intervention_history=history,
            max_clusters_per_level=args.max_clusters_per_level,
        )
        write_immutable_json(args.output, pack.model_dump(mode="json"))
        print(
            json.dumps(
                {
                    "status": "written",
                    "path": str(args.output.resolve()),
                    "sha256": canonical_sha256(pack),
                    "error_clusters": len(pack.error_clusters),
                }
            )
        )
        return 0
    if args.command == "init":
        protocol = load_protocol(args.protocol)
        store = RunStore.initialize(protocol, args.run_dir)
        print(store.current().model_dump_json())
        return 0
    store = RunStore(args.run_dir)
    if args.command == "status":
        print(store.current().model_dump_json())
        return 0
    if args.command == "verify":
        store.verify_integrity()
        print(json.dumps({"status": "valid", "events": len(store.events())}))
        return 0
    if args.command == "advance":
        event = store.advance(args.phase, parse_artifacts(args.artifact))
        print(event.model_dump_json())
        return 0
    raise AssertionError(args.command)
