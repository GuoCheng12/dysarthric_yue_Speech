from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .schema import RunProtocol, canonical_sha256
from .store import RunStore


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


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate":
        protocol = load_protocol(args.protocol)
        print(json.dumps({"status": "valid", "sha256": canonical_sha256(protocol)}))
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
