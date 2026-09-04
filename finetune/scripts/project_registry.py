#!/usr/bin/env python3
"""Read-only access to project dataset, baseline, and checkpoint registries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit(
        "PyYAML is required. Use the project qwen3-asr virtual environment."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_REGISTRY = REPO_ROOT / "data/registry/asr_dataset_settings_v1.yaml"
ARTIFACT_REGISTRY = REPO_ROOT / "artifacts/registry/asr_baselines_checkpoints_v1.yaml"
DEFAULT_CHECKPOINT_ROOT = Path("/data/qwen3-asr/finetune")
CHECKPOINT_RE = re.compile(r"^checkpoint-\d+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RegistryError(RuntimeError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RegistryError(f"registry not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RegistryError(f"registry root must be a mapping: {path}")
    return data


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_keys(item: dict[str, Any], keys: Iterable[str], context: str) -> list[str]:
    return [f"{context}: missing key '{key}'" for key in keys if key not in item]


def validate_registries(check_paths: bool) -> tuple[list[str], list[str]]:
    datasets = load_yaml(DATASET_REGISTRY)
    artifacts = load_yaml(ARTIFACT_REGISTRY)
    errors: list[str] = []
    notices: list[str] = []

    settings = datasets.get("settings")
    if not isinstance(settings, list):
        errors.append("dataset registry: 'settings' must be a list")
        settings = []

    setting_ids: set[str] = set()
    for index, setting in enumerate(settings):
        context = f"dataset settings[{index}]"
        if not isinstance(setting, dict):
            errors.append(f"{context}: entry must be a mapping")
            continue
        errors.extend(require_keys(setting, ("id", "name", "counts", "paths"), context))
        setting_id = setting.get("id")
        if not isinstance(setting_id, str):
            errors.append(f"{context}: id must be a string")
            continue
        if setting_id in setting_ids:
            errors.append(f"dataset registry: duplicate id '{setting_id}'")
        setting_ids.add(setting_id)

        counts = setting.get("counts", {})
        splits = counts.get("splits", {}) if isinstance(counts, dict) else {}
        if set(splits) != {"train", "dev", "test"}:
            errors.append(f"{setting_id}: splits must be exactly train/dev/test")
        split_total = 0
        for split_name, split in splits.items():
            split_context = f"{setting_id}.{split_name}"
            if not isinstance(split, dict):
                errors.append(f"{split_context}: count entry must be a mapping")
                continue
            errors.extend(
                require_keys(
                    split,
                    ("samples", "speakers", "prompts", "easy", "medium", "hard"),
                    split_context,
                )
            )
            samples = split.get("samples")
            if not isinstance(samples, int) or samples < 0:
                errors.append(
                    f"{split_context}: samples must be a non-negative integer"
                )
                continue
            split_total += samples
            bucket_counts = [split.get(name) for name in ("easy", "medium", "hard")]
            if (
                all(isinstance(value, int) for value in bucket_counts)
                and sum(bucket_counts) != samples
            ):
                errors.append(
                    f"{split_context}: easy+medium+hard does not equal samples"
                )
        if counts.get("total_samples") != split_total:
            errors.append(f"{setting_id}: total_samples does not equal train+dev+test")

        declared_hashes = setting.get("sha256")
        if declared_hashes is not None:
            if not isinstance(declared_hashes, dict):
                errors.append(f"{setting_id}: sha256 must be a mapping")
                declared_hashes = {}
            expected_hash_keys = {"manifest", "train", "dev", "test"}
            if set(declared_hashes) != expected_hash_keys:
                errors.append(
                    f"{setting_id}: sha256 keys must be exactly "
                    f"{sorted(expected_hash_keys)}"
                )
            for artifact_name, digest in declared_hashes.items():
                if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                    errors.append(
                        f"{setting_id}.sha256.{artifact_name}: invalid SHA-256"
                    )

        if check_paths:
            paths = setting.get("paths", {})
            for key in ("public_record", "private_csv_root", "private_jsonl_root"):
                value = paths.get(key)
                if not isinstance(value, str):
                    errors.append(f"{setting_id}.paths: missing string '{key}'")
                    continue
                if not resolve_path(value).exists():
                    errors.append(
                        f"{setting_id}.paths.{key}: missing path {resolve_path(value)}"
                    )
            for kind, root_key in (
                ("private_csv", "private_csv_root"),
                ("private_jsonl", "private_jsonl_root"),
            ):
                root_value = paths.get(root_key)
                files = paths.get(kind, {})
                if isinstance(root_value, str) and isinstance(files, dict):
                    for split_name, filename in files.items():
                        candidate = resolve_path(root_value) / str(filename)
                        if not candidate.is_file():
                            errors.append(
                                f"{setting_id}.{kind}.{split_name}: missing file {candidate}"
                            )
                            continue
                        expected_rows = splits.get(split_name, {}).get("samples")
                        if kind == "private_jsonl":
                            with candidate.open("rb") as handle:
                                actual_rows = sum(1 for line in handle if line.strip())
                        else:
                            with candidate.open(
                                "r", encoding="utf-8-sig", newline=""
                            ) as handle:
                                actual_rows = sum(1 for _ in csv.DictReader(handle))
                        if actual_rows != expected_rows:
                            errors.append(
                                f"{setting_id}.{kind}.{split_name}: row count "
                                f"{actual_rows} != {expected_rows}"
                            )
                        if (
                            kind == "private_jsonl"
                            and isinstance(declared_hashes, dict)
                            and split_name in declared_hashes
                            and sha256_file(candidate) != declared_hashes[split_name]
                        ):
                            errors.append(
                                f"{setting_id}.{kind}.{split_name}: SHA-256 mismatch"
                            )
            if isinstance(declared_hashes, dict) and "manifest" in declared_hashes:
                csv_root = paths.get("private_csv_root")
                if isinstance(csv_root, str):
                    manifest = resolve_path(csv_root) / f"{setting_id}_manifest.csv"
                    if not manifest.is_file():
                        errors.append(f"{setting_id}: missing manifest {manifest}")
                    elif sha256_file(manifest) != declared_hashes["manifest"]:
                        errors.append(f"{setting_id}: manifest SHA-256 mismatch")

    base_models = artifacts.get("base_models", [])
    base_model_ids: set[str] = set()
    for index, model in enumerate(base_models):
        context = f"base_models[{index}]"
        if not isinstance(model, dict):
            errors.append(f"{context}: entry must be a mapping")
            continue
        errors.extend(require_keys(model, ("id", "path"), context))
        model_id = model.get("id")
        if model_id in base_model_ids:
            errors.append(f"artifact registry: duplicate base model id '{model_id}'")
        if isinstance(model_id, str):
            base_model_ids.add(model_id)
        if (
            check_paths
            and isinstance(model.get("path"), str)
            and not resolve_path(model["path"]).is_dir()
        ):
            errors.append(
                f"{context}: missing model path {resolve_path(model['path'])}"
            )
        if check_paths and isinstance(model.get("fingerprint_file"), str):
            fingerprint = resolve_path(model["fingerprint_file"])
            declared = model.get("fingerprint_sha256")
            if not fingerprint.is_file():
                errors.append(f"{context}: missing fingerprint file {fingerprint}")
            elif not isinstance(declared, str) or SHA256_RE.fullmatch(declared) is None:
                errors.append(f"{context}: invalid fingerprint_sha256")
            elif sha256_file(fingerprint) != declared:
                errors.append(f"{context}: fingerprint SHA-256 mismatch")

    signatures = set(artifacts.get("model_signatures", {}).get("any_of", []))
    if not signatures:
        errors.append("artifact registry: model_signatures.any_of must not be empty")

    artifact_entries = artifacts.get("artifacts")
    if not isinstance(artifact_entries, list):
        errors.append("artifact registry: 'artifacts' must be a list")
        artifact_entries = []

    artifact_ids: set[str] = set()
    pinned_paths: set[Path] = set()
    required_metrics = (
        "sample_count",
        "hard_sample_count",
        "overall_cer",
        "hard_cer",
        "critical_error",
        "hard_critical_error",
        "patient_macro_cer",
    )
    for index, artifact in enumerate(artifact_entries):
        context = f"artifacts[{index}]"
        if not isinstance(artifact, dict):
            errors.append(f"{context}: entry must be a mapping")
            continue
        errors.extend(
            require_keys(
                artifact,
                (
                    "id",
                    "display_name",
                    "setting_id",
                    "role",
                    "model_ref",
                    "training",
                    "checkpoints",
                    "evaluation",
                ),
                context,
            )
        )
        artifact_id = artifact.get("id")
        if not isinstance(artifact_id, str):
            errors.append(f"{context}: id must be a string")
            continue
        if artifact_id in artifact_ids:
            errors.append(f"artifact registry: duplicate id '{artifact_id}'")
        artifact_ids.add(artifact_id)
        if artifact.get("setting_id") not in setting_ids:
            errors.append(
                f"{artifact_id}: unknown setting_id '{artifact.get('setting_id')}'"
            )
        eval_setting = artifact.get("evaluation_setting_id")
        if eval_setting is not None and eval_setting not in setting_ids:
            errors.append(
                f"{artifact_id}: unknown evaluation_setting_id '{eval_setting}'"
            )
        if artifact.get("model_ref") not in base_model_ids:
            errors.append(
                f"{artifact_id}: unknown model_ref '{artifact.get('model_ref')}'"
            )

        evaluation = artifact.get("evaluation", {})
        if not isinstance(evaluation, dict):
            errors.append(f"{artifact_id}: evaluation must be a mapping")
        else:
            errors.extend(
                require_keys(evaluation, required_metrics, f"{artifact_id}.evaluation")
            )
            if evaluation.get("patient_macro_cer") is None and not evaluation.get(
                "patient_macro_note"
            ):
                errors.append(
                    f"{artifact_id}: null patient_macro_cer requires patient_macro_note"
                )

        checkpoints = artifact.get("checkpoints", [])
        if not isinstance(checkpoints, list) or not checkpoints:
            errors.append(f"{artifact_id}: checkpoints must be a non-empty list")
            continue
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            cp_context = f"{artifact_id}.checkpoints[{checkpoint_index}]"
            if not isinstance(checkpoint, dict) or not isinstance(
                checkpoint.get("path"), str
            ):
                errors.append(f"{cp_context}: path must be a string")
                continue
            checkpoint_path = resolve_path(checkpoint["path"])
            if checkpoint_path in pinned_paths:
                errors.append(f"{cp_context}: duplicate pinned path {checkpoint_path}")
            pinned_paths.add(checkpoint_path)
            if check_paths:
                if not checkpoint_path.is_dir():
                    errors.append(
                        f"{cp_context}: missing checkpoint directory {checkpoint_path}"
                    )
                else:
                    present = {
                        child.name
                        for child in checkpoint_path.iterdir()
                        if child.is_file()
                    }
                    if signatures.isdisjoint(present):
                        errors.append(
                            f"{cp_context}: no recognized model signature in {checkpoint_path}"
                        )

        if check_paths:
            public_record = artifact.get("public_record")
            if (
                isinstance(public_record, str)
                and not resolve_path(public_record).exists()
            ):
                errors.append(
                    f"{artifact_id}: missing public record {resolve_path(public_record)}"
                )

    generator_entries = artifacts.get("speech_generators")
    if not isinstance(generator_entries, list):
        errors.append("artifact registry: 'speech_generators' must be a list")
        generator_entries = []
    generator_ids: set[str] = set()
    for index, generator in enumerate(generator_entries):
        context = f"speech_generators[{index}]"
        if not isinstance(generator, dict):
            errors.append(f"{context}: entry must be a mapping")
            continue
        errors.extend(
            require_keys(
                generator,
                (
                    "id",
                    "checkpoint",
                    "checkpoint_sha256",
                    "base_model",
                    "setting_id",
                    "reference_manifest",
                    "reference_manifest_sha256",
                    "public_record",
                ),
                context,
            )
        )
        generator_id = generator.get("id")
        if not isinstance(generator_id, str):
            errors.append(f"{context}: id must be a string")
            continue
        if generator_id in generator_ids:
            errors.append(
                f"artifact registry: duplicate speech generator id '{generator_id}'"
            )
        generator_ids.add(generator_id)
        if generator.get("setting_id") not in setting_ids:
            errors.append(
                f"{generator_id}: unknown setting_id '{generator.get('setting_id')}'"
            )
        declared_hash = generator.get("checkpoint_sha256")
        if (
            not isinstance(declared_hash, str)
            or SHA256_RE.fullmatch(declared_hash) is None
        ):
            errors.append(
                f"{generator_id}: checkpoint_sha256 must be lowercase SHA-256"
            )
        if check_paths:
            checkpoint = resolve_path(str(generator.get("checkpoint", "")))
            base_model = resolve_path(str(generator.get("base_model", "")))
            public_record = resolve_path(str(generator.get("public_record", "")))
            reference_manifest = resolve_path(
                str(generator.get("reference_manifest", ""))
            )
            reference_manifest_hash = generator.get("reference_manifest_sha256")
            if not checkpoint.is_file():
                errors.append(f"{generator_id}: missing checkpoint {checkpoint}")
            elif sha256_file(checkpoint) != declared_hash:
                errors.append(f"{generator_id}: checkpoint SHA-256 mismatch")
            if not base_model.is_dir():
                errors.append(f"{generator_id}: missing base model {base_model}")
            if not public_record.is_file():
                errors.append(f"{generator_id}: missing public record {public_record}")
            if not reference_manifest.is_file():
                errors.append(
                    f"{generator_id}: missing reference manifest {reference_manifest}"
                )
            elif (
                not isinstance(reference_manifest_hash, str)
                or SHA256_RE.fullmatch(reference_manifest_hash) is None
            ):
                errors.append(f"{generator_id}: invalid reference_manifest_sha256")
            elif sha256_file(reference_manifest) != reference_manifest_hash:
                errors.append(f"{generator_id}: reference manifest SHA-256 mismatch")

    notices.append(f"settings={len(setting_ids)}")
    notices.append(f"artifacts={len(artifact_ids)}")
    notices.append(f"speech_generators={len(generator_ids)}")
    notices.append(f"pinned_checkpoints={len(pinned_paths)}")
    return errors, notices


def pinned_checkpoint_map(
    artifact_registry: dict[str, Any],
) -> dict[Path, dict[str, str]]:
    pinned: dict[Path, dict[str, str]] = {}
    for artifact in artifact_registry.get("artifacts", []):
        for checkpoint in artifact.get("checkpoints", []):
            path = resolve_path(checkpoint["path"]).resolve()
            pinned[path] = {
                "artifact_id": artifact["id"],
                "setting_id": artifact["setting_id"],
                "role": artifact["role"],
            }
    return pinned


def classify_checkpoint(
    path: Path, pinned: dict[Path, dict[str, str]]
) -> tuple[str, str, str]:
    resolved = path.resolve()
    if resolved in pinned:
        entry = pinned[resolved]
        return entry["setting_id"], "pinned", entry["artifact_id"]

    path_text = str(path).lower()
    if "/private_analysis/" in path_text:
        return "private_eval_copy", "discovered_exploratory", "-"
    if "speaker_prompt_disjoint_v1" in path_text:
        return "setting_b", "discovered_exploratory", "-"
    if "setting_d" in path_text:
        return "setting_d", "discovered_exploratory", "-"
    if "prompt_disjoint_v1" in path_text:
        return "setting_a", "discovered_exploratory", "-"
    if any(token in path_text for token in ("/e2-", "/e3-", "/experiment2/")):
        return "legacy", "discovered_exploratory", "-"
    return "unknown", "discovered_exploratory", "-"


def discover_checkpoints(root: Path, signatures: set[str]) -> list[Path]:
    if not root.is_dir():
        raise RegistryError(f"checkpoint root not found: {root}")
    discovered: list[Path] = []
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        if CHECKPOINT_RE.fullmatch(current_path.name):
            dirs[:] = []
            if not signatures.isdisjoint(files):
                discovered.append(current_path)
    return sorted(discovered)


def command_datasets(_: argparse.Namespace) -> int:
    registry = load_yaml(DATASET_REGISTRY)
    print("id\tname\ttrain\tdev\ttest\tspeaker_disjoint\tprompt_disjoint")
    for setting in registry.get("settings", []):
        splits = setting["counts"]["splits"]
        policy = setting["split_policy"]
        print(
            f"{setting['id']}\t{setting['name']}\t{splits['train']['samples']}\t"
            f"{splits['dev']['samples']}\t{splits['test']['samples']}\t"
            f"{str(policy['speaker_disjoint']).lower()}\t{str(policy['prompt_disjoint']).lower()}"
        )
    return 0


def command_baselines(args: argparse.Namespace) -> int:
    registry = load_yaml(ARTIFACT_REGISTRY)
    print("id\tsetting\trole\tmethod\tseeds\tcheckpoint_count\tpublic_record")
    for artifact in registry.get("artifacts", []):
        if args.setting and artifact.get("setting_id") != args.setting:
            continue
        seeds = (
            ",".join(str(seed) for seed in artifact["training"].get("seeds", []))
            or "not_reported"
        )
        print(
            f"{artifact['id']}\t{artifact['setting_id']}\t{artifact['role']}\t"
            f"{artifact['training'].get('method', '-')}\t{seeds}\t{len(artifact['checkpoints'])}\t"
            f"{artifact.get('public_record', '-')}"
        )
    return 0


def command_generators(_: argparse.Namespace) -> int:
    registry = load_yaml(ARTIFACT_REGISTRY)
    print("id\tsetting\trole\tselected_epoch\tcheckpoint")
    for generator in registry.get("speech_generators", []):
        print(
            f"{generator['id']}\t{generator['setting_id']}\t{generator['role']}\t"
            f"{generator.get('selected_epoch', '-')}\t{generator['checkpoint']}"
        )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    errors, notices = validate_registries(args.check_paths)
    if errors:
        for error in errors:
            print(f"ERROR\t{error}", file=sys.stderr)
        print(f"validation_failed\terrors={len(errors)}", file=sys.stderr)
        return 1
    mode = "schema_and_paths" if args.check_paths else "schema"
    print(f"validation_ok\tmode={mode}\t" + "\t".join(notices))
    return 0


def command_scan_checkpoints(args: argparse.Namespace) -> int:
    registry = load_yaml(ARTIFACT_REGISTRY)
    signatures = set(registry["model_signatures"]["any_of"])
    pinned = pinned_checkpoint_map(registry)
    rows: list[dict[str, str]] = []
    for path in discover_checkpoints(Path(args.root), signatures):
        classification, status, artifact_id = classify_checkpoint(path, pinned)
        if (
            classification == "private_eval_copy"
            and not args.include_private_eval_copies
        ):
            continue
        rows.append(
            {
                "classification": classification,
                "status": status,
                "artifact_id": artifact_id,
                "checkpoint": path.name,
                "path": str(path),
            }
        )
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print("classification\tstatus\tartifact_id\tcheckpoint\tpath")
        for row in rows:
            print(
                "\t".join(
                    row[key]
                    for key in (
                        "classification",
                        "status",
                        "artifact_id",
                        "checkpoint",
                        "path",
                    )
                )
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    datasets_parser = subparsers.add_parser(
        "datasets", help="List retained dataset settings"
    )
    datasets_parser.set_defaults(func=command_datasets)

    baselines_parser = subparsers.add_parser(
        "baselines", help="List pinned baselines and candidates"
    )
    baselines_parser.add_argument(
        "--setting", choices=("setting_a", "setting_b", "setting_c", "setting_d")
    )
    baselines_parser.set_defaults(func=command_baselines)

    generators_parser = subparsers.add_parser(
        "generators", help="List frozen speech generators"
    )
    generators_parser.set_defaults(func=command_generators)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate registry schema and references"
    )
    validate_parser.add_argument(
        "--check-paths", action="store_true", help="Also verify local private paths"
    )
    validate_parser.set_defaults(func=command_validate)

    scan_parser = subparsers.add_parser(
        "scan-checkpoints", help="Discover model-bearing checkpoint directories"
    )
    scan_parser.add_argument("--root", default=str(DEFAULT_CHECKPOINT_ROOT))
    scan_parser.add_argument("--format", choices=("tsv", "json"), default="tsv")
    scan_parser.add_argument("--include-private-eval-copies", action="store_true")
    scan_parser.set_defaults(func=command_scan_checkpoints)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except RegistryError as exc:
        print(f"ERROR\t{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
