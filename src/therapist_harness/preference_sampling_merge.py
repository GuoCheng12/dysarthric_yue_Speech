"""Join real/synthetic sampling receipts without inventing a single sampling job."""

import json
from pathlib import Path

from .preference_data import build_preference_pairs, sample_seed, validate_sampling_inputs
from .store import sha256_file, write_immutable_json, write_immutable_bytes


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def jsonl(path, items):
    write_immutable_bytes(Path(path), ''.join(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n' for x in items).encode())


def expected_decode(protocol, parent_hash):
    c = protocol.preference_loop
    return {"current_asr_sha256": parent_hash, "base_model": protocol.base_model_dir,
            "base_model_sha256": c.review_asr_base_model_sha256,
            "samples_per_audio": c.samples_per_audio, "seed": protocol.lora.seed,
            "temperature": c.temperature, "top_p": c.top_p, "top_k": 0,
            "max_new_tokens": c.max_new_tokens, "language": "Cantonese", "prompt": "", "do_sample": True,
            "real_train_sha256": protocol.real_train.sha256, "feedback_sha256": protocol.feedback_split.sha256}


def shard_payload(directory, source, protocol, parent_hash):
    directory = Path(directory)
    paths = {name: directory / name for name in (
        "sampling_receipt.json", "sampling_config.json", "validated_inputs.jsonl", "candidates.jsonl", "preferences.jsonl")}
    receipt, config = read(paths["sampling_receipt.json"]), read(paths["sampling_config.json"])
    if receipt.get("status") != "complete" or receipt.get("parent_files_unchanged") is not True:
        raise ValueError("sampling shard did not finish on unchanged parent files")
    expected = expected_decode(protocol, parent_hash)
    if any(receipt.get(k) != v or config.get(k) != v for k, v in expected.items()):
        raise ValueError("sampling shard parent/model/decoding differs from the frozen round")
    if any(receipt.get(k) != v for k, v in config.items()):
        raise ValueError("sampling receipt differs from its source config")
    for field, name in (("candidates_sha256", "candidates.jsonl"), ("validated_inputs_sha256", "validated_inputs.jsonl")):
        if receipt.get(field) != sha256_file(paths[name]):
            raise ValueError("sampling shard file hash mismatch")
    inputs, candidates = rows(paths["validated_inputs.jsonl"]), rows(paths["candidates.jsonl"])
    if any(x["source"] != source for x in inputs):
        raise ValueError("sampling shard contains a different source")
    if receipt.get("candidate_count") != len(candidates):
        raise ValueError("sampling shard candidate count mismatch")
    if any(x.get("seed") != sample_seed(protocol.lora.seed, x["audio_id"], x["sample_index"]) for x in candidates):
        raise ValueError("sample random seed differs from frozen audio/sample identity")
    pairs = build_preference_pairs(inputs, candidates, parent_hash,
        expected_samples=protocol.preference_loop.samples_per_audio)[0] if inputs else []
    if not inputs and candidates:
        raise ValueError("empty sampling shard cannot contain candidates")
    if pairs != rows(paths["preferences.jsonl"]):
        raise ValueError("shard preference pairs differ from actual sampled errors")
    provenance = {"source": source, "files": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                                              for name, path in paths.items()}}
    return inputs, candidates, provenance, receipt


def validated_shards(shards, protocol, parent_hash, expected_inputs):
    inputs, candidates, provenance, receipts = [], [], [], []
    for source in ("real", "synthetic"):
        part_inputs, part_candidates, part_provenance, receipt = shard_payload(shards[source], source, protocol, parent_hash)
        inputs.extend(part_inputs); candidates.extend(part_candidates)
        provenance.append(part_provenance); receipts.append(receipt)
    # The exact same sampler implementation and base/parent files must produce both shards.
    for key in ("current_adapter", "source_sha256", "preference_code_sha256"):
        if receipts[0].get(key) != receipts[1].get(key):
            raise ValueError("sampling shards used different parent paths or code")
    if inputs != expected_inputs or len({x["audio_id"] for x in inputs}) != len(inputs):
        raise ValueError("merged shard sample identities/order differ from admitted round inputs")
    pairs, audit = build_preference_pairs(inputs, candidates, parent_hash,
        expected_samples=protocol.preference_loop.samples_per_audio)
    return inputs, candidates, pairs, audit, provenance


def merge_shards(root, number):
    from . import preference_loop as loop
    from .store import RunStore
    root = Path(root); p = RunStore(root).protocol(); d = loop.round_dir(root, number)
    parent_hash = loop.sha256_directory(loop.checkpoint(root, number - 1))
    inputs = validate_sampling_inputs(loop.rows(d / "sampling_inputs.jsonl"), loop.rows(p.real_train.path),
        loop.rows(p.feedback_split.path), loop.rows(d / "generation/reviews.jsonl"))
    inputs, candidates, pairs, audit, provenance = validated_shards(
        {source: d / f"sampling_{source}" for source in ("real", "synthetic")}, p, parent_hash, inputs)
    output = d / "sampling"; output.mkdir(exist_ok=False)
    for name, values in (("validated_inputs.jsonl", inputs), ("candidates.jsonl", candidates), ("preferences.jsonl", pairs)):
        jsonl(output / name, values)
    receipt = {**expected_decode(p, parent_hash), "schema_version": "current_asr_sampling_merged_v2", "status": "complete",
        "sampling_scope": "merge of separately executed real and synthetic sampling jobs; no single-job claim",
        "source_shards": provenance, "candidate_count": len(candidates),
        "validated_inputs_sha256": sha256_file(output / "validated_inputs.jsonl"),
        "candidates_sha256": sha256_file(output / "candidates.jsonl"),
        "preferences_sha256": sha256_file(output / "preferences.jsonl")}
    write_immutable_json(output / "sampling_receipt.json", receipt)
    write_immutable_json(output / "preference_audit.json", audit)
    validate_merged_receipt(receipt, p, parent_hash, inputs, candidates)
    return audit


def validate_merged_receipt(receipt, protocol, parent_hash, inputs, candidates):
    if receipt.get("schema_version") != "current_asr_sampling_merged_v2":
        raise ValueError("not a merged sampling receipt")
    if (receipt.get("status") != "complete" or receipt.get("candidate_count") != len(candidates)
            or any(receipt.get(k) != v for k, v in expected_decode(protocol, parent_hash).items())):
        raise ValueError("merged sampling metadata differs from the frozen round")
    entries = receipt.get("source_shards", [])
    if [x.get("source") for x in entries] != ["real", "synthetic"]:
        raise ValueError("merged receipt must preserve both source sampling jobs")
    directories = {}
    for entry in entries:
        files = entry["files"]
        for reference in files.values():
            if sha256_file(Path(reference["path"])) != reference["sha256"]:
                raise ValueError("merged source shard changed")
        directory = Path(files["sampling_receipt.json"]["path"]).parent
        if any(Path(ref["path"]) != directory / name for name, ref in files.items()):
            raise ValueError("merged source files must come from their recorded sampling job")
        directories[entry["source"]] = directory
    _, rebuilt, _, _, provenance = validated_shards(directories, protocol, parent_hash, inputs)
    if candidates != rebuilt or provenance != entries:
        raise ValueError("merged candidates differ from their original sampling jobs")
