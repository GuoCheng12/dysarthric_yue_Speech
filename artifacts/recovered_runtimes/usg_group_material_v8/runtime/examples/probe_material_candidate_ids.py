"""Retry an actual failed selection context with per-focus candidate constraints."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from therapist_harness.material_selection_offline import MaterialSelection, build_selector_request, validate_selection
from therapist_harness.real_sampling_mas import _call
from therapist_harness.schema import AgentConfig
from therapist_harness.laicai_credentials import load_laicai_key
from therapist_harness.store import write_immutable_json, sha256_file


def main(args):
    read = lambda p: json.loads(Path(p).read_text())
    protocol = read(args.protocol)
    old_request = read(args.failed_selection / "request.json")
    pack = json.loads(old_request["input"][-1]["content"])["context"]
    index_path = Path(protocol["material_curriculum"]["index_path"])
    index = read(index_path)
    request = build_selector_request(pack)
    assert json.loads(request["input"][-1]["content"]) == json.loads(old_request["input"][-1]["content"])
    args.out.mkdir(parents=True, exist_ok=False)
    files = [args.protocol, args.failed_selection / "request.json", args.failed_selection / "response.json",
             index_path, Path(__file__), REPO / "src/therapist_harness/material_selection_offline.py",
             REPO / "src/therapist_harness/prompts/material_selection_offline.md"]
    sources = {str(p): sha256_file(p) for p in files}
    write_immutable_json(args.out / "protocol.json", dict(failed_context_unchanged=True,
        source_sha256=sources, training_connected=False, per_focus_candidate_ids=True))
    config = AgentConfig.model_validate(protocol["agent"])
    load_laicai_key()
    result, receipt, seconds = _call(config, phase="select", context={}, directory=args.out / "selection",
        output_type=MaterialSelection, request=request, role="Material Selection Agent",
        validate=lambda output: validate_selection(index, pack, output.model_dump()))
    validated = validate_selection(index, pack, result.model_dump())
    record = dict(passed=True, actual_model=config.model, all_selected_ids_in_own_focus=True,
        failed_context_unchanged=True, unique_recordings=validated["unique_recordings"],
        training_decision_replayed=False, actual_api_calls=1, seconds=seconds, source_sha256=sources)
    write_immutable_json(args.out / "result.json", record)
    print(json.dumps({k:v for k,v in record.items() if k != "source_sha256"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-selection", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    main(parser.parse_args())
