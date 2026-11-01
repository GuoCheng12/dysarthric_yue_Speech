"""Explicit recovery of the recorded round-2 citation failure; no model-output edits."""

import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys

import therapist_crossed_dose_loop as loop


def prepare(root):
    revision = root / "revisions/citation_v2"
    store = loop.RunStore(root)
    assert (store.current().round_index, store.current().phase) == (2, "assessed")
    old_sources = loop.read(root / "sources.json")
    changed = {
        str(loop.PACKAGE / "schema.py"): revision / "before/therapist_harness/schema.py",
        str(Path(loop.__file__)): revision / "before/therapist_crossed_dose_loop.py",
    }
    for path, digest in old_sources.items():
        original = changed[path] if path in changed else Path(path)
        assert loop.sha256_file(original) == digest, path
    loop.save(revision / "sources.json", {str(p): loop.sha256_file(p)
        for p in [*[Path(p) for p in old_sources], Path(__file__)]})
    prompt = (root / "prompts/therapist.md").read_text()
    assert prompt.count("`intervention.<ledger_id>`") == 2
    prompt = prompt.replace("`intervention.<ledger_id>`",
        "the exact supplied `ledger_id` (e.g. `intervention.round_001`), without adding any prefix")
    loop.write_immutable_bytes(revision / "therapist.md", prompt.encode())
    untouched = [root / "protocol.json", root / "sources.json",
        root / "round_002/prescribe/request.json", root / "round_002/prescribe/response.json",
        root / "round_001/ledger_entry.json"]
    loop.save(revision / "authorization.json", {
        "reason": "User approved correction of duplicated intervention namespace and continuation",
        "resume_round": 2, "reuse_original_agent_response": True,
        "training_data_budget_and_hyperparameters_changed": False,
        "unchanged_artifacts": {str(p): loop.sha256_file(p) for p in untouched},
        "old_sources_sha256": loop.sha256_file(root / "sources.json"),
        "new_sources_sha256": loop.sha256_file(revision / "sources.json"),
        "new_round3_prompt_sha256": loop.sha256_file(revision / "therapist.md"),
    })
    print("[resume] revision recorded; original artifacts preserved", flush=True)


def run(root):
    revision = root / "revisions/citation_v2"
    for path, digest in loop.read(revision / "sources.json").items():
        assert loop.sha256_file(Path(path)) == digest, path
    for path, digest in loop.read(revision / "authorization.json")["unchanged_artifacts"].items():
        assert loop.sha256_file(Path(path)) == digest, path
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Gateway API key (not saved): ")
    for number in (2, 3):
        prompt = root / "prompts/therapist.md" if number == 2 else revision / "therapist.md"
        if number == 3:
            loop.assess(root, number)
        print(f"[resume] round={number} prescribe saved_response={number == 2}", flush=True)
        loop.prescribe(root, number, prompt_path=prompt, saved_response=number == 2)
        for stage in (loop.synthesize, loop.pair):
            print(f"[resume] round={number} phase={stage.__name__}", flush=True)
            stage(root, number)
        for arm in ("real_only", "mas"):
            print(f"[resume] round={number} train={arm}", flush=True)
            with (loop.directory(root, number) / f"{arm}.log").open("x") as log:
                subprocess.run([sys.executable, str(Path(loop.__file__)), str(root),
                    "--stage", "train", "--round", str(number), "--arm", arm,
                    "--source-manifest", str(revision / "sources.json")],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        loop.reassess(root, number, prescription_prompt=prompt)
        print(f"[resume] round={number} reflect", flush=True)
        loop.reflect(root, number)
    for path, digest in loop.read(revision / "authorization.json")["unchanged_artifacts"].items():
        assert loop.sha256_file(Path(path)) == digest, path
    loop.save(root / "run_summary.json", {"status": "three_rounds_committed", "rounds": [
        {"round": n, "arms": {a: loop.read(loop.directory(root,n)/a/"run_summary.json")
            for a in ("real_only", "mas")}, "ledger": loop.read(loop.directory(root,n)/"ledger_entry.json")}
        for n in (1,2,3)], "execution_revision": str(revision / "authorization.json")})
    print("[resume] all three rounds committed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    (prepare if args.prepare else run)(args.root.resolve())
