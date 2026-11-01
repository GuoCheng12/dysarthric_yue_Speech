"""Launch one content-centered Dynamic run with the existing continuous trainer."""

import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import real_curriculum_v2 as runner
from prepare_content_curriculum import build_request
from therapist_harness.agent import execute_therapist_request, serialize_response
from therapist_harness.content_curriculum import (ContentPrescription, ContentReflection,
    compile_lesson_block, delivery_report, enrich_catalog, evidence_pack, lesson_outcome,
    validate_lesson_reflection)
from therapist_harness.schema import AgentConfig


def call_agent(root, destination, payload, schema, kind):
    config = AgentConfig.model_validate(runner.read(root / "protocol.json")["agent"])
    assert config.max_retries == 0
    request = build_request(config.model_dump(mode="json"), payload, schema, kind)
    runner.save(destination / "request.json", request)
    started = time.monotonic()
    response = execute_therapist_request(SimpleNamespace(agent=config), request)
    runner.save(destination / "response.json", serialize_response(response))
    runner.save(destination / "timing.json", {"seconds": time.monotonic()-started,
        "usage": response.usage.model_dump(mode="json") if response.usage else None})
    assert response.status == "completed", response.status
    parsed = schema.model_validate_json(response.output_text)
    runner.save(destination / "parsed.json", parsed.model_dump(mode="json"))
    return parsed


class ContentCourse:
    def __init__(self, root, destination, mapping, coverage):
        self.root, self.destination = root, destination
        self.mapping, self.coverage = mapping, coverage
        self.proposal, self.delivery = None, None
        self.history = []

    def compile(self, end, entries, previous_manifests):
        # Enrich from this run's current predictions, never from smoke artifacts.
        contents = {r["content"]: r["clean_gt"] for r in self.mapping.values()}
        enriched = enrich_catalog({"recordings": entries, "contents": contents})
        manifest = compile_lesson_block(self.proposal, enriched, self.coverage[end], end, previous_manifests)
        self.delivery = delivery_report(self.proposal, enriched, self.coverage[end], manifest)
        runner.save(self.destination / f"delivery_{end}.json", self.delivery)
        return manifest

    def advance(self, step, catalog, manifest, before, after, feedback_before,
                feedback_after, feedback_metrics, publish):
        if self.proposal is not None:
            assert manifest[-1]["step"] == step
            outcome = lesson_outcome(step, self.proposal, self.delivery, manifest,
                self.mapping, before, after, feedback_before, feedback_after)
            runner.save(self.destination / f"outcome_{step}.json", outcome)
            publish("reflecting", step)
            reflection = call_agent(self.root, self.destination / f"reflect_{step}",
                {"task": "Reflect on this completed real-speech lesson block.",
                 "outcome": outcome, "history": self.history}, ContentReflection, "reflect")
            validate_lesson_reflection(reflection, outcome)
            episode = {"outcome": outcome, "reflection": reflection.model_dump(mode="json")}
            self.history.append(episode)
            runner.save(self.destination / f"episode_{step}.json", episode)
        if step < 100:
            pack, _ = evidence_pack(catalog, self.coverage[step+20], step, feedback_metrics, self.history)
            pack["task"] = "Prescribe the NEXT actual training block, using this run's current checkpoint and completed lesson history."
            runner.save(self.destination / f"content_evidence_{step}.json", pack)
            publish("prescribing", step)
            self.proposal = call_agent(self.root, self.destination / f"prescribe_{step}",
                pack, ContentPrescription, "prescribe")
            runner.save(self.destination / f"active_course_{step}.json", self.proposal.model_dump(mode="json"))


def report(root):
    result = runner.read(root / "dynamic/result.json")
    assert [r["step"] for r in result["history"]] == list(runner.ENDPOINTS)
    assert result["unique_train_recordings"] == 2117 and result["exposures"] == 3200
    result["best"] = min(result["history"], key=lambda r: r["feedback"]["overall"]["after"]["mean_utterance_cer"])
    runner.save(root / "results.json", result)
    lines = ["# Content-centered Dynamic results", "",
        "Setting D: 2117 real train / 511 merged feedback; own checkpoint decoded at every endpoint.", "",
        "|Step|Overall mean CER|Easy|Medium|Hard|Patient macro|Pooled CER|",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["history"]:
        feedback = row["feedback"]
        values = [feedback[b]["after"]["mean_utterance_cer"] for b in ("overall", "easy", "medium", "hard")]
        values += [feedback["overall"]["after"][key] for key in ("patient_macro_cer", "pooled_cer")]
        lines.append(f"|{row['step']}|" + "|".join(f"{v*100:.4f}%" for v in values) + "|")
    lines += ["", f"Lowest mean-utterance CER: step {result['best']['step']}. All five adapters retained."]
    runner.write_immutable_bytes(root / "report.md", ("\n".join(lines)+"\n").encode())


def worker(root):
    runner.save(root / "started.json", {"status": "authorized_content_dynamic_started",
        "pid": os.getpid(), "session_id": os.getsid(0), "started_unix": time.time(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]})
    try:
        runner.train_arm(root, "dynamic")
        report(root)
    except Exception as error:
        traceback.print_exc()
        runner.save(root / "background_exit.json", {"status": "failed", "exit_code": 1,
            "exception_type": type(error).__name__, "finished_unix": time.time()})
        return 1
    (root / "dynamic/status.json").write_text(json.dumps({"status": "complete", "step": 100})+"\n")
    runner.save(root / "background_exit.json", {"status": "complete", "exit_code": 0,
        "finished_unix": time.time()})
    return 0


def launch(root, gpu):
    cfg = runner.verify_frozen(root)
    assert cfg["course_design"] == "content_centered" and cfg["arms_execution_order"] == ["dynamic"]
    assert runner.read(root / "readiness.json")["checks_passed"]
    assert not (root / "started.json").exists() and not (root / "background_launch.json").exists()
    env = dict(os.environ)
    if not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = getpass.getpass("Gateway API key (memory only): ")
    assert env["OPENAI_API_KEY"]
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=f"{runner.REPO}:{runner.REPO / 'src'}", TOKENIZERS_PARALLELISM="false")
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--root", str(root)]
    with (root / "dynamic.log").open("x") as log:
        process = subprocess.Popen(command, cwd=runner.REPO, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    receipt = {"pid": process.pid, "gpu": gpu, "command": command,
        "start_new_session": True, "credentials_written": False, "started_unix": time.time(),
        "protocol_sha256": runner.sha256_file(root / "protocol.json")}
    runner.save(root / "background_launch.json", receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "launch", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    os.umask(0o077)
    root = args.root.resolve()
    if args.stage == "prepare":
        runner.prepare(root, content_centered=True)
    elif args.stage == "launch":
        launch(root, args.gpu)
    else:
        return worker(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
