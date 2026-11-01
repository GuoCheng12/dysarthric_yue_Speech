#!/usr/bin/env python3
"""Explicit stages for the MAS→SG→audio review→ASR preferences→DPO loop.

prepare never generates or trains. smoke runs one tiny intervention and builds
the next real-feedback proposal. Default transcript review reuses the MAS API.
"""

import argparse
import getpass
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from therapist_harness import preference_loop as loop
from therapist_harness.store import RunStore


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "migrate", "real-branch", "assess", "prescribe", "generate", "sample", "train", "evaluate", "reassess", "reflect", "run", "smoke", "status"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--round", type=int, help="defaults to the journal's current/next round")
    parser.add_argument("--arm", choices=(*loop.ARMS, "all"), default="all")
    parser.add_argument("--dpo-only", action="store_true", help="prepare/smoke: freeze only R-DPO and R+MAS-DPO branches")
    parser.add_argument("--synthetic-only", action="store_true", help="prepare/smoke: R-only starting model, only MAS TTS preferences; no real sampling or replay")
    parser.add_argument("--dual-gpu", action="store_true", help="prepare/smoke: persistent review ASR on GPU 0, real branch on GPU 1")
    parser.add_argument("--from-root", type=Path, help="migrate: stopped immutable v1 execution")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--parent-predictions", type=Path)
    parser.add_argument("--review-mode", choices=("asr_transcript", "audio"), default="asr_transcript",
                        help="Default: fixed R-only transcript plus target/reference texts, reviewed by the MAS text API.")
    parser.add_argument("--audio-review-model")
    parser.add_argument("--audio-review-base-url")
    parser.add_argument("--audio-review-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--synthetic-count", type=int, default=96)
    parser.add_argument("--unique-texts", type=int, default=6)
    parser.add_argument("--real-count", type=int, default=2117)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--real-per-step", type=int, default=32)
    parser.add_argument("--synthetic-per-step", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--generation-design", type=Path)
    parser.add_argument('--retention-design', type=Path, help='prepare/smoke: explicit real-KL + edit/EOS residual DPO execution v3')
    parser.add_argument("--checkpoint-acceptance", choices=['always','real_dev_v1','real_dev_initial_parent_v2'], default='always')
    parser.add_argument("--include-greedy", action='store_true')
    parser.add_argument("--review-allow-pending", action='store_true')
    parser.add_argument("--training-allocation", choices=['uniform','role_budget_v1','mas_audio_budget_v1'], default='uniform')
    args = parser.parse_args(argv)
    if args.retention_design:
        if args.stage not in {'prepare', 'smoke'} or args.dpo_only or args.dual_gpu:
            parser.error('--retention-design is frozen at prepare/smoke and uses the explicit MAS KL/DPO branch')
        args.synthetic_only = True
        args.include_greedy = True
        args.training_allocation = 'mas_audio_budget_v1'
        args.checkpoint_acceptance = 'real_dev_initial_parent_v2'
    if args.dpo_only and args.stage not in {"prepare", "smoke"}:
        parser.error("--dpo-only is frozen at prepare/smoke; run uses the saved protocol")
    if args.synthetic_only and (args.stage not in {"prepare", "smoke"} or args.dpo_only or args.dual_gpu):
        parser.error("--synthetic-only is frozen at prepare/smoke and cannot be combined with --dpo-only/--dual-gpu")
    if args.dual_gpu and args.stage not in {"prepare", "smoke"}:
        parser.error("--dual-gpu is frozen at prepare/smoke")
    if args.stage == "migrate" and args.from_root is None:
        parser.error("migrate requires --from-root")
    if args.stage in {"prepare", "smoke"} and args.review_mode == "audio" and (not args.audio_review_model or not args.audio_review_base_url):
        parser.error("prepare/smoke require the configured audio-review model and base URL")
    return args


def preflight(root):
    p = RunStore(root).protocol()
    required = sorted({p.agent.api_key_env, p.preference_loop.audio_review_api_key_env})
    missing = [key for key in required if not os.environ.get(key)]
    if missing and sys.stdin.isatty():
        for key in missing:
            value = getpass.getpass(f"{key} (memory only, not saved): ")
            if value:
                os.environ[key] = value
        missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise RuntimeError("configure credential environment variables before live execution: " + ", ".join(missing))


def finish_update(root, number):
    for arm in loop.enabled_arms(RunStore(root).protocol()):
        d = loop.round_dir(root, number) / arm
        if not (d / "training/training_result.json").exists() and not (d / "no_update.json").exists():
            loop.train_branch(root, number, arm)
        if not loop.predictions(root, number, arm).exists():
            loop.evaluate_branch(root, number, arm)


def run_round(root, number):
    store = RunStore(root)
    if store.protocol().preference_loop.execution_version == "persistent_dual_gpu_v2":
        from therapist_harness.preference_execution_v2 import run_parallel_round
        return run_parallel_round(root, number)
    phase = store.current().phase
    expected_round = 1 if phase == "initialized" else store.current().round_index + (phase == "committed")
    if number != expected_round:
        raise ValueError(f"journal expects round {expected_round}, received {number}")
    if phase in {"initialized", "committed"}:
        loop.assess(root, number)
    if store.current().phase in {"assessed", "proposed"}:
        loop.prescribe(root, number)
    if store.current().phase == "materialized":
        if not (loop.round_dir(root, number) / "synthesis_receipt.json").exists():
            loop.generate(root, number)
        if not loop.sample_and_pair(root, number):
            # Every admitted audio can still contribute to the SFT controls.
            # The DPO loop honestly terminates here; no negative or update is invented.
            finish_update(root, number)
            return False
    if store.current().phase == "synthesized":
        finish_update(root, number)
        loop.reassess(root, number)
    if store.current().phase == "trained":
        loop.reassess(root, number)
    if store.current().phase in {"reassessed", "reflected"}:
        loop.reflect(root, number)
    if store.current().phase != "committed":
        raise RuntimeError(f"round did not reach committed: {store.current().phase}")
    return True


def main(argv=None):
    args = parse_args(argv)
    root = args.root.resolve()
    if args.stage in {"migrate", "real-branch"}:
        from therapist_harness.preference_execution_v2 import migrate, real_branch
        result = migrate(args.from_root.resolve(), root) if args.stage == "migrate" else real_branch(root, args.round or 1)
        print(json.dumps({"status": "migrated" if args.stage == "migrate" else "real_branch_complete", "root": str(root)}))
        return
    if args.stage in {"prepare", "smoke"}:
        tiny = args.stage == "smoke"
        if args.review_mode == "asr_transcript":
            defaults = loop.load_protocol(REPO / "configs/therapist_setting_d_v1.yaml").agent
            args.audio_review_model = args.audio_review_model or defaults.model
            args.audio_review_base_url = args.audio_review_base_url or str(defaults.base_url)
        loop.prepare(root, parent=args.parent, parent_predictions=args.parent_predictions,
            review_model=args.audio_review_model, review_base_url=args.audio_review_base_url,
            review_key_env=args.audio_review_api_key_env, rounds_count=2 if tiny else args.rounds,
            synthetic_count=6 if tiny else args.synthetic_count, unique_texts=6 if tiny else args.unique_texts,
            real_count=8 if tiny else args.real_count, steps=1 if tiny else args.steps,
            real_per_step=2 if tiny else args.real_per_step, synthetic_per_step=2 if tiny else args.synthetic_per_step,
            seed=args.seed, learning_rate=args.lr, review_mode=args.review_mode,
            comparison="synthetic_only" if args.synthetic_only else ("dpo_only" if args.dpo_only or args.dual_gpu else "four_arms"),
            execution_version='persistent_kl_dpo_v3' if args.retention_design else ("persistent_synthetic_only_v1" if args.synthetic_only else ("persistent_dual_gpu_v2" if args.dual_gpu else "serial_v1")),
            generation_design_path=args.generation_design,checkpoint_acceptance=args.checkpoint_acceptance,
            include_greedy=args.include_greedy,review_allow_pending=args.review_allow_pending,
            training_allocation=args.training_allocation, retention_design_path=args.retention_design)
        if not tiny:
            print(json.dumps({"status": "prepared_only", "root": str(root)})); return
    if args.stage in {"run", "smoke"}:
        p = RunStore(root).protocol()
        last = 1 if args.stage == "smoke" else p.round_count
        current = RunStore(root).current()
        expected_start = 1 if current.phase == "initialized" else current.round_index + (current.phase == "committed")
        if args.round is not None and args.round != expected_start:
            raise ValueError(f"journal expects round {expected_start}, received {args.round}")
        start = expected_start
        if start <= last:
            preflight(root)
        for number in range(start, last + 1):
            if not run_round(root, number):
                print(json.dumps({"status": "stopped_no_preference_pairs", "round": number, "root": str(root)})); return
        if args.stage == "smoke":
            loop.assess(root, 2)
            loop.prescribe(root, 2)
            branch_status = {arm: "no_pairs_no_update" if (loop.round_dir(root, 1) / arm / "no_update.json").exists() else "updated" for arm in loop.enabled_arms(p)}
            loop.save(root / "smoke_result.json", {"status": "complete_live_loop_smoke", "round_1": "enabled branches evaluated; eligible branches updated; actual reflection committed", "branch_updates": branch_status,
                "round_2": "new proposal materialized from real updated evidence and history; no second update",
                "formal_experiment_started": False})
        print(json.dumps({"status": "complete_live_loop_smoke" if args.stage == "smoke" else "complete_fixed_rounds", "root": str(root)})); return
    if args.stage == "status":
        print(RunStore(root).current().model_dump_json()); return
    current = RunStore(root).current()
    default_round = max(1, current.round_index + (current.phase == "committed" and args.stage in {"assess", "prescribe"}))
    selected_round = args.round if args.round is not None else default_round
    if args.stage in {"train", "evaluate"}:
        action = loop.train_branch if args.stage == "train" else loop.evaluate_branch
        for arm in loop.enabled_arms(RunStore(root).protocol()) if args.arm == "all" else (args.arm,):
            action(root, selected_round, arm)
        return
    stages = {"assess": loop.assess, "prescribe": loop.prescribe, "generate": loop.generate,
              "sample": loop.sample_and_pair, "reassess": loop.reassess, "reflect": loop.reflect}
    stages[args.stage](root, selected_round)


if __name__ == "__main__":
    main()
