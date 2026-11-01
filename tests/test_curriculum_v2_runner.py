from pathlib import Path
import sys

import accelerate.utils.other as accelerate_other
import pytest
import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, default_data_collator, set_seed

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))
import real_curriculum_v2 as runner
from therapist_harness.curriculum_v2 import CourseProposal, CourseReflection, make_coverage
from finetune.scripts import qwen3_asr_lora_sft as training


class ToyPatientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 7)
        self.dropout = nn.Dropout(.1)

    def forward(self, x, **kwargs):
        return {"logits": self.proj(self.dropout(x))}


def test_three_arm_runner_reaches100_and_uses_dynamic_feedback(tmp_path, monkeypatch):
    """Different states/results; Random must finish before any Static files exist."""
    torch.set_num_threads(1)
    monkeypatch.setattr(accelerate_other, "is_deepspeed_available", lambda: False)
    mapping = {f"R{i+1:04d}": {"utt_id": f"private/{i}", "speaker_id": f"private_{i//33}",
                 "patient": f"P{i//33+1:03d}", "content": f"C{i%33+1:02d}", "duration": 2,
                 "clean_gt": "今日食飯", "prediction": "今日食", "zero_shot_bucket": ("easy", "medium", "hard")[i%3]}
               for i in range(2117)}
    # Different GT contents make the real catalog consistent with unit selectors.
    for i, r in enumerate(mapping.values()):
        r["clean_gt"] = "今日食飯" + chr(0x4e00+i%33)
        r["prediction"] = "今日食飯"
    # Catalog sorts by private utt_id; use lexical-padded IDs for stable Rxxxx.
    for i, r in enumerate(mapping.values()):
        r["utt_id"] = f"private/{i:04d}"
        r["speaker_id"] = f"private_{i//33:03d}"
    train = list(mapping.values())
    feedback = [{**r, "utt_id": "fb/"+r["utt_id"]} for r in train[:60]]
    runner.save(tmp_path / "local_map.json", mapping)
    runner.save(tmp_path / "zero_train.json", {r["utt_id"]: {**r, "raw_pred": r["prediction"]} for r in train})
    runner.save(tmp_path / "coverage.json", make_coverage(list(mapping)))
    runner.save_rows(tmp_path / "zero_train_predictions.jsonl", train)
    runner.save_rows(tmp_path / "zero_feedback_predictions.jsonl", feedback)
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})

    decoded = []

    def fake_decode(checkpoint, source, destination):
        arm = checkpoint.parent.parent.name
        step = int(checkpoint.name.split("-")[-1])
        assert destination.parent.parent.name == arm
        decoded.append((arm, step, source))
        result = train if source == runner.TRAIN else feedback
        if source == runner.FEEDBACK:
            text = {"static": "今日食飯", "random": "今日食", "dynamic": "今日"}[arm]
            result = [{**r, "prediction": text} for r in result]
        runner.save_rows(destination / "predictions.jsonl", result)
        return result
    monkeypatch.setattr(runner, "decode", fake_decode)
    from therapist_harness import curriculum_v2_training
    measured = []
    actual_measure = curriculum_v2_training.measure_losses

    def traced_measure(trainer, features):
        measured.append((Path(trainer.args.output_dir).parent.name, trainer.state.global_step))
        return actual_measure(trainer, features)

    monkeypatch.setattr(curriculum_v2_training, "measure_losses", traced_measure)
    calls = []

    def fake_agent(root, destination, payload, schema, kind):
        calls.append((kind, destination.name))
        if kind == "reflect":
            outcome = payload["outcome"]
            return CourseReflection(observed_changes="TEST fixture, not real measurement claim", next_guidance="fixture", units=[
                {"unit_id": key, "observation_ids": ids + outcome["global_observation_ids"],
                 "conclusion": "no_change", "explanation": "test", "next_action": "continue"}
                for key, ids in outcome["unit_observation_ids"].items()])
        units = []
        if payload["checkpoint_step"] == 20:
            for n in range(4):
                patients = [f"P{i:03d}" for i in range(4*n+1, 4*n+5)]
                units.append({"unit_id": f"U{n+1:03d}", "patients": patients, "contents": [], "target": "deletion",
                              "evidence_ids": [f"R{n*132+1:04d}"], "rationale": "fixture", "expected_change": "fixture"})
        else:
            assert payload["history"][-1]["reflection"]["next_guidance"] == "fixture"
        quotas = [60, 60, 60, 60] if units else [90, 70, 50, 30]
        return CourseProposal(diagnosis="TEST", hypothesis="TEST", risk="TEST", new_units=units,
            allocations=[{"unit_id": f"U{i+1:03d}", "exposures": n, "reason": "fixture"} for i, n in enumerate(quotas)])
    monkeypatch.setattr(runner, "call_agent", fake_agent)

    def fake_main():
        args = training.parse_args(sys.argv[1:])
        set_seed(42)
        model = get_peft_model(ToyPatientModel(), LoraConfig(r=2, lora_alpha=2, lora_dropout=.1, target_modules=["proj"]))
        # Deliberately unequal states emulate the formerly fatal cross-GPU case.
        arm = Path(args.output_dir).parent.name
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.add_({"static": 0, "random": 1e-4, "dynamic": 2e-4}[arm])
        features = [{"x": torch.randn(4, 3), "labels": torch.tensor([-100, 1+i%6, 2, 3])} for i in range(2117)]
        trainer = training.QwenASRLoRATrainer(model=model, train_dataset=features, data_collator=default_data_collator,
            args=TrainingArguments(output_dir=args.output_dir, use_cpu=True, per_device_train_batch_size=8,
                gradient_accumulation_steps=4, max_steps=100, learning_rate=2e-4, warmup_steps=2,
                dataloader_num_workers=0, save_steps=20, save_total_limit=0, report_to="none",
                remove_unused_columns=False, logging_steps=20, disable_tqdm=True))
        trainer.train()
    monkeypatch.setattr(training, "main", fake_main)
    original = training.QwenASRLoRATrainer
    try:
        for arm in ("random", "static", "dynamic"):
            runner.train_arm(tmp_path, arm)
        runner.report(tmp_path)
    finally:
        training.QwenASRLoRATrainer = original
    result = runner.read(tmp_path / "results.json")
    assert all(r["exposures"] == 3200 for r in result.values())
    assert len(calls) == 8  # initial prescription + three updates + four reflections
    assert runner.read(tmp_path / "static/state_20.json") != runner.read(tmp_path / "random/state_20.json")
    assert runner.read(tmp_path / "static/state_40.json") != runner.read(tmp_path / "dynamic/state_40.json")
    assert len(decoded) == 30
    assert set(decoded) == {(arm, step, split) for arm in runner.ARMS for step in runner.ENDPOINTS
                            for split in (runner.TRAIN, runner.FEEDBACK)}
    assert len(measured) == 15
    assert set(measured) == {(arm, step) for arm in runner.ARMS for step in runner.ENDPOINTS}
    for arm, expected in (("static", .2), ("random", .4), ("dynamic", .6)):
        for step in runner.ENDPOINTS:
            metric = runner.read(tmp_path / arm / f"metrics_{step}.json")
            assert Path(metric["checkpoint"]).parent.parent.name == arm
            assert metric["evaluation_policy"] == "own_checkpoint"
            assert metric["feedback"]["overall"]["after"]["mean_utterance_cer"] == pytest.approx(expected)
        assert not list((tmp_path / arm).glob("evaluation_reuse_*.json"))
    assert runner.read(tmp_path / "static/manifest_60.json") != runner.read(tmp_path / "dynamic/manifest_60.json")
    assert runner.read(tmp_path / "dynamic/episode_40.json")["reflection"]["next_guidance"] == "fixture"


def test_production_cli_cannot_load_m0_or_reset_schedule():
    parsed = [vars(training.parse_args(runner.training_arguments(Path("/tmp/test"), arm)[1:])) for arm in runner.ARMS]
    for args in parsed:
        assert args["model_path"] == str(runner.BASE)
        assert args["lora_init_adapter_path"] == args["resume_from"] == ""
        assert args["max_steps"] == 100 and args["lr"] == .0002 and args["warmup_ratio"] == .02
        assert args["batch_size"] == 8 and args["grad_acc"] == 4
        assert args["num_workers"] == 0
    assert {k for k in parsed[0] if parsed[0][k] != parsed[1][k]} == {"output_dir"}


def test_request_contract_and_private_data_rejection():
    config = {"model": "gpt-5.6-sol", "reasoning_effort": "xhigh", "max_output_tokens": 32768}
    request = runner.build_request(config, {"checkpoint_step": 20, "history": []}, CourseProposal, "prescribe")
    assert request["model"] == "gpt-5.6-sol" and request["reasoning"]["effort"] == "xhigh"
    assert request["store"] is False and request["text"]["format"]["strict"] is True
    reflection = runner.build_request(config, {"history": []}, CourseReflection, "reflect")
    assert "outcome.global_observation_ids" in reflection["input"][0]["content"]
    assert "at least one ID" in reflection["input"][0]["content"]
    citation_schema = reflection["text"]["format"]["schema"]["$defs"]["UnitReflection"]["properties"]["observation_ids"]
    assert "global_observation_ids" in citation_schema["description"]
    for fragment in ("/data/audio.wav", "vlink201"):
        with pytest.raises(AssertionError):
            runner.build_request(config, {"bad": fragment}, CourseProposal, "prescribe")
