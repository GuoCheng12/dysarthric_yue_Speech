"""Full training-barrier integration with a toy model; never a research result."""

from collections import Counter
from pathlib import Path
import sys

import accelerate.utils.other as accelerate_other
import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, default_data_collator, set_seed

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))
import content_curriculum_run as content_run
import real_curriculum_v2 as runner
from finetune.scripts import qwen3_asr_lora_sft as training
from therapist_harness.content_curriculum import ContentPrescription, ContentReflection


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 7)
        self.dropout = nn.Dropout(.1)

    def forward(self, x, **kwargs):
        return {"logits": self.proj(self.dropout(x))}


def test_content_dynamic_full_loop_reads_fresh_evidence_and_reflection(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(accelerate_other, "is_deepspeed_available", lambda: False)
    train = []
    for i in range(2117):
        text = f"今日食飯{i%48:02d}"
        correct = i//48 % 2 == 0
        train.append({"utt_id": f"private/{i:04d}", "speaker_id": f"private_{i%66:03d}",
            "clean_gt": text, "prediction": text if correct else text.replace("飯", ""),
            "zero_shot_bucket": "easy" if correct else "hard", "duration": 2})
    zero = {r["utt_id"]: {**r, "raw_pred": r["prediction"]} for r in train}
    cat, mapping = runner.catalog(train, zero)
    for row in cat["recordings"]:
        mapping[row["id"]] = {**mapping[row["id"]], "patient": row["patient"], "content": row["content"]}
    feedback = [{**r, "utt_id": "feedback/"+r["utt_id"], "clean_gt": "我想飲水", "prediction": "我想水",
                 "zero_shot_bucket": ("easy", "medium", "hard")[i%3]} for i, r in enumerate(train[:66])]
    runner.save(tmp_path / "local_map.json", mapping)
    runner.save(tmp_path / "zero_train.json", zero)
    runner.save(tmp_path / "coverage.json", runner.make_coverage(list(mapping)))
    runner.save_rows(tmp_path / "zero_train_predictions.jsonl", train)
    runner.save_rows(tmp_path / "zero_feedback_predictions.jsonl", feedback)
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {"course_design": "content_centered"})

    decoded, calls = [], []

    def decode(checkpoint, source, destination):
        step = int(checkpoint.name.split("-")[-1])
        decoded.append((step, source))
        result = train if source == runner.TRAIN else [
            {**r, "prediction": "我想飲水" if step == 40 else "我想水"} for r in feedback]
        runner.save_rows(destination / "predictions.jsonl", result)
        return result

    def agent(root, destination, payload, schema, kind):
        calls.append((kind, destination.name))
        if kind == "reflect":
            outcome = payload["outcome"]
            assert len(payload["history"]) == outcome["step"]//20-2
            assert sum(r["actual_exposures"] for group in outcome["delivery"].values() for r in group) == 240
            return ContentReflection(observed_changes="TEST fixture only", next_guidance="use next contents", lessons=[
                {"lesson_id": lid, "observation_ids": [ids[0]], "delivery_interpretation": "fixture",
                 "conclusion": "mixed", "next_action": "replace", "explanation": "fixture",
                 "reusable_lesson": "fixture response-to-intervention"}
                for lid, ids in outcome["lesson_observation_ids"].items()])
        step = payload["checkpoint_step"]
        assert payload["content_count"] == 48 and payload["train_rows"] == 2117
        assert len(payload["history"]) == step//20-1
        if step > 20:
            assert payload["history"][-1]["outcome"]["step"] == step
            assert payload["history"][-1]["reflection"]["next_guidance"] == "use next contents"
        lessons = []
        for li in range(2):
            practice = []
            for di, role in enumerate(("target", "target", "related", "related", "protection", "protection")):
                cid = f"C{(step//20-1)*12+li*6+di+1:02d}"
                practice.append({"content_id": cid, "role": role, "region": "whole", "exposures": 20,
                    "selection": "error" if role == "target" else "correct" if role == "protection" else "any",
                    "buckets": ["easy"] if role == "protection" else ["easy", "hard"], "reason": "TEST"})
            lessons.append({"lesson_id": f"L{li+1:03d}", "title": "fixture", "recognition_goal": "fixture",
                "evidence_ids": [f"E_{p['content_id']}" for p in practice[:2]], "hypothesis": "fixture",
                "practice": practice, "expected_transfer": "fixture", "revision_condition": "fixture"})
        return ContentPrescription(diagnosis="TEST", lessons=lessons, history_used="fixture", risk="fixture")

    monkeypatch.setattr(runner, "decode", decode)
    monkeypatch.setattr(content_run, "call_agent", agent)

    def main():
        args = training.parse_args(sys.argv[1:])
        set_seed(42)
        model = get_peft_model(ToyModel(), LoraConfig(r=2, lora_alpha=2, lora_dropout=.1, target_modules=["proj"]))
        features = [{"x": torch.randn(4, 3), "labels": torch.tensor([-100, 1+i%6, 2, 3])} for i in range(2117)]
        trainer = training.QwenASRLoRATrainer(model=model, train_dataset=features, data_collator=default_data_collator,
            args=TrainingArguments(output_dir=args.output_dir, use_cpu=True, per_device_train_batch_size=8,
                gradient_accumulation_steps=4, max_steps=100, learning_rate=2e-4, warmup_steps=2,
                dataloader_num_workers=0, save_steps=20, save_total_limit=0, report_to="none",
                remove_unused_columns=False, logging_steps=20, disable_tqdm=True))
        trainer.train()

    monkeypatch.setattr(training, "main", main)
    original = training.QwenASRLoRATrainer
    try:
        runner.train_arm(tmp_path, "dynamic")
    finally:
        training.QwenASRLoRATrainer = original
    content_run.report(tmp_path)
    result = runner.read(tmp_path / "results.json")
    assert result["unique_train_recordings"] == 2117 and result["exposures"] == 3200
    assert result["best"]["step"] == 40
    assert decoded == [(s, split) for s in runner.ENDPOINTS for split in (runner.TRAIN, runner.FEEDBACK)]
    assert calls == [("prescribe", "prescribe_20"), ("reflect", "reflect_40"),
        ("prescribe", "prescribe_40"), ("reflect", "reflect_60"), ("prescribe", "prescribe_60"),
        ("reflect", "reflect_80"), ("prescribe", "prescribe_80"), ("reflect", "reflect_100")]
    assert not (tmp_path / "first_course.json").exists()  # old unit prescription never reused
    for step in runner.ENDPOINTS:
        manifest = runner.read(tmp_path / f"dynamic/manifest_{step}.json")
        assert len(manifest) == 640 and max(Counter(r["id"] for r in manifest).values()) <= 2
        if step > 20:
            assert sum(r["stream"] == "course" for r in manifest) == 240
            delivery = runner.read(tmp_path / f"dynamic/delivery_{step}.json")
            assert all(r["selector_match_exposures"] == r["actual_exposures"] == 20
                       for group in delivery.values() for r in group)


def test_launcher_detaches_and_keeps_key_out_of_receipts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import subprocess

    runner.save(tmp_path / "protocol.json", {"course_design": "content_centered"})
    runner.save(tmp_path / "readiness.json", {"checks_passed": True})
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {
        "course_design": "content_centered", "arms_execution_order": ["dynamic"]})
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-credential")
    captured = {}

    def spawn(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(content_run.subprocess, "Popen", spawn)
    content_run.launch(tmp_path, "1")
    assert captured["start_new_session"] and captured["stdin"] == subprocess.DEVNULL
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert captured["env"]["OPENAI_API_KEY"] == "test-only-not-a-credential"
    assert "test-only-not-a-credential" not in (tmp_path / "background_launch.json").read_text()
    assert "test-only-not-a-credential" not in str(captured["command"])
