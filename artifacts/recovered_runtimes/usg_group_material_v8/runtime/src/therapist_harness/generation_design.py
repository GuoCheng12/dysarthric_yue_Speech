"""Small, explicit SG actions; reference facts are separate from error evidence."""

import hashlib
import json
from pathlib import Path


INSTRUCTIONS = {
    "default": "You are a helpful assistant.<|endofprompt|>",
    "cantonese_en": "You are a helpful assistant. Please express in Cantonese.<|endofprompt|>",
    "cantonese_zh": "You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
}


def load_generation_choices(path, assignments):
    payload = json.loads(Path(path).read_text())
    choices = {x["sample_id"]: x for x in payload["choices"]}
    if len(choices) != len(payload["choices"]) or set(choices) != set(assignments):
        raise ValueError("generation choices must bind each assignment exactly once")
    for sample_id, choice in choices.items():
        row = assignments[sample_id]
        if choice["mode"] not in {"zero_shot", "instruct2"} or choice["instruction_id"] not in INSTRUCTIONS:
            raise ValueError("unsupported generation choice")
        if choice["reference_wav_sha256"] != row["reference_wav_sha256"]:
            raise ValueError("generation choice reference differs from assignment")
        if choice["target_text_sha256"] != hashlib.sha256(row["target_text"].encode()).hexdigest():
            raise ValueError("generation choice target differs from assignment")
    return choices


class ConfiguredGenerator:
    def __init__(self, model, mode, instruction_id, reference_text):
        self.model, self.mode, self.sample_rate = model, mode, model.sample_rate
        # The old explicit Cantonese fallback remains explicit even for a default trial.
        if mode == "instruct2" and instruction_id == "default":
            instruction_id = "cantonese_zh"
        self.instruction_id = instruction_id
        self.effective_prompt = INSTRUCTIONS[instruction_id]
        if mode == "zero_shot":
            self.effective_prompt += reference_text

    def inference_zero_shot(self, target, _legacy_prompt, reference, **kwargs):
        method = self.model.inference_zero_shot if self.mode == "zero_shot" else self.model.inference_instruct2
        return method(target, self.effective_prompt, reference, **kwargs)


def role_delivery(assignments, reviews, pairs, exposures):
    """Count audio occurrences, never count an exposure once per error pair."""
    from collections import Counter
    pair_counts = Counter(x["audio_id"] for x in pairs)
    occurrences = {(x["step"], x["source"], x["audio_id"], x["occurrence"]) for x in exposures}
    exposed = Counter(x[2] for x in occurrences)
    by_id = {x.get("audio_id", x.get("sample_id")): x for x in reviews}
    out = {}
    for row in assignments:
        sample_id = row["sample_id"]
        role = row["role"]
        counts = out.setdefault(role, dict(requested=0, passed=0, pending=0,
                                         failed=0, paired_audio=0, pairs=0, exposures=0))
        counts["requested"] += 1
        verdict = by_id[sample_id]["verdict"]
        counts[{"pass": "passed", "pending": "pending", "obvious_failure": "failed"}[verdict]] += 1
        counts["paired_audio"] += bool(pair_counts[sample_id])
        counts["pairs"] += pair_counts[sample_id]
        counts["exposures"] += exposed[sample_id]
        if verdict != "pass" and (pair_counts[sample_id] or exposed[sample_id]):
            raise ValueError("unadmitted audio received DPO delivery")
    return out
