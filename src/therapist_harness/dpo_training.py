"""Audio-conditioned preference loss and source-balanced update windows.

Targets are raw transcripts. The language tag is conditioning, as it is during
forced-language ASR sampling. Only transcript tokens and EOS are scored. A DPO
reference is a frozen R-only *ASR adapter*, unrelated to SG reference speech.
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import random

import torch
import torch.nn.functional as F


def transcript_log_probs(logits, labels):
    """Return summed continuation log probabilities and token counts per row."""
    targets = labels[:, 1:].contiguous()
    mask = targets.ne(-100)
    counts = mask.sum(1)
    if torch.any(counts == 0):
        raise ValueError("each continuation requires a supervised token (including EOS)")
    token_nll = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
        ignore_index=-100, reduction="none",
    ).view_as(targets)
    return -token_nll.sum(1), counts


def dpo_losses(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta):
    """Standard DPO, without length normalization or an auxiliary SFT term."""
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be finite and positive")
    margin = (policy_chosen - policy_rejected) - (
        reference_chosen.detach() - reference_rejected.detach()
    )
    return -F.logsigmoid(beta * margin)


def preference_key(row):
    fields = {key: row.get(key, default) for key, default in (
        ("audio_sha256", ""), ("audio", ""), ("chosen", ""),
        ("rejected", ""), ("prompt", ""), ("language", "Cantonese"),
    )}
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@dataclass
class AudioPreferenceCollator:
    processor: object
    sampling_rate: int = 16000
    audio_loader: object = None

    def __call__(self, rows, *, include_rejected=True):
        # Reuse Qwen's actual chat template and audio preprocessing conventions.
        if self.audio_loader is None:
            from qwen_asr.inference.utils import normalize_audio_input
            if self.sampling_rate != 16000:
                raise ValueError("Qwen ASR sampling and preference scoring require 16 kHz audio")
            loader = normalize_audio_input
        else:
            loader = self.audio_loader
        audios = [loader(row["audio"]) for row in rows]
        prefixes = []
        for row in rows:
            messages = [
                {"role": "system", "content": row.get("prompt", "")},
                {"role": "user", "content": [{"type": "audio", "audio": None}]},
            ]
            prefix = self.processor.apply_chat_template(
                [messages], add_generation_prompt=True, tokenize=False,
            )[0]
            prefixes.append(prefix + "language " + row.get("language", "Cantonese") + "<asr_text>")
        targets = [row["chosen"] for row in rows]
        if include_rejected:
            targets += [row["rejected"] for row in rows]
            prefixes = prefixes + prefixes
            audios = audios + audios
        eos = self.processor.tokenizer.eos_token
        if not eos:
            raise ValueError("Qwen ASR tokenizer requires an EOS token")
        kwargs = dict(audio=audios, return_tensors="pt", padding=True,
                      padding_side="right", truncation=False)
        full = self.processor(text=[p + t + eos for p, t in zip(prefixes, targets)], **kwargs)
        prefix = self.processor(text=prefixes, **kwargs)
        labels = full["input_ids"].clone()
        for index, length in enumerate(prefix["attention_mask"].sum(1).tolist()):
            if not torch.equal(full["input_ids"][index, :length], prefix["input_ids"][index, :length]):
                raise ValueError("tokenized conditioning must be an exact prefix of the continuation")
            labels[index, :length] = -100
        # Attention masks, not pad-token identity: EOS can share an ID with PAD.
        labels[full["attention_mask"].eq(0)] = -100
        full["labels"] = labels
        return full


def group_training_rows(rows, objective):
    if objective not in {"dpo", "sft"}:
        raise ValueError("objective must be dpo or sft")
    groups = {"real": {}, "synthetic": {}}
    for original in rows:
        row = dict(original)
        row["chosen"] = row.get("chosen", row.get("text", ""))
        if not row["chosen"] or row["chosen"].startswith("language "):
            raise ValueError("chosen must be a nonempty raw transcript, without the ASR language prefix")
        source = row.get("source")
        if source not in groups:
            raise ValueError("training source must be real or synthetic")
        if objective == "dpo" and ("rejected" not in row or row["rejected"] == row["chosen"]):
            raise ValueError("DPO rows require a distinct, actually sampled rejected transcript")
        key = row.get("audio_id", row.get("utt_id", row["audio"]))
        groups[source].setdefault(key, []).append(row)
    for source, audio_groups in groups.items():
        for audio_id, group in audio_groups.items():
            conditioning = {(r["audio"], r["chosen"], r.get("prompt", ""), r.get("language", "Cantonese")) for r in group}
            if len(conditioning) != 1:
                raise ValueError(f"audio group has inconsistent conditioning: {audio_id}")
            if objective == "sft":
                # Reusing a preference manifest for SFT never repeats chosen once per error.
                groups[source][audio_id] = [{**group[0], "pair_weight": 1.0}]
                continue
            if len({r["rejected"] for r in group}) != len(group):
                raise ValueError(f"duplicate rejected transcripts in audio group: {audio_id}")
            for row in group:
                row.setdefault("pair_weight", 1.0 / len(group))
                if not math.isfinite(row["pair_weight"]) or row["pair_weight"] <= 0:
                    raise ValueError("pair weights must be finite and positive")
            if not math.isclose(sum(r["pair_weight"] for r in group), 1.0, abs_tol=1e-6):
                raise ValueError("pair weights must sum to one per audio")
    if not any(groups.values()):
        raise ValueError("empty training data requires an explicit no-update branch")
    return groups


class SourceBalancedWindows:
    """Independent R/S streams keep real order identical in paired controls.

Each sampled audio contributes all its unique pairs, whose weights sum to one.
The update objective is mean(real audio loss) + lambda * mean(synthetic loss).
Minibatches partition the already-normalized window, including its final tail.
"""

    def __init__(self, groups, real_per_step, synthetic_per_step, synthetic_weight, seed, *, synthetic_schedule=None):
        if real_per_step < 0 or synthetic_per_step < 0 or synthetic_weight < 0:
            raise ValueError("invalid source exposure counts or synthetic weight")
        if groups["real"] and real_per_step == 0:
            raise ValueError("real input requires positive real_per_step")
        if groups["synthetic"] and synthetic_per_step == 0:
            raise ValueError("mixed input requires positive synthetic_per_step")
        self.groups = groups
        self.counts = {"real": real_per_step if groups["real"] else 0,
                       "synthetic": synthetic_per_step if groups["synthetic"] else 0}
        self.weights = {"real": 1.0, "synthetic": synthetic_weight}
        self.rng = {"real": random.Random(seed), "synthetic": random.Random(seed + 1009)}
        self.remaining = defaultdict(list)
        self.synthetic_schedule, self.step = synthetic_schedule, 0
        if synthetic_schedule is not None and (not synthetic_schedule or any(
                len(step) != self.counts['synthetic'] or any(x not in groups['synthetic'] for x in step)
                for step in synthetic_schedule)):
            raise ValueError('synthetic exposure schedule must use eligible audio and fixed per-step counts')

    def next(self):
        window = []
        for source, count in self.counts.items():
            for occurrence in range(count):
                if source == 'synthetic' and self.synthetic_schedule is not None:
                    audio_id = self.synthetic_schedule[self.step][occurrence]
                else:
                    if not self.remaining[source]:
                        self.remaining[source] = sorted(self.groups[source])
                        self.rng[source].shuffle(self.remaining[source])
                    audio_id = self.remaining[source].pop()
                for row in self.groups[source][audio_id]:
                    window.append({**row, "_audio_group": audio_id, "_occurrence": occurrence,
                                   "_loss_weight": self.weights[source] * row["pair_weight"] / count})
        self.step += 1
        return window


def move_batch(batch, device, dtype):
    return {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            for key, value in batch.items()}


def batch_log_probs(model, batch):
    labels = batch["labels"]
    result = model(**{key: value for key, value in batch.items() if key != "labels"})
    logits = result["logits"] if isinstance(result, dict) else result.logits
    return transcript_log_probs(logits, labels)


@torch.no_grad()
def frozen_reference_log_probs(model, batch):
    """Fail if accidentally handed a trainable reference; never toggle policy state."""
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("DPO reference must be a separate frozen eval-mode R-only ASR")
    return batch_log_probs(model, batch)[0].detach()
