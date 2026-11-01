"""Explicit sequence objectives for the IPO and character-risk pilots.

These training costs never participate in SG coarse review.
"""

import math

import torch
import torch.nn.functional as F

from .text import edit_distance, text_normalize


def mean_preference_loss(policy_logp, reference_logp, token_counts, *, objective, beta):
    """One audio's chosen/rejected pair, full transcript and EOS in both scores.

    IPO is 0.5 * beta**2 times the usual squared IPO loss. This positive
    scale preserves its finite optimum and matches DPO's initial derivative
    at policy == reference. It is recorded explicitly in the pilot protocol.
    DPO here uses mean token scores too; it is not standard summed-score DPO.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if policy_logp.shape != (2,) or reference_logp.shape != (2,) or token_counts.shape != (2,):
        raise ValueError("expected one chosen/rejected pair")
    if torch.any(token_counts <= 0):
        raise ValueError("each completion requires at least EOS")
    ratios = (policy_logp - reference_logp.detach()) / token_counts
    margin = ratios[0] - ratios[1]
    scaled = beta * margin
    if objective == "dpo":
        loss = -F.logsigmoid(scaled)
    elif objective == "ipo":
        loss = 0.5 * (scaled - 0.5).square()
    else:
        raise ValueError("objective must be dpo or ipo")
    return loss, margin


def character_costs(reference, hypotheses, *, normalized):
    """Character edit counts, or per-utterance CER, with evaluation normalization."""
    target = text_normalize(reference)
    if not target or not hypotheses:
        raise ValueError("a nonempty normalized reference and hypotheses are required")
    counts = [edit_distance(target, text_normalize(h)) for h in hypotheses]
    divisor = len(target) if normalized else 1
    return [c / divisor for c in counts]


def minimum_character_risk(sequence_logp, costs):
    """One deduplicated N-best list; raw sequence probabilities, no length mean.

    q remains differentiable. Ground-truth costs and the unweighted N-best
    baseline are constants. Equal costs therefore yield exactly zero risk
    gradient; a separate CE term can still learn from the ground truth.
    """
    if sequence_logp.ndim != 1 or sequence_logp.numel() == 0:
        raise ValueError("expected a nonempty vector of sequence log probabilities")
    costs = torch.as_tensor(costs, dtype=sequence_logp.dtype, device=sequence_logp.device).detach()
    if costs.shape != sequence_logp.shape or not torch.isfinite(costs).all():
        raise ValueError("costs must be finite and match the candidate count")
    q = torch.softmax(sequence_logp, dim=0)
    return (q * (costs - costs.mean())).sum(), (q * costs).sum(), q
