"""Document-level fusion rules over sentence-level probabilities.

Every rule receives ``p_i = P(change-oriented | sentence i)`` for the sentences of one post,
except :func:`direct`, which receives the single probability of the whole post.

TEF (Section 3 of the paper) weights each sentence by its normalized information gain
``w_i = 1 - H(p_i) / log 2`` and sums the weighted, clipped log-odds of both positions,
``Score(c) = sum_i w_i * clip(log(p_i(c) / (1 - p_i(c))), -M, M)``, with probabilities clipped to
``[epsilon, 1 - epsilon]`` before the logit. The paper uses ``epsilon = 1e-6`` and ``M = 10``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from .schema import CHANGE, STABILITY

EPSILON = 1e-6
CLIP_BOUND = 10.0
CONFIDENCE_TEMPERATURE = 5.0


@dataclass(frozen=True)
class FusionResult:
    """Decision of a fusion rule for one post and one dimension.

    ``change_score`` is what the rule compares: Score(change) for TEF and w/o Entropy, the share of
    change-oriented votes for Majority Vote, and the (weighted) mean probability for the other rules.
    """

    prediction: str
    confidence: float
    change_score: float


def _check(probabilities: Sequence[float]) -> list[float]:
    values = [float(p) for p in probabilities]
    if not values:
        raise ValueError("at least one probability is required")
    for p in values:
        if not 0.0 <= p <= 1.0 or math.isnan(p):
            raise ValueError(f"probability out of range: {p}")
    return values


def clipped_log_odds(p: float, epsilon: float = EPSILON, clip_bound: float = CLIP_BOUND) -> float:
    p = min(max(p, epsilon), 1.0 - epsilon)
    return max(-clip_bound, min(clip_bound, math.log(p / (1.0 - p))))


def binary_entropy(p: float) -> float:
    return -sum(q * math.log(q) for q in (p, 1.0 - p) if q > 0.0)


def information_gain_weight(p: float) -> float:
    """Normalized information gain ``1 - H(p) / log 2``, in [0, 1]."""
    return max(0.0, 1.0 - binary_entropy(p) / math.log(2.0))


def _decide(change_value: float, stability_value: float) -> str:
    return CHANGE if change_value >= stability_value else STABILITY


def _softmax_confidence(change_score: float, stability_score: float, prediction: str, temperature: float) -> float:
    diff = (change_score - stability_score) / temperature
    p_change = 1.0 / (1.0 + math.exp(-diff)) if diff >= 0 else math.exp(diff) / (1.0 + math.exp(diff))
    return p_change if prediction == CHANGE else 1.0 - p_change


def direct(probabilities: Sequence[float]) -> FusionResult:
    """Direct prediction from the probability of the whole post."""
    values = _check(probabilities)
    if len(values) != 1:
        raise ValueError("direct prediction expects exactly one document-level probability")
    p = values[0]
    prediction = _decide(p, 1.0 - p)
    return FusionResult(prediction, max(p, 1.0 - p), p)


def majority_vote(probabilities: Sequence[float]) -> FusionResult:
    """Count sentence-level hard labels; ties go to the label that received a vote first."""
    values = _check(probabilities)
    votes = Counter(_decide(p, 1.0 - p) for p in values)
    prediction = votes.most_common(1)[0][0]
    share = votes[prediction] / len(values)
    return FusionResult(prediction, share, votes[CHANGE] / len(values))


def soft_vote(probabilities: Sequence[float]) -> FusionResult:
    """Average sentence-level probabilities with equal weights."""
    values = _check(probabilities)
    mean = sum(values) / len(values)
    prediction = _decide(mean, 1.0 - mean)
    return FusionResult(prediction, max(mean, 1.0 - mean), mean)


def _log_odds_fusion(
    values: list[float], weights: list[float], epsilon: float, clip_bound: float, temperature: float
) -> FusionResult:
    change_score = sum(w * clipped_log_odds(p, epsilon, clip_bound) for w, p in zip(weights, values))
    stability_score = sum(w * clipped_log_odds(1.0 - p, epsilon, clip_bound) for w, p in zip(weights, values))
    prediction = _decide(change_score, stability_score)
    return FusionResult(prediction, _softmax_confidence(change_score, stability_score, prediction, temperature), change_score)


def tef(
    probabilities: Sequence[float],
    epsilon: float = EPSILON,
    clip_bound: float = CLIP_BOUND,
    temperature: float = CONFIDENCE_TEMPERATURE,
) -> FusionResult:
    """Tempered Evidence Fusion.

    The confidence, used only by the calibration analysis, is the softmax of the two position
    scores at ``temperature``.
    """
    values = _check(probabilities)
    return _log_odds_fusion(values, [information_gain_weight(p) for p in values], epsilon, clip_bound, temperature)


def tef_without_entropy(
    probabilities: Sequence[float],
    epsilon: float = EPSILON,
    clip_bound: float = CLIP_BOUND,
    temperature: float = CONFIDENCE_TEMPERATURE,
) -> FusionResult:
    """Ablation "w/o Entropy": equal weights, clipped log-odds sum."""
    values = _check(probabilities)
    return _log_odds_fusion(values, [1.0] * len(values), epsilon, clip_bound, temperature)


def tef_without_log_odds(probabilities: Sequence[float]) -> FusionResult:
    """Ablation "w/o Log-Odds": entropy-weighted probability averaging."""
    values = _check(probabilities)
    weights = [information_gain_weight(p) for p in values]
    total = sum(weights)
    mean = sum(w * p for w, p in zip(weights, values)) / total if total > 0.0 else 0.5
    prediction = _decide(mean, 1.0 - mean)
    return FusionResult(prediction, max(mean, 1.0 - mean), mean)


FUSION_RULES: dict[str, Callable[[Sequence[float]], FusionResult]] = {
    "majority_vote": majority_vote,
    "soft_vote": soft_vote,
    "tef": tef,
    "tef_without_entropy": tef_without_entropy,
    "tef_without_log_odds": tef_without_log_odds,
}

METHOD_NAMES: dict[str, str] = {
    "direct": "Direct",
    "majority_vote": "Majority Vote",
    "soft_vote": "Soft Vote",
    "tef": "TEF",
    "tef_without_entropy": "w/o Entropy",
    "tef_without_log_odds": "w/o Log-Odds",
}
