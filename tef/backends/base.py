"""Shared types and the extraction of P(change-oriented) from answer-token log-probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from ..prompts import ANSWER_TOKENS
from ..schema import CHANGE, STABILITY


@dataclass(frozen=True)
class AnswerScore:
    """Probability of the change-oriented answer and how it was obtained.

    ``source`` is ``"logprobs"`` when both answer tokens appear among the top log-probabilities,
    ``"generated_text"`` when only the generated letter is usable, and ``"invalid"`` otherwise
    (the probability is then 0.5).
    """

    p_change: float
    source: str


class Backend(Protocol):
    name: str

    def score(self, prompts: Sequence[str]) -> list[AnswerScore]:
        """Return one :class:`AnswerScore` per prompt, in order."""


def _variants(token: str) -> tuple[str, ...]:
    return (token, " " + token, token + ".", " " + token + ".", token + ":", " " + token + ":")


def probability_from_logprobs(top_logprobs: Mapping[str, float], generated_text: str = "") -> AnswerScore:
    """Renormalize the two answer-token probabilities over the label set."""
    found: dict[str, float] = {}
    for side in (CHANGE, STABILITY):
        for variant in _variants(ANSWER_TOKENS[side]):
            if variant in top_logprobs:
                found[side] = float(top_logprobs[variant])
                break
    if len(found) == 2:
        top = max(found.values())
        change = math.exp(found[CHANGE] - top)
        stability = math.exp(found[STABILITY] - top)
        return AnswerScore(change / (change + stability), "logprobs")
    text = generated_text.strip().upper()
    if text.startswith(ANSWER_TOKENS[CHANGE]):
        return AnswerScore(1.0, "generated_text")
    if text.startswith(ANSWER_TOKENS[STABILITY]):
        return AnswerScore(0.0, "generated_text")
    return AnswerScore(0.5, "invalid")
