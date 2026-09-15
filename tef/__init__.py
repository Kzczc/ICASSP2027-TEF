"""Tempered Evidence Fusion (TEF) of sentence-level LLM judgments and the MIND benchmark."""

from .fusion import (
    FUSION_RULES,
    FusionResult,
    direct,
    majority_vote,
    soft_vote,
    tef,
    tef_without_entropy,
    tef_without_log_odds,
)
from .schema import CHANGE, DIMENSIONS, SIDES, STABILITY

__all__ = [
    "CHANGE",
    "DIMENSIONS",
    "FUSION_RULES",
    "FusionResult",
    "SIDES",
    "STABILITY",
    "direct",
    "majority_vote",
    "soft_vote",
    "tef",
    "tef_without_entropy",
    "tef_without_log_odds",
]

__version__ = "1.0.0"
