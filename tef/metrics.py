"""Evaluation metrics: accuracy, macro-F1, expected calibration error, overconfidence gap."""

from __future__ import annotations

from typing import Sequence

from .schema import SIDES


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _same_length(y_true, y_pred)
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = SIDES) -> float:
    """Unweighted mean of per-label F1 over ``labels``."""
    _same_length(y_true, y_pred)
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    """ECE with ``n_bins`` equal-width bins over [0, 1]; the last bin includes 1.0."""
    _same_length(confidences, correct)
    n = len(confidences)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        lower, upper = b / n_bins, (b + 1) / n_bins
        members = [
            i for i, c in enumerate(confidences)
            if (lower <= c < upper) or (b == n_bins - 1 and c == upper)
        ]
        if members:
            mean_conf = sum(confidences[i] for i in members) / len(members)
            mean_acc = sum(bool(correct[i]) for i in members) / len(members)
            ece += len(members) / n * abs(mean_acc - mean_conf)
    return ece


def overconfidence_gap(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean confidence minus accuracy; positive values indicate overconfidence."""
    _same_length(confidences, correct)
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences) - sum(bool(c) for c in correct) / len(correct)


def _same_length(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
