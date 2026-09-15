"""Cached inference records and their evaluation.

A cache file ``<output_dir>/<model>/<prompt>/<lang>.jsonl`` stores, for every evaluation item,
the document-level probability and the sentence-level probabilities. All fusion rules, ablations
and calibration statistics are computed from these files without further model calls.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .data import read_jsonl
from .fusion import FUSION_RULES, FusionResult, direct
from .metrics import accuracy, expected_calibration_error, macro_f1, overconfidence_gap
from .schema import TABLE_ORDER


def cache_path(output_dir: str | os.PathLike, model: str, prompt: str, lang: str) -> Path:
    return Path(output_dir) / model / prompt / f"{lang}.jsonl"


@dataclass(frozen=True)
class Record:
    post_id: str
    lang: str
    dimension: str
    side: str
    direct: float
    sentences: tuple[float, ...]


def load_records(path: str | os.PathLike) -> list[Record]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; run scripts/infer.py first")
    return [
        Record(r["post_id"], r["lang"], r["dimension"], r["side"], float(r["direct"]), tuple(float(p) for p in r["sentences"]))
        for r in read_jsonl(path)
    ]


def fuse(record: Record, method: str) -> FusionResult:
    if method == "direct":
        return direct([record.direct])
    return FUSION_RULES[method](record.sentences)


@dataclass(frozen=True)
class Scores:
    accuracy: float
    macro_f1: float
    ece: float
    overconfidence_gap: float
    n: int


def score_records(records: Iterable[Record], method: str) -> Scores:
    records = list(records)
    results = [fuse(r, method) for r in records]
    y_true = [r.side for r in records]
    y_pred = [res.prediction for res in results]
    confidences = [res.confidence for res in results]
    correct = [t == p for t, p in zip(y_true, y_pred)]
    return Scores(
        accuracy=100.0 * accuracy(y_true, y_pred),
        macro_f1=100.0 * macro_f1(y_true, y_pred),
        ece=expected_calibration_error(confidences, correct),
        overconfidence_gap=overconfidence_gap(confidences, correct),
        n=len(records),
    )


def by_dimension(records: Iterable[Record]) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.dimension].append(record)
    return {d: groups[d] for d in TABLE_ORDER if d in groups}


def dimension_table(records: Iterable[Record], methods: Iterable[str]) -> dict[str, dict[str, Scores]]:
    """Scores per method and dimension, plus the unweighted average over dimensions under ``"Average"``."""
    groups = by_dimension(records)
    table: dict[str, dict[str, Scores]] = {}
    for method in methods:
        row = {dimension: score_records(recs, method) for dimension, recs in groups.items()}
        if row:
            k = len(row)
            row["Average"] = Scores(
                accuracy=sum(s.accuracy for s in row.values()) / k,
                macro_f1=sum(s.macro_f1 for s in row.values()) / k,
                ece=sum(s.ece for s in row.values()) / k,
                overconfidence_gap=sum(s.overconfidence_gap for s in row.values()) / k,
                n=sum(s.n for s in row.values()),
            )
        table[method] = row
    return table
