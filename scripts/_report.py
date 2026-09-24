"""Helpers shared by the evaluation scripts: argument parsing and Markdown tables."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tef.config import load_model_config  # noqa: E402
from tef.data import LANGUAGES  # noqa: E402
from tef.prompts import PROMPT_VARIANTS  # noqa: E402
from tef.schema import DIMENSION_BY_NAME, TABLE_ORDER  # noqa: E402

PAPER_MODELS = ("qwen2.5-7b", "llama3-8b", "qwen3-14b", "deepseek-v3.2", "gpt-4o-mini")
LANG_NAMES = {"zh": "Chinese", "en": "English"}


def base_parser(description: str, prompt: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(PAPER_MODELS), help="model keys in configs/models.yaml")
    parser.add_argument("--lang", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    parser.add_argument("--output-dir", default="outputs", help="directory with the caches written by infer.py")
    parser.add_argument("--report", default=None, help="write the Markdown report to this file")
    if prompt:
        parser.add_argument("--prompt", default="original", choices=PROMPT_VARIANTS)
    return parser


@lru_cache(maxsize=None)
def display_name(model: str) -> str:
    """Model name as written in the paper, taken from ``display_name`` in configs/models.yaml."""
    try:
        return load_model_config(model).get("display_name", model)
    except KeyError:
        return model


def acc_f1_header(*leading: str) -> list[str]:
    columns = [DIMENSION_BY_NAME[d].abbreviation for d in TABLE_ORDER] + ["Avg."]
    return list(leading) + [f"{c} {m}" for c in columns for m in ("Acc", "F1")]


def acc_f1_cells(row: dict) -> list[str]:
    """Accuracy and macro-F1 per dimension in table order, then the average; "-" where missing."""
    cells = []
    for dimension in (*TABLE_ORDER, "Average"):
        s = row.get(dimension)
        cells += [f"{s.accuracy:.1f}", f"{s.macro_f1:.1f}"] if s else ["-", "-"]
    return cells


def table_payload(table: dict) -> dict:
    return {method: {dimension: asdict(s) for dimension, s in row.items()} for method, row in table.items()}


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def emit(text: str, report: str | None, payload: dict | None = None) -> None:
    print(text)
    if report:
        path = Path(report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        if payload is not None:
            path.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
