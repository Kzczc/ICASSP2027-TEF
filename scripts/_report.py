"""Helpers shared by the evaluation scripts: argument parsing and Markdown tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tef.data import LANGUAGES  # noqa: E402
from tef.schema import DIMENSION_BY_NAME, TABLE_ORDER  # noqa: E402

PAPER_MODELS = ("qwen2.5-7b", "llama3-8b", "qwen3-14b", "deepseek-v3.2", "gpt-4o-mini")
LANG_NAMES = {"zh": "Chinese", "en": "English"}


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(PAPER_MODELS), help="model keys in configs/models.yaml")
    parser.add_argument("--lang", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    parser.add_argument("--output-dir", default="outputs", help="directory with the caches written by infer.py")
    parser.add_argument("--report", default=None, help="write the Markdown report to this file")
    return parser


def dimension_columns() -> list[str]:
    return [DIMENSION_BY_NAME[d].abbreviation for d in TABLE_ORDER]


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
