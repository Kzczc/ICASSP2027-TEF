#!/usr/bin/env python3
"""Ablation: TEF against equal weights (w/o Entropy) and probability averaging (w/o Log-Odds).

Example:
    python scripts/ablation.py --models qwen2.5-7b deepseek-v3.2 --report reports/ablation.md
"""

from __future__ import annotations

from dataclasses import asdict

from _report import LANG_NAMES, base_parser, dimension_columns, emit, markdown_table
from tef.fusion import METHOD_NAMES
from tef.pipeline import cache_path, dimension_table, load_records
from tef.schema import TABLE_ORDER

METHODS = ("tef", "tef_without_entropy", "tef_without_log_odds")


def main() -> None:
    parser = base_parser(__doc__)
    parser.set_defaults(models=["qwen2.5-7b", "deepseek-v3.2"])
    parser.add_argument("--prompt", default="original")
    args = parser.parse_args()

    header = ["Model", "Language", "Method"] + [f"{c} {m}" for c in dimension_columns() + ["Avg."] for m in ("Acc", "F1")]
    rows, payload = [], {}
    for model in args.models:
        for lang in args.lang:
            table = dimension_table(load_records(cache_path(args.output_dir, model, args.prompt, lang)), METHODS)
            payload.setdefault(model, {})[lang] = {m: {d: asdict(s) for d, s in row.items()} for m, row in table.items()}
            for method in METHODS:
                cells = []
                for dimension in list(TABLE_ORDER) + ["Average"]:
                    s = table[method].get(dimension)
                    cells += [f"{s.accuracy:.1f}", f"{s.macro_f1:.1f}"] if s else ["-", "-"]
                rows.append([model, LANG_NAMES[lang], METHOD_NAMES[method]] + cells)
    emit(markdown_table(header, rows), args.report, payload)


if __name__ == "__main__":
    main()
