#!/usr/bin/env python3
"""Main results: accuracy and macro-F1 of Direct, Majority Vote, Soft Vote and TEF.

Example:
    python scripts/evaluate.py --report reports/main_results.md
"""

from __future__ import annotations

from dataclasses import asdict

from _report import LANG_NAMES, base_parser, dimension_columns, emit, markdown_table
from tef.fusion import METHOD_NAMES
from tef.pipeline import cache_path, dimension_table, load_records
from tef.schema import TABLE_ORDER

METHODS = ("direct", "majority_vote", "soft_vote", "tef")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--prompt", default="original")
    args = parser.parse_args()

    sections, payload = [], {}
    for lang in args.lang:
        header = ["Model", "Method"] + [f"{c} {m}" for c in dimension_columns() + ["Avg."] for m in ("Acc", "F1")]
        rows = []
        for model in args.models:
            table = dimension_table(load_records(cache_path(args.output_dir, model, args.prompt, lang)), METHODS)
            payload.setdefault(lang, {})[model] = {m: {d: asdict(s) for d, s in row.items()} for m, row in table.items()}
            for method in METHODS:
                cells = []
                for dimension in list(TABLE_ORDER) + ["Average"]:
                    s = table[method].get(dimension)
                    cells += [f"{s.accuracy:.1f}", f"{s.macro_f1:.1f}"] if s else ["-", "-"]
                rows.append([model, METHOD_NAMES[method]] + cells)
        sections.append(f"### {LANG_NAMES[lang]}\n\n" + markdown_table(header, rows))
    emit("\n\n".join(sections), args.report, payload)


if __name__ == "__main__":
    main()
