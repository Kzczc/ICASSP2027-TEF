#!/usr/bin/env python3
"""Main results: accuracy and macro-F1 of Direct, Majority Vote, Soft Vote and TEF.

Example:
    python scripts/evaluate.py --report reports/main_results.md
"""

from __future__ import annotations

from _report import LANG_NAMES, acc_f1_cells, acc_f1_header, base_parser, display_name, emit, markdown_table, table_payload
from tef.fusion import METHOD_NAMES
from tef.pipeline import cache_path, dimension_table, load_records

METHODS = ("direct", "majority_vote", "soft_vote", "tef")


def main() -> None:
    args = base_parser(__doc__, prompt=True).parse_args()
    sections, payload = [], {}
    for lang in args.lang:
        rows = []
        for model in args.models:
            table = dimension_table(load_records(cache_path(args.output_dir, model, args.prompt, lang)), METHODS)
            payload.setdefault(lang, {})[model] = table_payload(table)
            rows += [[display_name(model), METHOD_NAMES[m]] + acc_f1_cells(table[m]) for m in METHODS]
        sections.append(f"### {LANG_NAMES[lang]}\n\n" + markdown_table(acc_f1_header("Model", "Method"), rows))
    emit("\n\n".join(sections), args.report, payload)


if __name__ == "__main__":
    main()
