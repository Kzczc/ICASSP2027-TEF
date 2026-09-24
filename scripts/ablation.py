#!/usr/bin/env python3
"""Ablation: TEF against equal weights (w/o Entropy) and probability averaging (w/o Log-Odds).

Example:
    python scripts/ablation.py --models qwen2.5-7b deepseek-v3.2 --report reports/ablation.md
"""

from __future__ import annotations

from _report import LANG_NAMES, acc_f1_cells, acc_f1_header, base_parser, display_name, emit, markdown_table, table_payload
from tef.fusion import METHOD_NAMES
from tef.pipeline import cache_path, dimension_table, load_records

METHODS = ("tef", "tef_without_entropy", "tef_without_log_odds")


def main() -> None:
    parser = base_parser(__doc__, prompt=True)
    parser.set_defaults(models=["qwen2.5-7b", "deepseek-v3.2"])
    args = parser.parse_args()

    rows, payload = [], {}
    for model in args.models:
        for lang in args.lang:
            table = dimension_table(load_records(cache_path(args.output_dir, model, args.prompt, lang)), METHODS)
            payload.setdefault(model, {})[lang] = table_payload(table)
            rows += [[display_name(model), LANG_NAMES[lang], METHOD_NAMES[m]] + acc_f1_cells(table[m]) for m in METHODS]
    emit(markdown_table(acc_f1_header("Model", "Language", "Method"), rows), args.report, payload)


if __name__ == "__main__":
    main()
