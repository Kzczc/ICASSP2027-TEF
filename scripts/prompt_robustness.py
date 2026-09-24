#!/usr/bin/env python3
"""Prompt robustness: average accuracy over both languages under three prompts.

Run ``infer.py`` with ``--prompt original``, ``--prompt verbose`` and ``--prompt minimal`` first.
The strongest baseline is the best of Direct, Majority Vote and Soft Vote under the same prompt.
Accuracy is averaged over the six dimensions and then over the two languages.

Example:
    python scripts/prompt_robustness.py --report reports/prompt_robustness.md
"""

from __future__ import annotations

from _report import base_parser, display_name, emit, markdown_table
from tef.pipeline import cache_path, dimension_table, load_records
from tef.prompts import PROMPT_VARIANTS

BASELINES = ("direct", "majority_vote", "soft_vote")
METHODS = BASELINES + ("tef",)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    header = ["Model"] + [f"Strongest baseline ({p})" for p in PROMPT_VARIANTS] + [f"TEF ({p})" for p in PROMPT_VARIANTS]
    rows, payload = [], {}
    for model in args.models:
        baseline_cells, tef_cells = [], []
        for prompt in PROMPT_VARIANTS:
            averages = {m: [] for m in METHODS}
            for lang in args.lang:
                table = dimension_table(load_records(cache_path(args.output_dir, model, prompt, lang)), METHODS)
                for method in METHODS:
                    averages[method].append(table[method]["Average"].accuracy)
            mean = {m: sum(v) / len(v) for m, v in averages.items()}
            best = max(BASELINES, key=lambda m: mean[m])
            payload.setdefault(model, {})[prompt] = {"strongest_baseline": best, **mean}
            baseline_cells.append(f"{mean[best]:.1f}")
            tef_cells.append(f"{mean['tef']:.1f}")
        rows.append([display_name(model)] + baseline_cells + tef_cells)
    emit(markdown_table(header, rows), args.report, payload)


if __name__ == "__main__":
    main()
