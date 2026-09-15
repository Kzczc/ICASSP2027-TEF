#!/usr/bin/env python3
"""Calibration: expected calibration error and overconfidence gap of document-level confidence.

Confidence is the probability of the predicted position: the model probability for Direct, the
vote share for Majority Vote, the mean probability for Soft Vote, and the softmax of the two
position scores at temperature 5 for TEF. Posts of all dimensions are pooled per language.

Example:
    python scripts/calibration.py --models qwen2.5-7b --plot reports/calibration.pdf
"""

from __future__ import annotations

from dataclasses import asdict

from _report import LANG_NAMES, base_parser, emit, markdown_table
from tef.fusion import METHOD_NAMES
from tef.pipeline import cache_path, load_records, score_records

METHODS = ("direct", "majority_vote", "soft_vote", "tef")


def plot(results: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    langs = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    width = 0.8 / max(len(langs), 1)
    for ax, key, title in ((axes[0], "ece", "(a) Expected calibration error"), (axes[1], "overconfidence_gap", "(b) Overconfidence gap")):
        for j, lang in enumerate(langs):
            values = [results[lang][m][key] * 100 for m in METHODS]
            ax.bar([i + j * width for i in range(len(METHODS))], values, width, label=LANG_NAMES[lang])
        ax.set_xticks([i + width * (len(langs) - 1) / 2 for i in range(len(METHODS))])
        ax.set_xticklabels([METHOD_NAMES[m] for m in METHODS])
        ax.set_ylabel("%")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    parser = base_parser(__doc__)
    parser.set_defaults(models=["qwen2.5-7b"])
    parser.add_argument("--prompt", default="original")
    parser.add_argument("--plot", default=None, help="save a bar chart of the first model to this file")
    args = parser.parse_args()

    header = ["Model", "Language", "Method", "Acc", "ECE (%)", "Overconfidence gap (%)", "N"]
    rows, payload = [], {}
    for model in args.models:
        for lang in args.lang:
            records = load_records(cache_path(args.output_dir, model, args.prompt, lang))
            for method in METHODS:
                s = score_records(records, method)
                payload.setdefault(model, {}).setdefault(lang, {})[method] = asdict(s)
                rows.append([model, LANG_NAMES[lang], METHOD_NAMES[method], f"{s.accuracy:.1f}", f"{100 * s.ece:.1f}", f"{100 * s.overconfidence_gap:.1f}", str(s.n)])
    emit(markdown_table(header, rows), args.report, payload)
    if args.plot:
        plot(payload[args.models[0]], args.plot)


if __name__ == "__main__":
    main()
