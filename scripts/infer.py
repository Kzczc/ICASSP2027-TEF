#!/usr/bin/env python3
"""Query a model on every evaluation item of MIND and cache the answer probabilities.

For each (post, dimension) item the script stores the document-level probability used by
Direct and the sentence-level probabilities used by Majority Vote, Soft Vote, TEF and the
ablations. Existing records are skipped, so an interrupted run can be resumed.

Example:
    python scripts/infer.py --model qwen2.5-7b --lang zh en --prompt original
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tef.backends import create_backend  # noqa: E402
from tef.config import load_model_config  # noqa: E402
from tef.data import LANGUAGES, append_jsonl, evaluation_items, load_posts, read_jsonl  # noqa: E402
from tef.pipeline import cache_path  # noqa: E402
from tef.prompts import PROMPT_VARIANTS, build_prompt  # noqa: E402
from tef.schema import get_dimension  # noqa: E402
from tef.segment import split_sentences  # noqa: E402

logger = logging.getLogger("infer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model key in configs/models.yaml")
    parser.add_argument("--lang", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    parser.add_argument("--prompt", default="original", choices=PROMPT_VARIANTS)
    parser.add_argument("--data-dir", default=None, help="directory with mind_zh.jsonl and mind_en.jsonl")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-config", default=None, help="path to a models.yaml file")
    parser.add_argument("--batch-items", type=int, default=256, help="items per backend call")
    parser.add_argument("--limit", type=int, default=None, help="only the first N items per language")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_model_config(args.model, args.model_config)
    backend = create_backend(config)

    for lang in args.lang:
        items = evaluation_items(load_posts(lang, args.data_dir))
        if args.limit:
            items = items[: args.limit]
        out = cache_path(args.output_dir, args.model, args.prompt, lang)
        done = {(r["post_id"], r["dimension"]) for r in read_jsonl(out)} if out.exists() else set()
        todo = [item for item in items if (item.post_id, item.dimension) not in done]
        logger.info("%s | %s | %s: %d items, %d cached, %d to run", args.model, args.prompt, lang, len(items), len(done), len(todo))
        sentence_cache: dict[str, list[str]] = {}
        sources: Counter = Counter()

        for start in range(0, len(todo), args.batch_items):
            batch = todo[start : start + args.batch_items]
            prompts: list[str] = []
            spans = []
            for item in batch:
                dimension = get_dimension(item.dimension)
                sentences = sentence_cache.setdefault(item.post_id, split_sentences(item.text, lang))
                first = len(prompts)
                prompts.append(build_prompt(item.text, dimension, lang, args.prompt))
                prompts.extend(build_prompt(s, dimension, lang, args.prompt) for s in sentences)
                spans.append((first, len(prompts)))
            scores = backend.score(prompts)
            sources.update(s.source for s in scores)
            append_jsonl(out, (
                {
                    "post_id": item.post_id,
                    "lang": item.lang,
                    "dimension": item.dimension,
                    "side": item.side,
                    "direct": scores[a].p_change,
                    "sentences": [s.p_change for s in scores[a + 1 : b]],
                }
                for item, (a, b) in zip(batch, spans)
            ))
            logger.info("%s: %d/%d items written", lang, min(start + len(batch), len(todo)), len(todo))
        if sources:
            logger.info("%s: answer extraction sources %s", lang, dict(sources))


if __name__ == "__main__":
    main()
