#!/usr/bin/env python3
"""Efficiency: tokens, time to first token (TTFT) and cost of Direct, voting and TEF.

Requests go through an OpenAI-compatible endpoint with streaming, and TTFT is measured on the
client as the time until the first streamed content arrives. For local models, start a vLLM
server first, e.g. ``vllm serve $QWEN25_7B_PATH --served-model-name Qwen2.5-7B-Instruct``.
Majority Vote and Soft Vote issue the same sentence-level queries and are reported together as
"Voting"; the TEF run issues the same queries again and additionally times its fusion step.

Cost is GPU hours times ``--gpu-price-per-hour`` when that option is given, otherwise tokens
times the per-million-token prices.

Example:
    python scripts/efficiency.py --model qwen2.5-7b-server --gpu-price-per-hour 0.8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _report import emit, markdown_table  # noqa: E402
from tef.backends.base import probability_from_logprobs  # noqa: E402
from tef.config import load_model_config  # noqa: E402
from tef.data import LANGUAGES, evaluation_items, load_posts  # noqa: E402
from tef.fusion import tef  # noqa: E402
from tef.prompts import build_prompt  # noqa: E402
from tef.schema import get_dimension  # noqa: E402
from tef.segment import split_sentences  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="an OpenAI-compatible model key in configs/models.yaml")
    parser.add_argument("--lang", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--limit", type=int, default=None, help="only the first N items per language")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--gpu-price-per-hour", type=float, default=None, help="USD per GPU hour for local models")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--price-input", type=float, default=0.0, help="USD per million input tokens")
    parser.add_argument("--price-output", type=float, default=0.0, help="USD per million output tokens")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


class Client:
    def __init__(self, config: dict) -> None:
        from openai import OpenAI

        key = os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"))
        if not key:
            raise RuntimeError(f"environment variable {config.get('api_key_env')} is not set")
        self.model = config["model"]
        self.top_logprobs = int(config.get("top_logprobs", 5))
        self.client = OpenAI(api_key=key, base_url=config.get("base_url") or None)

    def request(self, prompt: str) -> dict:
        start = time.perf_counter()
        first = None
        text, logprobs, usage = "", {}, None
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                if first is None:
                    first = time.perf_counter()
                text += choice.delta.content
            if choice.logprobs is not None and choice.logprobs.content and not logprobs:
                logprobs = {e.token: e.logprob for e in choice.logprobs.content[0].top_logprobs or []}
        end = time.perf_counter()
        return {
            "ttft_ms": 1000.0 * ((first or end) - start),
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "p_change": probability_from_logprobs(logprobs, text).p_change,
        }


def run(client: Client, groups: list[list[str]], workers: int, fuse: bool) -> dict:
    flat = [p for group in groups for p in group]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(client.request, flat))
    fusion_seconds = 0.0
    if fuse:
        offset = 0
        for group in groups:
            t0 = time.perf_counter()
            tef([r["p_change"] for r in results[offset : offset + len(group)]])
            fusion_seconds += time.perf_counter() - t0
            offset += len(group)
    wall = time.perf_counter() - start
    return {
        "requests": len(results),
        "input_tokens": sum(r["input_tokens"] for r in results),
        "output_tokens": sum(r["output_tokens"] for r in results),
        "mean_ttft_ms": sum(r["ttft_ms"] for r in results) / len(results),
        "wall_seconds": wall,
        "fusion_ms": 1000.0 * fusion_seconds,
    }


def main() -> None:
    args = parse_args()
    config = load_model_config(args.model, args.model_config)
    client = Client(config)
    direct_groups, sentence_groups = [], []
    for lang in args.lang:
        for item in evaluation_items(load_posts(lang, args.data_dir))[: args.limit]:
            dimension = get_dimension(item.dimension)
            direct_groups.append([build_prompt(item.text, dimension, lang)])
            sentence_groups.append([build_prompt(s, dimension, lang) for s in split_sentences(item.text, lang)])

    measured = {
        "Direct": run(client, direct_groups, args.workers, fuse=False),
        "Voting": run(client, sentence_groups, args.workers, fuse=False),
        "TEF": run(client, sentence_groups, args.workers, fuse=True),
    }
    rows = []
    for method, m in measured.items():
        if args.gpu_price_per_hour is not None:
            cost = m["wall_seconds"] / 3600.0 * args.num_gpus * args.gpu_price_per_hour
        else:
            cost = (m["input_tokens"] * args.price_input + m["output_tokens"] * args.price_output) / 1e6
        m["cost_usd"] = cost
        tokens = (m["input_tokens"] + m["output_tokens"]) / 1e6
        rows.append([args.model, method, f"{tokens:.2f}", f"{m['mean_ttft_ms']:.1f}", f"{cost:.2f}", str(m["requests"])])
    table = markdown_table(["Model", "Method", "Tokens (M)", "TTFT (ms)", "Cost ($)", "Requests"], rows)
    emit(table, args.report, measured)
    if not args.report:
        print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    main()
