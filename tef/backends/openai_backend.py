"""Inference through an OpenAI-compatible chat completions API (OpenAI, DeepSeek, or a vLLM server)."""

from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from .base import AnswerScore, probability_from_logprobs

logger = logging.getLogger(__name__)


class OpenAIBackend:
    def __init__(
        self,
        model: str,
        name: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        top_logprobs: int = 5,
        max_workers: int = 8,
        max_retries: int = 6,
        timeout: float = 60.0,
    ) -> None:
        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"environment variable {api_key_env} is not set (see .env.example)")
        self.name = name or model
        self.model = model
        self.top_logprobs = top_logprobs
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    def _score_one(self, prompt: str) -> AnswerScore:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=self.top_logprobs,
                )
                choice = response.choices[0]
                logprobs: dict[str, float] = {}
                content = choice.logprobs.content if choice.logprobs is not None else None
                if content:
                    for entry in content[0].top_logprobs or []:
                        logprobs.setdefault(entry.token, entry.logprob)
                return probability_from_logprobs(logprobs, choice.message.content or "")
            except Exception as exc:  # network errors, rate limits, server errors
                if attempt == self.max_retries - 1:
                    raise
                delay = min(60.0, 2.0 ** attempt) + random.random()
                logger.warning("request failed (%s); retrying in %.1f s", exc, delay)
                time.sleep(delay)
        raise RuntimeError("unreachable")

    def score(self, prompts: Sequence[str]) -> list[AnswerScore]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._score_one, prompts))
