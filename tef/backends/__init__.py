"""Inference backends that return P(change-oriented) for each prompt."""

from __future__ import annotations

from typing import Any

from .base import AnswerScore, Backend, probability_from_logprobs

__all__ = ["AnswerScore", "Backend", "create_backend", "probability_from_logprobs"]


def create_backend(config: dict[str, Any]) -> Backend:
    """Instantiate a backend from one entry of ``configs/models.yaml``."""
    kind = config.get("backend")
    if kind == "vllm":
        from .vllm_backend import VLLMBackend

        return VLLMBackend(
            model_path=config["model_path"],
            name=config.get("name"),
            tensor_parallel_size=int(config.get("tensor_parallel_size", 1)),
            max_model_len=int(config.get("max_model_len", 4096)),
            gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.9)),
            top_logprobs=int(config.get("top_logprobs", 20)),
            enable_thinking=bool(config.get("enable_thinking", False)),
        )
    if kind == "openai":
        from .openai_backend import OpenAIBackend

        return OpenAIBackend(
            model=config["model"],
            name=config.get("name"),
            api_key_env=config.get("api_key_env", "OPENAI_API_KEY"),
            base_url=config.get("base_url") or None,
            top_logprobs=int(config.get("top_logprobs", 5)),
            max_workers=int(config.get("max_workers", 8)),
            max_retries=int(config.get("max_retries", 6)),
            timeout=float(config.get("timeout", 60)),
        )
    raise ValueError(f"unknown backend {kind!r} in model config {config.get('name')!r}")
