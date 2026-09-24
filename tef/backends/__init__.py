"""Inference backends that return P(change-oriented) for each prompt."""

from __future__ import annotations

from typing import Any, Callable

from .base import AnswerScore, Backend, probability_from_logprobs

__all__ = ["AnswerScore", "Backend", "create_backend", "probability_from_logprobs"]


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes")


_VLLM_OPTIONS: dict[str, Callable[[Any], Any]] = {
    "tensor_parallel_size": int,
    "max_model_len": int,
    "gpu_memory_utilization": float,
    "top_logprobs": int,
    "enable_thinking": _as_bool,
}
_OPENAI_OPTIONS: dict[str, Callable[[Any], Any]] = {
    "api_key_env": str,
    "base_url": str,
    "top_logprobs": int,
    "max_workers": int,
    "max_retries": int,
    "timeout": float,
}


def _options(config: dict[str, Any], casts: dict[str, Callable[[Any], Any]]) -> dict[str, Any]:
    # Environment expansion turns numbers into strings, and an unset variable into "".
    return {key: cast(config[key]) for key, cast in casts.items() if config.get(key) not in (None, "")}


def create_backend(config: dict[str, Any]) -> Backend:
    """Instantiate a backend from one entry of ``configs/models.yaml``."""
    kind = config.get("backend")
    if kind == "vllm":
        from .vllm_backend import VLLMBackend

        return VLLMBackend(config["model_path"], name=config.get("name"), **_options(config, _VLLM_OPTIONS))
    if kind == "openai":
        from .openai_backend import OpenAIBackend

        return OpenAIBackend(config["model"], name=config.get("name"), **_options(config, _OPENAI_OPTIONS))
    raise ValueError(f"unknown backend {kind!r} in model config {config.get('name')!r}")
