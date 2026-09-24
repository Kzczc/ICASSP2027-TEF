"""Local inference with vLLM: one greedy answer token and its top log-probabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .base import AnswerScore, probability_from_logprobs


class VLLMBackend:
    def __init__(
        self,
        model_path: str,
        name: str | None = None,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        top_logprobs: int = 20,
        enable_thinking: bool = False,
        seed: int = 0,
    ) -> None:
        if not model_path:
            raise ValueError("model_path is empty; set it in configs/models.yaml or through the referenced environment variable")
        from vllm import LLM, SamplingParams

        self.name = name or Path(model_path).name
        self._llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            seed=seed,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=top_logprobs)
        self._enable_thinking = enable_thinking

    def _format(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            # Only Qwen3 templates read enable_thinking; older tokenizers reject the extra keyword.
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=self._enable_thinking
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def score(self, prompts: Sequence[str]) -> list[AnswerScore]:
        outputs = self._llm.generate([self._format(p) for p in prompts], self._params, use_tqdm=False)
        scores = []
        for output in outputs:
            completion = output.outputs[0] if output.outputs else None
            logprobs: dict[str, float] = {}
            if completion is not None and completion.logprobs:
                for token_id, entry in completion.logprobs[0].items():
                    token = entry.decoded_token if getattr(entry, "decoded_token", None) is not None else self._tokenizer.decode([token_id])
                    logprobs.setdefault(token, entry.logprob)
            scores.append(probability_from_logprobs(logprobs, completion.text if completion is not None else ""))
        return scores
