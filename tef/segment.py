"""Sentence segmentation used by all sentence-level fusion rules."""

from __future__ import annotations

import re

DELIMITERS: dict[str, tuple[str, ...]] = {
    "zh": ("。", "！", "？"),
    "en": (". ", "! ", "? "),
}
MIN_SENTENCE_LENGTH = 10


def split_sentences(text: str, lang: str, min_length: int = MIN_SENTENCE_LENGTH) -> list[str]:
    """Split a post into sentences at sentence-final punctuation.

    Pieces shorter than ``min_length`` characters are dropped. A post without any piece of
    sufficient length is returned as a single sentence, so every post yields at least one query.
    """
    if lang not in DELIMITERS:
        raise ValueError(f"unsupported language {lang!r}; expected one of {sorted(DELIMITERS)}")
    delimiters = DELIMITERS[lang]
    pattern = "(" + "|".join(re.escape(d) for d in delimiters) + ")"
    sentences: list[str] = []
    current = ""
    for part in re.split(pattern, text):
        current += part
        if part in delimiters:
            sentence = current.strip()
            if len(sentence) >= min_length:
                sentences.append(sentence)
            current = ""
    tail = current.strip()
    if len(tail) >= min_length:
        sentences.append(tail)
    return sentences or [text.strip()]
