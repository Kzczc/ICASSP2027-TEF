import pytest

from tef.segment import MIN_SENTENCE_LENGTH, split_sentences


def test_english_sentences():
    text = (
        "The market will naturally push companies to adapt. Why? "
        "Enterprise behavior always follows consumer demand! "
        "The free market can determine the allocation of resources."
    )
    assert split_sentences(text, "en") == [
        "The market will naturally push companies to adapt.",
        "Enterprise behavior always follows consumer demand!",
        "The free market can determine the allocation of resources.",
    ]


def test_chinese_sentences():
    text = "看到运动员在赛场上的表现，心里非常感动。好！这不仅是个人的成功，更是集体力量的展现。"
    assert split_sentences(text, "zh") == [
        "看到运动员在赛场上的表现，心里非常感动。",
        "这不仅是个人的成功，更是集体力量的展现。",
    ]


def test_short_text_is_kept_as_one_sentence():
    assert split_sentences("  Too short.  ", "en", min_length=100) == ["Too short."]
    assert MIN_SENTENCE_LENGTH == 10


def test_unsupported_language():
    with pytest.raises(ValueError):
        split_sentences("text", "fr")
