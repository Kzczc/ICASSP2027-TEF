import pytest

from tef.data import LANGUAGES
from tef.prompts import ANSWER_TOKENS, PROMPT_VARIANTS, build_prompt, describe_side
from tef.schema import CHANGE, DIMENSIONS, SIDES, STABILITY, get_dimension


@pytest.mark.parametrize("lang", LANGUAGES)
@pytest.mark.parametrize("variant", PROMPT_VARIANTS)
@pytest.mark.parametrize("dimension", DIMENSIONS, ids=lambda d: d.name)
def test_prompt_names_dimension_positions_and_answer_tokens(lang, variant, dimension):
    prompt = build_prompt("  An example post.  ", dimension, lang, variant)
    assert "An example post." in prompt
    assert (dimension.name_zh if lang == "zh" else dimension.name) in prompt
    for sub in dimension.subcategories:
        for side in SIDES:
            position = sub.position(side)
            assert (position.name_zh if lang == "zh" else position.name) in prompt
    for token in ANSWER_TOKENS.values():
        assert token in prompt


def test_describe_side():
    politics = get_dimension("Politics")
    assert describe_side(politics, CHANGE, "en") == "Revolutionism (Policy Orientations); Globalism (Diplomatic Strategies)"
    assert describe_side(politics, STABILITY, "zh") == "改良主义（政策导向）；孤立主义（外交战略）"


def test_variants_differ_in_length():
    dimension = get_dimension("Society")
    lengths = [len(build_prompt("text", dimension, "en", v)) for v in ("minimal", "original", "verbose")]
    assert lengths == sorted(lengths)


def test_invalid_arguments():
    dimension = get_dimension("Culture")
    with pytest.raises(ValueError):
        build_prompt("text", dimension, "fr")
    with pytest.raises(ValueError):
        build_prompt("text", dimension, "en", "long")
    assert len(set(ANSWER_TOKENS.values())) == 2
