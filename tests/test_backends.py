import math

import pytest

from tef.backends import probability_from_logprobs


def test_renormalizes_answer_tokens():
    score = probability_from_logprobs({"C": math.log(0.6), "S": math.log(0.2), "The": math.log(0.1)})
    assert score.source == "logprobs"
    assert score.p_change == pytest.approx(0.75)


def test_accepts_token_variants():
    score = probability_from_logprobs({" C": math.log(0.3), "S.": math.log(0.3)})
    assert score.p_change == pytest.approx(0.5)


def test_falls_back_to_generated_text():
    score = probability_from_logprobs({"C": -0.1}, generated_text=" s")
    assert (score.p_change, score.source) == (0.0, "generated_text")
    assert probability_from_logprobs({}, "C").p_change == 1.0


def test_invalid_answer():
    score = probability_from_logprobs({}, "I cannot tell")
    assert (score.p_change, score.source) == (0.5, "invalid")
