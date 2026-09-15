import math

import pytest

from tef.fusion import (
    CLIP_BOUND,
    FUSION_RULES,
    METHOD_NAMES,
    binary_entropy,
    clipped_log_odds,
    direct,
    information_gain_weight,
    majority_vote,
    soft_vote,
    tef,
    tef_without_entropy,
    tef_without_log_odds,
)
from tef.schema import CHANGE, STABILITY


def test_information_gain_weight_bounds():
    assert binary_entropy(0.5) == pytest.approx(math.log(2))
    assert information_gain_weight(0.5) == pytest.approx(0.0)
    assert information_gain_weight(1.0) == pytest.approx(1.0)
    assert information_gain_weight(0.0) == pytest.approx(1.0)
    assert 0.0 < information_gain_weight(0.9) < 1.0


def test_log_odds_are_clipped():
    assert clipped_log_odds(1.0) == CLIP_BOUND
    assert clipped_log_odds(0.0) == -CLIP_BOUND
    assert clipped_log_odds(0.5) == pytest.approx(0.0)
    assert clipped_log_odds(0.9) == pytest.approx(math.log(9.0))


def test_decisive_sentence_outweighs_uncertain_majority():
    # One decisive change-oriented sentence and four uncertain stability-leaning sentences.
    probabilities = [0.9, 0.35, 0.35, 0.35, 0.35]
    assert majority_vote(probabilities).prediction == STABILITY
    assert soft_vote(probabilities).prediction == STABILITY
    assert tef(probabilities).prediction == CHANGE


def test_uncertain_sentences_contribute_nothing():
    result = tef([0.5, 0.5, 0.5])
    assert result.change_score == pytest.approx(0.0)
    assert result.confidence == pytest.approx(0.5)
    assert tef([0.8, 0.5, 0.5]).change_score == pytest.approx(tef([0.8]).change_score)


def test_tef_is_symmetric_in_the_two_positions():
    probabilities = [0.8, 0.3, 0.65]
    flipped = [1.0 - p for p in probabilities]
    assert tef(probabilities).prediction != tef(flipped).prediction
    assert tef(probabilities).change_score == pytest.approx(-tef(flipped).change_score)


def test_tef_score_matches_equation():
    probabilities = [0.9, 0.2]
    expected = sum(information_gain_weight(p) * math.log(p / (1 - p)) for p in probabilities)
    assert tef(probabilities).change_score == pytest.approx(expected)


def test_ablations():
    probabilities = [0.9, 0.2, 0.6]
    assert tef_without_entropy(probabilities).change_score == pytest.approx(sum(math.log(p / (1 - p)) for p in probabilities))
    weights = [information_gain_weight(p) for p in probabilities]
    expected_mean = sum(w * p for w, p in zip(weights, probabilities)) / sum(weights)
    assert tef_without_log_odds(probabilities).change_score == pytest.approx(expected_mean)
    assert tef_without_log_odds([0.5, 0.5]).change_score == pytest.approx(0.5)


@pytest.mark.parametrize("rule", list(FUSION_RULES.values()))
def test_confidence_is_probability_of_prediction(rule):
    for probabilities in ([0.9], [0.1, 0.2, 0.7], [0.55, 0.45], [1.0, 0.0, 0.0]):
        result = rule(probabilities)
        assert result.prediction in (CHANGE, STABILITY)
        assert 0.5 <= result.confidence <= 1.0


def test_direct_and_voting_confidence():
    assert direct([0.3]).prediction == STABILITY
    assert direct([0.3]).confidence == pytest.approx(0.7)
    assert majority_vote([0.9, 0.8, 0.2]).confidence == pytest.approx(2 / 3)
    assert soft_vote([0.9, 0.8, 0.2]).confidence == pytest.approx(19 / 30)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        tef([])
    with pytest.raises(ValueError):
        tef([1.2])
    with pytest.raises(ValueError):
        direct([0.4, 0.6])


def test_method_names_cover_all_rules():
    assert set(METHOD_NAMES) == set(FUSION_RULES) | {"direct"}
