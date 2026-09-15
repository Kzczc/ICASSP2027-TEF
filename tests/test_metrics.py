import pytest

from tef.metrics import accuracy, expected_calibration_error, macro_f1, overconfidence_gap
from tef.schema import CHANGE, STABILITY


def test_accuracy_and_macro_f1():
    y_true = [CHANGE, CHANGE, STABILITY, STABILITY]
    y_pred = [CHANGE, STABILITY, STABILITY, STABILITY]
    assert accuracy(y_true, y_pred) == pytest.approx(0.75)
    # F1(change) = 2/3, F1(stability) = 4/5
    assert macro_f1(y_true, y_pred) == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_macro_f1_counts_absent_predictions_as_zero():
    assert macro_f1([CHANGE, STABILITY], [CHANGE, CHANGE]) == pytest.approx((2 / 3 + 0.0) / 2)


def test_expected_calibration_error():
    assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert expected_calibration_error([0.8] * 5, [True] * 4 + [False]) == pytest.approx(0.0)
    assert expected_calibration_error([0.9] * 10, [True] * 6 + [False] * 4) == pytest.approx(0.3)
    # Two bins with different gaps, weighted by bin size.
    confidences = [0.55] * 2 + [0.95] * 2
    correct = [True, True, True, False]
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.5 * 0.45 + 0.5 * 0.45)


def test_overconfidence_gap():
    assert overconfidence_gap([0.9] * 10, [True] * 6 + [False] * 4) == pytest.approx(0.3)
    assert overconfidence_gap([0.6, 0.6], [True, True]) == pytest.approx(-0.4)


def test_length_mismatch():
    with pytest.raises(ValueError):
        accuracy([CHANGE], [])
