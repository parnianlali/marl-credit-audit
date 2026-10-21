"""Week 3 Task 4 -- tests for the new eval/metrics.py functions:
credit_error(scale=...), normalized_regret, spearman.

Every expected value below is copied from the Week 3 task spec table (hand-
derived), or derived on paper in the comment next to it. None comes from
running the implementation.
"""

import math

import numpy as np
import pytest

from eval.metrics import credit_error, normalized_regret, spearman
from games.generator import make_mono_game


# ---------- spearman ----------

def test_spearman_perfect_agreement():
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0, abs=1e-12)


def test_spearman_perfect_reversal():
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0, abs=1e-12)


def test_spearman_one_swap():
    # ranks (1,2,3) vs (1,3,2): 1 - 6*sum(d^2)/(n(n^2-1)) = 1 - 6*2/24 = 0.5
    assert spearman([1, 2, 3], [1, 3, 2]) == pytest.approx(0.5, abs=1e-12)


def test_spearman_ties_use_average_ranks():
    # x ranks (1.5, 1.5, 3), y ranks (1, 2, 3). Centered: x (-.5,-.5,1),
    # y (-1,0,1). cov sum = 1.5, |x|^2 = 1.5, |y|^2 = 2 -> 1.5/sqrt(3) = sqrt(3)/2.
    assert spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(math.sqrt(3) / 2, abs=1e-12)


def test_spearman_constant_input_is_nan():
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert math.isnan(spearman([1, 2, 3], [5, 5, 5]))


# ---------- normalized_regret ----------

def test_normalized_regret_qtran_matrix():
    # payoff.max() = 8, mean = -40/9, so denominator = 8 + 40/9 = 112/9.
    # (1,1) has k = 0 agents on the target -> payoff 0 -> regret 8.
    # 8 / (112/9) = 72/112 = 9/14.
    payoff = make_mono_game(2, 3, penalty=12, high=8.0)["payoff"]
    assert normalized_regret(payoff, (1, 1)) == pytest.approx(9 / 14, abs=1e-12)


def test_normalized_regret_optimal_is_zero():
    payoff = make_mono_game(2, 3, penalty=12, high=8.0)["payoff"]
    assert normalized_regret(payoff, (0, 0)) == pytest.approx(0.0, abs=1e-12)


def test_normalized_regret_constant_payoff_raises():
    with pytest.raises(ValueError):
        normalized_regret(np.full((3, 3), 2.0), (0, 0))


# ---------- credit_error scale ----------

def test_credit_error_scaled_l2():
    # diff = (3, 4) -> raw_l2 = 5; scale 2.5 -> 2.
    out = credit_error([3.0, 4.0], [0.0, 0.0], scale=2.5)
    assert out["scaled_l2"] == pytest.approx(2.0, abs=1e-12)
    assert out["raw_l2"] == pytest.approx(5.0, abs=1e-12)


def test_credit_error_without_scale_has_no_scaled_key_and_old_keys_unchanged():
    out = credit_error([3.0, 4.0], [1.0, 1.0])
    assert set(out) == {"normalized_l2", "raw_l2", "sup"}
    # diff = (2, 3): raw = sqrt(13); |phi_true| = sqrt(2); sup = 3.
    assert out["raw_l2"] == pytest.approx(math.sqrt(13), abs=1e-12)
    assert out["normalized_l2"] == pytest.approx(math.sqrt(13 / 2), abs=1e-12)
    assert out["sup"] == pytest.approx(3.0, abs=1e-12)


def test_credit_error_scaled_is_finite_when_phi_true_is_zero():
    # The reason D8 uses payoff.std(): normalized_l2 is NaN here, scaled is not.
    out = credit_error([1.0, 0.0], [0.0, 0.0], scale=4.0)
    assert math.isnan(out["normalized_l2"])
    assert out["scaled_l2"] == pytest.approx(0.25, abs=1e-12)
