"""Week 2 Task 8 -- tests for ground_truth/baselines.py:policy_marginal.

Both expected values are derived by hand from the spec's math, not from
running policy_marginal and copying its output (CLAUDE.md's test-
independence rule):

- test_greedy_delta_zero_semivalue_at_optimum: F5 says the greedy-delta
  baseline (param=0) makes v(S) constant whenever the decentralized greedy
  joint action is the payoff's global optimum. The payoff and q_tables
  below are hand-picked (not from games/generator.py) so that argmax(q_i)
  is literally the payoff's argmax, and the "v(S) = if payoff.max() for every
  S" argument is written out below rather than taken on faith.
- test_epsilon_one_equals_uniform: read directly off policy_marginal's own
  formula in the task spec -- epsilon=1 reduces B_i to (0*delta + 1*uniform)
  -- so it's an algebraic identity in the spec, not a repo-derived value.
"""

import numpy as np

from ground_truth.baselines import policy_marginal, uniform
from ground_truth.enumerate import coalition_values
from ground_truth.semivalues import banzhaf_weights, loo_weights, semivalue, shapley_weights


def test_greedy_delta_zero_semivalue_at_optimum():
    # n=2, |A|=2. Global max is payoff[1, 1] = 10.0, strictly greater than
    # every other entry, so no tie-breaking ambiguity.
    payoff = np.array([[1.0, 2.0],
                        [3.0, 10.0]])
    assert payoff.max() == payoff[1, 1]

    # q_tables chosen so each agent's own argmax is action 1 -- i.e. the
    # decentralized greedy joint action is (1, 1), the global optimum.
    q_tables = np.array([[0.0, 1.0],
                          [0.0, 1.0]])

    B = policy_marginal(q_tables, "epsilon_greedy", 0.0)
    # By hand: epsilon=0 collapses the mixture to a point mass on each
    # agent's greedy action, i.e. B = [[0, 1], [0, 1]].
    np.testing.assert_array_equal(B, np.array([[0.0, 1.0], [0.0, 1.0]]))

    v = coalition_values(payoff, B)
    # By hand (F5's argument): with every excluded agent fixed at its
    # greedy action, v(S) = max_{a_S} payoff(a_S, greedy_{-S}). Because
    # greedy = (1, 1) is the payoff's global argmax, no choice of a_S can
    # beat it and a_S = greedy_S attains it, so v(S) = payoff.max() = 10.0
    # for every one of the 2**2 = 4 coalitions, including S = empty set
    # (v = payoff[1,1] directly, no max taken) and S = {0, 1} (v =
    # payoff[1,1] exactly, no marginalization).
    np.testing.assert_allclose(v, np.full(4, 10.0), atol=1e-12)

    # Hence every semivalue phi_i = sum_S weight(|S|) * (v(S+i) - v(S))
    # = sum_S weight(|S|) * 0 = 0, for any weight vector -- Shapley,
    # Banzhaf, and LOO alike.
    for weights in (shapley_weights(2), banzhaf_weights(2), loo_weights(2)):
        phi = semivalue(v, 2, weights)
        np.testing.assert_allclose(phi, np.zeros(2), atol=1e-12)


def test_epsilon_one_equals_uniform():
    rng = np.random.default_rng(0)
    n, n_actions = 4, 5
    q_tables = rng.normal(size=(n, n_actions))

    B = policy_marginal(q_tables, "epsilon_greedy", 1.0)
    np.testing.assert_array_equal(B, uniform(n, n_actions))
