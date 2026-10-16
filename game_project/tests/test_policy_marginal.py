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
- test_epsilon_greedy_interior_scales_uniform_credit: the interior case,
  0 < epsilon < 1. Derivation (written out again at the call site): in an
  additive (lambda=0) game, payoff(a) = sum_j f_j[a_j], so for any baseline
  with independent per-agent marginals, v(S) = sum_{j in S} max_a f_j[a] +
  sum_{j not in S} E_{B_j}[f_j] (outside agents are fixed by the max, inside
  agents are each maxed independently). The marginal contribution of adding
  agent i to any coalition S not containing i is then v(S+i) - v(S) =
  max f_i - E_{B_i}[f_i], independent of S -- so every semivalue (whose
  weights satisfy sum_s C(n-1,s)*w(s) = 1) gives phi_i = max f_i -
  E_{B_i}[f_i], regardless of which one. Substituting the epsilon-greedy
  mixture, E_{B_i}[f_i] = (1-epsilon)*max f_i + epsilon*mean f_i, so
  phi_i = epsilon*(max f_i - mean f_i) -- exactly epsilon times the
  epsilon=1 (uniform) case's own phi_i = max f_i - mean f_i.
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


def test_epsilon_greedy_interior_scales_uniform_credit():
    # n=3, |A|=3, hand-written additive (lambda=0) payoff: payoff(a) =
    # f[0,a0] + f[1,a1] + f[2,a2]. Each row's max is unique (strictly
    # greater than that row's other two entries), so the greedy joint
    # action is unambiguous.
    f = np.array([
        [1.0, 2.0, 0.0],   # agent 0: unique max 2.0 at action 1
        [0.0, 3.0, 1.0],   # agent 1: unique max 3.0 at action 1
        [2.0, 2.0, 5.0],   # agent 2: unique max 5.0 at action 2
    ])
    n, n_actions = f.shape
    payoff = f[0][:, None, None] + f[1][None, :, None] + f[2][None, None, :]
    assert [int(np.argmax(f[i])) for i in range(n)] == [1, 1, 2]

    # q_tables == f, so each agent's own greedy action matches the row
    # argmax used in the derivation above.
    epsilon = 0.05
    B = policy_marginal(f, "epsilon_greedy", epsilon)
    v = coalition_values(payoff, B)

    # By hand (derivation in the module docstring): phi_i =
    # epsilon * (max_a f[i, a] - mean_a f[i, a]).
    phi_expected = epsilon * (f.max(axis=1) - f.mean(axis=1))
    np.testing.assert_allclose(
        phi_expected, np.array([0.05, 0.08333333333333333, 0.1]), atol=1e-12
    )

    for weights in (shapley_weights(n), banzhaf_weights(n), loo_weights(n)):
        phi = semivalue(v, n, weights)
        np.testing.assert_allclose(phi, phi_expected, atol=1e-12)
