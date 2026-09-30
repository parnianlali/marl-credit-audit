"""Correctness oracle for the ground-truth computation.
"""

import itertools

import numpy as np
import pytest

from games.generator import (
    make_game,
    make_mono_game,
    marginal_greedy_action,
    joint_argmax,
    sample_reward,
)
from ground_truth.baselines import fixed_action, uniform, fixed_default
from ground_truth.enumerate import coalition_values
from ground_truth.semivalues import shapley, banzhaf, loo, shapley_by_permutations

SEEDS = list(range(10))

# Shared parametrization grid for the axiom tests below. Small n/|A| keep
# coalition enumeration (2**n) and permutation enumeration (n!) cheap while
# still covering n=2 and n=3.
GRID = [(2, 2), (2, 3), (3, 2), (3, 3)]
LAMBDAS = [0.0, 1.0, 2.0]
BASELINE_NAMES = ["fixed", "uniform", "asymmetric"]


def brute_force_coalition_values(payoff, baseline):
    """v(S) for every coalition, by explicit enumeration. Reference implementation.
    """
    n, n_actions = baseline.shape
    v = np.empty(2 ** n)

    for S in range(2 ** n):
        inside = [i for i in range(n) if (S >> i) & 1]
        outside = [j for j in range(n) if not (S >> j) & 1]
        # common pattern to searching for a maximum value
        best = -np.inf
        # generate every possible action combinations to coalition members.
        for a_in in itertools.product(range(n_actions), repeat=len(inside)):
            total = 0.0
            # Expectation over every assignment to the agents OUTSIDE it,
            for a_out in itertools.product(range(n_actions), repeat=len(outside)):
                a = [None] * n
                for pos, i in enumerate(inside):
                    a[i] = a_in[pos]
                weight = 1.0
                for pos, j in enumerate(outside):
                    a[j] = a_out[pos]
                    weight *= baseline[j, a_out[pos]]
                total += weight * payoff[tuple(a)]
            best = max(best, total)
        v[S] = best

    return v


def asymmetric_baseline(n, n_actions, seed=0):
    """A baseline with a DIFFERENT distribution for every agent.
    """
    rng = np.random.default_rng(1000 + 10 * n + n_actions + seed)
    raw = rng.random((n, n_actions)) + 0.05  # +0.05 To make sure no probability becomes zero.
    # normalizes each row so it sums to 1
    return fixed_default(raw / raw.sum(axis=1, keepdims=True))


def _baseline_by_name(name, n, n_actions):
    """Look up one of the three baselines used across the axiom grid by name."""
    if name == "fixed":
        return fixed_action(n, n_actions)
    elif name == "uniform":
        return uniform(n, n_actions)
    else:
        return asymmetric_baseline(n, n_actions)


@pytest.mark.parametrize("n,n_actions", [(2, 2), (2, 3), (3, 2), (3, 3)])
@pytest.mark.parametrize("lam", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("baseline_name", ["fixed", "uniform", "asymmetric"])
def test_coalition_values_match_bruteforce(n, n_actions, lam, baseline_name):
    """coalition_values agrees with the independent reference on every coalition."""
    if baseline_name == "fixed":
        B = fixed_action(n, n_actions)
    elif baseline_name == "uniform":
        B = uniform(n, n_actions)
    else:
        B = asymmetric_baseline(n, n_actions)

    for seed in SEEDS:
        game = make_game(n, n_actions, lam, seed)
        fast = coalition_values(game["payoff"], B)
        slow = brute_force_coalition_values(game["payoff"], B)

        assert fast.shape == (2 ** n,)
        np.testing.assert_allclose(
            fast, slow, atol=1e-12,
            err_msg="coalition_values disagrees with brute force at "
                    "n=%d, |A|=%d, lam=%g, baseline=%s, seed=%d"
                    % (n, n_actions, lam, baseline_name, seed),
        )


# ---------------------------------------------------------------------------
# Task 5 axiom tests. Every expected value below comes from the definitions
# and closed forms in the task spec itself, never by running semivalues.py
# and reading off what it produces.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,n_actions", GRID)
@pytest.mark.parametrize("lam", LAMBDAS)
@pytest.mark.parametrize("baseline_name", BASELINE_NAMES)
def test_grand_coalition_equals_max(n, n_actions, lam, baseline_name):
    """v(full coalition) is a plain max (no agent is marginalized out), so it
    must equal payoff.max() exactly, not just approximately."""
    B = _baseline_by_name(baseline_name, n, n_actions)
    for seed in SEEDS:
        game = make_game(n, n_actions, lam, seed)
        v = coalition_values(game["payoff"], B)
        assert v[2 ** n - 1] == game["payoff"].max(), (
            "v[full] = %.17g but payoff.max() = %.17g at n=%d, |A|=%d, "
            "lam=%g, baseline=%s, seed=%d"
            % (v[2 ** n - 1], game["payoff"].max(), n, n_actions, lam, baseline_name, seed)
        )


@pytest.mark.parametrize("n,n_actions", GRID)
@pytest.mark.parametrize("lam", LAMBDAS)
@pytest.mark.parametrize("baseline_name", BASELINE_NAMES)
def test_monotone_in_coalition(n, n_actions, lam, baseline_name):
    """v(S) must not decrease when S gains a member, under the
    max-over-coalition/expectation-under-baseline definition."""
    B = _baseline_by_name(baseline_name, n, n_actions)
    for seed in SEEDS:
        game = make_game(n, n_actions, lam, seed)
        v = coalition_values(game["payoff"], B)
        for S in range(2 ** n):
            for i in range(n):
                if (S >> i) & 1:
                    continue
                assert v[S] <= v[S | (1 << i)] + 1e-10, (
                    "monotonicity violated at n=%d, |A|=%d, lam=%g, baseline=%s, "
                    "seed=%d: v[%d]=%.17g > v[%d]=%.17g"
                    % (n, n_actions, lam, baseline_name, seed, S, v[S], S | (1 << i), v[S | (1 << i)])
                )


@pytest.mark.parametrize("n,n_actions", GRID)
@pytest.mark.parametrize("lam", LAMBDAS)
@pytest.mark.parametrize("baseline_name", BASELINE_NAMES)
def test_shapley_efficiency(n, n_actions, lam, baseline_name):
    """Efficiency axiom: sum_i phi_i == v(N) - v(empty). Deliberately keeps
    v(empty) rather than assuming it's zero (Decision table, Week 1 spec)."""
    B = _baseline_by_name(baseline_name, n, n_actions)
    for seed in SEEDS:
        game = make_game(n, n_actions, lam, seed)
        v = coalition_values(game["payoff"], B)
        phi = shapley(v, n)
        np.testing.assert_allclose(
            phi.sum(), v[-1] - v[0], atol=1e-10,
            err_msg="Shapley efficiency violated at n=%d, |A|=%d, lam=%g, "
                    "baseline=%s, seed=%d" % (n, n_actions, lam, baseline_name, seed),
        )


def test_banzhaf_not_efficient_on_synergistic_game():
    """Banzhaf values have no reason to sum to v(N) - v(empty) once there is
    real synergy; if they do, the test suite has degenerated (everything
    collapsing to the same answer)."""
    n, n_actions, lam = 3, 3, 1.0
    B = uniform(n, n_actions)
    for seed in SEEDS:
        game = make_game(n, n_actions, lam, seed)
        v = coalition_values(game["payoff"], B)
        phi = banzhaf(v, n)
        gap = abs(phi.sum() - (v[-1] - v[0]))
        assert gap > 1e-6, (
            "banzhaf is suspiciously (near-)efficient at seed=%d: gap=%.3e" % (seed, gap)
        )


def test_dummy_player():
    """If agent 2's action cannot affect the payoff at all, then for every
    coalition S not containing 2, v(S u {2}) == v(S) exactly (maximizing or
    marginalizing over an irrelevant axis changes nothing). Every semivalue
    is a weighted sum of these all-zero marginal contributions, so Shapley,
    Banzhaf, and LOO must all give agent 2 exactly 0 credit -- independent of
    the weights used, and independent of anything else about the game."""
    n, n_actions, lam = 3, 3, 1.0
    B = uniform(n, n_actions)
    for seed in SEEDS:
        game = make_game(n, n_actions, lam, seed)
        # Force every slice along agent 2's axis to be identical, so the
        # payoff provably does not depend on agent 2's action.
        payoff = np.broadcast_to(game["payoff"][:, :, :1], game["payoff"].shape).copy()
        v = coalition_values(payoff, B)
        for credit_fn, name in [(shapley, "shapley"), (banzhaf, "banzhaf"), (loo, "loo")]:
            phi = credit_fn(v, n)
            assert abs(phi[2]) < 1e-10, (
                "%s gives dummy agent 2 credit %.3e at seed=%d, expected exactly 0"
                % (name, phi[2], seed)
            )


def test_symmetry():
    """f[0] = f[1] and g symmetrized under the 0<->1 axis swap must force
    phi_0 == phi_1 for every semivalue. Averaging an already-orthogonalized G
    with its own 0<->1 swap keeps it orthogonalized (the cross term reduces
    to another agent's zero-mean condition), so no re-normalization is
    needed here; uniform is used as the baseline because it's automatically
    symmetric across agents, so it needs no separate symmetrization."""
    n, n_actions, lam = 3, 3, 1.0
    B = uniform(n, n_actions)
    for seed in SEEDS:
        game = make_game(n, n_actions, lam, seed)
        f = game["f"].copy()
        f[1] = f[0]
        G = game["g"]
        G_sym = (G + np.swapaxes(G, 0, 1)) / 2.0

        payoff = np.zeros((n_actions,) * n)
        for i in range(n):
            shape = [1] * n
            shape[i] = n_actions
            payoff = payoff + f[i].reshape(shape)
        payoff = payoff + lam * G_sym

        v = coalition_values(payoff, B)
        for credit_fn, name in [(shapley, "shapley"), (banzhaf, "banzhaf"), (loo, "loo")]:
            phi = credit_fn(v, n)
            np.testing.assert_allclose(
                phi[0], phi[1], atol=1e-10,
                err_msg="%s not symmetric under 0<->1 swap at seed=%d: phi=%s"
                        % (name, seed, phi),
            )


@pytest.mark.parametrize("n,n_actions", GRID)
@pytest.mark.parametrize("baseline_name", BASELINE_NAMES)
def test_permutation_equivariance(n, n_actions, baseline_name):
    """Relabeling agents (permuting payoff axes and baseline rows) and
    recomputing must relabel the credit vector identically. The
    highest-frequency real bug class here is an axis-order mistake.

    Parametrized over all three baselines, not just "fixed": fixed_action's
    rows are identical across agents, so B[list(perm), :] == B and permuting
    it is a no-op -- that case alone cannot detect a baseline-row-to-payoff-
    axis mispairing, which is exactly the bug this test exists to catch. The
    asymmetric baseline (a different distribution per agent) is what actually
    exercises that failure mode."""
    lam = 1.0
    B = _baseline_by_name(baseline_name, n, n_actions)
    for seed in SEEDS:
        game = make_game(n, n_actions, lam, seed)
        payoff = game["payoff"]
        v = coalition_values(payoff, B)
        phi = shapley(v, n)

        for perm in itertools.permutations(range(n)):
            payoff_perm = np.transpose(payoff, perm)
            B_perm = B[list(perm), :]
            v_perm = coalition_values(payoff_perm, B_perm)
            phi_perm = shapley(v_perm, n)
            expected = phi[list(perm)]
            np.testing.assert_allclose(
                phi_perm, expected, atol=1e-10,
                err_msg="permutation equivariance violated for perm=%s at "
                        "n=%d, |A|=%d, baseline=%s, seed=%d"
                        % (perm, n, n_actions, baseline_name, seed),
            )


@pytest.mark.parametrize("n,n_actions", GRID)
@pytest.mark.parametrize("baseline_name", BASELINE_NAMES)
def test_lambda_zero_closed_form(n, n_actions, baseline_name):
    """At lam=0, r(a) = sum_i f_i(a_i) and the baseline is a product
    distribution, so agent i's marginal contribution is constant across
    every coalition: delta_i = max_{a_i} f_i(a_i) - E_{B_i}[f_i]. This is
    computed directly from game['f'] and B, never from coalition_values or
    semivalue, so it is a genuinely independent oracle."""
    B = _baseline_by_name(baseline_name, n, n_actions)
    for seed in SEEDS:
        game = make_game(n, n_actions, 0.0, seed)
        v = coalition_values(game["payoff"], B)
        phi = shapley(v, n)
        delta = np.array([
            game["f"][i].max() - B[i] @ game["f"][i]
            for i in range(n)
        ])
        np.testing.assert_allclose(
            phi, delta, atol=1e-10,
            err_msg="Shapley does not match the lam=0 closed form at n=%d, "
                    "|A|=%d, baseline=%s, seed=%d" % (n, n_actions, baseline_name, seed),
        )


@pytest.mark.parametrize("n,n_actions", GRID)
@pytest.mark.parametrize("baseline_name", BASELINE_NAMES)
def test_all_semivalues_agree_at_lambda_zero(n, n_actions, baseline_name):
    """Every semivalue's weight vector is normalized, so at lam=0 (where
    marginal contributions are constant across coalitions, per the closed
    form above) Shapley, Banzhaf, and LOO must all agree exactly."""
    B = _baseline_by_name(baseline_name, n, n_actions)
    for seed in SEEDS:
        game = make_game(n, n_actions, 0.0, seed)
        v = coalition_values(game["payoff"], B)
        phi_s = shapley(v, n)
        phi_b = banzhaf(v, n)
        phi_l = loo(v, n)
        np.testing.assert_allclose(
            phi_s, phi_b, atol=1e-10,
            err_msg="shapley/banzhaf disagree at lam=0, n=%d, |A|=%d, "
                    "baseline=%s, seed=%d" % (n, n_actions, baseline_name, seed),
        )
        np.testing.assert_allclose(
            phi_s, phi_l, atol=1e-10,
            err_msg="shapley/loo disagree at lam=0, n=%d, |A|=%d, "
                    "baseline=%s, seed=%d" % (n, n_actions, baseline_name, seed),
        )


@pytest.mark.parametrize("n_actions", [2, 3])
@pytest.mark.parametrize("lam", [1.0, 2.0])
def test_shapley_matches_permutation_bruteforce(n_actions, lam):
    """shapley_by_permutations is a genuinely independent code path
    (average marginal contributions over all n! orderings); it must agree
    with the coalition-weight formula at n=4, lam > 0."""
    n = 4
    B = uniform(n, n_actions)
    for seed in SEEDS:
        game = make_game(n, n_actions, lam, seed)
        v = coalition_values(game["payoff"], B)
        np.testing.assert_allclose(
            shapley(v, n), shapley_by_permutations(v, n), atol=1e-10,
            err_msg="shapley disagrees with permutation bruteforce at n=4, "
                    "|A|=%d, lam=%g, seed=%d" % (n_actions, lam, seed),
        )


def test_semivalues_disagree_at_high_lambda():
    """At high synergy, Shapley/Banzhaf/LOO have no reason to agree; if they
    do (beyond float noise) the test suite has degenerated to one trivial
    answer for every solution concept."""
    n, n_actions, lam = 3, 3, 3.0
    B = uniform(n, n_actions)
    for seed in SEEDS:
        game = make_game(n, n_actions, lam, seed)
        v = coalition_values(game["payoff"], B)
        phi_s = shapley(v, n)
        phi_b = banzhaf(v, n)
        phi_l = loo(v, n)
        max_gap = max(
            np.abs(phi_s - phi_b).max(),
            np.abs(phi_b - phi_l).max(),
            np.abs(phi_s - phi_l).max(),
        )
        assert max_gap > 1e-6, (
            "semivalues suspiciously agree at high lambda, seed=%d: max_gap=%.3e"
            % (seed, max_gap)
        )


def test_qtran_counterexample_reproduced():
    """make_mono_game(2, 3, penalty=12, target=0, high=8) is exactly the
    published QTRAN counterexample matrix:
        (0,0) -> 8, exactly one agent on 0 -> -12, neither on 0 -> 0.
    Hand-computed under uniform(2, 3): a single agent's expected payoff for
    playing action 0 is (8 - 12 - 12) / 3 = -16/3 ~= -5.33; for action 1 or 2
    it's (-12 + 0 + 0) / 3 = -4. Since -4 > -16/3, and np.argmax breaks the
    1-vs-2 tie toward the lower index, marginal_greedy_action must return
    [1, 1], while joint_argmax must return the unique global optimum (0, 0).
    This reproduces the classic marginal-greedy mismatch as a free external
    check on games/generator.py's diagnostics."""
    game = make_mono_game(2, 3, penalty=12, target=0, high=8.0)
    payoff = game["payoff"]
    B = uniform(2, 3)

    greedy = marginal_greedy_action(payoff, B)
    optimum = joint_argmax(payoff)

    np.testing.assert_array_equal(greedy, np.array([1, 1]))
    assert optimum == (0, 0)
    assert tuple(greedy) != optimum, "expected a marginal-greedy mismatch"


def test_hand_computed_oracle_mono_game():
    """Independent, hand-derived oracle -- computed on paper, external to
    this repo, before this test was written. Every other lam > 0 check in
    this file cross-checks one code path in this repo against another
    (coalition_values vs brute_force_coalition_values, shapley vs
    shapley_by_permutations); none of them is anchored to a value that
    didn't come from running this code. This test is.

    Game: make_mono_game(2, 3, penalty=12, target=0, high=8.0) -- the QTRAN
    counterexample matrix, small enough (n=2) to hand-enumerate every
    coalition.

    Under uniform(2, 3), each outside agent contributes its average row
    (1/3, 1/3, 1/3):
        v(empty) = mean of all 9 entries = (8 - 12*4) / 9 = -40/9
        v({0})   = max_a0 mean_a1[payoff(a0,a1)]
                 = max(-16/3, -4, -4) = -4   (agent 0 plays 1 or 2, not 0)
        v({1})   = -4, by the symmetric argument
        v({0,1}) = payoff.max() = 8
    At n=2 the Shapley and Banzhaf weight vectors are identical (both put
    weight 1/2 on s=0 and s=1), so they must coincide exactly:
        phi_0 = 1/2*(v({0})-v(empty)) + 1/2*(v({0,1})-v({1}))
              = 1/2*(-4 + 40/9) + 1/2*12 = 2/9 + 6 = 56/9
        Shapley = Banzhaf = (56/9, 56/9), by symmetry of the game in the
        two agents. LOO only counts the grand-coalition-minus-i marginal:
        LOO_0 = v({0,1}) - v({1}) = 8 - (-4) = 12, so LOO = (12, 12) --
        the only one of the three that isn't degenerate with the others
        here, which is why all three are asserted.

    Under fixed_action(2, 3) (delta at action 0 for both agents), every
    coalition's out-of-coalition agents are pinned to action 0, which is
    already the joint argmax, so v(S) = 8 for all four S (see the
    concept_spread degenerate case in eval/week1_smoke.py for the general
    argument). Every marginal contribution is then exactly 0, so all three
    semivalues are exactly (0, 0).
    """
    n, n_actions = 2, 3
    game = make_mono_game(n, n_actions, penalty=12, target=0, high=8.0)
    payoff = game["payoff"]

    # ---- uniform baseline ----
    B = uniform(n, n_actions)
    v = coalition_values(payoff, B)
    v_expected = np.array([-40.0 / 9.0, -4.0, -4.0, 8.0])
    np.testing.assert_allclose(
        v, v_expected, atol=1e-12,
        err_msg="coalition values under uniform(2,3) do not match the "
                "hand-computed oracle",
    )

    shapley_expected = np.array([56.0 / 9.0, 56.0 / 9.0])
    loo_expected = np.array([12.0, 12.0])
    np.testing.assert_allclose(shapley(v, n), shapley_expected, atol=1e-12,
                                err_msg="shapley disagrees with hand-computed oracle")
    np.testing.assert_allclose(banzhaf(v, n), shapley_expected, atol=1e-12,
                                err_msg="banzhaf disagrees with hand-computed oracle "
                                        "(must equal shapley at n=2)")
    np.testing.assert_allclose(loo(v, n), loo_expected, atol=1e-12,
                                err_msg="loo disagrees with hand-computed oracle")

    # ---- fixed baseline ----
    Bf = fixed_action(n, n_actions)
    v_fixed = coalition_values(payoff, Bf)
    np.testing.assert_allclose(
        v_fixed, np.full(4, 8.0), atol=1e-12,
        err_msg="coalition values under fixed_action(2,3) do not match the "
                "hand-computed oracle (expected constant 8 for every S)",
    )

    zero = np.zeros(2)
    np.testing.assert_allclose(shapley(v_fixed, n), zero, atol=1e-12)
    np.testing.assert_allclose(banzhaf(v_fixed, n), zero, atol=1e-12)
    np.testing.assert_allclose(loo(v_fixed, n), zero, atol=1e-12)


def test_baselines_change_ground_truth():
    """Changing the baseline must change at least some coalition values --
    guards against enumerate.py silently ignoring its baseline argument."""
    n, n_actions, lam = 3, 3, 1.0
    for seed in SEEDS[:3]:
        game = make_game(n, n_actions, lam, seed)
        v_fixed = coalition_values(game["payoff"], fixed_action(n, n_actions))
        v_uniform = coalition_values(game["payoff"], uniform(n, n_actions))
        assert not np.allclose(v_fixed, v_uniform, atol=1e-10), (
            "coalition values identical under fixed vs uniform baseline at seed=%d" % seed
        )


@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("n_actions", [2, 3])
def test_shapes(n, n_actions):
    """Guards against silent broadcasting: every output must have the shape
    its docstring promises, at every supported n and |A|."""
    lam, seed = 1.0, 0
    game = make_game(n, n_actions, lam, seed)
    B = uniform(n, n_actions)
    v = coalition_values(game["payoff"], B)
    assert v.shape == (2 ** n,)
    assert shapley(v, n).shape == (n,)
    assert banzhaf(v, n).shape == (n,)
    assert loo(v, n).shape == (n,)


def test_noise_not_in_ground_truth():
    """Noise is applied at sample time only (Decision table, Week 1 spec):
    calling sample_reward must never mutate the payoff tensor, and the
    ground truth computed from it must be identical before and after."""
    n, n_actions, lam, seed = 3, 3, 1.0, 0
    game = make_game(n, n_actions, lam, seed)
    B = uniform(n, n_actions)
    payoff_before = game["payoff"].copy()
    v_before = coalition_values(payoff_before, B)

    rng = np.random.default_rng(123)
    for _ in range(20):
        joint_action = tuple(int(a) for a in rng.integers(0, n_actions, size=n))
        sample_reward(game["payoff"], joint_action, sigma=1.0, rng=rng)

    assert np.array_equal(game["payoff"], payoff_before), "sample_reward mutated payoff"
    v_after = coalition_values(game["payoff"], B)
    assert np.array_equal(v_after, v_before), "ground truth changed after noisy sampling"
