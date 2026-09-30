"""Exact coalition values v(S) for all 2**n coalitions, by full enumeration.
"""

from string import ascii_lowercase

import numpy as np

from ground_truth.baselines import check_baseline

_MONOTONE_ATOL = 1e-12


def _axis_letters(n):
    """One distinct letter per agent axis, e.g. n=3 -> 'abc'.
    """
    assert n <= len(ascii_lowercase), "einsum subscripts support at most 26 agents"
    return ascii_lowercase[:n]


def coalition_values(payoff, baseline):
    """Exact v(S) for every coalition S, indexed by bitmask.

    Parameters:
    payoff : ndarray, shape (n_actions,) * n
    baseline : ndarray, shape (n, n_actions)

    Returns:
    ndarray, shape (2**n,)
        v[S] for the coalition whose membership is the bitmask S.
    """
    payoff = np.asarray(payoff)
    baseline = np.asarray(baseline)

    if baseline.ndim != 2:
        raise ValueError(
            "baseline must be 2-D of shape (n_agents, n_actions), got ndim %d "
            "with shape %s." % (baseline.ndim, baseline.shape)
        )
    n, n_actions = baseline.shape

    # this verifies that the baseline is actually a valid probability distribution.
    check_baseline(baseline, n, n_actions)

    if payoff.shape != (n_actions,) * n:
        raise ValueError(
            "payoff has shape %s but the baseline implies %d agents with %d "
            "actions, i.e. payoff should be %s."
            % (payoff.shape, n, n_actions, (n_actions,) * n)
        )

    letters = _axis_letters(n)
    n_coalitions = 2 ** n
    v = np.empty(n_coalitions)

    for S in range(n_coalitions):
        # Membership by bitmask: bit i set if agent i is in the coalition.
        inside = [i for i in range(n) if (S >> i) & 1]
        outside = [j for j in range(n) if not (S >> j) & 1]

        # calculating value for coalition while outsiders following the baseline.
        in_subs = [letters] + [letters[j] for j in outside]
        out_subs = "".join(letters[i] for i in inside)
        subs = ",".join(in_subs) + "->" + out_subs
        operands = [payoff] + [baseline[j] for j in outside]

        expected = np.einsum(subs, *operands)

        assert expected.shape == (n_actions,) * len(inside), (
            "coalition %d: marginalized tensor has shape %s, expected %s"
            % (S, expected.shape, (n_actions,) * len(inside))
        )

        v[S] = expected.max()

    assert v.shape == (n_coalitions,), (
        "v has shape %s, expected %s" % (v.shape, (n_coalitions,))
    )

    # Grand coalition: the max number of the agents in the coalition
    full_mask = n_coalitions - 1
    # Does the value of the grand coalition equal the maximum possible payoff?
    assert v[full_mask] == payoff.max(), (
        "v[full coalition] = %.17g but payoff.max() = %.17g; they must be "
        "bitwise equal because no marginalization happens for the grand "
        "coalition" % (v[full_mask], payoff.max())
    )

    # Monotonicity. varify that for example: v({1, 2}) >= v({1})
    # What happens if I add agent i?
    for S in range(n_coalitions):
        for i in range(n):
            # Don't add an agent who's already there
            if (S >> i) & 1:
                continue 
            # Take coalition S and turn agent i's bit to 1.(mask)
            S_with_i = S | (1 << i)
            if v[S] > v[S_with_i] + _MONOTONE_ATOL:
                raise AssertionError(
                    "monotonicity violated: v[%d] = %.17g > v[%d] = %.17g "
                    "(adding agent %d), by %.3e. v(S) must not decrease when a "
                    "coalition gains a member under the max-with-baseline "
                    "definition."
                    % (S, v[S], S_with_i, v[S_with_i], i, v[S] - v[S_with_i])
                )

    return v
