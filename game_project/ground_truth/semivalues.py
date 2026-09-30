"""Semivalue solution concepts (Shapley, Banzhaf, LOO) over a precomputed v.
"""

import itertools
from math import comb

import numpy as np

_WEIGHT_SUM_ATOL = 1e-12


def _assert_weights_sum_to_one(weights, n):
    """Sum_s C(n-1, s) * w(s) must equal 1 (normalization condition).
    """
    total = sum(comb(n - 1, s) * weights[s] for s in range(n))
    assert abs(total - 1.0) < _WEIGHT_SUM_ATOL, (
        "weight vector %s for n=%d does not satisfy the semivalue "
        "normalization identity: sum_s C(n-1,s)*w(s) = %.17g, expected 1"
        % (weights, n, total)
    )


def shapley_weights(n):
    """w(s) = s! (n-1-s)! / n!, for s = 0..n-1 (Shapley 1953's coalition-size weight).
    """
    weights = np.array([1.0 / (n * comb(n - 1, s)) for s in range(n)])
    _assert_weights_sum_to_one(weights, n)
    return weights


def banzhaf_weights(n):
    """w(s) = 1 / 2^(n-1) for every s.
    """
    weights = np.full(n, 1.0 / (2 ** (n - 1)))
    _assert_weights_sum_to_one(weights, n)
    return weights


def loo_weights(n):
    """w(s) = 1 if s == n-1 else 0: only the grand-coalition-minus-i marginal counts.
    agent i's credit is just v(N) - v(N\\{i}).
    """
    weights = np.zeros(n)
    weights[n - 1] = 1.0
    _assert_weights_sum_to_one(weights, n)
    return weights


def semivalue(v, n, weights):
    """phi_i = sum_{S subset of N\\{i}} weights[|S|] * (v[S+i] - v[S]), for every i.
    """
    v = np.asarray(v)
    assert v.shape == (2 ** n,), (
        "v has shape %s, expected %s for n=%d" % (v.shape, (2 ** n,), n)
    )

    phi = np.zeros(n)
    for i in range(n):
        bit_i = 1 << i
        for S in range(2 ** n):
            # Only coalitions S that do NOT already contain i are valid
            if S & bit_i:
                continue
            s = bin(S).count("1")  # |S|, i.e. how many agents are in S
            phi[i] += weights[s] * (v[S | bit_i] - v[S])

    assert phi.shape == (n,), "phi has shape %s, expected (%d,)" % (phi.shape, n)
    return phi


def shapley(v, n):
    return semivalue(v, n, shapley_weights(n))


def banzhaf(v, n):
    return semivalue(v, n, banzhaf_weights(n))


def loo(v, n):
    return semivalue(v, n, loo_weights(n))


def shapley_by_permutations(v, n):
    """Shapley value by brute-force averaging over all n! orderings.
    """
    v = np.asarray(v)
    assert v.shape == (2 ** n,), (
        "v has shape %s, expected %s for n=%d" % (v.shape, (2 ** n,), n)
    )

    phi = np.zeros(n)
    n_orderings = 0
    for order in itertools.permutations(range(n)):
        prefix = 0  # bitmask of agents seen so far in this ordering
        for agent in order:
            phi[agent] += v[prefix | (1 << agent)] - v[prefix]
            prefix |= 1 << agent
        n_orderings += 1

    phi /= n_orderings
    assert phi.shape == (n,), "phi has shape %s, expected (%d,)" % (phi.shape, n)
    return phi
