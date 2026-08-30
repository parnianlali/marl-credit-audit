"""Counterfactual baselines: what agents OUTSIDE a coalition are assumed to do.

A baseline is a single ndarray `B` of shape `(n, n_actions)`, row-stochastic:
row `j` is agent `j`'s action distribution, entries are >= 0, and each row sums
to 1. Agents are assumed independent, so the joint distribution over any subset
of agents is the product of their rows.

WHY ONE MATRIX, AND WHY THAT MATTERS
------------------------------------
`Project_Structure.md` Section 5 lists four options for what out-of-coalition
agents do -- no-op, uniform random, policy-marginal, and fixed default policy --
and treats the choice as a first-class experimental variable, quite possibly the
location of the project's headline finding. Representing a baseline as one
row-stochastic matrix turns all four of those options into DATA rather than
code paths. Nothing downstream branches on which one you picked.

THIS INTERFACE IS WHAT POLICY-MARGINAL WILL USE IN WEEK 2. When the learner
arrives, it supplies `B` -- its per-agent action distribution -- and passes it to
exactly the same functions. NOTHING ABOUT `enumerate.py` CHANGES WHEN THAT
HAPPENS. That is the whole point of fixing the representation now, before there
is anything to be tempted to special-case.

NAMING
------
`fixed_action` builds a FIXED DESIGNATED ACTION baseline. It is deliberately not
called a "no-op" baseline. In a randomly generated game, action 0 is not a null
action, does not mean "do nothing", and has no special status of any kind -- it
is simply the action at index 0. Calling it a no-op would be a false claim, and a
false claim in the code becomes a false claim in the paper.

THIS WEEK'S TWO BASELINES
-------------------------
Fixed designated action, and uniform random. Policy-marginal is not here because
it needs a policy and none exists until Week 2.

A caveat worth knowing while reading this file: under the UNIFORM baseline, the
game family's orthogonalized synergy term is invisible to singleton coalitions by
construction, while under the FIXED-ACTION baseline it is not. See
`reports/notes_baselines.md`.
"""

import numpy as np

# Tolerance for both the row-sum and the non-negativity checks.
#
# Exact `== 1` is unusable: summing n_actions floats leaves error of roughly
# n_actions * 2.2e-16 (~7e-16 at |A| = 3), so even a correctly built uniform
# baseline misses 1.0 by a few ULP at some sizes. 1e-12 sits about three orders
# above realistic float error and eight orders below any genuine normalization
# bug, and it matches the tolerance already used for the payoff reconstruction
# assertion in games/generator.py.
#
# Applying the same tolerance to non-negativity means a -1e-17 round-off
# artifact is not reported as a negative probability, while a real negative
# entry still is.
_TOL = 1e-12


def check_baseline(B, n, n_actions):
    """Validate that `B` is a legal baseline for `n` agents with `n_actions` actions.

    Checks exactly the three conditions in the Week 1 task spec: shape, all
    entries non-negative, every row sums to 1.

    Raises ValueError naming the failed condition, the offending row, and the
    offending value. Returns `B` unchanged on success, so it can be used inline.

    WHY IT RAISES RATHER THAN RETURNING A BOOL: a validator that returns False
    and gets ignored at one call site is worse than no validator at all. A
    malformed baseline does not crash anything downstream -- it flows into
    `coalition_values` and produces credit vectors that look entirely reasonable
    and are wrong. That is the "plausible-looking wrong ground truth" failure
    mode named in `Project_Structure.md` Section 13, and the only defence is to
    make the malformed case impossible to ignore.

    This function is the SINGLE place row-stochasticity is enforced.
    `games/generator.py:marginal_greedy_action` deliberately shape-asserts only
    and defers to this.
    """
    B = np.asarray(B)

    if B.shape != (n, n_actions):
        raise ValueError(
            "baseline has shape %s, expected (%d, %d) = (n_agents, n_actions). "
            "A baseline is one row per agent, one column per action."
            % (B.shape, n, n_actions)
        )

    # Report the single worst offender rather than "some entry is negative", so
    # the caller can find it. argmin over the flattened array, then unravel.
    if B.min() < -_TOL:
        worst = np.unravel_index(int(np.argmin(B)), B.shape)
        raise ValueError(
            "baseline has a negative entry at [agent %d, action %d]: %.6g. "
            "Entries are probabilities and must be >= 0."
            % (worst[0], worst[1], B[worst])
        )

    row_sums = B.sum(axis=1)
    errors = np.abs(row_sums - 1.0)
    if errors.max() > _TOL:
        bad = int(np.argmax(errors))
        raise ValueError(
            "baseline row %d sums to %.12g, not 1 (|error| = %.3e > tol %.0e). "
            "Each row must be a probability distribution over the %d actions."
            % (bad, row_sums[bad], errors[bad], _TOL, n_actions)
        )

    return B


def fixed_action(n, n_actions, index=0):
    """Fixed designated action baseline: every agent plays `index` with probability 1.

    NOT a "no-op" baseline -- see the module docstring. `index` designates an
    action; it does not mean the agent does nothing.

    Returns a (n, n_actions) delta matrix.
    """
    if not (0 <= index < n_actions):
        raise ValueError(
            "index %d is out of range for n_actions = %d; the designated action "
            "must be a real action index in [0, %d]."
            % (index, n_actions, n_actions - 1)
        )

    B = np.zeros((n, n_actions))
    B[:, index] = 1.0

    # Self-validating. Redundant by construction, but it costs one line and
    # catches a typo here rather than three tasks downstream, where it would
    # surface as wrong ground truth instead of an exception.
    return check_baseline(B, n, n_actions)


def uniform(n, n_actions):
    """Uniform random baseline: every agent plays every action with probability 1/|A|.

    Returns a (n, n_actions) matrix of 1/n_actions.
    """
    if n_actions < 1:
        raise ValueError("n_actions must be >= 1, got %d" % n_actions)

    B = np.full((n, n_actions), 1.0 / n_actions)
    return check_baseline(B, n, n_actions)


def fixed_default(dist):
    """Validated pass-through for a caller-supplied reference behaviour.

    `dist` is any (n, n_actions) row-stochastic array. `n` and `n_actions` are
    inferred from its shape, so a 1-D or otherwise malformed input fails in
    `check_baseline` with a clear message rather than by silently broadcasting
    somewhere downstream.

    THIS IS THE ENTRY POINT WEEK 2's POLICY-MARGINAL BASELINE WILL USE: the
    learner hands over its per-agent action distribution and gets back a
    validated baseline. No other code changes.

    Returns an independent float64 COPY, not the caller's array. "Pass-through"
    here is about semantics, not object identity -- if the returned baseline
    stayed aliased to the caller's array, mutating that array afterwards would
    silently change the baseline that ground truth had already been computed
    under, and nothing would report it. Copying also makes all three
    constructors in this module behave alike.
    """
    dist = np.asarray(dist)

    # Shape is validated first, by check_baseline, so that a 1-D input produces
    # the shape error rather than an IndexError from unpacking below.
    if dist.ndim != 2:
        raise ValueError(
            "baseline must be 2-D of shape (n_agents, n_actions), got ndim %d "
            "with shape %s." % (dist.ndim, dist.shape)
        )

    n, n_actions = dist.shape
    B = np.array(dist, dtype=float)  # np.array copies by default
    return check_baseline(B, n, n_actions)
