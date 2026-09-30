"""it says what agents OUTSIDE a coalition are assumed to do.( atention to the
marginal_greedy_action in generator.py)

A baseline is a single ndarray `B` of shape `(n, n_actions)`
"""

import numpy as np

_TOL = 1e-12

def check_baseline(B, n, n_actions):
    """Validate that `B` is a legal baseline for `n` agents with `n_actions` actions.
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

    # Report the single worst offender rather than "some entry is negative"
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
    """Fixed designated action baseline: every agent plays action of `index` with
      probability 1.
    """
    if not (0 <= index < n_actions):
        raise ValueError(
            "index %d is out of range for n_actions = %d; the designated action "
            "must be a real action index in [0, %d]."
            % (index, n_actions, n_actions - 1)
        )

    B = np.zeros((n, n_actions))
    B[:, index] = 1.0
    return check_baseline(B, n, n_actions)


def uniform(n, n_actions):
    """Uniform random baseline: every agent plays every action with probability 1/|A|.
    """
    if n_actions < 1:
        raise ValueError("n_actions must be >= 1, got %d" % n_actions)

    B = np.full((n, n_actions), 1.0 / n_actions)
    return check_baseline(B, n, n_actions)


def fixed_default(dist):
    """Validated pass-through for a caller-supplied reference behaviour.
    """
    dist = np.asarray(dist)

    if dist.ndim != 2:
        raise ValueError(
            "baseline must be 2-D of shape (n_agents, n_actions), got ndim %d "
            "with shape %s." % (dist.ndim, dist.shape)
        )

    n, n_actions = dist.shape
    B = np.array(dist, dtype=float)  # np.array copies by default
    return check_baseline(B, n, n_actions)
