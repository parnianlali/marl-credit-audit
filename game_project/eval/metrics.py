"""Generic metrics for comparing extracted credit against ground truth, and
Q_tot against the true payoff. No knowledge of how either side was produced.
"""

import numpy as np


def credit_error(phi_hat, phi_true, scale=None):
    """{normalized_l2, raw_l2, sup} between an extracted and a true credit
    vector. normalized_l2 is NaN -- never 0 -- when phi_true is the zero
    vector, so a degenerate seed can't look like perfect recovery.

    If `scale` is given, also return scaled_l2 = raw_l2 / scale (D8's primary
    credit error, with scale = payoff.std()). Unlike normalized_l2 it has the
    same denominator for every baseline and lambda, and no 0/0.
    """
    phi_hat = np.asarray(phi_hat, dtype=np.float64)
    phi_true = np.asarray(phi_true, dtype=np.float64)
    assert phi_hat.shape == phi_true.shape, (
        "credit_error: shapes differ, %s vs %s" % (phi_hat.shape, phi_true.shape)
    )

    diff = phi_hat - phi_true
    raw_l2 = float(np.linalg.norm(diff))
    sup = float(np.max(np.abs(diff)))
    true_norm = np.linalg.norm(phi_true)
    normalized_l2 = float(raw_l2 / true_norm) if true_norm != 0 else float("nan")

    out = {"normalized_l2": normalized_l2, "raw_l2": raw_l2, "sup": sup}
    if scale is not None:
        out["scaled_l2"] = float(raw_l2 / scale)
    return out


def fit_error(q_tot, payoff):
    """{rmse, sup} between a model's Q_tot tensor and the true payoff tensor,
    both shape (n_actions,)*n.
    """
    q_tot = np.asarray(q_tot, dtype=np.float64)
    payoff = np.asarray(payoff, dtype=np.float64)
    assert q_tot.shape == payoff.shape, (
        "fit_error: shapes differ, %s vs %s" % (q_tot.shape, payoff.shape)
    )

    diff = q_tot - payoff
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    sup = float(np.max(np.abs(diff)))
    return {"rmse": rmse, "sup": sup}


def regret(payoff, greedy):
    """payoff.max() - payoff[greedy], the gap between the true optimum and the
    joint-greedy action's payoff.
    """
    payoff = np.asarray(payoff, dtype=np.float64)
    greedy = np.asarray(greedy)
    assert greedy.shape == (payoff.ndim,), (
        "regret: greedy has shape %s, expected (%d,)" % (greedy.shape, payoff.ndim)
    )
    return float(payoff.max() - payoff[tuple(greedy)])


def normalized_regret(payoff, greedy):
    """regret / (payoff.max() - payoff.mean()): the fraction of the gain over
    uniform-random play that the greedy action fails to capture. 0 is optimal,
    1 is as bad as the uniform-random average. Raises ValueError when the
    payoff is (numerically) constant, since there is no gain to normalize by.
    """
    payoff = np.asarray(payoff, dtype=np.float64)
    denom = float(payoff.max() - payoff.mean())
    if denom < 1e-12:
        raise ValueError(
            "normalized_regret: payoff.max() - payoff.mean() = %g < 1e-12" % denom
        )
    return regret(payoff, greedy) / denom


def _average_ranks(x):
    """1-based ranks of a 1-D array, tied values sharing the mean of the
    ranks they span (e.g. [1, 1, 2] -> [1.5, 1.5, 3]). Done by hand because
    the project doesn't use scipy.
    """
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1  # extend the run of equal values
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # mean of ranks i+1..j+1
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation: Pearson correlation of average ranks.
    NaN if either rank vector has zero variance (a constant input).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert x.ndim == 1 and x.shape == y.shape, (
        "spearman: need two 1-D arrays of equal length, got %s and %s"
        % (x.shape, y.shape)
    )
    rx = _average_ranks(x) - _average_ranks(x).mean()
    ry = _average_ranks(y) - _average_ranks(y).mean()
    # Ranks of a constant vector are all equal, so centered they are exactly 0.
    if not rx.any() or not ry.any():
        return float("nan")
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))
