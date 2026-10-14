"""Generic metrics for comparing extracted credit against ground truth, and
Q_tot against the true payoff. No knowledge of how either side was produced.
"""

import numpy as np


def credit_error(phi_hat, phi_true):
    """{normalized_l2, raw_l2, sup} between an extracted and a true credit
    vector. normalized_l2 is NaN -- never 0 -- when phi_true is the zero
    vector, so a degenerate seed can't look like perfect recovery.
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

    return {"normalized_l2": normalized_l2, "raw_l2": raw_l2, "sup": sup}


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
