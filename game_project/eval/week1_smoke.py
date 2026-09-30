"""Week 1 smoke artifact -- ground truth vs. ground truth, not the audit.

n=3, |A|=3, lam in {0, 0.25, 0.5, 1.0, 2.0}, 10 seeds, both Week 1 baselines
(fixed designated action, uniform). Computes Shapley, Banzhaf, and LOO under
each baseline; reports, per lam, mean + 95% CI over seeds of:

  - solution-concept spread, all under the FIXED baseline:
        pair ratios  ||phi_A - phi_B||_2 / ||phi_Shapley||_2
        for (Shapley,Banzhaf), (Shapley,LOO), (Banzhaf,LOO), plus their
        per-seed MAX. Shapley and Banzhaf are the closest pair of the three
        (they coincide exactly whenever both weight n=2 games identically);
        reporting only that pair would bias the artifact toward finding
        concept spread small relative to baseline spread. The max over all
        three pairs is the headline column; the three pairs are also kept
        individually so the max's composition is inspectable.
  - baseline spread: ||phi_Shapley^fixed - phi_Shapley^uniform||_2 / ||phi_Shapley^uniform||_2.

The UNIFORM baseline is the reference (denominator) for baseline spread, and
phi_Shapley^fixed is the reference (denominator) for every concept-spread
pair -- kept as a single consistent reference rather than re-normalizing each
pair by its own two vectors, so the three pair columns stay comparable to
each other and to the original Shapley-Banzhaf definition.

Degenerate fixed-baseline cases: under the FIXED (delta-at-index-0) baseline,
v(S) is an exact index lookup rather than a genuine average, so v(S) can be
exactly constant across every coalition -- specifically whenever the game's
unconstrained joint argmax happens to land on action 0 for every agent. When
that happens every marginal contribution v[S u {i}] - v[S] is exactly 0,
which forces EVERY semivalue (any weighting -- Shapley, Banzhaf, LOO alike)
to be the exact zero vector, and every concept-spread pair ratio becomes
0/0. This is the same exact-index-lookup fragility of the fixed baseline
already flagged in the Task 3 section of docs/reports/Week_1_Report.md.
Resolved by asserting the zero-vector argument holds for all three
semivalues and substituting 0.0 (both vectors in every pair are provably
identical) rather than nan. The count of seeds this happens on, per lam, is
written to the CSV (`n_degenerate_fixed`) so a reader can see how much of a
low concept-spread mean is real agreement vs. substituted zeros.

CI: Student-t, not the normal (1.96) approximation -- at 10 seeds
(dof=9) the correct two-sided 95% multiplier is t(0.975, 9) = 2.262, about
13% wider than 1.96. Hardcoded rather than via scipy: this project's Task 0
dependencies are numpy, pytest, matplotlib only, and every use in this file
is at the spec-mandated 10 seeds, so a single dof=9 table entry covers it.

This compares ground truth to ground truth. It is not the audit (Phase 2) and
not the baseline-dominance experiment (Week 5); it is a free end-to-end smoke
check and an early look at whether Section 5's effect exists at all. Subject
to the orthogonalization caveat in reports/notes_baselines.md: under a
uniform baseline, synergy is invisible to singleton coalitions by
construction, so part of any baseline-spread signal seen here is structural
to the game family rather than discovered.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from games.generator import make_game
from ground_truth.baselines import fixed_action, uniform
from ground_truth.enumerate import coalition_values
from ground_truth.semivalues import banzhaf, loo, shapley

N = 3
N_ACTIONS = 3
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]
SEEDS = list(range(10))

CSV_PATH = "results/week1_spread.csv"
PNG_PATH = "results/week1_spread.png"

_DEGENERATE_ATOL = 1e-9

# t(0.975, dof) two-sided 95% critical values, keyed by degrees of freedom.
# len(SEEDS) is fixed at 10 throughout this file, so dof = 9 is the only
# entry needed; _mean_ci95 raises rather than silently falling back to the
# normal approximation if that ever changes.
_T975 = {9: 2.262}


def _mean_ci95(values):
    """Mean and 95% CI half-width over seeds, via Student-t (not the
    normal/1.96 approximation, which understates the interval at n=10)."""
    values = np.asarray(values)
    dof = len(values) - 1
    if dof not in _T975:
        raise ValueError(
            "no hardcoded t(0.975, dof=%d) critical value in _T975; add one "
            "(or install scipy and use scipy.stats.t.ppf(0.975, dof))" % dof
        )
    mean = values.mean()
    sem = values.std(ddof=1) / np.sqrt(len(values))
    return mean, _T975[dof] * sem


def main():
    B_fixed = fixed_action(N, N_ACTIONS)
    B_uniform = uniform(N, N_ACTIONS)

    rows = []
    for lam in LAMBDAS:
        sb_spreads, sl_spreads, bl_spreads, max_spreads = [], [], [], []
        baseline_spreads = []
        n_degenerate = 0

        for seed in SEEDS:
            game = make_game(N, N_ACTIONS, lam, seed)
            payoff = game["payoff"]

            v_fixed = coalition_values(payoff, B_fixed)
            v_uniform = coalition_values(payoff, B_uniform)

            phi_s_fixed = shapley(v_fixed, N)
            phi_b_fixed = banzhaf(v_fixed, N)
            phi_l_fixed = loo(v_fixed, N)

            phi_s_uniform = shapley(v_uniform, N)
            banzhaf(v_uniform, N)  # computed per spec; not part of any reported ratio
            loo(v_uniform, N)      # ditto

            shapley_norm = np.linalg.norm(phi_s_fixed)
            if shapley_norm < _DEGENERATE_ATOL:
                assert np.linalg.norm(phi_b_fixed) < _DEGENERATE_ATOL, (
                    "phi_Shapley_fixed ~= 0 but phi_Banzhaf_fixed is not, at "
                    "lam=%g seed=%d -- the constant-v argument does not hold "
                    "here, investigate before trusting this artifact"
                    % (lam, seed)
                )
                assert np.linalg.norm(phi_l_fixed) < _DEGENERATE_ATOL, (
                    "phi_Shapley_fixed ~= 0 but phi_LOO_fixed is not, at "
                    "lam=%g seed=%d -- the constant-v argument does not hold "
                    "here, investigate before trusting this artifact"
                    % (lam, seed)
                )
                sb, sl, bl = 0.0, 0.0, 0.0
                n_degenerate += 1
            else:
                sb = np.linalg.norm(phi_s_fixed - phi_b_fixed) / shapley_norm
                sl = np.linalg.norm(phi_s_fixed - phi_l_fixed) / shapley_norm
                bl = np.linalg.norm(phi_b_fixed - phi_l_fixed) / shapley_norm

            sb_spreads.append(sb)
            sl_spreads.append(sl)
            bl_spreads.append(bl)
            max_spreads.append(max(sb, sl, bl))

            baseline_spreads.append(
                np.linalg.norm(phi_s_fixed - phi_s_uniform)
                / np.linalg.norm(phi_s_uniform)
            )

        sb_mean, sb_ci = _mean_ci95(sb_spreads)
        sl_mean, sl_ci = _mean_ci95(sl_spreads)
        bl_mean, bl_ci = _mean_ci95(bl_spreads)
        max_mean, max_ci = _mean_ci95(max_spreads)
        bs_mean, bs_ci = _mean_ci95(baseline_spreads)
        rows.append((
            lam,
            sb_mean, sb_ci, sl_mean, sl_ci, bl_mean, bl_ci,
            max_mean, max_ci, n_degenerate,
            bs_mean, bs_ci,
        ))

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lam",
            "spread_shapley_banzhaf_mean", "spread_shapley_banzhaf_ci95",
            "spread_shapley_loo_mean", "spread_shapley_loo_ci95",
            "spread_banzhaf_loo_mean", "spread_banzhaf_loo_ci95",
            "concept_spread_max_mean", "concept_spread_max_ci95",
            "n_degenerate_fixed",
            "baseline_spread_mean", "baseline_spread_ci95",
        ])
        writer.writerows(rows)

    lams = [r[0] for r in rows]
    max_means = [r[7] for r in rows]
    max_cis = [r[8] for r in rows]
    bs_means = [r[10] for r in rows]
    bs_cis = [r[11] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(lams, max_means, yerr=max_cis, marker="o", capsize=3,
                label="solution-concept spread (max of Shapley/Banzhaf/LOO "
                      "pairs, fixed baseline)")
    ax.errorbar(lams, bs_means, yerr=bs_cis, marker="s", capsize=3,
                label="baseline spread (Shapley: fixed vs uniform)")
    ax.set_xlabel("lambda (synergy dial)")
    ax.set_ylabel("relative L2 spread")
    ax.set_title("Week 1 smoke: ground truth vs ground truth\n(n=3, |A|=3, 10 seeds)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)

    print("wrote", CSV_PATH)
    print("wrote", PNG_PATH)
    for lam, sb_mean, sb_ci, sl_mean, sl_ci, bl_mean, bl_ci, max_mean, max_ci, n_deg, bs_mean, bs_ci in rows:
        print(f"lam={lam:.2f}  "
              f"SB={sb_mean:.4f}+/-{sb_ci:.4f}  SL={sl_mean:.4f}+/-{sl_ci:.4f}  "
              f"BL={bl_mean:.4f}+/-{bl_ci:.4f}  max={max_mean:.4f}+/-{max_ci:.4f}  "
              f"n_degenerate={n_deg}/10  baseline={bs_mean:.4f}+/-{bs_ci:.4f}")


if __name__ == "__main__":
    main()
