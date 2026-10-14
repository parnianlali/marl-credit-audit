"""Week 2 Task 7: pilot, D2 gate criteria, lambda=0 reporting run, gate verdict.

n=3, |A|=3, lambda=0, VDN and QMIX, uniform training weights, full_batch,
sigma=0 (D3). Trains once per seed per method and extracts credit under both
the fixed and the uniform baseline (training does not depend on the
baseline). Primary extraction per D1: VDN -> extract_A (the baseline-relative
gap), QMIX -> extract_D (semivalue on the learned Q_tot, Shapley weights).
F2 says A and D coincide for VDN to 1e-9, so using Shapley-weighted D as the
common ground-truth comparandum for both methods' primary line is exact, not
an approximation.

Two seed pools, never mixed (Task 7's own rule):
  - PILOT_SEEDS = 100..104, used only to fix STEPS/LR and read off the fit
    floor each method reaches (D2a).
  - SEEDS = 0..9, the reporting run and the gate.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from eval.metrics import credit_error, fit_error, regret
from games.generator import make_game
from ground_truth.baselines import fixed_action, uniform
from ground_truth.enumerate import coalition_values
from ground_truth.semivalues import semivalue, shapley_weights
from methods.extraction import extract_A, extract_B, extract_C, extract_D
from methods.qmix import QMIX
from methods.train import joint_from_product, seed_to_int, train
from methods.vdn import VDN

N = 3
N_ACTIONS = 3
LAM = 0.0

PILOT_SEEDS = [100, 101, 102, 103, 104]
SEEDS = list(range(10))

# D2(a): fixed by the pilot (see main()'s printed pilot summary). Both
# methods use the same step/optimizer family (D4): Adam + cosine decay to 0.
TRAIN_CONFIG = {
    "VDN": {"steps": 20000, "lr": 0.1, "log_every": 1000},
    "QMIX": {"steps": 20000, "lr": 0.1, "log_every": 1000},
}

# D2(a): tau_fit, set above each method's pilot floor (never from 0-9 itself).
# Pilot (seeds 100-104, steps=20000, lr=0.1): VDN floor mean=8.08e-17
# (max=1.21e-16); QMIX floor mean=2.79e-4 (max=4.27e-4, min=1.19e-4).
# VDN's threshold sits ~8 orders above its floor (plain machine precision on
# a convex fit). QMIX's sits ~2.3x above the worst pilot seed, clear of the
# floor band but well short of the known stalled-optimizer scale (~1e-2,
# see D4's RMSprop check in the report) -- a QMIX seed landing there is a
# real optimization failure, not fit-floor noise.
TAU_FIT = {"VDN": 1e-8, "QMIX": 1e-3}

CSV_PATH = "results/week2_lambda0.csv"
PNG_PATH = "results/week2_lambda0.png"

_DEGENERATE_ATOL = 1e-9

# t(0.975, dof) two-sided 95% critical values. dof=9 for the 10-seed
# reporting run, dof=4 for the 5-seed pilot (same convention as
# eval/week1_smoke.py: hardcoded table, not scipy -- see that file's note).
_T975 = {4: 2.776, 9: 2.262}


def _mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    dof = len(values) - 1
    if dof not in _T975:
        raise ValueError("no hardcoded t(0.975, dof=%d) critical value in _T975" % dof)
    mean = values.mean()
    sem = values.std(ddof=1) / np.sqrt(len(values))
    return mean, _T975[dof] * sem


def _nanmean_ci95(values):
    """Like _mean_ci95, but drops NaN entries (degenerate seeds, Key
    Learning 3) first and reports how many were dropped."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[~np.isnan(values)]
    n_dropped = len(values) - len(finite)
    if len(finite) < 2:
        return float("nan"), float("nan"), n_dropped
    mean, ci = _mean_ci95(finite)
    return mean, ci, n_dropped


## Note on the credit-error figure's band (write_plot, below): a first
## attempt computed the CI in log10 space (geometric mean +/- log-space CI)
## instead of linear mean +/- CI, to fix the linear band running to the
## bottom of a log-scale axis. That made it worse, not better: at lambda=0
## several VDN seeds recover credit to *exactly* 0.0 (perfect recovery at
## machine precision -- confirmed in results/week2_lambda0.csv), mixed with
## other seeds at ~1e-16. log10(0) is undefined, so flooring those exact
## zeros at any epsilon manufactures an arbitrary, enormous fake spread in
## log space (the epsilon choice alone produced a 10^-216-to-10^40 y-axis).
## Parametric statistics -- linear or log -- don't fit data with legitimate
## exact zeros mixed with near-zero floats. write_plot uses the per-seed
## lines + median instead, which has no such failure mode.


def _run(ModelClass, seed, steps, lr, log_every):
    game = make_game(N, N_ACTIONS, LAM, seed)
    payoff = game["payoff"]
    train_weights = joint_from_product(uniform(N, N_ACTIONS))
    init_seed, data_seed = np.random.SeedSequence(seed).spawn(2)
    model = ModelClass(N, N_ACTIONS, seed_to_int(init_seed))
    history = train(
        model, payoff, weights=train_weights, mode="full_batch",
        steps=steps, lr=lr, sigma=0.0, data_seed=data_seed, log_every=log_every,
    )
    return payoff, history


def _primary_extraction(method_name, snapshot, B, w_shapley):
    if method_name == "VDN":
        return extract_A(snapshot, B)
    return extract_D(snapshot, B, w_shapley)


def run_pilot():
    """D2(a) evidence: fit floor each method reaches under the committed
    TRAIN_CONFIG, over the pilot seeds only."""
    print("=== Pilot (seeds 100-104, never 0-9) ===")
    floors = {}
    for method_name, ModelClass in [("VDN", VDN), ("QMIX", QMIX)]:
        cfg = TRAIN_CONFIG[method_name]
        seed_floors = []
        for seed in PILOT_SEEDS:
            payoff, history = _run(ModelClass, seed, cfg["steps"], cfg["lr"], cfg["log_every"])
            rmse_trajectory = [fit_error(h["q_tot_tensor"], payoff)["rmse"] for h in history]
            seed_floors.append(rmse_trajectory[-1])
            print(f"  {method_name} seed={seed}: final rmse={rmse_trajectory[-1]:.3e}, "
                  f"trajectory={['%.1e' % r for r in rmse_trajectory[::len(rmse_trajectory)//4 or 1]]}")
        mean, ci = _mean_ci95(seed_floors)
        floors[method_name] = {"mean": mean, "ci95": ci, "max": max(seed_floors), "min": min(seed_floors)}
        print(f"  {method_name} floor: mean={mean:.3e} +/- {ci:.3e}  (max={max(seed_floors):.3e}, "
              f"tau_fit={TAU_FIT[method_name]:.1e})")
        assert mean < TAU_FIT[method_name], (
            f"{method_name} pilot floor {mean:.3e} is not below tau_fit {TAU_FIT[method_name]:.1e}; "
            "raise steps/lr or lower tau_fit before trusting the reporting run"
        )
    return floors


def run_reporting():
    """n=3, |A|=3, lambda=0, seeds 0-9. Train once per seed per method;
    extract under both baselines. Returns the credit-error step-curves (for
    the figure) and the final-step rows (for the CSV). The fit-error-vs-step
    figure lives in eval/week2_diagnostics.py, not here -- see that file."""
    B_fixed = fixed_action(N, N_ACTIONS)
    B_uniform = uniform(N, N_ACTIONS)
    baselines = {"fixed": B_fixed, "uniform": B_uniform}
    w_shapley = shapley_weights(N)

    curves = {}   # (method, baseline) -> list over log steps of per-seed normalized credit error
    csv_rows = []

    for method_name, ModelClass in [("VDN", VDN), ("QMIX", QMIX)]:
        cfg = TRAIN_CONFIG[method_name]
        for bname in baselines:
            curves[(method_name, bname)] = None  # filled per-seed below, shape (n_steps, n_seeds)

        per_baseline_step_errors = {bname: [] for bname in baselines}

        for seed in SEEDS:
            payoff, history = _run(ModelClass, seed, cfg["steps"], cfg["lr"], cfg["log_every"])
            steps = [h["step"] for h in history]

            for bname, B in baselines.items():
                v_true = coalition_values(payoff, B)
                phi_true = semivalue(v_true, N, w_shapley)

                step_errors = []
                for h in history:
                    phi_hat = _primary_extraction(method_name, h, B, w_shapley)
                    step_errors.append(credit_error(phi_hat, phi_true)["normalized_l2"])
                per_baseline_step_errors[bname].append(step_errors)

                # Final-step row for the CSV: all four extractions.
                final = history[-1]
                fe = fit_error(final["q_tot_tensor"], payoff)
                greedy = final["greedy_action"]
                reg = regret(payoff, greedy)
                n_degenerate = int(np.linalg.norm(phi_true) < _DEGENERATE_ATOL)

                for ext_name, ext_fn in [
                    ("A", lambda h, B: extract_A(h, B)),
                    ("B", lambda h, B: extract_B(h, B)),
                    ("C", lambda h, B: extract_C(h, B)),
                    ("D", lambda h, B: extract_D(h, B, w_shapley)),
                ]:
                    phi_hat = ext_fn(final, B)
                    ce = credit_error(phi_hat, phi_true)
                    csv_rows.append({
                        "method": method_name, "baseline": bname, "extraction": ext_name,
                        "seed": seed, "step": final["step"],
                        "fit_rmse": fe["rmse"], "fit_sup": fe["sup"],
                        "credit_normalized_l2": ce["normalized_l2"],
                        "credit_raw_l2": ce["raw_l2"], "credit_sup": ce["sup"],
                        "regret": reg, "n_degenerate": n_degenerate,
                        "is_primary": int(
                            (method_name == "VDN" and ext_name == "A")
                            or (method_name == "QMIX" and ext_name == "D")
                        ),
                    })

            curves_steps = steps

        for bname in baselines:
            arr = np.array(per_baseline_step_errors[bname])  # (n_seeds, n_steps)
            curves[(method_name, bname)] = (curves_steps, arr)

    return curves, csv_rows


def write_csv(csv_rows):
    fieldnames = [
        "method", "baseline", "extraction", "seed", "step",
        "fit_rmse", "fit_sup", "credit_normalized_l2", "credit_raw_l2", "credit_sup",
        "regret", "n_degenerate", "is_primary",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print("wrote", CSV_PATH)


def write_plot(curves):
    """Per-seed lines (thin, low alpha) plus the per-step median (bold) --
    not a parametric mean +/- CI band. See the note above this function's
    old band helper: at lambda=0 several VDN seeds hit exactly 0.0 credit
    error, mixed with others at ~1e-16, and neither a linear nor a log10
    mean+CI handles that mix (linear collapses to the axis floor; log
    explodes once the zeros are floored). The median has neither failure
    mode, and keeping the raw per-seed lines visible (rather than
    replacing them with any single-number summary) means no spread
    information is hidden behind a statistic that doesn't fit this data.
    matplotlib silently omits non-positive points on a log axis, so a
    seed's exact-0.0 steps just don't plot -- the correct behavior here,
    not a bug."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {
        ("VDN", "fixed"): dict(color="C0", marker="o", linestyle="-"),
        ("VDN", "uniform"): dict(color="C0", marker="o", linestyle="--"),
        ("QMIX", "fixed"): dict(color="C1", marker="s", linestyle="-"),
        ("QMIX", "uniform"): dict(color="C1", marker="s", linestyle="--"),
    }
    for (method_name, bname), (steps, arr) in curves.items():
        style = styles[(method_name, bname)]
        for row in arr:
            ax.plot(steps, row, color=style["color"], linestyle=style["linestyle"],
                    linewidth=0.6, alpha=0.25, zorder=1)
        median = np.nanmedian(arr, axis=0)
        ax.plot(steps, median, label=f"{method_name} ({bname}) median",
                linewidth=2, zorder=2, **style)

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("normalized credit error (L2, primary extraction)")
    ax.set_title("Week 2: lambda=0 credit recovery (n=3, |A|=3, 10 seeds)\n"
                  "bold = per-step median, thin = individual seeds")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    print("wrote", PNG_PATH)


def gate_verdict(csv_rows):
    """D2 applied to the reporting run's numbers."""
    print("\n=== D2 gate verdict ===")
    print("D2(d) aggregation: mean +/- t(0.975,9) CI over seeds 0-9, both "
          "baselines. Not per-seed pass/fail (see D2 writeup in the report "
          "for why).")

    rows_by_key = {}
    for r in csv_rows:
        if not r["is_primary"]:
            continue
        key = (r["method"], r["baseline"])
        rows_by_key.setdefault(key, []).append(r)

    overall_pass = True
    for method_name in ["VDN", "QMIX"]:
        fit_values = [r["fit_rmse"] for r in rows_by_key[(method_name, "uniform")]]
        fit_mean, fit_ci = _mean_ci95(fit_values)
        fit_pass = fit_mean + fit_ci < TAU_FIT[method_name]
        print(f"{method_name} fit: mean={fit_mean:.3e} +/- {fit_ci:.3e}  "
              f"tau_fit={TAU_FIT[method_name]:.1e}  -> {'PASS' if fit_pass else 'FAIL'}")

        for bname in ["fixed", "uniform"]:
            key = (method_name, bname)
            credit_values = [r["credit_normalized_l2"] for r in rows_by_key[key]]
            c_mean, c_ci, n_dropped = _nanmean_ci95(credit_values)
            # D2(b): for extraction D (QMIX) and A (VDN, == D by F2), F3's
            # bound is automatic: credit_sup <= 2*fit_sup + 1e-9, per seed.
            bound_values = [r["credit_sup"] <= 2 * r["fit_sup"] + 1e-9 for r in rows_by_key[key]]
            bound_pass = all(bound_values)
            print(f"  {method_name}/{bname}: credit normalized_l2 mean={c_mean:.3e} "
                  f"+/- {c_ci:.3e}  (n_degenerate={n_dropped}/10)  "
                  f"F3 bound (per-seed, sup-norm) holds: {bound_pass}")
            overall_pass = overall_pass and fit_pass and bound_pass

    print(f"\nOverall gate (D2(c): binds both VDN and QMIX, per Project_Structure "
          f"Section 10): {'PASS' if overall_pass else 'FAIL'}")
    return overall_pass


def main():
    run_pilot()
    curves, csv_rows = run_reporting()
    write_csv(csv_rows)
    write_plot(curves)
    gate_verdict(csv_rows)


if __name__ == "__main__":
    main()
