"""Week 2 diagnostics -- requested directly in the project chat after seeing
Task 7's results, not part of the original Task 7 task-spec deliverables
(those are eval/week2_lambda0.py's CSV/PNG/gate verdict).

Two pieces:

1. Fit-error vs training step, for both methods (moved out of
   eval/week2_lambda0.py, which now stays focused on the credit-error
   figure, the CSV, and the gate verdict). Same config Task 7's reporting
   run used: n=3, |A|=3, lambda=0, seeds 0-9, steps=20000. Baseline-
   independent, since training never sees a baseline. This is the direct
   view of why VDN's and QMIX's credit-error curves sit so far apart: F3
   bounds credit error by <= 2x fit error, so whatever gap shows up here
   propagates into the credit-error figure almost unchanged.

   This re-trains rather than reading back saved history: the gate script
   (eval/week2_lambda0.py) only persists final-step rows to its CSV, not
   full per-step histories. Training is deterministic (methods.train.
   seed_to_int(SeedSequence(seed))), so re-running it reproduces Task 7's
   run bit-for-bit -- confirmed by the floor numbers below matching Task
   7's pilot output exactly.

2. QMIX long-run check: Task 7's reporting run saw QMIX plateau around a
   ~1e-4 fit floor at 20000 steps. Is that a hard floor, or does it keep
   dropping with more steps? Trains QMIX at lambda=0 for 100k steps (5x
   Task 7's budget) on the PILOT seeds (100-104, never 0-9) only -- this is
   a diagnostic about the optimizer's behavior, not a new reporting-run
   result, so it must never touch the seeds the gate was computed on. Also
   plots the logged learning rate alongside fit error, as a direct visual
   check that methods.train.train()'s CosineAnnealingLR is actually
   decaying (the closed-form check lives in
   tests/test_methods.py::test_lr_schedule_logged_matches_cosine_closed_form).
"""

import numpy as np
import matplotlib.pyplot as plt

from eval.metrics import fit_error
from games.generator import make_game
from ground_truth.baselines import uniform
from methods.qmix import QMIX
from methods.train import joint_from_product, seed_to_int, train
from methods.vdn import VDN

from eval.week2_lambda0 import (
    LAM, N, N_ACTIONS, PILOT_SEEDS, SEEDS, TAU_FIT, TRAIN_CONFIG,
    _mean_ci95, _run,
)

FIT_PNG_PATH = "results/week2_diagnostics_fit.png"
QMIX_LONGRUN_PNG_PATH = "results/week2_diagnostics_qmix_100k.png"

QMIX_LONGRUN_STEPS = 100_000
QMIX_LONGRUN_LR = 0.1
QMIX_LONGRUN_LOG_EVERY = 2000


def run_fit_curves():
    """Fit rmse (Q_tot vs true payoff) vs step, both methods, Task 7's own
    seeds/config (seeds 0-9, steps=20000, lr=0.1)."""
    fit_curves = {}
    for method_name, ModelClass in [("VDN", VDN), ("QMIX", QMIX)]:
        cfg = TRAIN_CONFIG[method_name]
        per_seed = []
        steps = None
        for seed in SEEDS:
            payoff, history = _run(ModelClass, seed, cfg["steps"], cfg["lr"], cfg["log_every"])
            per_seed.append([fit_error(h["q_tot_tensor"], payoff)["rmse"] for h in history])
            steps = [h["step"] for h in history]
        fit_curves[method_name] = (steps, np.array(per_seed))
    return fit_curves


def write_fit_plot(fit_curves):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {
        "VDN": dict(color="C0", marker="o"),
        "QMIX": dict(color="C1", marker="s"),
    }
    for method_name, (steps, arr) in fit_curves.items():
        means, cis = [], []
        for col in range(arr.shape[1]):
            m, ci = _mean_ci95(arr[:, col])
            means.append(m)
            cis.append(ci)
        means = np.array(means)
        cis = np.array(cis)
        style = styles[method_name]
        ax.plot(steps, means, label=method_name, **style)
        ax.fill_between(steps, means - cis, means + cis, alpha=0.15, color=style["color"])

    for method_name, tau in TAU_FIT.items():
        ax.axhline(tau, color=styles[method_name]["color"], linestyle=":", linewidth=1,
                    label=f"{method_name} tau_fit={tau:.0e}")

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("fit RMSE (Q_tot vs true payoff)")
    ax.set_title("Week 2: lambda=0 fit error (n=3, |A|=3, 10 seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIT_PNG_PATH, dpi=150)
    print("wrote", FIT_PNG_PATH)


def run_qmix_longrun():
    """Does QMIX's ~1e-4 fit floor (Task 7, 20000 steps) keep dropping with
    5x more steps? Pilot seeds only (100-104, never 0-9)."""
    train_weights = joint_from_product(uniform(N, N_ACTIONS))
    per_seed_fit, per_seed_lr, steps_logged = [], [], None

    print("=== QMIX 100k-step long run (seeds 100-104, never 0-9) ===")
    for seed in PILOT_SEEDS:
        payoff = make_game(N, N_ACTIONS, LAM, seed)["payoff"]
        init_seed, data_seed = np.random.SeedSequence(seed).spawn(2)
        model = QMIX(N, N_ACTIONS, seed_to_int(init_seed))
        history = train(
            model, payoff, weights=train_weights, mode="full_batch",
            steps=QMIX_LONGRUN_STEPS, lr=QMIX_LONGRUN_LR, sigma=0.0,
            data_seed=data_seed, log_every=QMIX_LONGRUN_LOG_EVERY,
        )
        steps_logged = [h["step"] for h in history]
        per_seed_fit.append([fit_error(h["q_tot_tensor"], payoff)["rmse"] for h in history])
        per_seed_lr.append([h["lr"] for h in history])
        nearest_20k = min(range(len(steps_logged)), key=lambda i: abs(steps_logged[i] - 20000))
        print(f"  seed={seed}: final rmse={per_seed_fit[-1][-1]:.3e}  "
              f"(step {steps_logged[nearest_20k]} value was "
              f"{per_seed_fit[-1][nearest_20k]:.3e})")

    fit_arr = np.array(per_seed_fit)
    lr_arr = np.array(per_seed_lr)
    mean, ci = _mean_ci95(fit_arr[:, -1])
    print(f"  floor at {QMIX_LONGRUN_STEPS} steps: mean={mean:.3e} +/- {ci:.3e}")
    return steps_logged, fit_arr, lr_arr


def write_qmix_longrun_plot(steps, fit_arr, lr_arr):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    means, cis = [], []
    for col in range(fit_arr.shape[1]):
        m, ci = _mean_ci95(fit_arr[:, col])
        means.append(m)
        cis.append(ci)
    means = np.array(means)
    cis = np.array(cis)
    ax1.plot(steps, means, color="C1", marker="s", markersize=3, label="QMIX fit rmse")
    ax1.fill_between(steps, means - cis, means + cis, alpha=0.15, color="C1")
    ax1.axhline(TAU_FIT["QMIX"], color="C1", linestyle=":", linewidth=1,
                label=f"tau_fit={TAU_FIT['QMIX']:.0e}")
    ax1.axvline(20000, color="gray", linestyle="--", linewidth=1,
                label="Task 7's 20000-step budget")
    ax1.set_yscale("log")
    ax1.set_ylabel("fit RMSE")
    ax1.set_title(f"QMIX at lambda=0, {QMIX_LONGRUN_STEPS} steps (pilot seeds 100-104)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    # lr schedule is identical across seeds (deterministic given steps/lr0),
    # so any one seed's column already is the schedule; averaging just
    # guards against that assumption silently breaking.
    lr_mean = lr_arr.mean(axis=0)
    ax2.plot(steps, lr_mean, color="C2", marker=".", markersize=3)
    ax2.set_ylabel("learning rate")
    ax2.set_xlabel("training step")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(QMIX_LONGRUN_PNG_PATH, dpi=150)
    print("wrote", QMIX_LONGRUN_PNG_PATH)


def main():
    fit_curves = run_fit_curves()
    write_fit_plot(fit_curves)

    steps, fit_arr, lr_arr = run_qmix_longrun()
    write_qmix_longrun_plot(steps, fit_arr, lr_arr)


if __name__ == "__main__":
    main()
