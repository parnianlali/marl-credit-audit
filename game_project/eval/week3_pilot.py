"""Week 3 Task 7 -- pilot. Seeds 100-104 ONLY; seeds 0-9 are never touched here.

n = 3, |A| = 3, both D6 regimes (R1 uniform, R2 central weighting alpha = 0.01),
lambda_max = 4 (D7). Four parts, one table each:

  7a  VDN final q_tot vs vdn_lstsq_oracle under the regime's weights.
  7b  QMIX budget: steps in {20k, 50k, 100k}, lr 0.1, cosine (train() default).
  7c  QMIX restarts: 5 inits per pilot game, largest budget.
  7d  QMIX embedding: 8 vs 32, largest budget.

Credit = extract_D, Shapley weights, uniform baseline, scaled by
payoff.std() (D8 primary); truth = semivalue(coalition_values(payoff, uniform)).

Seeding: restart 0 is exactly the Week 2 convention,
    init_seed, data_seed = SeedSequence(s).spawn(2)
Restarts k >= 1 take the *next* children of the same SeedSequence object
(spawn() keeps a counter, so a second spawn call returns children 2, 3, ...),
which leaves restart 0 untouched.

Run from game_project/:  python -m eval.week3_pilot
"""
#runs multiple experiments in parallel
import csv
import multiprocessing as mp
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")          # no display needed: figures are only written to PNG files
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval.metrics import credit_error, fit_error, normalized_regret, regret
from games.generator import make_game
from ground_truth.baselines import uniform
from ground_truth.enumerate import coalition_values
from ground_truth.semivalues import semivalue, shapley_weights
from methods.extraction import extract_D
from methods.qmix import QMIX
from methods.train import central_weights, joint_from_product, seed_to_int, train
from methods.vdn import VDN, vdn_lstsq_oracle

N, N_ACTIONS = 3, 3
PILOT_SEEDS = [100, 101, 102, 103, 104]
LAMS = [0.0, 1.0, 4.0]            # {0, 1, lambda_max}; lambda_max = 4 from D7
LAMS_RESTART = [1.0, 4.0]         # 7c and 7d
ALPHA = 0.01                      # D6
LR = 0.1
BUDGETS = [20000, 50000, 100000]
VDN_BUDGETS = [1000, 2000, 5000, 10000, 20000]
VDN_TOL = 1e-8                    # "exact convergence" for 7a (max abs diff to oracle)
N_RESTARTS = 5
EMBEDS = [8, 32]

B_UNIFORM = uniform(N, N_ACTIONS)
W_SHAPLEY = shapley_weights(N)


def regime_weights(regime, payoff):
    if regime == "R1":
        return joint_from_product(B_UNIFORM)
    return central_weights(payoff, ALPHA)

# This function derives two reproducible integer seeds — 
    # one for weight initialization (init) and one for data. 
def init_and_data_seeds(s, restart):
    ss = np.random.SeedSequence(s)
    init0, data = ss.spawn(2)                       # restart 0: unchanged convention
    if restart == 0:
        return seed_to_int(init0), seed_to_int(data)
    extra = ss.spawn(restart)                       # children 2 .. restart+1
    return seed_to_int(extra[restart - 1]), seed_to_int(data)


def run_one(job):
    """One training run -> flat result dict."""
    torch.set_num_threads(1)
    method, regime, lam, s, restart, steps, embed = job
    payoff = make_game(N, N_ACTIONS, lam, s)["payoff"]
    w = regime_weights(regime, payoff)
    # determine Neural-network initialization and Training data randomness.
    init_seed, data_seed = init_and_data_seeds(s, restart)
    if method == "VDN":
        model = VDN(N, N_ACTIONS, init_seed)
    else:
        model = QMIX(N, N_ACTIONS, init_seed, embed_dim=embed)
    # Train the model 
    hist = train(model, payoff, weights=w, mode="full_batch", steps=steps,
                 lr=LR, sigma=0.0, data_seed=data_seed, log_every=steps)
    fin = hist[-1]
    # Calculate the theoretical credit allocation
    phi_true = semivalue(coalition_values(payoff, B_UNIFORM), N, W_SHAPLEY)
    # Extract the learned credit allocation
    phi_hat = extract_D(fin, B_UNIFORM, W_SHAPLEY)
    # Calculate the evaluation metrics
    out = {
        "job": job,
        "loss": fin["train_loss"],
        "rmse": fit_error(fin["q_tot_tensor"], payoff)["rmse"],
        "credit": credit_error(phi_hat, phi_true, scale=payoff.std())["scaled_l2"],
        "regret": regret(payoff, fin["greedy_action"]),
        "nregret": normalized_regret(payoff, fin["greedy_action"]),
    }
    # calculates the oracle solution under the same weights used for training -> Oracle
    if method == "VDN":
        _, oracle = vdn_lstsq_oracle(payoff, w)
        out["oracle_gap"] = float(np.abs(fin["q_tot_tensor"] - oracle).max())
    return out

# The helper function: This formats a number in scientific notation
def fmt(x):
    return "%.2e" % x

# helper function:
# suppose five games produce these credit errors:[0.1, 0.2, 0.15, 0.3, 0.25]
    # This calculates: mean[max, min]
def ms(vals):
    """mean [min, max] -- 5 pilot games, so no CI (pilot, not a claim)."""
    v = np.asarray(vals, dtype=np.float64)
    return "%s [%s, %s]" % (fmt(v.mean()), fmt(v.min()), fmt(v.max()))


# ---------------------------------------------------------------------------
# Saving results (added beyond the Task 7 spec; flagged in the week report).
# The CSV is the real output: plots and tables can be rebuilt from it without
# re-training.  Run from game_project/ with:
#   python -m eval.week3_pilot | tee results/week3_pilot.txt
#   python -m eval.week3_pilot --plot-only      (re-draw figures from the CSV)
# ---------------------------------------------------------------------------
RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
CSV_PATH = RESULTS_DIR / "week3_pilot.csv"
FIELDS = ["method", "regime", "lam", "seed", "restart", "steps", "embed",
          "loss", "rmse", "credit", "regret", "nregret", "oracle_gap"]


def write_csv(results, path=CSV_PATH):
    """One row per training run. oracle_gap is empty for QMIX (VDN-only metric)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(results, key=lambda r: r["job"]):
            method, regime, lam, s, restart, steps, embed = r["job"]
            w.writerow({"method": method, "regime": regime, "lam": lam, "seed": s,
                        "restart": restart, "steps": steps, "embed": embed,
                        "loss": r["loss"], "rmse": r["rmse"], "credit": r["credit"],
                        "regret": r["regret"], "nregret": r["nregret"],
                        "oracle_gap": r.get("oracle_gap", "")})


def read_csv(path=CSV_PATH):
    """Read the CSV back with proper number types."""
    rows = []
    with open(path) as fh:
        for d in csv.DictReader(fh):
            rows.append({
                "method": d["method"], "regime": d["regime"], "lam": float(d["lam"]),
                "seed": int(d["seed"]), "restart": int(d["restart"]),
                "steps": int(d["steps"]), "embed": int(d["embed"]),
                "loss": float(d["loss"]), "rmse": float(d["rmse"]),
                "credit": float(d["credit"]), "regret": float(d["regret"]),
                "nregret": float(d["nregret"]),
                "oracle_gap": float(d["oracle_gap"]) if d["oracle_gap"] != "" else None,
            })
    return rows


def _pick(rows, **kw):
    """Rows whose fields equal every given keyword, e.g. _pick(rows, method="VDN", lam=0.0)."""
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def make_plots(rows, out_dir=RESULTS_DIR):
    """Four figures, one per pilot part. Only 5 games per cell, so we show
    mean with a min-max band (or the raw points), never a confidence interval."""
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {0.0: "tab:blue", 1.0: "tab:orange", 4.0: "tab:red"}
    regimes = ("R1", "R2")

    # 7a: VDN worst-case gap to the oracle vs steps (log scale), tolerance as a line.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, regime in zip(axes, regimes):
        for lam in LAMS:
            worst = [max(r["oracle_gap"] for r in _pick(rows, method="VDN", regime=regime,
                                                        lam=lam, steps=b)) for b in VDN_BUDGETS]
            ax.plot(VDN_BUDGETS, worst, "o-", color=colors[lam], label="lam=%g" % lam)
        ax.axhline(VDN_TOL, color="k", ls="--", lw=1, label="tolerance")
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_title("7a  VDN vs oracle, %s" % regime)
        ax.set_xlabel("training steps"); ax.legend()
    axes[0].set_ylabel("max |q_tot - oracle| (worst of 5 games)")
    fig.tight_layout(); fig.savefig(out_dir / "week3_pilot_7a_vdn_oracle.png", dpi=150); plt.close(fig)

    # 7b: QMIX loss and D credit error vs budget: mean line, min-max band over 5 games.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for col, regime in enumerate(regimes):
        for row, (key, label) in enumerate([("loss", "final weighted loss"),
                                            ("credit", "D credit error (scaled L2)")]):
            ax = axes[row, col]
            for lam in LAMS:
                vals = np.array([[r[key] for r in _pick(rows, method="QMIX", regime=regime, lam=lam,
                                                        restart=0, steps=b, embed=32)]
                                 for b in BUDGETS])          # shape (n_budgets, 5 games)
                ax.plot(BUDGETS, vals.mean(axis=1), "o-", color=colors[lam], label="lam=%g" % lam)
                ax.fill_between(BUDGETS, vals.min(axis=1), vals.max(axis=1),
                                color=colors[lam], alpha=0.2)
            if key == "loss":
                ax.set_yscale("log")
            ax.set_title("7b  QMIX, %s" % regime); ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("training steps")
            ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "week3_pilot_7b_qmix_budget.png", dpi=150); plt.close(fig)

    # 7c: restarts. One dot per init, grouped by game; shows how much init alone moves the credit.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, regime in zip(axes, regimes):
        for j, lam in enumerate(LAMS_RESTART):
            for g, s in enumerate(PILOT_SEEDS):
                ys = [r["credit"] for r in _pick(rows, method="QMIX", regime=regime, lam=lam, seed=s,
                                                 steps=BUDGETS[-1], embed=32)]
                ax.scatter([g + 0.15 * (j - 0.5)] * len(ys), ys, color=colors[lam], s=18,
                           label="lam=%g" % lam if g == 0 else None)
        ax.set_xticks(range(len(PILOT_SEEDS))); ax.set_xticklabels(PILOT_SEEDS)
        ax.set_title("7c  restarts, %s (%d steps)" % (regime, BUDGETS[-1]))
        ax.set_xlabel("pilot game (seed)"); ax.legend()
    axes[0].set_ylabel("D credit error (scaled L2), one dot per init")
    fig.tight_layout(); fig.savefig(out_dir / "week3_pilot_7c_restarts.png", dpi=150); plt.close(fig)

    # 7d: embedding 8 vs 32, init 0, one dot per game, black bar = mean.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, regime in zip(axes, regimes):
        for j, lam in enumerate(LAMS_RESTART):
            for k, e in enumerate(EMBEDS):
                ys = [r["credit"] for r in _pick(rows, method="QMIX", regime=regime, lam=lam, restart=0,
                                                 steps=BUDGETS[-1], embed=e)]
                x = j * 3 + k
                ax.scatter([x] * len(ys), ys, color=colors[lam], s=18)
                ax.hlines(np.mean(ys), x - 0.3, x + 0.3, color="k")
        ax.set_xticks([0, 1, 3, 4]); ax.set_xticklabels(["lam=1\nemb 8", "lam=1\nemb 32",
                                                         "lam=4\nemb 8", "lam=4\nemb 32"])
        ax.set_title("7d  embedding, %s (%d steps)" % (regime, BUDGETS[-1]))
    axes[0].set_ylabel("D credit error (scaled L2)")
    fig.tight_layout(); fig.savefig(out_dir / "week3_pilot_7d_embedding.png", dpi=150); plt.close(fig)


def main():
    """ this will have 340 runs, based on the 7 configuration tests"""
    jobs = []
    # R1 —> Uniform regime: Every joint action is equally likely.
    # R2 —> Central weighting regime (alpha=0.01)
    for regime in ("R1", "R2"):
        for lam in LAMS:
            for s in PILOT_SEEDS:
                for b in VDN_BUDGETS:                                   # 7a
                    jobs.append(("VDN", regime, lam, s, 0, b, 32))
                for b in BUDGETS:     
                    # QMIX embedding experiments          # 7b (also restart 0 + embed 32 of 7c/7d)
                    jobs.append(("QMIX", regime, lam, s, 0, b, 32))
        for lam in LAMS_RESTART:
            for s in PILOT_SEEDS:
                for k in range(1, N_RESTARTS):                          # 7c
                    jobs.append(("QMIX", regime, lam, s, k, BUDGETS[-1], 32))
                jobs.append(("QMIX", regime, lam, s, 0, BUDGETS[-1], 8))  # 7d
    # builds a dict using each item as a key
    jobs = list(dict.fromkeys(jobs))
    print("%d training runs" % len(jobs), flush=True)

    # How does the code actually execute 340 runs? 
    # spawns 4 worker processes to run tasks in parallel
    with mp.Pool(4) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1)):
            results.append(r)
            if (i + 1) % 20 == 0:
                # prints progress every 20 completed runs.
                print("  done %d/%d" % (i + 1, len(jobs)), file=sys.stderr, flush=True)
    by_job = {r["job"]: r for r in results}
    write_csv(results)                       # raw per-run numbers, saved before anything else
    print("wrote %s" % CSV_PATH, flush=True)
    make_plots(read_csv())                   # read back from disk so the plots match the CSV exactly
    print("wrote 4 figures to %s" % RESULTS_DIR, flush=True)

    def get(method, regime, lam, restart, steps, embed):
        return [by_job[(method, regime, lam, s, restart, steps, embed)] for s in PILOT_SEEDS]

    print("\n=== 7a  VDN vs vdn_lstsq_oracle: max|q_tot - oracle|, mean [min, max] over 5 games ===")
    print("(converged = all 5 games below %g)" % VDN_TOL)
    for regime in ("R1", "R2"):
        for lam in LAMS:
            cells = []
            first = None
            for b in VDN_BUDGETS:
                gaps = [r["oracle_gap"] for r in get("VDN", regime, lam, 0, b, 32)]
                # This identifies the first tested budget for
                #  which all five games have an oracle gap below the telorance
                if first is None and max(gaps) < VDN_TOL:
                    first = b
                cells.append("%dk:%s" % (b // 1000, fmt(max(gaps))))
            print("%s lam=%-4g  worst gap by budget  %s   -> first converged budget: %s"
                  % (regime, lam, "  ".join(cells), first))

    print("\n=== 7b  QMIX budget (embed 32, init 0): mean [min, max] over 5 games ===")
    print("regime lam   steps   loss                          rmse                          D credit (scaled)             regret")
    # in Qmix For every combination, it retrieves the five pilot games
    for regime in ("R1", "R2"):
        for lam in LAMS:
            for b in BUDGETS:
                rs = get("QMIX", regime, lam, 0, b, 32)
                print("%-6s %-4g %6d   %-28s  %-28s  %-28s  %s" % (
                    regime, lam, b, ms([r["loss"] for r in rs]), ms([r["rmse"] for r in rs]),
                    ms([r["credit"] for r in rs]), [round(r["regret"], 2) for r in rs]))

    print("\n=== 7c  QMIX restarts (%d inits, %d steps, embed 32): per game, spread across inits ===" % (N_RESTARTS, BUDGETS[-1]))
    print("regime lam  game   loss min..max                 regret per init            credit min..max")
    # the output contains loss, credit, regret, RMSE . 
    for regime in ("R1", "R2"):
        for lam in LAMS_RESTART:
            for s in PILOT_SEEDS:
                rs = [by_job[("QMIX", regime, lam, s, k, BUDGETS[-1], 32)] for k in range(N_RESTARTS)]
                l = [r["loss"] for r in rs]
                c = [r["credit"] for r in rs]
                print("%-6s %-4g %4d   %s..%s   %-26s  %.3f..%.3f" % (
                    regime, lam, s, fmt(min(l)), fmt(max(l)),
                    [round(r["regret"], 2) for r in rs], min(c), max(c)))

    print("\n=== 7d  QMIX embedding 8 vs 32 (init 0, %d steps): mean [min, max] over 5 games ===" % BUDGETS[-1])
    print("regime lam  embed  loss                          D credit (scaled)")
    # 7d
    for regime in ("R1", "R2"):
        for lam in LAMS_RESTART:
            # It reports the Final loss and D credit error.
            for e in EMBEDS:
                rs = get("QMIX", regime, lam, 0, BUDGETS[-1], e)
                print("%-6s %-4g %5d  %-28s  %s" % (
                    regime, lam, e, ms([r["loss"] for r in rs]), ms([r["credit"] for r in rs])))


if __name__ == "__main__":
    if "--plot-only" in sys.argv:            # redraw figures from the saved CSV, no training
        make_plots(read_csv())
    else:
        main()
