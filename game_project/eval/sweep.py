"""Week 3 Task 9 -- reporting sweep (seeds 0-9). Run it ONCE.

Grid: D7 lambda x {VDN, QMIX} x D6 regimes {R1, R2} x games 0-9, with D11 settings
(QMIX: 50k steps, lr 0.1, cosine, embed 32, 5 inits per game; VDN: 20k steps, 1 init).

Per trained model we log, for every D10 baseline (fixed, uniform, policy-marginal
eps = 0.05) x semivalue (Shapley, Banzhaf, LOO) x extraction (A, B, C, D):
phi_true, phi_hat, scaled_l2 (scale = payoff.std()), normalized_l2, raw_l2, sup, Spearman,
plus fit RMSE / sup, final loss, the D11 last-10%-of-steps loss-change flag, regret,
normalized regret, n_degenerate, and (R2) the per-game WQMIX Thm 1 bound.

Assertions during the run:
  * VDN:   max |q_tot - vdn_lstsq_oracle| < 1e-8 (D11), every run.
  * F3:    credit_sup <= 2 * fit_sup + 1e-9 for extraction D, every semivalue,
           every baseline, every lambda, both methods.
  * F4 (R1 + VDN only): extraction D equals semivalue(coalition_values(A, B)) to 1e-8,
           where A = sum_i f[i, a_i] is rebuilt from the game's own f (an independent
           route: it never looks at the trained model).

Seeding is the Week 3 pilot's, unchanged: init 0 = Week 2 convention
(SeedSequence(s).spawn(2) -> init, data); inits k >= 1 take later children.

Run from game_project/:
    python -m eval.sweep | tee results/week3_sweep.txt
    python -m eval.sweep --plot-only          (re-do gates + figures from the CSV)
"""

import csv
import multiprocessing as mp
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval.metrics import credit_error, fit_error, normalized_regret, regret, spearman
from eval.week2_lambda0 import TAU_FIT, _mean_ci95        # one source for tau_fit and the t-CI
from eval.week3_pilot import init_and_data_seeds, regime_weights
from games.generator import make_game
from ground_truth.baselines import fixed_action, policy_marginal, uniform
from ground_truth.enumerate import coalition_values
from ground_truth.semivalues import banzhaf_weights, loo_weights, semivalue, shapley_weights
from methods.extraction import extract_A, extract_B, extract_C, extract_D
from methods.qmix import QMIX
from methods.train import train
from methods.vdn import VDN, vdn_lstsq_oracle

N, N_ACTIONS = 3, 3
SEEDS = list(range(10))
LAMS = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]            # D7
REGIMES = ["R1", "R2"]                                    # D6
ALPHA = 0.01                                              # D6
LR = 0.1
QMIX_STEPS, QMIX_EMBED, QMIX_INITS = 50000, 32, 5         # D11
VDN_STEPS = 20000                                         # D11
VDN_ORACLE_TOL = 1e-8                                     # D11
PM_EPS = 0.05                                             # D5 / D10
DEGENERATE_ATOL = 1e-9

# D9 thresholds
REGRET_MAX, CREDIT_MIN = 0.05, 0.10

BASELINES = ["fixed", "uniform", "policy_marginal"]
SEMIVALUES = {"shapley": shapley_weights(N), "banzhaf": banzhaf_weights(N), "loo": loo_weights(N)}
EXTRACTIONS = ["A", "B", "C", "D"]
PRIMARY = {"VDN": "A", "QMIX": "D"}                       # D1 / D9 primary extraction

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
CSV_PATH = RESULTS_DIR / "week3_sweep.csv"

FIELDS = (
    ["regime", "lam", "method", "seed", "init", "steps", "baseline", "semivalue", "extraction",
     "is_primary", "scaled_l2", "normalized_l2", "raw_l2", "sup", "spearman"]
    + ["phi_true_%d" % i for i in range(N)] + ["phi_hat_%d" % i for i in range(N)]
    + ["fit_rmse", "fit_sup", "loss", "loss_rel_change_last10", "regret", "nregret",
       "n_degenerate", "lowest_loss_init", "thm1_bound", "alpha_satisfies_bound"]
)


# ---------------------------------------------------------------------------
# one training run -> 36 rows (3 baselines x 3 semivalues x 4 extractions)
# ---------------------------------------------------------------------------
def additive_tensor(f):
    """A(a) = sum_i f[i, a_i], rebuilt from the game's f (used only by the F4 check)."""
    A = np.zeros((N_ACTIONS,) * N)
    for i in range(N):
        shape = [1] * N
        shape[i] = N_ACTIONS
        A = A + f[i].reshape(shape)
    return A


def thm1_bound(payoff):
    """WQMIX Thm 1 sufficient bound at gamma = 0:  alpha < Delta^2 / (R_max^2 * |A|^n).
    FLAGGED reading: Delta = best minus second-best joint payoff, R_max = payoff range
    (max - min).  On the QTRAN matrix this gives 64/(400*9) = 0.0178, i.e. the ~0.018
    quoted in D6, which is how this reading was chosen."""
    flat = np.sort(payoff.reshape(-1))
    delta = flat[-1] - flat[-2]
    r_max = flat[-1] - flat[0]
    return float(delta ** 2 / (r_max ** 2 * payoff.size))


def run_one(job):
    """job = (method, regime, lam, seed, init, steps, embed) -> list of row dicts."""
    torch.set_num_threads(1)
    method, regime, lam, seed, init, steps, embed = job

    game = make_game(N, N_ACTIONS, lam, seed)
    payoff = game["payoff"]
    w = regime_weights(regime, payoff)
    init_seed, data_seed = init_and_data_seeds(seed, init)
    model = VDN(N, N_ACTIONS, init_seed) if method == "VDN" else QMIX(N, N_ACTIONS, init_seed, embed_dim=embed)

    # log_every = steps // 10 -> history holds step 0, 10%, ..., 90%, 100%: the 90% entry
    # gives D11's "relative loss change over the last 10% of steps".
    hist = train(model, payoff, weights=w, mode="full_batch", steps=steps, lr=LR,
                 sigma=0.0, data_seed=data_seed, log_every=steps // 10)
    fin, at90 = hist[-1], hist[-2]
    assert fin["step"] == steps and at90["step"] == steps - steps // 10, "unexpected history layout"

    fit = fit_error(fin["q_tot_tensor"], payoff)
    loss_rel = abs(fin["train_loss"] - at90["train_loss"]) / max(at90["train_loss"], 1e-12)
    reg, nreg = regret(payoff, fin["greedy_action"]), normalized_regret(payoff, fin["greedy_action"])

    if method == "VDN":                                       # D11 assertion
        _, oracle = vdn_lstsq_oracle(payoff, w)
        gap = float(np.abs(fin["q_tot_tensor"] - oracle).max())
        assert gap < VDN_ORACLE_TOL, "VDN != oracle: %s gap %.3e" % (job, gap)

    if regime == "R2":
        bound = thm1_bound(payoff)
        satisfies = int(ALPHA < bound)
    else:
        bound, satisfies = "", ""

    baselines = {
        "fixed": fixed_action(N, N_ACTIONS),
        "uniform": uniform(N, N_ACTIONS),
        "policy_marginal": policy_marginal(fin["q_tables"], "epsilon_greedy", PM_EPS),  # this model's own greedy
    }
    A_true = additive_tensor(game["f"]) if (method == "VDN" and regime == "R1") else None
    scale = payoff.std()

    rows = []
    for bname, B in baselines.items():
        abc = {"A": extract_A(fin, B), "B": extract_B(fin, B), "C": extract_C(fin, B)}
        v_true = coalition_values(payoff, B)
        for sname, sw in SEMIVALUES.items():
            phi_true = semivalue(v_true, N, sw)
            phi_D = extract_D(fin, B, sw)
            ce_D = credit_error(phi_D, phi_true)
            # F3: a theorem for nonnegative semivalue weights, so it holds at every lambda.
            assert ce_D["sup"] <= 2 * fit["sup"] + 1e-9, (
                "F3 violated %s %s %s: credit_sup %.3e > 2*fit_sup %.3e"
                % (job, bname, sname, ce_D["sup"], 2 * fit["sup"]))
            if A_true is not None:                            # F4, R1 + VDN
                phi_add = semivalue(coalition_values(A_true, B), N, sw)
                assert np.abs(phi_D - phi_add).max() < 1e-8, (
                    "F4 violated %s %s %s: %.3e" % (job, bname, sname, np.abs(phi_D - phi_add).max()))
            for ext in EXTRACTIONS:
                phi_hat = phi_D if ext == "D" else abc[ext]
                ce = credit_error(phi_hat, phi_true, scale=scale)
                row = {
                    "regime": regime, "lam": lam, "method": method, "seed": seed, "init": init,
                    "steps": steps, "baseline": bname, "semivalue": sname, "extraction": ext,
                    "is_primary": int(ext == PRIMARY[method] and sname == "shapley"
                                      and bname in ("fixed", "uniform")),
                    "scaled_l2": ce["scaled_l2"], "normalized_l2": ce["normalized_l2"],
                    "raw_l2": ce["raw_l2"], "sup": ce["sup"], "spearman": spearman(phi_hat, phi_true),
                    "fit_rmse": fit["rmse"], "fit_sup": fit["sup"], "loss": fin["train_loss"],
                    "loss_rel_change_last10": loss_rel, "regret": reg, "nregret": nreg,
                    "n_degenerate": int(np.linalg.norm(phi_true) < DEGENERATE_ATOL),
                    "lowest_loss_init": "",                   # filled in after all inits are known
                    "thm1_bound": bound, "alpha_satisfies_bound": satisfies,
                }
                for i in range(N):
                    row["phi_true_%d" % i] = phi_true[i]
                    row["phi_hat_%d" % i] = phi_hat[i]
                rows.append(row)
    return rows


def make_jobs(seeds=SEEDS, lams=LAMS, qmix_steps=QMIX_STEPS, vdn_steps=VDN_STEPS, n_inits=QMIX_INITS):
    jobs = []
    for regime in REGIMES:
        for lam in lams:
            for s in seeds:
                jobs.append(("VDN", regime, lam, s, 0, vdn_steps, QMIX_EMBED))
                for k in range(n_inits):
                    jobs.append(("QMIX", regime, lam, s, k, qmix_steps, QMIX_EMBED))
    # longest first so the pool is not left waiting on a few QMIX runs at the end
    return sorted(jobs, key=lambda j: -j[5])


def mark_lowest_loss(rows):
    """Secondary statistic (D11): the init with the lowest final loss, per (method, regime, lam,
    seed). Chosen by loss only -- never by credit or regret."""
    best = {}
    for r in rows:
        k = (r["method"], r["regime"], r["lam"], r["seed"])
        if k not in best or (r["loss"], r["init"]) < best[k]:
            best[k] = (r["loss"], r["init"])
    for r in rows:
        r["lowest_loss_init"] = int(best[(r["method"], r["regime"], r["lam"], r["seed"])][1] == r["init"])


def run_sweep(jobs, processes=4):
    rows, t0 = [], time.time()
    with mp.Pool(processes) as pool:
        for i, out in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows.extend(out)
            if i % 20 == 0 or i == len(jobs):
                print("  %d/%d runs, %.0f s" % (i, len(jobs), time.time() - t0), flush=True)
    rows.sort(key=lambda r: (r["regime"], r["lam"], r["method"], r["seed"], r["init"],
                             BASELINES.index(r["baseline"]), list(SEMIVALUES).index(r["semivalue"]),
                             r["extraction"]))
    mark_lowest_loss(rows)
    return rows


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def write_csv(rows, path=CSV_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", path)


_STR = {"regime", "method", "baseline", "semivalue", "extraction"}


def read_csv(path=CSV_PATH):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (v if k in _STR else (float(v) if v != "" else float("nan")))
                         for k, v in r.items()})
    return rows


# ---------------------------------------------------------------------------
# aggregation: per-game mean over inits, then mean + t-CI over the 10 games (D11)
# ---------------------------------------------------------------------------
def build_index(rows):
    idx = {}
    for r in rows:
        idx.setdefault((r["regime"], r["lam"], r["method"], r["baseline"], r["semivalue"],
                        r["extraction"]), []).append(r)
    return idx


def per_game(idx, key, metric, lowest_only=False):
    """List (one entry per seed, sorted) of the metric averaged over that game's inits."""
    by_seed = {}
    for r in idx[key]:
        if lowest_only and r["lowest_loss_init"] != 1:
            continue
        by_seed.setdefault(r["seed"], []).append(r[metric])
    return [float(np.mean(by_seed[s])) for s in sorted(by_seed)]


def stat(values):
    """(mean, t-CI half-width); CI needs exactly 10 games (t table is hard-coded)."""
    return _mean_ci95(values)


def d9_holds(idx, method, regime, lam, lowest_only=False):
    """D9 for one (method, regime, lambda): under BOTH baselines, mean + CI of normalized
    regret < 0.05 and mean - CI of primary credit error (scaled_l2) > 0.10."""
    ext = PRIMARY[method]
    out = {}
    ok = True
    for b in ("fixed", "uniform"):
        k = (regime, lam, method, b, "shapley", ext)
        rm, rc = stat(per_game(idx, k, "nregret", lowest_only))
        cm, cc = stat(per_game(idx, k, "scaled_l2", lowest_only))
        out[b] = (rm, rc, cm, cc)
        ok = ok and (rm + rc < REGRET_MAX) and (cm - cc > CREDIT_MIN)
    return ok, out


# ---------------------------------------------------------------------------
# gates (printed)
# ---------------------------------------------------------------------------
def gate_d2(idx):
    """lambda = 0 rows: D2 per method per regime. Fit: mean + CI < tau_fit; F3 per run
    (already asserted in the run, re-counted here)."""
    print("\n=== Gate: lambda = 0, D2 per method per regime ===")
    all_pass = True
    for method in ("VDN", "QMIX"):
        for regime in REGIMES:
            k = (regime, 0.0, method, "uniform", "shapley", "D")
            m, c = stat(per_game(idx, k, "fit_rmse"))
            fit_ok = m + c < TAU_FIT[method]
            f3_ok = all(r["sup"] <= 2 * r["fit_sup"] + 1e-9
                        for b in BASELINES for s in SEMIVALUES
                        for r in idx[(regime, 0.0, method, b, s, "D")])
            all_pass = all_pass and fit_ok and f3_ok
            print("  %-4s %s  fit_rmse mean=%.3e +/- %.3e  tau_fit=%.0e -> %s   F3 all runs: %s"
                  % (method, regime, m, c, TAU_FIT[method], "PASS" if fit_ok else "FAIL", f3_ok))
    print("  D2 gate:", "PASS" if all_pass else "FAIL -> STOP, go to the phase chat")
    return all_pass


def gate_d9(idx):
    print("\n=== D9 decoupling: lambdas where it holds (regret<%.2f and credit>%.2f, both baselines) ==="
          % (REGRET_MAX, CREDIT_MIN))
    for method in ("VDN", "QMIX"):
        for regime in REGIMES:
            holds = []
            print("  %s %s   lam: nregret mean+CI (fixed) | credit mean-CI (fixed, uniform)" % (method, regime))
            for lam in LAMS:
                ok, d = d9_holds(idx, method, regime, lam)
                if ok:
                    holds.append(lam)
                print("    %-6g reg+ci=%.3f  credit-ci fixed=%.3f uniform=%.3f  %s"
                      % (lam, d["fixed"][0] + d["fixed"][1], d["fixed"][2] - d["fixed"][3],
                         d["uniform"][2] - d["uniform"][3], "HOLDS" if ok else ""))
            print("    -> lambdas where D9 holds:", holds if holds else "none")
    # Secondary (D11): lowest-final-loss init per game only, QMIX.
    print("  Secondary, QMIX lowest-loss init only:")
    for regime in REGIMES:
        holds = [lam for lam in LAMS if d9_holds(idx, "QMIX", regime, lam, lowest_only=True)[0]]
        print("    QMIX %s -> %s" % (regime, holds if holds else "none"))


def print_flags(rows):
    """D11: the loss-change flag, summarised (logged, not a criterion)."""
    print("\n=== D11 flag: relative loss change over the last 10% of steps (QMIX, per run) ===")
    seen = {}
    for r in rows:
        if r["method"] == "QMIX" and r["baseline"] == "uniform" and r["semivalue"] == "shapley" and r["extraction"] == "D":
            seen.setdefault((r["regime"], r["lam"]), []).append(r["loss_rel_change_last10"])
    for (regime, lam), v in sorted(seen.items()):
        print("  %s lam=%-6g median=%.2e max=%.2e  runs>1%%: %d/%d"
              % (regime, lam, np.median(v), np.max(v), sum(x > 0.01 for x in v), len(v)))


# ---------------------------------------------------------------------------
# figures (English labels, mean +/- t-CI)
# ---------------------------------------------------------------------------
# lambda is plotted at equal spacing of grid index: the D7 grid is log-spaced and
# includes 0, which a log axis cannot show.
XPOS = {lam: i for i, lam in enumerate(LAMS)}
LAM_LABELS = ["%g" % lam for lam in LAMS]
C_CREDIT, C_REGRET = "#1b6ca8", "#d1495b"


def _band(ax, xs, stats, color, label, marker="o", ls="-"):
    m = np.array([s[0] for s in stats])
    c = np.array([s[1] for s in stats])
    ax.errorbar(xs, m, yerr=c, color=color, marker=marker, ls=ls, capsize=3, lw=1.6, ms=4, label=label)


def _setup_x(ax):
    ax.set_xticks(range(len(LAMS)))
    ax.set_xticklabels(LAM_LABELS)
    ax.set_xlabel("synergy strength lambda")


def fig_central(idx, path):
    """Per method x regime (rows), fixed / uniform baseline (columns): primary credit error
    (left axis) and normalized regret (right axis) against lambda, Shapley."""
    fig, axes = plt.subplots(4, 2, figsize=(10, 13), sharex=True)
    xs = list(range(len(LAMS)))
    for r_i, (method, regime) in enumerate([(m, g) for m in ("VDN", "QMIX") for g in REGIMES]):
        ext = PRIMARY[method]
        for c_i, b in enumerate(("fixed", "uniform")):
            ax = axes[r_i, c_i]
            cred = [stat(per_game(idx, (regime, lam, method, b, "shapley", ext), "scaled_l2")) for lam in LAMS]
            regr = [stat(per_game(idx, (regime, lam, method, b, "shapley", ext), "nregret")) for lam in LAMS]
            _band(ax, xs, cred, C_CREDIT, "credit error (scaled L2)")
            ax.axhline(CREDIT_MIN, color=C_CREDIT, ls=":", lw=1)
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_ylabel("credit error", color=C_CREDIT)
            ax2 = ax.twinx()
            _band(ax2, xs, regr, C_REGRET, "normalized regret", marker="s", ls="--")
            ax2.axhline(REGRET_MAX, color=C_REGRET, ls=":", lw=1)
            ax2.set_ylabel("normalized regret", color=C_REGRET)
            ax2.set_ylim(bottom=0)
            ax.set_title("%s, %s, %s baseline (extraction %s)" % (method, regime, b, ext), fontsize=9)
            ax.grid(alpha=0.3)
            _setup_x(ax)
    fig.suptitle("Credit error and task return against lambda (Shapley; mean +/- 95% t-CI over 10 games)\n"
                 "dotted lines: D9 thresholds (credit 0.10, regret 0.05)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def fig_vdn_targets(idx, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    colors = {"shapley": "#1b6ca8", "banzhaf": "#e08a1e", "loo": "#2a9d5c"}
    xs = list(range(len(LAMS)))
    for r_i, regime in enumerate(REGIMES):
        for c_i, b in enumerate(("fixed", "uniform")):
            ax = axes[r_i, c_i]
            for sname in SEMIVALUES:
                st = [stat(per_game(idx, (regime, lam, "VDN", b, sname, "A"), "scaled_l2")) for lam in LAMS]
                _band(ax, xs, st, colors[sname], sname)
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_title("VDN gap, %s, %s baseline" % (regime, b), fontsize=9)
            ax.set_ylabel("scaled L2 distance to target")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            _setup_x(ax)
    fig.suptitle("VDN gap against the three semivalue targets (mean +/- 95% t-CI over 10 games)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def fig_qmix_extractions(idx, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    colors = {"A": "#1b6ca8", "B": "#e08a1e", "C": "#2a9d5c", "D": "#8a4fb0"}
    xs = list(range(len(LAMS)))
    for r_i, regime in enumerate(REGIMES):
        for c_i, b in enumerate(("fixed", "uniform")):
            ax = axes[r_i, c_i]
            for ext in EXTRACTIONS:
                st = [stat(per_game(idx, (regime, lam, "QMIX", b, "shapley", ext), "scaled_l2")) for lam in LAMS]
                _band(ax, xs, st, colors[ext], "extraction " + ext)
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_title("QMIX, %s, %s baseline (Shapley target)" % (regime, b), fontsize=9)
            ax.set_ylabel("scaled L2 credit error")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            _setup_x(ax)
    fig.suptitle("QMIX extractions A-D against lambda (mean +/- 95% t-CI over 10 games)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def report(rows):
    idx = build_index(rows)
    fig_central(idx, RESULTS_DIR / "week3_sweep_1_central.png")
    fig_vdn_targets(idx, RESULTS_DIR / "week3_sweep_2_vdn_targets.png")
    fig_qmix_extractions(idx, RESULTS_DIR / "week3_sweep_3_qmix_extractions.png")
    ok = gate_d2(idx)
    gate_d9(idx)
    print_flags(rows)
    return ok


def main():
    if "--plot-only" in sys.argv:
        ok = report(read_csv())
    else:
        jobs = make_jobs()
        print("sweep: %d training runs" % len(jobs), flush=True)
        rows = run_sweep(jobs)
        write_csv(rows)                  # written before the gate so nothing is lost on a FAIL
        ok = report(rows)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
