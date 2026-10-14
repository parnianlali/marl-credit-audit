"""Week 2 Task 6 -- tests for methods/{train,vdn,qmix,extraction}.py and
eval/metrics.py.

Every expected value in this file is sourced from the spec's math (F1-F7 in
Phase1_Week2_Tasks.md, the D1 formula table, the metric definitions) or from
an *independent* computation (ground_truth.semivalues/enumerate on the true
payoff, vdn_lstsq_oracle's closed-form solve) -- never by running the
trained models and copying whatever number came out. That is what makes
these correctness checks rather than change-detectors.

Training configs are fixed here, not tuned against the still-pending Task 7
pilot (CLAUDE.md: "No test may depend on the pilot"):
- STEPS_FAST/LR_FAST: used wherever the property under test holds at *any*
  point of training (determinism, VDN's four extractions coinciding, the F3
  error bound), so a short run is enough and keeps the suite fast.
- STEPS_FIT/LR_FIT: used wherever the test needs the model actually close to
  converged (matching the lstsq oracle, F4's lambda-invariance, and the
  "greedy is optimal" precondition for extraction C).
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from eval.metrics import credit_error, fit_error, regret
from games.generator import make_game
from ground_truth.baselines import fixed_action, uniform
from ground_truth.enumerate import coalition_values
from ground_truth.semivalues import banzhaf_weights, loo_weights, semivalue, shapley_weights
from methods.extraction import extract_A, extract_B, extract_C, extract_D
from methods.qmix import QMIX
from methods.train import enumerate_joint_actions, joint_from_product, seed_to_int, train
from methods.vdn import VDN, vdn_lstsq_oracle

# Reuse Week 1's asymmetric baseline and game-size grid instead of
# redefining them -- same tool, same reason: a symmetric baseline can't
# detect an axis-order bug (Key Learning 5).
from tests.test_axioms import GRID, asymmetric_baseline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STEPS_FAST, LR_FAST = 150, 0.1
STEPS_FIT, LR_FIT = 2000, 0.1


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _train(model_cls, payoff, weights, steps, lr, seed, mode="full_batch",
           sigma=0.0, log_every=None, **model_kwargs):
    """Build a model and train it, mirroring the seeding convention recorded
    in Week2_Report.md's Task 3: init_seed and data_seed are independent
    spawns of one SeedSequence, and the model gets a plain int (its
    torch.Generator can't consume a SeedSequence directly)."""
    n = payoff.ndim
    n_actions = payoff.shape[0]
    init_seed, data_seed = np.random.SeedSequence(seed).spawn(2)
    model = model_cls(n, n_actions, seed_to_int(init_seed), **model_kwargs)
    history = train(
        model, payoff, weights=weights, mode=mode, steps=steps, lr=lr,
        sigma=sigma, data_seed=seed_to_int(data_seed),
        log_every=log_every or steps,
    )
    return model, history


def _true_counterfactual_credit(payoff, baseline, greedy):
    """The D1 table's formula for C -- Q_tot(a*) - E_{a_i~B_i}[Q_tot(a_i,
    a*_-i)] -- applied to the *true* payoff instead of a learned Q_tot.
    Independent of extraction.py: this is extraction C's own formula, read
    off the spec table, re-applied to the ground-truth tensor so it serves
    as C's ground truth rather than a second copy of the implementation."""
    n = baseline.shape[0]
    val_at_greedy = payoff[tuple(greedy)]
    phi = np.zeros(n)
    for i in range(n):
        slicer = tuple(int(greedy[j]) if j != i else slice(None) for j in range(n))
        vary_i = payoff[slicer]
        phi[i] = val_at_greedy - (baseline[i] * vary_i).sum()
    return phi


def _permute_snapshot(snapshot, baseline, perm):
    """Relabel agents by `perm`: new agent i is old agent perm[i]. Every
    per-agent array is row-indexed by perm; the joint tensor's axis i must
    come from the old axis perm[i], which is exactly np.transpose's
    `axes=perm` convention."""
    perm = np.asarray(perm)
    permuted = {
        "q_tables": snapshot["q_tables"][perm],
        "greedy_action": snapshot["greedy_action"][perm],
        "mixer_grad": snapshot["mixer_grad"][perm],
        "q_tot_tensor": np.transpose(snapshot["q_tot_tensor"], axes=perm),
    }
    return permuted, baseline[perm]


# --------------------------------------------------------------------------
# 1. float64 everywhere
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ModelClass", [VDN, QMIX])
def test_all_parameters_float64(ModelClass):
    model = ModelClass(3, 3, seed_to_int(np.random.SeedSequence(0)))
    for p in model.parameters():
        assert p.dtype == torch.float64

    payoff = make_game(3, 3, lam=0.0, seed=0)["payoff"]
    weights = joint_from_product(uniform(3, 3))
    _, history = _train(ModelClass, payoff, weights, STEPS_FAST, LR_FAST, seed=1)
    snapshot = history[-1]
    assert snapshot["q_tables"].dtype == np.float64
    assert snapshot["q_tot_tensor"].dtype == np.float64
    assert snapshot["mixer_grad"].dtype == np.float64


# --------------------------------------------------------------------------
# 2. ground_truth/games never import torch or methods
# --------------------------------------------------------------------------

def test_ground_truth_imports_no_torch():
    code = (
        "import sys\n"
        "import games.generator\n"
        "import ground_truth.baselines\n"
        "import ground_truth.enumerate\n"
        "import ground_truth.semivalues\n"
        "assert 'torch' not in sys.modules, sorted(sys.modules)\n"
        "assert not any(m == 'methods' or m.startswith('methods.') for m in sys.modules), sorted(sys.modules)\n"
    )
    env = dict(os.environ)
    # Set PYTHONPATH explicitly rather than relying on cwd: `python -c`
    # always puts cwd on sys.path[0], which would make the import work even
    # with no PYTHONPATH at all and silently re-hide the trailing-colon bug
    # Task 0b removed (Progress_Log.md). Running from a neutral cwd makes
    # PYTHONPATH the only thing that can make this import succeed.
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT.parent),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------
# 3. determinism
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ModelClass", [VDN, QMIX])
@pytest.mark.parametrize("mode,sigma", [("full_batch", 0.0), ("sampled", 0.1)])
def test_determinism(ModelClass, mode, sigma):
    payoff = make_game(3, 3, lam=1.0, seed=0)["payoff"]
    weights = joint_from_product(uniform(3, 3))
    finals = []
    for _ in range(2):
        _, history = _train(ModelClass, payoff, weights, STEPS_FAST, LR_FAST,
                             seed=7, mode=mode, sigma=sigma)
        finals.append(history[-1]["q_tot_tensor"])
    np.testing.assert_array_equal(finals[0], finals[1])


# --------------------------------------------------------------------------
# 4. VDN matches the closed-form lstsq oracle
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.0, 1.0, 2.0])
def test_vdn_matches_lstsq(lam):
    payoff = make_game(3, 3, lam=lam, seed=0)["payoff"]
    weights = joint_from_product(uniform(3, 3))
    _, history = _train(VDN, payoff, weights, STEPS_FIT, LR_FIT, seed=0)
    trained = history[-1]["q_tot_tensor"]

    # Independent computation: a linear-algebra solve, not the trained
    # forward pass.
    _, oracle_tensor = vdn_lstsq_oracle(payoff, weights)
    np.testing.assert_allclose(trained, oracle_tensor, atol=1e-6)


# --------------------------------------------------------------------------
# 5. F4: VDN's fitted Q_tot equals the additive part under uniform data
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.0, 1.0, 2.0])
def test_vdn_lambda_invariance_uniform(lam):
    game = make_game(3, 3, lam=lam, seed=0)
    payoff, f = game["payoff"], game["f"]
    n, n_actions = f.shape

    # Rebuild the additive part directly from f -- A(a) = sum_i f[i, a_i] --
    # rather than importing games.generator's private _additive_tensor.
    A = np.zeros((n_actions,) * n)
    for i in range(n):
        shape = [1] * n
        shape[i] = n_actions
        A = A + f[i].reshape(shape)

    weights = joint_from_product(uniform(n, n_actions))
    _, history = _train(VDN, payoff, weights, STEPS_FIT, LR_FIT, seed=0)
    trained = history[-1]["q_tot_tensor"]
    np.testing.assert_allclose(trained, A, atol=1e-4)


# --------------------------------------------------------------------------
# 6. F1: raw Q_i(a_i*) is gauge-dependent, the gap is not
# --------------------------------------------------------------------------

def test_vdn_gap_gauge_invariant():
    rng = np.random.default_rng(0)
    n, n_actions = 3, 4
    q_tables = rng.normal(size=(n, n_actions))
    greedy = np.array([int(np.argmax(q_tables[i])) for i in range(n)])
    B = uniform(n, n_actions)

    snapshot = {"q_tables": q_tables, "greedy_action": greedy}
    gap_before = extract_A(snapshot, B)

    # F1's own construction: a per-agent shift that sums to zero, so Q_tot
    # (= sum_i Q_i[a_i] for VDN) is unchanged but raw Q_i(a_i*) moves.
    c = np.array([5.0, -2.0, -3.0])
    assert abs(c.sum()) < 1e-12
    shifted = q_tables + c[:, None]
    snapshot_shifted = {"q_tables": shifted, "greedy_action": greedy}
    gap_after = extract_A(snapshot_shifted, B)

    np.testing.assert_allclose(gap_after, gap_before, atol=1e-10)
    raw_before = q_tables[np.arange(n), greedy]
    raw_after = shifted[np.arange(n), greedy]
    np.testing.assert_allclose(raw_after - raw_before, c, atol=1e-12)


# --------------------------------------------------------------------------
# 7. F2: VDN's four extractions coincide, at every lambda / baseline / semivalue
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.0, 1.0, 2.0])
def test_vdn_extractions_coincide(lam):
    n, n_actions = 3, 3
    baselines = {
        "fixed": fixed_action(n, n_actions),
        "uniform": uniform(n, n_actions),
        "asymmetric": asymmetric_baseline(n, n_actions),
    }
    weight_vectors = {
        "shapley": shapley_weights(n),
        "banzhaf": banzhaf_weights(n),
        "loo": loo_weights(n),
    }
    train_weights = joint_from_product(uniform(n, n_actions))

    for seed in range(3):
        payoff = make_game(n, n_actions, lam=lam, seed=seed)["payoff"]
        # STEPS_FAST is enough: the identity holds for any Q_tables of a VDN
        # model (additive Q_tot, mixer_grad == 1), not only a converged one.
        _, history = _train(VDN, payoff, train_weights, STEPS_FAST, LR_FAST, seed=seed)
        snapshot = history[-1]

        for bname, B in baselines.items():
            a = extract_A(snapshot, B)
            b = extract_B(snapshot, B)
            c = extract_C(snapshot, B)
            for wname, w in weight_vectors.items():
                d = extract_D(snapshot, B, w)
                label = f"lam={lam} seed={seed} baseline={bname} weights={wname}"
                np.testing.assert_allclose(a, b, atol=1e-9, err_msg=f"A vs B, {label}")
                np.testing.assert_allclose(a, c, atol=1e-9, err_msg=f"A vs C, {label}")
                np.testing.assert_allclose(a, d, atol=1e-9, err_msg=f"A vs D, {label}")


# --------------------------------------------------------------------------
# 8. F3: credit error <= 2 * fit error, for D always and for C at lambda=0
#    when greedy is optimal
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ModelClass", [VDN, QMIX])
@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_credit_error_bounded_by_fit_D(ModelClass, lam):
    n, n_actions = 3, 3
    payoff = make_game(n, n_actions, lam=lam, seed=0)["payoff"]
    train_weights = joint_from_product(uniform(n, n_actions))
    # The bound is unconditional (v is a max of expectations, both
    # non-expansive, and the semivalue weights sum to 1), so it must hold
    # even at a short, unconverged training run.
    _, history = _train(ModelClass, payoff, train_weights, STEPS_FAST, LR_FAST, seed=0)
    snapshot = history[-1]

    B = uniform(n, n_actions)
    v_true = coalition_values(payoff, B)
    for wname, w in [("shapley", shapley_weights(n)),
                      ("banzhaf", banzhaf_weights(n)),
                      ("loo", loo_weights(n))]:
        phi_true = semivalue(v_true, n, w)
        phi_hat = extract_D(snapshot, B, w)
        ce = credit_error(phi_hat, phi_true)["sup"]
        fe = fit_error(snapshot["q_tot_tensor"], payoff)["sup"]
        assert ce <= 2 * fe + 1e-9, (ModelClass.__name__, lam, wname, ce, fe)


def test_credit_error_bounded_by_fit_C_at_lambda0_greedy_optimal():
    n, n_actions = 3, 3
    payoff = make_game(n, n_actions, lam=0.0, seed=0)["payoff"]
    train_weights = joint_from_product(uniform(n, n_actions))
    B = uniform(n, n_actions)

    for ModelClass in (VDN, QMIX):
        # Needs the "greedy is optimal" precondition, which (unlike the D
        # bound above) does depend on convergence -- hence STEPS_FIT.
        _, history = _train(ModelClass, payoff, train_weights, STEPS_FIT, LR_FIT, seed=0)
        snapshot = history[-1]
        greedy = snapshot["greedy_action"]
        assert regret(payoff, greedy) < 1e-6, (ModelClass.__name__, greedy)

        phi_true = _true_counterfactual_credit(payoff, B, greedy)
        phi_hat = extract_C(snapshot, B)
        ce = credit_error(phi_hat, phi_true)["sup"]
        fe = fit_error(snapshot["q_tot_tensor"], payoff)["sup"]
        assert ce <= 2 * fe + 1e-9, (ModelClass.__name__, ce, fe)


# --------------------------------------------------------------------------
# 9. QMIX non-negativity is enforced on the forward path, not just defined
# --------------------------------------------------------------------------

def test_qmix_nonnegative_on_forward_path():
    n, n_actions = 2, 3
    model = QMIX(n, n_actions, seed_to_int(np.random.SeedSequence(0)))
    with torch.no_grad():
        # Force the *raw* parameters negative. If the forward path (and the
        # autograd path through it) didn't actually route through .abs(),
        # this would make dQ_tot/dQ_i negative somewhere -- the
        # "unreachable code path" failure mode named in CLAUDE.md.
        model.W1.copy_(-model.W1.abs() - 1.0)
        model.W2.copy_(-model.W2.abs() - 1.0)

    for a in enumerate_joint_actions(n, n_actions):
        grad = model.mixer_grad(a)
        assert np.all(grad >= -1e-9), (a, grad)


# --------------------------------------------------------------------------
# 10. QMIX's decentralized greedy action is a joint argmax (IGM)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n,n_actions", GRID)
def test_qmix_igm(n, n_actions):
    # Structural: holds for any Q_tables, so an untrained model is enough.
    seed = np.random.SeedSequence(1000 * n + n_actions)
    model = QMIX(n, n_actions, seed_to_int(seed))
    tensor = model.q_tot_tensor()
    joint_amax = np.unravel_index(int(np.argmax(tensor)), tensor.shape)
    assert tuple(int(x) for x in model.greedy_action()) == joint_amax


# --------------------------------------------------------------------------
# 11. q_tot_tensor()'s axis order matches q_tot(actions)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ModelClass", [VDN, QMIX])
@pytest.mark.parametrize("n,n_actions", GRID)
def test_q_tot_tensor_axis_order(ModelClass, n, n_actions):
    seed = np.random.SeedSequence(2000 * n + n_actions)
    model = ModelClass(n, n_actions, seed_to_int(seed))

    actions_np = enumerate_joint_actions(n, n_actions)
    actions_t = torch.from_numpy(actions_np).long()
    with torch.no_grad():
        vals = model.q_tot(actions_t).cpu().numpy()

    tensor = model.q_tot_tensor()
    # Index the tensor at every row of actions_np in the same order vals is in.
    expected = tensor[tuple(actions_np.T)]
    np.testing.assert_allclose(vals, expected, atol=1e-12)


# --------------------------------------------------------------------------
# 12. extraction is equivariant under relabeling agents (needs an
#     asymmetric baseline -- Key Learning 5)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ModelClass", [VDN, QMIX])
def test_extraction_permutation_equivariance(ModelClass):
    n, n_actions = 3, 3
    payoff = make_game(n, n_actions, lam=1.0, seed=0)["payoff"]
    train_weights = joint_from_product(uniform(n, n_actions))
    _, history = _train(ModelClass, payoff, train_weights, STEPS_FAST, LR_FAST, seed=0)
    snapshot = history[-1]

    B = asymmetric_baseline(n, n_actions)
    perm = [1, 2, 0]  # non-identity; new agent i == old agent perm[i]
    permuted_snapshot, B_perm = _permute_snapshot(snapshot, B, perm)
    w = shapley_weights(n)

    extractors = {
        "A": lambda snap, b: extract_A(snap, b),
        "B": lambda snap, b: extract_B(snap, b),
        "C": lambda snap, b: extract_C(snap, b),
        "D": lambda snap, b: extract_D(snap, b, w),
    }
    for name, fn in extractors.items():
        original = fn(snapshot, B)
        permuted = fn(permuted_snapshot, B_perm)
        np.testing.assert_allclose(permuted, original[perm], atol=1e-9, err_msg=name)


# --------------------------------------------------------------------------
# 13. normalized credit error is NaN, never 0, when phi_true is zero
# --------------------------------------------------------------------------

def test_normalized_error_nan_on_zero_truth():
    # Hand-computed: ||[1,2,3]|| = sqrt(14), sup|diff| = 3.
    result = credit_error(np.array([1.0, 2.0, 3.0]), np.zeros(3))
    assert math.isnan(result["normalized_l2"])
    assert math.isclose(result["raw_l2"], math.sqrt(14.0), rel_tol=1e-12)
    assert math.isclose(result["sup"], 3.0, rel_tol=1e-12)


# --------------------------------------------------------------------------
# 14. added after Task 7 at the user's request (not in the original Task 6
#     table): train() now logs "lr" at every history step. Checked against
#     PyTorch's own documented closed form for CosineAnnealingLR with
#     eta_min=0, lr_t = base_lr * (1 + cos(pi * t / T_max)) / 2 -- an
#     independent oracle (it's the scheduler's published formula, not
#     anything read off our trainer's output), not just "it went down."
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ModelClass", [VDN, QMIX])
def test_lr_schedule_logged_matches_cosine_closed_form(ModelClass):
    n, n_actions = 3, 3
    payoff = make_game(n, n_actions, lam=0.0, seed=0)["payoff"]
    train_weights = joint_from_product(uniform(n, n_actions))
    steps, lr0, log_every = 200, 0.1, 20

    _, history = _train(ModelClass, payoff, train_weights, steps, lr0, seed=0,
                         log_every=log_every)

    logged_steps = [h["step"] for h in history]
    logged_lrs = [h["lr"] for h in history]

    expected = [lr0 * (1.0 + math.cos(math.pi * t / steps)) / 2.0 for t in logged_steps]
    np.testing.assert_allclose(logged_lrs, expected, atol=1e-12, rtol=1e-9)

    # Monotonically non-increasing (cosine from 0 to T_max is), base lr at
    # step 0, and annealed to ~0 by the final logged step.
    assert logged_lrs[0] == pytest.approx(lr0)
    assert all(a >= b - 1e-12 for a, b in zip(logged_lrs, logged_lrs[1:]))
    assert logged_lrs[-1] == pytest.approx(0.0, abs=1e-12)
