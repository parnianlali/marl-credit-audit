"""Shared training module for VDN and QMIX.

This module contains the common training procedure used by both
value-decomposition methods in our project. The goal is simple:
train the model so that its predicted Q_tot matches the game's payoff
under the chosen joint-action weighting.

The module also supports the two training modes used in our experiments:
- full_batch: train on every joint action using the exact weighted MSE.
- sampled: sample one joint action from the chosen weighting and train
  on its noisy reward.

The weighting here is a probability distribution over COMPLETE joint
actions. This is different from the per-agent baseline used in the
ground-truth credit calculations.

For our current project decisions:
- D3: the training weighting is uniform.
- D4: use Adam with cosine learning-rate decay to zero.
- The same trainer is reused for VDN and QMIX; the model itself handles
  how the individual Q-values are combined into Q_tot.

The training history stores the model's state at selected steps so that
the later credit-evaluation stage can use the trained values without
having to retrain the model.
"""

from string import ascii_lowercase

import numpy as np
import torch

from games.generator import sample_reward

_WEIGHTS_TOL = 1e-12


def _axis_letters(n):
    assert n <= len(ascii_lowercase)
    return ascii_lowercase[:n]


def enumerate_joint_actions(n, n_actions):
    """Generates all possible joint actions for n agents
    """
    shape = (n_actions,) * n
    idx = np.indices(shape) 
    actions = idx.reshape(n, -1).T   
    assert actions.shape == (n_actions ** n, n), (
        "enumerate_joint_actions produced shape %s, expected %s"
        % (actions.shape, (n_actions ** n, n))
    )
    return actions


def check_joint_weights(w, n, n_actions, allow_zeros=False):
    """Check that the joint-action training weights are a valid probability
distribution.

The weights must have one entry for every possible joint action, be
non-negative, finite, and sum to 1.

In our current setup (D3), these weights are uniform, so every joint
action is trained equally.

Full support is required by default because later credit extraction
evaluates Q_tot for every joint action. A zero-weight action would
therefore never be trained in full-batch mode.

Returns a validated float64 copy.
"""
    w = np.asarray(w, dtype=np.float64)
    shape = (n_actions,) * n

    if w.shape != shape:
        raise ValueError(
            "joint weights has shape %s, expected %s = (n_actions,)*n"
            % (w.shape, shape)
        )

    # Checks whether all elements of w are finite
    if not np.all(np.isfinite(w)):
        raise ValueError("joint weights contains NaN or inf")

    if w.min() < -_WEIGHTS_TOL:
        # Finds the index of the minimum element in w
        worst = np.unravel_index(int(np.argmin(w)), w.shape)
        raise ValueError(
            "joint weights has a negative entry at %s: %.6g. Entries are "
            "probabilities and must be >= 0." % (worst, w[worst])
        )

    # sum of the weights should be 1.
    total = w.sum()
    if abs(total - 1.0) > _WEIGHTS_TOL:
        raise ValueError(
            "joint weights sums to %.12g, not 1 (|error| = %.3e > tol %.0e)"
            % (total, abs(total - 1.0), _WEIGHTS_TOL)
        )

    # rise error if w.min was zero or < 0
    if not allow_zeros and w.min() <= 0.0:
        worst = np.unravel_index(int(np.argmin(w)), w.shape)
        raise ValueError(
            "joint weights has a non-positive entry at %s: %.6g. A "
            "zero-weight joint action leaves Q_tot unconstrained there "
            "under full_batch training, and extraction D reads Q_tot at "
            "every joint action. Pass allow_zeros=True to permit this "
            "explicitly." % (worst, w[worst])
        )

    return w.copy()


def joint_from_product(B):
    B = np.asarray(B, dtype=np.float64)
    n, n_actions = B.shape
    letters = _axis_letters(n)
    # take n vectors, each labeled by its own letter,
    #   and produce an n-dim output using all letters. gives: "a,b,c->abc"
    subs = ",".join(letters) + "->" + letters
    # produces a joint tensor over all n agents outside a coalition.
    tensor = np.einsum(subs, *[B[i] for i in range(n)])
    assert tensor.shape == (n_actions,) * n, (
        "joint_from_product produced shape %s, expected %s"
        % (tensor.shape, (n_actions,) * n)
    )
    return tensor


def seed_to_int(seed_sequence):
    """Turns a SeedSequence into a single reproducible integer seed
    """
    return int(seed_sequence.generate_state(1)[0])

def train(model, payoff, *, weights, mode, steps, lr, sigma, data_seed, log_every):
    """Train `model` (VDN or QMIX) so its q_tot matches `payoff` under
`weights`, with Adam + cosine-decay-to-zero (D4), for `steps` steps.

Parameters:
model : VDN or QMIX instance, with the shared q_tables/q_tot/q_tot_tensor/
    greedy_action/mixer_grad contract.
payoff : ndarray, shape (n_actions,)*n -- the ground-truth game.
weights : ndarray, shape (n_actions,)*n -- a joint-action distribution
    (validated by check_joint_weights); what "full_batch" weights its MSE
    by, and what "sampled" draws joint actions from.
mode : "full_batch" -- exact weighted MSE over every joint action, every
    step (data_seed unused). "sampled" -- one joint action drawn from
    `weights` per step, trained on that action's reward + N(0, sigma^2)
    noise (games.generator.sample_reward), using a np.random.Generator
    seeded by `data_seed` (an int -- pass seed_to_int(data_seed), not a
    bare SeedSequence, so "sampled" draws the same stream here as in
    tests/test_methods.py).
steps, lr : optimizer step count and initial learning rate. The scheduler
    is torch.optim.lr_scheduler.CosineAnnealingLR(T_max=steps, eta_min=0.0).
log_every : log a history entry at step 0, every `log_every` steps, and the
    final step.

Returns:
history : list of dicts, one per logged step, with keys "step", "q_tables",
    "q_tot_tensor", "greedy_action", "mixer_grad", "train_loss" (the
    full-batch weighted MSE, computed even in "sampled" mode, so fit error
    is always comparable across both modes), and "lr" (the optimizer's
    current learning rate, read off `opt.param_groups`, not the scheduler,
    since `sched.step()` already mutated it in place by log time).
"""
    # Point to the phase1-week2-D3
    if mode not in ("full_batch", "sampled"):
        raise ValueError("mode must be 'full_batch' or 'sampled', got %r" % mode)

    payoff = np.asarray(payoff, dtype=np.float64)
    n = payoff.ndim
    n_actions = payoff.shape[0]
    assert payoff.shape == (n_actions,) * n, (
        "payoff has shape %s, not square-per-axis as (n_actions,)*n requires"
        % (payoff.shape,)
    )
    weights = check_joint_weights(weights, n, n_actions)

    actions_np = enumerate_joint_actions(n, n_actions)             # (M, n)
    # Converts a NumPy array actions_np into a PyTorch tensor integers 
    actions_t = torch.from_numpy(actions_np).long()
    payoff_flat = payoff.reshape(-1)                                # (M,)
    y_t = torch.from_numpy(payoff_flat)
    w_flat = weights.reshape(-1)                                    # (M,), same flat order as actions_np/payoff_flat
    w_t = torch.from_numpy(w_flat)

    # phase1-week2-D4 (Adam + cosine lr rate optimizer)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)

    # authorize to use sample training later
    rng = np.random.default_rng(data_seed) if mode == "sampled" else None

    def full_batch_weighted_mse():
        with torch.no_grad():
            qtot = model.q_tot(actions_t)
            # MSE loss
            return (w_t * (qtot - y_t) ** 2).sum().item()

    history = []

    def log(step):
        q_tables_np = model.q_tables().detach().cpu().numpy().astype(np.float64).copy()
        greedy = np.asarray(model.greedy_action(), dtype=int)
        # gradient of the mixer output related to QMIX, no need to use nessecariy,
        # because we are using semivalue for QMIX
        mgrad = np.asarray(model.mixer_grad(greedy), dtype=np.float64)
        history.append({
            "step": step,
            "q_tables": q_tables_np,
            "q_tot_tensor": model.q_tot_tensor(),
            "greedy_action": greedy,
            "mixer_grad": mgrad,
            "train_loss": full_batch_weighted_mse(),
            # Current optimizer lr -- read off opt directly, not sched,
            # since sched.step() already mutates opt.param_groups in place;
            # this is the lr that will be used for this iteration's update.
            "lr": opt.param_groups[0]["lr"],
        })

    # run log only and only at the first step of the game and log every steps.
    for t in range(steps):
        if t == 0 or t % log_every == 0:
            log(t)

        # clear old gradients
        opt.zero_grad()
        if mode == "full_batch":
            qtot = model.q_tot(actions_t)
            # Calculate the loss MSE like above
            loss = (w_t * (qtot - y_t) ** 2).sum()
        else:
            # even if it was sample training, calculate the loss
            idx = rng.choice(w_flat.shape[0], p=w_flat)
            a = actions_np[idx]
            r = sample_reward(payoff, a, sigma, rng)
            a_t = torch.from_numpy(a).long().unsqueeze(0)           # (1, n)
            r_t = torch.tensor(r, dtype=torch.float64)
            loss = (model.q_tot(a_t)[0] - r_t) ** 2
        # calculate gradients
        loss.backward()
        # update the model's parameters using those gradients
        opt.step()
        # update the learning rate according to the scheduler
        sched.step()

    log(steps)   # always capture the final trained state
    return history
