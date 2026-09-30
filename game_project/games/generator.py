

from string import ascii_lowercase

import numpy as np

# Tolerance for the orthogonality assertion on G.
_ORTHOGONALITY_ATOL = 1e-10

_STD_FLOOR = 1e-12


# --------------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------------

def _axis_letters(n):
    """ describe each agent with a letter, exmaple: agent1 = a"""
    assert n <= len(ascii_lowercase), "it supports at most 26 agents"
    return ascii_lowercase[:n]


def _broadcast_shape(n, i, n_actions):
    """ create a tuple of the actions of the agents
        turns one agent's vector into a shape that NumPy can broadcast
    """
    shape = [1] * n
    shape[i] = n_actions
    return tuple(shape)


def _additive_tensor(f, n, n_actions):
    """Build A(a) = sum(f(a)) as a tensor. 
    """
    A = np.zeros((n_actions,) * n)
    for i in range(n):
        A = A + f[i].reshape(_broadcast_shape(n, i, n_actions))
    return A


def _other_axes(n, i):
    """Every axis except agent i's. 
    Used to average over everybody else's actions while keeping agent i's action fixed."""
    return tuple(j for j in range(n) if j != i)



# --------------------------------------------------------------------------
# general view:
# --------------------------------------------------------------------------

# _axis_letters()
#        ↓
# creates labels like "abc"
#        ↓
# used later by einsum


# _broadcast_shape()
#        ↓
# turns one agent's vector into a shape
# that NumPy can broadcast


# _additive_tensor()
#        ↓
# f[0] + f[1] + ... + f[n-1]
#        ↓
# full A(a) tensor


# _other_axes()
#        ↓
# "everyone except agent i"
#        ↓
# used to calculate conditional means







# --------------------------------------------------------------------------
# the game family
# --------------------------------------------------------------------------

def make_game(n, n_actions, lam, seed):
    """Random n-agent, |A|-action one-step cooperative game with synergy dial `lam`.
    """

    rng = np.random.default_rng(seed)

    # ---- steps 2-3: additive part -------------------------------------
    f_raw = rng.normal(size=(n, n_actions))
    A_raw = _additive_tensor(f_raw, n, n_actions)

    # ---- step 4: raw synergy ------------------------------------------
    G_raw = rng.normal(size=(n_actions,) * n)

    # ---- step 5: remove the additive projection ------------------------
    m = G_raw.mean()
    P = np.full((n_actions,) * n, -(n - 1) * m)
    for i in range(n):
        M_i = G_raw.mean(axis=_other_axes(n, i))  # shape (n_actions,)
        P = P + M_i.reshape(_broadcast_shape(n, i, n_actions))
    G = G_raw - P

    # ---- step 6: normalize, guarding a zero standard deviation ---------
    std_A = A_raw.std()
    if std_A < _STD_FLOOR:
        raise ValueError(
            "additive tensor A has degenerate standard deviation %.3e (< %.0e), "
            "so it cannot be normalized. This means every agent's f_i is "
            "effectively constant across its actions and the game carries no "
            "additive signal. Check that n_actions > 1."
            % (std_A, _STD_FLOOR)
        )
    f = f_raw / std_A
    A = _additive_tensor(f, n, n_actions)  # rebuilt from rescaled f: exact

    std_G = G.std()
    if std_G < _STD_FLOOR:
        raise ValueError(
            "synergy tensor G has degenerate standard deviation %.3e (< %.0e), "
            "so it cannot be normalized. The additive projection annihilated G "
            "entirely. This happens whenever n == 1: a single agent's payoff is "
            "purely additive by definition, so there is no synergy subspace left "
            "to keep. Got n = %d."
            % (std_G, _STD_FLOOR, n)
        )
    G = G / std_G

    # ---- step 7 --------------------------------------------------------
    payoff = A + lam * G


# --------------------------------------------------------------------------
    # ---- spec-mandated assertions --------------------------------------

    #condition of omiting additive part of G
    for i in range(n):
        residual = G.mean(axis=_other_axes(n, i))
        assert residual.shape == (n_actions,), (
            "orthogonality residual for agent %d has shape %s, expected %s"
            % (i, residual.shape, (n_actions,))
        )
        assert np.allclose(residual, 0.0, atol=_ORTHOGONALITY_ATOL), (
            "G is not orthogonal to the additive subspace for agent %d: "
            "max |residual| = %.3e" % (i, np.abs(residual).max())
        )

    assert payoff.shape == (n_actions,) * n, (
        "payoff has shape %s, expected %s" % (payoff.shape, (n_actions,) * n)
    )
    # a simple testing that we sure the true payoff doesn't change meanwhile the procces
    np.testing.assert_allclose(
        payoff,
        _additive_tensor(f, n, n_actions) + lam * G,
        atol=1e-12,
        err_msg="payoff is not exactly sum_i f[i, a_i] + lam * g[a]",
    )

    # At lam = 0 the synergy term must be gone exactly, not approximately.
    if lam == 0:
        assert np.array_equal(payoff, A), "payoff != A at lam = 0"

    return {
        "payoff": payoff,
        "f": f,
        "g": G,
        "n": n,
        "n_actions": n_actions,
        "lam": lam,
        "seed": seed,
    }


def sample_reward(payoff, joint_action, sigma, rng):
    """Draw one noisy reward for `joint_action`.
    """
    return float(payoff[tuple(joint_action)] + sigma * rng.normal())


def make_mono_game(n, n_actions, penalty, target=0, high=8.0):
    """another game: Monotonicity-violation game to test IGM

    k(a) = the number of agents playing target:

        payoff = high       if k == n     (everyone coordinates)
        payoff = -penalty   if 0 < k < n  (someone defected -- the trap)
        payoff = 0          if k == 0     (nobody tried)

    """
    shape = (n_actions,) * n

    idx = np.indices(shape)
    k = (idx == target).sum(axis=0)

    # idx[0] is telling: what action agent 0 is taking at every joint-action location.
    payoff = np.where(k == n, float(high), np.where(k == 0, 0.0, -float(penalty)))

    assert payoff.shape == shape, (
        "payoff has shape %s, expected %s" % (payoff.shape, shape)
    )

    return {
        "payoff": payoff,
        "n": n,
        "n_actions": n_actions,
        "penalty": penalty,
        "target": target,
        "high": high,
    }


# --------------------------------------------------------------------------
# diagnostics (NOT metrics, tools for examining structural properties of the game.)
# --------------------------------------------------------------------------

def marginal_greedy_action(payoff, baseline):
    """For each agent, what action looks best if everyone else follows
       the baseline distribution? (for each agent not globally)

    For agent i:   argmax_{a_i} E_{a_{-i} ~ B}[ payoff(a) ]
    above formulation says: For agent i, fix its action to
    a_i, let everyone else act according to the baseline
    distribution, calculate the expected team reward, and choose
    whichever a_i gives the highest expected reward.
    """
    baseline = np.asarray(baseline)
    assert baseline.ndim == 2, "baseline must be 2-D, got ndim %d" % baseline.ndim
    n, n_actions = baseline.shape
    assert payoff.shape == (n_actions,) * n, (
        "payoff shape %s is inconsistent with baseline shape %s"
        % (payoff.shape, baseline.shape)
    )

    letters = _axis_letters(n)
    out = np.empty(n, dtype=int)

    for i in range(n):
        others = [j for j in range(n) if j != i] 
        in_subs = [letters] + [letters[j] for j in others]
        subs = ",".join(in_subs) + "->" + letters[i]
        operands = [payoff] + [baseline[j] for j in others]

        expected = np.einsum(subs, *operands)  # shape (n_actions,)
        assert expected.shape == (n_actions,), (
            "marginal for agent %d has shape %s, expected %s"
            % (i, expected.shape, (n_actions,))
        )
        out[i] = int(np.argmax(expected))

    return out


def joint_argmax(payoff):
    """it asks Which complete joint action has the highest payoff?
    """
    flat_index = int(np.argmax(payoff))
    return tuple(int(a) for a in np.unravel_index(flat_index, payoff.shape))
