"""Parameterized one-step cooperative game family.

Team reward:

    r(a) = A(a) + lam * G(a) + noise(sigma),   A(a) = sum_i f_i(a_i)

with three independent dials: `lam` (synergy / non-additivity), `sigma` (reward
noise), and a separate monotonicity-violation construction (`make_mono_game`).

Two conventions this module is deliberate about, both from the Week 1 task spec:

1. NOISE IS APPLIED AT SAMPLE TIME ONLY, NEVER IN THE PAYOFF TENSOR.
   `sigma` appears in exactly one place: `sample_reward`. The `payoff` array
   returned by the constructors is the noise-free game, so every coalition value
   and credit vector computed downstream is ground truth for the game you
   intended, not for one particular noise draw.

2. THE DIAGNOSTIC IS CALLED "MARGINAL-GREEDY MISMATCH", NOT AN IGM VIOLATION.
   `marginal_greedy_action` vs `joint_argmax` measures a property of a raw game.
   IGM (Individual-Global-Max) is a property of a *factorization* -- it needs a
   decomposer to even be defined, and none exists here. Conflating the two in
   code becomes conflating them in the paper.

The synergy term G is orthogonalized against the additive subspace (see
`make_game` step 5). Without that, raising `lam` would scale overall signal
magnitude rather than isolating non-additivity, so `lam` would not be the clean
independent variable the experiments treat it as. There is a known caveat to
this choice recorded in `reports/notes_baselines.md`.

Pure numpy. No PyTorch anywhere in Week 1.
"""

from string import ascii_lowercase

import numpy as np

# Tolerance for the orthogonality assertion on G. The projection removal is
# exact in real arithmetic, so anything above float round-off is a real bug.
_ORTHOGONALITY_ATOL = 1e-10

# A standard deviation below this is treated as degenerate. The normalizing
# division is then REFUSED with a clear error -- not silently performed, and not
# silently skipped. Dividing by ~0 would fill the tensor with inf/nan; skipping
# the division would quietly hand back a game whose scale does not match what
# every downstream caller assumes. Neither failure is detectable by eyeballing
# the output, so it raises instead.
#
# This branch is reachable: at n = 1 the additive projection annihilates G
# entirely, so G.std() is exactly 0 and make_game(1, ...) raises.
_STD_FLOOR = 1e-12


# --------------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------------

def _axis_letters(n):
    """One distinct einsum subscript letter per agent axis, e.g. n=3 -> 'abc'."""
    assert n <= len(ascii_lowercase), "einsum subscripts support at most 26 agents"
    return ascii_lowercase[:n]


def _broadcast_shape(n, i, n_actions):
    """Shape that places a length-`n_actions` vector on axis `i` and 1 elsewhere.

    Reshaping to this and adding lets numpy broadcast a per-agent vector across
    the full joint-action tensor without ever materializing an index grid.
    For n=3, i=1, n_actions=3 this is (1, 3, 1).
    """
    shape = [1] * n
    shape[i] = n_actions
    return tuple(shape)


def _additive_tensor(f, n, n_actions):
    """Build A(a) = sum_i f[i, a_i] as a full (n_actions,)*n tensor.

    Each agent's row f[i] is reshaped to sit on its own axis and broadcast
    against the others, so the sum over agents becomes a sum of n broadcast
    arrays rather than a loop over |A|^n joint actions.
    """
    A = np.zeros((n_actions,) * n)
    for i in range(n):
        A = A + f[i].reshape(_broadcast_shape(n, i, n_actions))
    return A


def _other_axes(n, i):
    """Every axis except agent i's. Used to average over 'everyone but i'."""
    return tuple(j for j in range(n) if j != i)


# --------------------------------------------------------------------------
# the game family
# --------------------------------------------------------------------------

def make_game(n, n_actions, lam, seed):
    """Random n-agent, |A|-action one-step cooperative game with synergy dial `lam`.

    Returns a dict with keys:
        payoff     (n_actions,)*n  -- noise-free team reward for every joint action
        f          (n, n_actions)  -- per-agent additive part, ALREADY RESCALED
                                      (see the scaling note below)
        g          (n_actions,)*n  -- normalized, orthogonalized synergy term
        n, n_actions, lam, seed    -- the arguments, echoed back

    ON THE SCALING OF `f`
    ---------------------
    Step 6 normalizes the additive tensor A by its standard deviation. Because
    A is a sum of per-agent terms, dividing A by a scalar s is *exactly* the
    same as dividing every f_i by s -- nothing is lost by pushing the scale down
    into f. We do that, and rebuild A from the rescaled f, so the identity

        payoff(a) == sum_i f[i, a_i]      (at lam = 0)

    holds exactly rather than only to float round-off. This matters because the
    Week 1 lam = 0 closed form is stated as

        Delta_i = max_{a_i} f_i(a_i) - E_{B_i}[f_i]

    with no scale factor in it, which is only true if the returned `f` is the
    rescaled one. Returning raw draws instead would make that closed form off by
    a constant factor and the most valuable test in the suite would fail for a
    reason that has nothing to do with the code being tested.

    The same applies to `g`: it is the post-projection, post-normalization array,
    not the raw draw. So `f` and `g` are the EXACT components of the returned
    `payoff`, and any caller may reconstruct it:

        payoff[a] == sum_i f[i, a_i] + lam * g[a]

    That identity is asserted in the constructor rather than assumed.

    Raises
    ------
    ValueError
        If either normalization would divide by a degenerate standard deviation
        (< _STD_FLOOR). Notably `make_game(1, ...)` raises, because the additive
        projection annihilates G entirely when there is only one agent.
    """
    rng = np.random.default_rng(seed)

    # ---- steps 2-3: additive part -------------------------------------
    f_raw = rng.normal(size=(n, n_actions))
    A_raw = _additive_tensor(f_raw, n, n_actions)

    # ---- step 4: raw synergy ------------------------------------------
    G_raw = rng.normal(size=(n_actions,) * n)

    # ---- step 5: remove the additive projection ------------------------
    # G_raw generically contains an additive component. If we left it in, then
    # raising lam would partly just add more additive signal, and lam would no
    # longer isolate non-additivity. So we subtract off P, the projection of
    # G_raw onto the additive subspace {sum_i h_i(a_i)}.
    #
    #   m       = grand mean of G_raw
    #   M_i(c)  = mean of G_raw over ALL OTHER agents' actions, given a_i = c
    #   P(a)    = sum_i M_i(a_i) - (n-1) * m
    #
    # The -(n-1)*m term corrects for the fact that each of the n conditional
    # means M_i already carries the grand mean m inside it, so naively summing
    # them counts m n times instead of once.
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

    # ---- spec-mandated assertions --------------------------------------
    # Orthogonality. Note carefully WHICH mean is asserted to vanish: for each
    # agent i, the mean of G over ALL AXES EXCEPT i must be the zero vector.
    # The superficially similar "mean over axis i is zero" is a DIFFERENT and
    # FALSE statement for n >= 3, and an assertion written that way would pass
    # happily while the projection was wrong.
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

    # The returned f and g must be the exact components of the returned payoff.
    # This is what lets Task 5's lam = 0 closed form be written against the
    # returned arrays with no scale factor in it. Asserted, not assumed, because
    # a future refactor that normalized payoff without normalizing f and g would
    # otherwise pass every other check in this file.
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

    THE ONLY place sigma enters the codebase. `payoff` is not modified, so the
    ground truth stays the ground truth for the noise-free game no matter how
    many samples are drawn from it.

    `rng` is supplied by the caller rather than created here, so that a caller
    drawing many samples advances one stream instead of restarting an identical
    one on every call.
    """
    return float(payoff[tuple(joint_action)] + sigma * rng.normal())


def make_mono_game(n, n_actions, penalty, target=0, high=8.0):
    """Monotonicity-violation game: full agreement pays, partial agreement punishes.

    With k(a) = the number of agents playing `target`:

        payoff = high       if k == n     (everyone coordinates)
        payoff = -penalty   if 0 < k < n  (someone defected -- the trap)
        payoff = 0          if k == 0     (nobody tried)

    `penalty` is the dial. Raising it deepens the trap around the joint optimum,
    so an agent reasoning about `target` in isolation sees it as increasingly
    bad while the joint argmax stays at all-target.

    Returns a dict with the same "payoff" key as `make_game`, so downstream code
    handles both constructors identically.
    """
    shape = (n_actions,) * n

    # np.indices(shape)[i] is the tensor whose entry at joint action a is a_i.
    # Comparing to `target` and summing over the leading agent axis gives k(a)
    # for every joint action at once, with no Python loop over |A|^n.
    idx = np.indices(shape)
    k = (idx == target).sum(axis=0)

    # Nested np.where: the outer picks off k == n, the inner separates k == 0
    # from the 0 < k < n middle band.
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
# diagnostics (NOT metrics -- see the module docstring)
# --------------------------------------------------------------------------

def marginal_greedy_action(payoff, baseline):
    """Each agent's best action assuming everyone else acts from `baseline`.

    For agent i:   argmax_{a_i} E_{a_{-i} ~ B}[ payoff(a) ]

    `baseline` is an (n, n_actions) row-stochastic matrix; row j is agent j's
    action distribution. Validation of row-stochasticity is NOT done here --
    `ground_truth/baselines.check_baseline` owns that. Only shapes are asserted.

    Returns an integer array of shape (n,).

    IMPLEMENTATION NOTE -- why einsum with a generated subscript string:
    the obvious alternative, contracting the other agents' axes one at a time
    with `tensordot`, is a known silent failure mode for this project. After
    each contraction the array loses an axis, so every remaining axis index
    shifts down by one; code that keeps using the original indices still runs
    and still returns numbers, and they are wrong. einsum names the axes by
    letter instead of by position, so nothing shifts.

    For n = 3 and i = 1 the generated call is literally:
        np.einsum('abc,a,c->b', payoff, baseline[0], baseline[2])
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
        # Build the subscript pieces as a list FIRST, then join. Joining a
        # possibly-empty list of others yields 'a->a' at n = 1, whereas
        # string-concatenating a comma unconditionally would yield the
        # malformed 'a,->a'.
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
    """The joint action attaining payoff.max(), as a tuple of ints.

    Ties are broken by the first occurrence in C (row-major) order, which is
    what np.argmax does on the flattened array.
    """
    flat_index = int(np.argmax(payoff))
    return tuple(int(a) for a in np.unravel_index(flat_index, payoff.shape))
