"""VDN: Q_tot(a) = sum_i Q_i[a_i]
"""

import numpy as np
import torch
import torch.nn as nn

from methods.train import enumerate_joint_actions


class VDN(nn.Module):
    def __init__(self, n, n_actions, init_seed):
        super().__init__()
        self.n = n
        self.n_actions = n_actions
        gen = torch.Generator().manual_seed(int(init_seed))
        init = torch.randn(n, n_actions, generator=gen, dtype=torch.float64)
        self.Q = nn.Parameter(init)
        assert self.Q.dtype == torch.float64

    def q_tables(self):
        """The live, autograd-attached (n, n_actions) parameter."""
        return self.Q

    def q_tot(self, actions):
        B = actions.shape[0]
        # Creates an agent-index tensor of shape (B, n)
        agent_idx = torch.arange(self.n).unsqueeze(0).expand(B, -1)
        gathered = self.Q[agent_idx, actions]   # (B, n)
        return gathered.sum(dim=1)

    def q_tot_tensor(self):
        Q = self.Q.detach().cpu().numpy()
        shape = (self.n_actions,) * self.n
        total = np.zeros(shape)
        for i in range(self.n):
            bshape = [1] * self.n
            bshape[i] = self.n_actions
            total = total + Q[i].reshape(bshape)
        assert total.shape == shape, (
            "q_tot_tensor produced shape %s, expected %s" % (total.shape, shape)
        )
        return total

    def greedy_action(self):
        return self.Q.detach().cpu().numpy().argmax(axis=1).astype(int)

    def mixer_grad(self, a):
        """a sanity check, dQ_tot/dQ_i should be 1 for VDN
        """
        a = np.asarray(a)
        assert a.shape == (self.n,), (
            "mixer_grad: a has shape %s, expected (%d,)" % (a.shape, self.n)
        )
        return np.ones(self.n, dtype=np.float64)


def vdn_lstsq_oracle(payoff, weights):
    """calculates the best VDN decomposition of the payoff table,
      (The function finds the best individual Q-values for VDN)
    """
    from methods.train import check_joint_weights

    payoff = np.asarray(payoff, dtype=np.float64)
    n = payoff.ndim
    n_actions = payoff.shape[0]
    assert payoff.shape == (n_actions,) * n
    weights = check_joint_weights(weights, n, n_actions)

    actions = enumerate_joint_actions(n, n_actions)      # (M, n)
    M = actions.shape[0]
    w_flat = weights.reshape(-1)                          # (M,)
    y = payoff.reshape(-1)                                 # (M,), same flat order as actions

    X = np.zeros((M, n * n_actions))
    rows = np.arange(M)
    for i in range(n):
        # for every joint-action row, put a 1 in the column corresponding to
        #  the action chosen by agent i
        X[rows, i * n_actions + actions[:, i]] = 1.0

    sw = np.sqrt(w_flat)
    # Finds the individual Q-values that minimize the weighted squared error
    #  between actual payoffs and VDN's predicted payoffs
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)

    # The individual Q-tables. Each agent has a value for each of its actions.
    Q_oracle = beta.reshape(n, n_actions)
    # Each row represents one joint-action combination
    q_tot_oracle_tensor = (X @ beta).reshape((n_actions,) * n)
    # Adds the individual Q-values for every joint action to reconstruct
    #  VDN's predicted joint-value tensor
    return Q_oracle, q_tot_oracle_tensor
