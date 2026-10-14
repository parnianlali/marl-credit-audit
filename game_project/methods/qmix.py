"""QMIX: Q_tot(a) = mixer(Q_1[a_1], ..., Q_n[a_n])
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.train import enumerate_joint_actions


class QMIX(nn.Module):
    def __init__(self, n, n_actions, init_seed, embed_dim=32):
        super().__init__()
        self.n = n
        self.n_actions = n_actions
        self.embed_dim = embed_dim
        gen = torch.Generator().manual_seed(int(init_seed))

        q_init = torch.randn(n, n_actions, generator=gen, dtype=torch.float64)
        self.Q = nn.Parameter(q_init)

        # Mixer: raw parameters, non-negativity applied on the forward path
        # (see _mix), not stored pre-transformed. Biases are unconstrained,
        # so no abs() is ever applied to b1/b2.
        self.W1 = nn.Parameter(torch.randn(embed_dim, n, generator=gen, dtype=torch.float64))
        self.b1 = nn.Parameter(torch.randn(embed_dim, generator=gen, dtype=torch.float64))
        self.W2 = nn.Parameter(torch.randn(1, embed_dim, generator=gen, dtype=torch.float64))
        self.b2 = nn.Parameter(torch.randn(1, generator=gen, dtype=torch.float64))

        assert self.Q.dtype == torch.float64
        for p in (self.W1, self.b1, self.W2, self.b2):
            assert p.dtype == torch.float64

    def q_tables(self):
        return self.Q

    def _mix(self, q):
        """individual agents' Q-values for one joint action and produces Q-total
        design 2layer networks
        """
        W1_eff = self.W1.abs()
        W2_eff = self.W2.abs()
        assert torch.all(W1_eff >= 0) and torch.all(W2_eff >= 0), (
            "mixer weights not non-negative on the forward path"
        )
        hidden = F.elu(q @ W1_eff.T + self.b1)          # (B, embed_dim)
        out = hidden @ W2_eff.T + self.b2                # (B, 1)
        return out.squeeze(-1)                            # (B,)

    def q_tot(self, actions):
        B = actions.shape[0]
        # Creates an agent-index tensor of shape (B, n)
        agent_idx = torch.arange(self.n).unsqueeze(0).expand(B, -1)
        gathered = self.Q[agent_idx, actions]   # (B, n)
        return self._mix(gathered)

    # calculates Q-t for EVERY possible joint action and puts the results into
    #  a tensor
    def q_tot_tensor(self):
        # numpy action
        actions_np = enumerate_joint_actions(self.n, self.n_actions)
        #convert numpy action to torch tensor
        actions_t = torch.from_numpy(actions_np).long()
        with torch.no_grad():
            vals = self.q_tot(actions_t).cpu().numpy().astype(np.float64)
        shape = (self.n_actions,) * self.n
        total = vals.reshape(shape)
        assert total.shape == shape, (
            "q_tot_tensor produced shape %s, expected %s" % (total.shape, shape)
        )

        # IGM check, inside the module: monotonic mixing of independent
        greedy = self.greedy_action()
        at_greedy = total[tuple(greedy)]
        # checks whether two values (at_greedy, total.max()) are approximately
        #  equal, within a tolerance or not.
        assert np.isclose(at_greedy, total.max(), rtol=1e-9, atol=1e-9), (
            "IGM check failed: q_tot_tensor at greedy_action %s = %.17g, max = %.17g"
            % (greedy.tolist(), at_greedy, total.max())
        )
        return total

    def greedy_action(self):
        return self.Q.detach().cpu().numpy().argmax(axis=1).astype(int)

    def mixer_grad(self, a):
        """this function calls _mix() to get that output, then calculates
            its gradients with respect to the input Q_i values.
            do this with dQ_tot/dQ_i gradient.
        """
        a = np.asarray(a)
        assert a.shape == (self.n,), (
            "mixer_grad: a has shape %s, expected (%d,)" % (a.shape, self.n)
        )
        # Extract each agent's Q-value
        with torch.no_grad():
            q_vals = self.Q[torch.arange(self.n), torch.from_numpy(a).long()].clone()
        # Enable gradient tracking
        q_vals.requires_grad_(True)
        out = self._mix(q_vals.unsqueeze(0)).squeeze(0)
        (grad,) = torch.autograd.grad(out, q_vals)
        return grad.detach().cpu().numpy().astype(np.float64)
