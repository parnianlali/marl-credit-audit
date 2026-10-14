"""Credit assignments in all ways,
Reads off the four candidates named in D1 (decisions.md).
"""

import numpy as np

from ground_truth.baselines import check_baseline
from ground_truth.enumerate import coalition_values
from ground_truth.semivalues import semivalue

#snapshot is a dict, You build it by running train.py's training loop.
# snapshot = {
#    "step": 999,                     
 #   "q_tables": <(n_agents, n_actions) array>,  
  #  "q_tot_tensor": <array>,        
   # "greedy_action": <(n_agents,) int array>,   
    #"mixer_grad": <(n_agents,) array>,          
    #"train_loss": 0.0001,   }        

def extract_A(snapshot, B):
    """Utility gap: Q_i(a_i*) - E_{B_i}[Q_i]."""
    q_tables = snapshot["q_tables"]
    n, n_actions = q_tables.shape
    check_baseline(B, n, n_actions)

    greedy = snapshot["greedy_action"]
    qi_at_astar = q_tables[np.arange(n), greedy]        # (n,)
    baseline_exp = (B * q_tables).sum(axis=1)             # (n,)
    return qi_at_astar - baseline_exp


def extract_B(snapshot, B):
    """Gradient-scaled gap: dQ_tot/dQ_i at a*, times A."""
    q_tables = snapshot["q_tables"]
    n, n_actions = q_tables.shape
    check_baseline(B, n, n_actions)

    return snapshot["mixer_grad"] * extract_A(snapshot, B)


def extract_C(snapshot, B):
    """COMA-style counterfactual: Q_tot(a*) - E_{a_i~B_i}[Q_tot(a_i, a*_-i)]."""
    q_tables = snapshot["q_tables"]
    n, n_actions = q_tables.shape
    check_baseline(B, n, n_actions)

    q_tot_tensor = snapshot["q_tot_tensor"]
    greedy = snapshot["greedy_action"]
    qtot_at_astar = q_tot_tensor[tuple(greedy)]

    phi = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # fix every agent but i at its greedy action, vary i's own action
        slicer = tuple(int(greedy[j]) if j != i else slice(None) for j in range(n))
        qtot_vary_i = q_tot_tensor[slicer]                 # (n_actions,)
        expected_i = (B[i] * qtot_vary_i).sum()
        phi[i] = qtot_at_astar - expected_i
    return phi


def extract_D(snapshot, B, weights):
    """Semivalue on the learned Q_tot: semivalue(coalition_values(Q_tot, B), weights).

    `weights` is required -- which semivalue (Shapley/Banzhaf/LOO) is not
    pre-registered anywhere, so there is no default to pick silently.
    """
    q_tables = snapshot["q_tables"]
    n, n_actions = q_tables.shape
    check_baseline(B, n, n_actions)

    v = coalition_values(snapshot["q_tot_tensor"], B)
    return semivalue(v, n, weights)
