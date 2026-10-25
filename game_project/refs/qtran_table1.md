# QTRAN reference values (Task 1a / 6b)

Source: Son et al. 2019, "QTRAN: Learning to Factorize with Transformation for
Cooperative Multi-Agent Reinforcement Learning", ICML, arXiv:1905.05408.
Section 3.5, Table 1. Setup in Suppl. C.2, Table 3.

Actions A, B, C = indices 0, 1, 2.

## Table 1a -- payoff matrix
```
[[  8, -12, -12],
 [-12,   0,   0],
 [-12,   0,   0]]
```
Checked against `make_mono_game(2, 3, penalty=12, high=8.0)["payoff"]`: identical, same index order.

## Table 1f -- QMIX (trained)
- Q1 = [-1.02, 0.11, 0.10]   (agent 1, A/B/C)
- Q2 = [-0.92, 0.00, 0.01]   (agent 2, A/B/C)
- Qjt:
```
[[-8.08, -8.08, -8.08],
 [-8.08,  0.01,  0.03],
 [-8.08,  0.01,  0.02]]
```
- Greedy joint action: (B, C) = (1, 2), true payoff 0. Qjt(A,A) = -8.08 vs true 8.
- B and C are near-tied (0.11 vs 0.10, 0.00 vs 0.01), so (1,2) is not special.

## Table 1e -- VDN (context)
Greedy (C, B), Qjt(C,B) = -3.02.

## Paper's setup (Section 3.5, Suppl. C.2, Table 3)
epsilon = 1 (full uniform exploration), 20,000 steps, Adam lr 5e-4, replay
buffer 20,000, batch 32. Agent nets: 2 hidden layers of width 32. QMIX mixer:
1 hidden layer of width 32 with hypernetworks. ReLU. Single run, no seeds or
error bars reported.
