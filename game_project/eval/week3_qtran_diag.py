"""Week 3 Task 6b -- diagnostic, not a test. QMIX on the QTRAN matrix, 10
inits, under each D6 regime (R1 uniform, R2 central weighting alpha = 0.01).
Reports the greedy action and regret per init.

Expected direction (Task 1a): uniform misses (0,0); central finds it.
"""

import numpy as np

from eval.metrics import regret
from games.generator import make_mono_game
from ground_truth.baselines import uniform
from methods.qmix import QMIX
from methods.train import central_weights, joint_from_product, seed_to_int, train

STEPS, LR = 20000, 0.1          # smallest Task 7b budget; not a D11 decision
INIT_SEEDS = range(100, 110)    # pilot-range seeds; seeds 0-9 stay untouched
ALPHA = 0.01                    # D6

payoff = make_mono_game(2, 3, penalty=12, high=8.0)["payoff"]
regimes = {
    "R1 uniform": joint_from_product(uniform(2, 3)),
    "R2 central": central_weights(payoff, ALPHA),
}

for name, w in regimes.items():
    print("\n%s  (QMIX, %d steps, lr %g)" % (name, STEPS, LR))
    print("init  greedy  regret  final_loss")
    n_found = 0
    for s in INIT_SEEDS:
        init_seed, data_seed = np.random.SeedSequence(s).spawn(2)
        model = QMIX(2, 3, seed_to_int(init_seed))
        hist = train(model, payoff, weights=w, mode="full_batch", steps=STEPS,
                     lr=LR, sigma=0.0, data_seed=seed_to_int(data_seed),
                     log_every=STEPS)
        g = hist[-1]["greedy_action"]
        r = regret(payoff, g)
        n_found += r == 0
        print("%4d  %-6s  %6.2f  %.3e" % (s, tuple(int(x) for x in g), r,
                                          hist[-1]["train_loss"]))
        qt = np.asarray(hist[-1]["q_tot_tensor"])
        qs = hist[-1]["q_tables"]
        print("      Q1=%s  Q2=%s" % (np.round(qs[0], 2), np.round(qs[1], 2)))
        print("      Qjt(0,0)=%.2f  Qjt=%s" % (qt[0, 0], np.round(qt, 2).tolist()))
    print("optimal (0,0) found in %d/10 inits" % n_found)
