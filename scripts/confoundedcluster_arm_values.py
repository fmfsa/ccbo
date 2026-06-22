"""
Ground-truth arm values for ConfoundedCluster, cited in the paper (Tier-2).

Brute-force Monte Carlo over the true SEM: for every candidate intervention
target S, grid-search min_x E[Y | do(X_S = x)] within the interventional
ranges. Establishes the Tier-2 story:

  * the best single-target intervention is {B} (the optimal singleton);
  * the WrongBC misspecification makes do(B) non-identifiable and deletes
    {B} from the fine exploration set, leaving {C} / {A} as the best
    *surviving* singletons -- an unrecoverable gap;
  * the coarse {B,C} cluster dominates every singleton and is invariant to
    the intra-cluster misspecification (see test_bow_do_effect.py).

Run from the repo root:
    PYTHONPATH=. python scripts/confoundedcluster_arm_values.py
"""

import itertools

import numpy as np
import pandas as pd

from ccbo.tests.test_do_accuracy import true_do_mc
from ccbo.run_experiment import get_original_graph

ARMS = [['A'], ['B'], ['C'], ['B', 'C'], ['A', 'B', 'C']]
# Targets deleted from the fine exploration set under ConfoundedCluster_WrongBC
# (do(B) is non-identifiable via the B->C + B<->C bow).
DELETED_UNDER_WRONGBC = {('B',)}
GRID_1D, GRID_2D, GRID_3D = 25, 13, 9
MC_1D, MC_2D, MC_3D = 8000, 4000, 3000


def main():
    obs = pd.read_pickle('ccbo/cbo/data/ConfoundedCluster/observations.pkl')[:100]
    g = get_original_graph('ConfoundedCluster', obs, None)
    sem, ranges = g.define_SEM, g.get_interventional_ranges()

    print(f"{'arm':14s} {'mu*':>9s}  {'argmin':>14s}  note")
    best_overall = (np.inf, None)
    best_surviving_singleton = (np.inf, None)
    for arm in ARMS:
        pts = {1: GRID_1D, 2: GRID_2D, 3: GRID_3D}[len(arm)]
        mc = {1: MC_1D, 2: MC_2D, 3: MC_3D}[len(arm)]
        grids = [np.linspace(*ranges[v], pts) for v in arm]
        best = (np.inf, None)
        for vals in itertools.product(*grids):
            m, _ = true_do_mc(sem, arm, list(vals), num_samples=mc)
            if m < best[0]:
                best = (m, vals)
        note = ''
        if tuple(arm) in DELETED_UNDER_WRONGBC:
            note = 'DELETED under WrongBC'
        if best[0] < best_overall[0]:
            best_overall = (best[0], arm)
        if len(arm) == 1 and tuple(arm) not in DELETED_UNDER_WRONGBC \
                and best[0] < best_surviving_singleton[0]:
            best_surviving_singleton = (best[0], arm)
        print(f"{str(arm):14s} {best[0]:9.3f}  {str(np.round(best[1], 2)):>14s}  {note}")

    print()
    print(f"Best overall target          : {best_overall[1]}  (mu* = {best_overall[0]:.3f})")
    print(f"Best singleton surviving WrongBC: {best_surviving_singleton[1]}  "
          f"(mu* = {best_surviving_singleton[0]:.3f})")


if __name__ == '__main__':
    main()
