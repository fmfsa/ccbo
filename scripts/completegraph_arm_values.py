"""
Ground-truth arm values for CompleteGraph, cited in the paper (Sec. 6.2).

Brute-force Monte Carlo over the true SEM: for every arm S in the finest
exploration set, grid-search min_x E[Y | do(X_S = x)] within the
interventional ranges. Establishes:

  * the optimum is the joint arm {B, D} (mu* = -3.60 at B=-5, D=3.3);
  * forcing E destroys it (mu*({B,D,E}) = -0.57) -- natural E compensates
    U1 via A (Lee-Bareinboim "where to intervene" phenomenon);
  * the Prop-4 floor of partition {B},{D,E} is mu*({B}) = -1.31, since
    do(B, D) is not a union of clusters there.

Run from the repo root:
    PYTHONPATH=. python scripts/completegraph_arm_values.py
"""

import itertools

import numpy as np
import pandas as pd

from ccbo.tests.test_do_accuracy import true_do_mc
from ccbo.run_experiment import get_original_graph

ARMS = [['B'], ['D'], ['E'],
        ['B', 'D'], ['B', 'E'], ['D', 'E'],
        ['B', 'D', 'E']]
GRID_1D, GRID_2D, GRID_3D = 25, 13, 9
MC_1D, MC_2D, MC_3D = 8000, 4000, 3000


def main():
    obs = pd.read_pickle('ccbo/cbo/data/CompleteGraph/observations.pkl')[:100]
    g = get_original_graph('CompleteGraph', obs, None)
    sem, ranges = g.define_SEM, g.get_interventional_ranges()

    print(f"{'arm':14s} {'mu*':>9s}  argmin")
    for arm in ARMS:
        pts = {1: GRID_1D, 2: GRID_2D, 3: GRID_3D}[len(arm)]
        mc = {1: MC_1D, 2: MC_2D, 3: MC_3D}[len(arm)]
        grids = [np.linspace(*ranges[v], pts) for v in arm]
        best = (np.inf, None)
        for vals in itertools.product(*grids):
            m, _ = true_do_mc(sem, arm, list(vals), num_samples=mc)
            if m < best[0]:
                best = (m, vals)
        print(f"{str(arm):14s} {best[0]:9.3f}  {np.round(best[1], 2)}")


if __name__ == '__main__':
    main()
