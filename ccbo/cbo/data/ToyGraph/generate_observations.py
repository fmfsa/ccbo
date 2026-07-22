"""Generate ``observations.pkl`` for the ToyGraph benchmark (CBO-2020 toy).

Draws ``--n`` rows (default 1000) from ``ToyGraph.define_SEM()`` and stores
the observable columns X, Z, Y only — the latent confounder U (U -> X,
U -> Y) is sampled but dropped from the on-disk file, so the observational
data *carries* the confounding without exposing it.  Asserts corr(X, Y) is
clearly positive (the U-induced dependence), so the bidirected X <-> Y edge
declared in ``ccbo/coarsening.py::get_hidden_confounders('ToyGraph')`` is
actually reflected in the data.

Protocol note: the CBO-2020 paper's "N = 100 observational samples" is the
*initial* slice — the runners pass ``initial_num_obs_samples=100`` and use
only ``observations.pkl[:100]`` to fit priors at trial 0.  The remaining
rows are the pool the CBO loop's epsilon-greedy *observe* trials draw from
(20 fresh rows per observation), exactly like the other vendored benchmarks
(CompleteGraph ships 500 rows, SimplifiedCoralGraph 1000, Tier1Graph 5000).
A pool of exactly 100 rows starves the observe steps (they become no-ops)
and leaves the do(Z) prior with no coverage below Z ~= -2.5, which repels
the search from the optimum z* ~= -3.2.

Run from the repo root:
    python -m ccbo.cbo.data.ToyGraph.generate_observations [--n 1000 --seed 0]
"""

import os
import argparse
import numpy as np
import pandas as pd

from ccbo.generic_do import sample_from_model


N_SAMPLES = 1000  # full pool; runners slice [:100] (the paper's N=100)
SEED = 0
MIN_XY_CORR = 0.2


def main():
    parser = argparse.ArgumentParser(
        description='Generate ToyGraph observational data')
    parser.add_argument('--n', type=int, default=N_SAMPLES,
                        help='Number of observational samples (default 1000)')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed (default 0)')
    args = parser.parse_args()

    from ccbo.cbo.graphs.ToyGraph import ToyGraph

    bootstrap = pd.DataFrame({c: [] for c in ('X', 'Z', 'Y')})
    sem_fn = ToyGraph(bootstrap).define_SEM

    np.random.seed(args.seed)
    samples = []
    for _ in range(args.n):
        # 4 i.i.d. standard normals: one per node in SEM order (U, X, Z, Y).
        eps = np.random.randn(4)
        samples.append(sample_from_model(sem_fn(), epsilon=eps))

    df = pd.DataFrame(samples)
    # Observable columns only (drop the latent confounder U from the on-disk
    # file -- CBO does not observe it).
    df = df[['X', 'Z', 'Y']]

    xy_corr = float(df['X'].corr(df['Y']))
    print(f"Generated {len(df)} samples; corr(X, Y) = {xy_corr:.3f}")
    assert xy_corr > MIN_XY_CORR, (
        f"corr(X, Y) = {xy_corr:.3f} is too low (need > {MIN_XY_CORR}); "
        f"the latent confounder U -> {{X, Y}} is not visible in the "
        f"observational data. Check the SEM in ccbo/cbo/graphs/ToyGraph.py "
        f"or increase --n."
    )

    out_path = os.path.join(os.path.dirname(__file__), 'observations.pkl')
    df.to_pickle(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
