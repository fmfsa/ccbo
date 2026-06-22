"""Generate ``observations.pkl`` for the ConfoundedCluster benchmark.

Draws ``N_SAMPLES`` rows from ``ConfoundedCluster.define_SEM()`` and asserts
``corr(B, C) > MIN_BC_CORR``.  The B-C correlation (induced by the shared
latent confounder U) is the lever that makes the ``ConfoundedCluster_WrongBC``
misspecification actually bite: without observational correlation, the
spurious intra-cluster edge B -> C would not change CBO's identifiability
verdict for do(B).

Run from the repo root:
    python -m ccbo.cbo.data.ConfoundedCluster.generate_observations
"""

import os
import numpy as np
import pandas as pd

from ccbo.generic_do import sample_from_model


N_SAMPLES = 5000
SEED = 0
MIN_BC_CORR = 0.4


def main():
    from ccbo.cbo.graphs.ConfoundedCluster import ConfoundedCluster

    bootstrap = pd.DataFrame({c: [] for c in ('A', 'B', 'C', 'Y')})
    sem_fn = ConfoundedCluster(bootstrap).define_SEM

    np.random.seed(SEED)
    samples = []
    for _ in range(N_SAMPLES):
        # 5 i.i.d. standard normals: one per node in SEM order
        # (U, A, B, C, Y). The SEM scales each by its sigma.
        eps = np.random.randn(5)
        samples.append(sample_from_model(sem_fn(), epsilon=eps))

    df = pd.DataFrame(samples)
    # Observable columns only (drop the latent confounder U from the on-disk
    # file -- CBO does not observe it).
    df = df[['A', 'B', 'C', 'Y']]

    bc_corr = float(df['B'].corr(df['C']))
    print(f"Generated {len(df)} samples; corr(B, C) = {bc_corr:.3f}")
    assert bc_corr > MIN_BC_CORR, (
        f"corr(B, C) = {bc_corr:.3f} is too low (need > {MIN_BC_CORR}); "
        f"the WrongBC misspecification would not affect CBO without this "
        f"observational correlation. Increase _SIGMA_U in "
        f"ccbo/cbo/graphs/ConfoundedCluster.py or N_SAMPLES."
    )

    out_path = os.path.join(os.path.dirname(__file__), 'observations.pkl')
    df.to_pickle(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
