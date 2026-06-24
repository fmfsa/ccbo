"""Generate ``observations.pkl`` for the Tier1Graph benchmark.

Draws ``N_SAMPLES`` rows from ``Tier1Graph.define_SEM()`` and asserts that B and
the mediator C are observationally correlated (the B -> C edge), so that deleting
it (``Tier1Graph_NoBC``) actually changes the assumed prior means of the
B-containing arms.

Run from the repo root:
    python -m ccbo.cbo.data.Tier1Graph.generate_observations
"""

import os
import numpy as np
import pandas as pd

from ccbo.generic_do import sample_from_model


N_SAMPLES = 5000
SEED = 0
MIN_BC_ABS_CORR = 0.3


def main():
    from ccbo.cbo.graphs.Tier1Graph import Tier1Graph

    bootstrap = pd.DataFrame({c: [] for c in ('B', 'D', 'E', 'C', 'Y')})
    sem_fn = Tier1Graph(bootstrap).define_SEM

    np.random.seed(SEED)
    samples = []
    for _ in range(N_SAMPLES):
        # 5 i.i.d. standard normals: one per node in SEM order (B, D, E, C, Y).
        eps = np.random.randn(5)
        samples.append(sample_from_model(sem_fn(), epsilon=eps))

    df = pd.DataFrame(samples)[['B', 'D', 'E', 'C', 'Y']]

    bc_corr = float(df['B'].corr(df['C']))
    print(f"Generated {len(df)} samples; corr(B, C) = {bc_corr:.3f}")
    assert abs(bc_corr) > MIN_BC_ABS_CORR, (
        f"|corr(B, C)| = {abs(bc_corr):.3f} is too low (need > "
        f"{MIN_BC_ABS_CORR}); the NoBC misspecification would not bite without "
        f"observational B-C dependence. Increase _A_BC in "
        f"ccbo/cbo/graphs/Tier1Graph.py or N_SAMPLES."
    )

    out_path = os.path.join(os.path.dirname(__file__), 'observations.pkl')
    df.to_pickle(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
