"""
Generate ``observations.pkl`` for the LightTunnel benchmark.

Draws 5000 rows from ``LightTunnel.define_SEM()`` (which routes the inputs
through ``lt.Deterministic`` for Y) and asserts ``corr(R, G) > 0.4``. The
correlation is the lever that makes the ``LightTunnel_WrongRG`` misspec
actually bite -- if the data did not show R-G correlation, the spurious
intra-cluster edge would not affect CBO's adjustment.

Run from the repo root:
    python -m ccbo.cbo.data.LightTunnel.generate_observations
"""

import os
import numpy as np
import pandas as pd

from ccbo.generic_do import sample_from_model


N_SAMPLES = 5000
SEED = 0
MIN_RG_CORR = 0.4


def main():
    # Bootstrap with an empty DataFrame so LightTunnel.__init__ has the
    # required columns; we discard the bootstrap object and rebuild the SEM
    # from a fresh empty-data instance for sampling.
    from ccbo.cbo.graphs.LightTunnel import LightTunnel

    bootstrap = pd.DataFrame({c: [] for c in ('R', 'G', 'B', 'P1', 'P2', 'Y')})
    sem_fn = LightTunnel(bootstrap).define_SEM

    np.random.seed(SEED)
    samples = []
    for _ in range(N_SAMPLES):
        # 7 i.i.d. standard normals: one per node in the SEM order
        # (U_color, R, G, B, P1, P2, Y). The SEM scales each by its sigma.
        eps = np.random.randn(7)
        samples.append(sample_from_model(sem_fn(), epsilon=eps))

    df = pd.DataFrame(samples)
    # Observable columns only (drop U_color from the on-disk file -- it is
    # a latent confounder and CBO does not observe it).
    df = df[['R', 'G', 'B', 'P1', 'P2', 'Y']]

    rg_corr = float(df['R'].corr(df['G']))
    print(f"Generated {len(df)} samples; corr(R, G) = {rg_corr:.3f}")
    assert rg_corr > MIN_RG_CORR, (
        f"corr(R, G) = {rg_corr:.3f} is too low (need > {MIN_RG_CORR}); "
        f"the WrongRG misspecification would not affect CBO without this "
        f"observational correlation. Increase _SIGMA_U_COLOR in "
        f"ccbo/cbo/graphs/LightTunnel.py or N_SAMPLES."
    )

    out_path = os.path.join(os.path.dirname(__file__), 'observations.pkl')
    df.to_pickle(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
