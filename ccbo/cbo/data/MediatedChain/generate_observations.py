"""Generate ``observations.pkl`` for the MediatedChain SCM (MinimalBench).

Draws ``N_SAMPLES`` rows from ``MediatedChain.define_SEM()`` and runs the
oracle gate:

  G1  the price identity: MC best-in-box E[Y | do(X2)] minus MC
      E[Y | do(X1 = C/BETA)] matches ``minibench.mc_price()`` = 3.96.
  G2  the mechanism: X2's natural response under do(X1 = C/BETA) exceeds
      the policy box (BETA * x1* = 4 > 2), so the whole-cluster arm cannot
      reach y*.

Run from the repo root:
    python -m ccbo.cbo.data.MediatedChain.generate_observations
"""

import os
import numpy as np
import pandas as pd

from ccbo.generic_do import sample_from_model, compute_do_generic
from ccbo import minibench as mb


N_SAMPLES = 5000
SEED = 0
TOL = 0.05
MC_SAMPLES = 200000


def main():
    from ccbo.cbo.graphs.MediatedChain import MediatedChain

    bootstrap = pd.DataFrame({c: [] for c in mb.MC_NODES})
    sem_fn = MediatedChain(bootstrap).define_SEM

    np.random.seed(SEED)
    samples = []
    for _ in range(N_SAMPLES):
        # 3 i.i.d. standard normals: eX1, eX2, eY.
        eps = np.random.randn(3)
        samples.append(sample_from_model(sem_fn(), epsilon=eps))

    df = pd.DataFrame(samples)[mb.MC_NODES]
    print(f"Generated {len(df)} samples; X2 in "
          f"[{df['X2'].min():.2f}, {df['X2'].max():.2f}] "
          f"(policy box {mb.MC_X2_BOX})")

    # --- G2: the fine optimum leaves the policy box ------------------------
    x1_star = mb.ORACLE[mb.MC_NAME]['x1_star']
    assert mb.MC_BETA * x1_star > mb.MC_X2_BOX[1], (
        "the fine optimum must drive X2 outside the policy box")

    # --- G1: MC price identity --------------------------------------------
    mc_fine, _ = compute_do_generic(
        sem_fn, ['X1'], df, {}, x1_star, num_mc_samples=MC_SAMPLES, seed=1)
    mc_coarse, _ = compute_do_generic(
        sem_fn, ['X2'], df, {}, mb.MC_X2_BOX[1],
        num_mc_samples=MC_SAMPLES, seed=1)
    price_mc = mc_coarse - mc_fine
    print(f"MC fine optimum   E[Y|do(X1={x1_star})]      = {mc_fine:+.4f} "
          f"(analytic {mb.ORACLE[mb.MC_NAME]['y_star']:+.4f})")
    print(f"MC coarse optimum E[Y|do(X2={mb.MC_X2_BOX[1]})] = "
          f"{mc_coarse:+.4f} (analytic {mb.ORACLE[mb.MC_NAME]['v_pi']:+.4f})")
    print(f"MC price = {price_mc:.4f} (analytic {mb.mc_price():.4f})")
    assert abs(mc_fine - mb.ORACLE[mb.MC_NAME]['y_star']) < TOL
    assert abs(mc_coarse - mb.ORACLE[mb.MC_NAME]['v_pi']) < TOL
    assert abs(price_mc - mb.mc_price()) < 2 * TOL

    out_path = os.path.join(os.path.dirname(__file__), 'observations.pkl')
    df.to_pickle(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
