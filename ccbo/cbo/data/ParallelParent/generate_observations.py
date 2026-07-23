"""Generate ``observations.pkl`` for the ParallelParent SCM (MinimalBench).

Draws ``N_SAMPLES`` rows from ``ParallelParent.define_SEM()`` and runs the
oracle gate:

  G1  corr(X1, X2) > MIN_CORR — the latent-U correlation is the lever that
      makes the A2 bow misspecification bite (analytic value 0.64).
  G2  Monte-Carlo E[Y | do(arm)] matches the closed forms in ccbo.minibench
      at the optimum of every arm (|MC - analytic| < TOL):
          do(X1=A, X2=B) = 0        do(X1=A) = 4.25        do(X2=B) = 4.25 LAM

Run from the repo root:
    python -m ccbo.cbo.data.ParallelParent.generate_observations
"""

import os
import numpy as np
import pandas as pd

from ccbo.generic_do import sample_from_model, compute_do_generic
from ccbo import minibench as mb


N_SAMPLES = 5000
SEED = 0
MIN_CORR = 0.4
TOL = 0.05
MC_SAMPLES = 200000


def main():
    from ccbo.cbo.graphs.ParallelParent import ParallelParent

    bootstrap = pd.DataFrame({c: [] for c in mb.PP_NODES})
    sem_fn = ParallelParent(bootstrap).define_SEM

    np.random.seed(SEED)
    samples = []
    for _ in range(N_SAMPLES):
        # 4 i.i.d. standard normals: U, eX1, eX2, eY.
        eps = np.random.randn(4)
        samples.append(sample_from_model(sem_fn(), epsilon=eps))

    df = pd.DataFrame(samples)[mb.PP_NODES]  # drop latent U

    # --- G1: confounder-induced correlation --------------------------------
    corr = float(df['X1'].corr(df['X2']))
    print(f"Generated {len(df)} samples; corr(X1, X2) = {corr:.3f} "
          f"(analytic {mb.PP_CORR:.2f})")
    assert corr > MIN_CORR, (
        f"corr(X1, X2) = {corr:.3f} too low (need > {MIN_CORR}); the A2 bow "
        f"misspecification would not bite. Increase PP_SIGMA_U in "
        f"ccbo/minibench.py.")

    # --- G2: MC arm values vs closed forms ---------------------------------
    checks = [
        (['X1', 'X2'], np.array([mb.PP_A, mb.PP_B]), 0.0),
        (['X1'], mb.PP_A, mb.ORACLE[mb.PP_NAME]['best_do_x1']),
        (['X2'], mb.PP_B, mb.ORACLE[mb.PP_NAME]['best_do_x2']),
    ]
    for arm, value, analytic in checks:
        mc_mean, _ = compute_do_generic(
            sem_fn, arm, df, {}, value, num_mc_samples=MC_SAMPLES, seed=1)
        print(f"E[Y|do({','.join(arm)})] at optimum: MC {mc_mean:+.4f} "
              f"vs analytic {analytic:+.4f}")
        assert abs(mc_mean - analytic) < TOL, (
            f"MC/oracle mismatch for do({arm}): {mc_mean:.4f} vs "
            f"{analytic:.4f} (tol {TOL})")

    out_path = os.path.join(os.path.dirname(__file__), 'observations.pkl')
    df.to_pickle(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
