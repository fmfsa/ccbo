"""Generate ``observations.pkl`` for the FrontDoor SCM (MinimalBench).

Draws ``N_SAMPLES`` rows from ``FrontDoor.define_SEM()`` and runs the
oracle gate:

  G1  confounding is visible: corr(X1, Y-residual channel) via U requires
      corr(X1, M) high (mechanism) and E[Y | M] biased relative to the
      backdoor functional — we check the naive regression minimum differs
      from the closed form (the lever that makes the B1 tier drop bite).
  G2  Monte-Carlo E[Y | do(arm)] matches the closed forms in ccbo.minibench:
      do(M=C) = 0, do(X1=C/B) = the noise-smoothed well value (~0.336),
      do(X1,M=C) = 0 (lossless cluster arm).

Run from the repo root:
    python -m ccbo.cbo.data.FrontDoor.generate_observations
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
    from ccbo.cbo.graphs.FrontDoor import FrontDoor

    bootstrap = pd.DataFrame({c: [] for c in mb.FD_NODES})
    sem_fn = FrontDoor(bootstrap).define_SEM

    np.random.seed(SEED)
    samples = []
    for _ in range(N_SAMPLES):
        # 4 i.i.d. standard normals: U, eX1, eM, eY.
        eps = np.random.randn(4)
        samples.append(sample_from_model(sem_fn(), epsilon=eps))

    df = pd.DataFrame(samples)[mb.FD_NODES]  # drop latent U

    corr = float(df['X1'].corr(df['M']))
    mean_y = float(df['Y'].mean())
    print(f"Generated {len(df)} samples; corr(X1, M) = {corr:.3f}; "
          f"E[Y] = {mean_y:.3f} "
          f"(analytic {mb.ORACLE[mb.FD_NAME]['obs_mean_y']:.3f})")
    assert corr > 0.9, "the X1 -> M mechanism should dominate M"
    assert abs(mean_y - mb.ORACLE[mb.FD_NAME]['obs_mean_y']) < TOL

    # --- G2: MC arm values vs closed forms ---------------------------------
    o = mb.ORACLE[mb.FD_NAME]
    checks = [
        (['M'], mb.FD_C, o['y_star']),
        (['X1'], mb.FD_C / mb.FD_B, o['best_do_x1']),
        (['X1', 'M'], np.array([0.0, mb.FD_C]), o['v_pi']),
    ]
    for arm, value, analytic in checks:
        mc_mean, _ = compute_do_generic(
            sem_fn, arm, df, {}, value, num_mc_samples=MC_SAMPLES, seed=1)
        print(f"E[Y|do({','.join(arm)})] at optimum: MC {mc_mean:+.4f} "
              f"vs analytic {analytic:+.4f}")
        assert abs(mc_mean - analytic) < TOL, (arm, mc_mean, analytic)

    # --- G1: the naive (confounded) regression is biased -------------------
    # E[Y | M = C] != E[Y | do(M = C)] because M proxies X1 hence U.
    band = df[(df['M'] - mb.FD_C).abs() < 0.1]
    naive = float(band['Y'].mean())
    print(f"naive E[Y | M~C] = {naive:+.4f} vs do-value {o['y_star']:+.4f} "
          f"(bias {naive - o['y_star']:+.4f})")
    assert abs(naive - o['y_star']) > 0.05, (
        "observational confounding too weak: the backdoor adjustment (and "
        "hence the B1 prior corruption) would not bite")

    out_path = os.path.join(os.path.dirname(__file__), 'observations.pkl')
    df.to_pickle(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
