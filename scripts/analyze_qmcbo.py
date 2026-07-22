"""Score QMCBO/MCBO trial CSVs (E0/E1/E2 of the QMCBO pilot).

Reads trial_results_<ALGO>_<ENV>_<SEED>[ _e2 ].csv files from a directory and
prints, per (env, algo, variant): final best (MCBO maximizes; envs negate
minimization targets), best-so-far at budget checkpoints, mean wall-clock,
and — for E2 pairs — whether the correct-vs-perturbed trajectories are
byte-identical per seed.

Run:  PYTHONPATH=. conda run -n ccbo python scripts/analyze_qmcbo.py <dir>
"""

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd


def collect(d):
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, "trial_results_*.csv"))):
        m = re.match(r"trial_results_(MCBO|QMCBO)_([A-Za-z_0-9]+)_(\d+)\.csv",
                     os.path.basename(p))
        if not m:
            continue
        algo, env, seed = m.group(1), m.group(2), int(m.group(3))
        variant = "e2" if os.path.basename(d).endswith("e2") or "_e2" in d else ""
        tr = pd.read_csv(p)["current_optimal"].astype(float).tolist()
        info_p = p.replace(".csv", "_info.json")
        secs = misspec = None
        if os.path.exists(info_p):
            info = json.load(open(info_p))
            secs, misspec = info.get("secs"), info.get("misspec", "")
        rows.setdefault((env, algo, misspec or ""), {})[seed] = (tr, secs)
    return rows


def main():
    dirs = sys.argv[1:] or ["results/qmcbo"]
    all_rows = {}
    for d in dirs:
        all_rows.update(collect(d))

    for (env, algo, misspec), seeds in sorted(all_rows.items()):
        trajs = [t for t, _ in seeds.values()]
        n = min(len(t) for t in trajs)
        arr = np.array([t[:n] for t in trajs])
        secs = [s for _, s in seeds.values() if s]
        tag = f" [misspec {misspec}]" if misspec else ""
        cps = [c for c in (10, 25, 50, 100) if c < n]
        cp_str = "  ".join(f"@{c}={arr[:, c].mean():+.3f}" for c in cps)
        sem = (arr[:, -1].std(ddof=1) / np.sqrt(len(trajs))
               if len(trajs) > 1 else 0.0)
        secs_str = f"  ({np.mean(secs):.0f}s/unit)" if secs else ""
        print(f"{env:12s} {algo:6s}{tag:18s} n={len(trajs)} "
              f"final={arr[:, -1].mean():+.4f}±{sem:.4f}  {cp_str}{secs_str}")

    # Byte-identity check across variants of the same (env, algo, seed).
    by_key = {}
    for (env, algo, misspec), seeds in all_rows.items():
        for seed, (tr, _) in seeds.items():
            by_key.setdefault((env, algo, seed), {})[misspec] = tr
    for (env, algo, seed), variants in sorted(by_key.items()):
        if len(variants) > 1:
            base = variants.get("")
            for mv, tr in variants.items():
                if mv == "" or base is None:
                    continue
                ident = (len(tr) == len(base)
                         and all(a == b for a, b in zip(tr, base)))
                print(f"IDENTITY {env} {algo} seed{seed} correct-vs-[{mv}]: "
                      f"{'BYTE-IDENTICAL' if ident else 'DIFFERS'}")


if __name__ == "__main__":
    main()
