"""Release gate for the ToyGraph MCBO claim of the family suite (Exp. D).

The paper reports MCBO ToyGraph final 1.143 +- 0.271 (5 seeds, T=100) with a
best-so-far trajectory that remains flat across the 100 BO iterations. The
checked-in aggregate (results/qmcbo_pilot.json) is not sufficient evidence on
its own; this script verifies the claim against the raw per-seed CSVs, which
live on the experiment server (results/ is gitignored).

Checks, all of which must pass for the claim to be release-ready:
  (a) trial_results_MCBO_ToyGraph_{0..4}.csv and their _info.json exist;
  (b) each CSV has exactly --trials finite records;
  (c) the best-so-far sequence is monotone non-decreasing (maximization);
  (d) the recomputed final mean +- s.e. matches the published numbers;
  (e) flatness: max_t y_t^best - y_0^best == 0 per seed.

Note on (e): the runner records best_score during BO iterations, so the first
CSV row may already follow the first BO evaluation. A passing (e) supports
"the recorded best-so-far trajectory remains flat across the BO iterations",
NOT "MCBO never improves on its initial design". The stronger claim needs the
instrumented rerun described in RUN_TODO.md (log the initial-design incumbent
before BO, each proposed intervention, the raw objective value, and the
updated incumbent).

Run:  python scripts/verify_qmcbo_release.py [--dir results/qmcbo]
Exit status is nonzero on any failure.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

SEEDS = (0, 1, 2, 3, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/qmcbo")
    ap.add_argument("--env", default="ToyGraph")
    ap.add_argument("--algo", default="MCBO")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--expected-mean", type=float, default=1.143)
    ap.add_argument("--expected-sem", type=float, default=0.271)
    ap.add_argument("--tol", type=float, default=5e-3,
                    help="absolute tolerance on the recomputed mean/s.e. "
                         "against the published (3-decimal) numbers")
    args = ap.parse_args()

    failures = []
    finals = []

    for seed in SEEDS:
        csv = os.path.join(
            args.dir, f"trial_results_{args.algo}_{args.env}_{seed}.csv")
        info = csv.replace(".csv", "_info.json")

        # (a) presence
        if not os.path.exists(csv):
            failures.append(f"(a) missing {csv}")
            continue
        if not os.path.exists(info):
            failures.append(f"(a) missing {info}")
        else:
            with open(info) as fh:
                meta = json.load(fh)
            if "secs" not in meta:
                failures.append(f"(a) {info} lacks 'secs' (incomplete run?)")

        y = pd.read_csv(csv)["current_optimal"].to_numpy(float)

        # (b) count + finiteness
        if len(y) != args.trials:
            failures.append(
                f"(b) seed {seed}: {len(y)} records, expected {args.trials}")
        if not np.all(np.isfinite(y)):
            failures.append(f"(b) seed {seed}: non-finite best-so-far values")
            continue

        # (c) monotone under maximization
        if np.any(np.diff(y) < 0):
            t = int(np.argmax(np.diff(y) < 0))
            failures.append(
                f"(c) seed {seed}: best-so-far decreases at trial {t + 1}")

        # (e) flatness of the recorded trajectory
        if y.max() - y[0] != 0.0:
            failures.append(
                f"(e) seed {seed}: trajectory improves by {y.max() - y[0]:.6g}"
                " — the paper's flat-trajectory sentence must be revised")

        finals.append(y[-1])

    # (d) recompute the published aggregate
    if len(finals) == len(SEEDS):
        mean = float(np.mean(finals))
        sem = float(np.std(finals, ddof=1) / math.sqrt(len(finals)))
        if abs(mean - args.expected_mean) > args.tol:
            failures.append(
                f"(d) recomputed mean {mean:.4f} != {args.expected_mean}")
        if abs(sem - args.expected_sem) > args.tol:
            failures.append(
                f"(d) recomputed s.e. {sem:.4f} != {args.expected_sem}")
        print(f"recomputed final: {mean:.3f} +- {sem:.3f} "
              f"(published {args.expected_mean} +- {args.expected_sem})")
    else:
        failures.append("(d) cannot recompute aggregate: missing seeds")

    if failures:
        print(f"\nRELEASE GATE FAILED ({len(failures)} problem(s)):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("release gate passed: MCBO ToyGraph raw trajectories support the "
          "flat-trajectory claim")


if __name__ == "__main__":
    main()
