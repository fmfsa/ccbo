"""Historical reproduction check for five old ToyGraph MCBO trajectories.

This is NOT a scientific correctness/release gate. Those historical trajectories
used the upstream intervention-coordinate bug. Flatness and exact historical
means only identify the archived result; they are never acceptance requirements
for corrected experiments. Explicit --historical-reproduction is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_MEAN = 1.1433154226418354
EXPECTED_SEM = 0.27128671442865404


def verify(results_dir: Path, trials: int, tolerance: float) -> None:
    finals = []
    errors = []
    for seed in range(5):
        stem = results_dir / f"trial_results_MCBO_ToyGraph_{seed}"
        csv_path = stem.with_suffix(".csv")
        info_path = Path(f"{stem}_info.json")
        if not csv_path.exists():
            errors.append(f"missing {csv_path}")
            continue
        if not info_path.exists():
            errors.append(f"missing {info_path}")
            continue

        df = pd.read_csv(csv_path)
        if "current_optimal" not in df:
            errors.append(f"{csv_path}: missing current_optimal column")
            continue
        values = df["current_optimal"].to_numpy(float)
        if len(values) != trials:
            errors.append(f"{csv_path}: expected {trials} rows, found {len(values)}")
        if not np.isfinite(values).all():
            errors.append(f"{csv_path}: non-finite current_optimal value")
        if np.any(np.diff(values) < -tolerance):
            errors.append(f"{csv_path}: best-so-far decreases under maximization")
        if np.ptp(values) > tolerance:
            errors.append(
                f"{csv_path}: recorded trajectory is not flat "
                f"(range={np.ptp(values):.6g})")

        with info_path.open() as fh:
            info = json.load(fh)
        if info.get("env") != "ToyGraph" or info.get("algo") != "MCBO":
            errors.append(f"{info_path}: unexpected env/algo metadata")
        if int(info.get("num_trials", -1)) != trials:
            errors.append(f"{info_path}: unexpected num_trials metadata")
        finals.append(values[-1])

    if len(finals) == 5:
        arr = np.asarray(finals)
        mean = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        if not np.isclose(mean, EXPECTED_MEAN, atol=tolerance, rtol=0):
            errors.append(f"final mean {mean:.12g} != {EXPECTED_MEAN:.12g}")
        if not np.isclose(sem, EXPECTED_SEM, atol=tolerance, rtol=0):
            errors.append(f"final s.e. {sem:.12g} != {EXPECTED_SEM:.12g}")
    else:
        mean = sem = float("nan")

    if errors:
        raise SystemExit("Historical MCBO reproduction mismatch:\n- " + "\n- ".join(errors))
    print(
        f"PASS: 5 ToyGraph MCBO trajectories, {trials} rows each, "
        f"flat recorded best-so-far, final={mean:.6f} +/- {sem:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("results/qmcbo"))
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--historical-reproduction", action="store_true",
                        help="Confirm that this is only an old-archive identity check")
    args = parser.parse_args()
    if not args.historical_reproduction:
        parser.error("This flat-trajectory check only identifies historical outputs. "
                     "Use --historical-reproduction for archives; corrected experiments "
                     "must pass protocol/coordinate/scorer validity tests instead.")
    verify(args.dir, args.trials, args.tolerance)


if __name__ == "__main__":
    main()
