"""Summarize the exact-population MinimalBench rerun.

The script validates the complete 30-seed trace grid and emits the numerical
aggregates used by the controlled-experiment figures and manuscript. Row zero
is the shared initial incumbent and is excluded from cumulative incumbent
regret, consistently with :mod:`ccbo.metrics`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ccbo import minibench as mb
from ccbo.metrics import cumulative_regret


GROUPS = [
    *( (mb.PP_NAME, cond, method)
       for cond in ("A0", "A1", "A2", "A3")
       for method in ("CBO", "QCBO") ),
    *( (mb.FD_NAME, cond, method)
       for cond in ("B0", "B1")
       for method in ("CBO", "QCBO") ),
    *( (mb.MC_NAME, "C0", method)
       for method in ("CBO", "QCBO", "HQCBO") ),
]


def _se(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _load(root: Path, scm: str, cond: str, method: str, seed: int) -> pd.DataFrame:
    path = root / f"{scm}_{cond}_{method}_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _paired_deviation(root: Path, scm: str, base: str, cond: str) -> dict:
    sup = 0.0
    decisions_equal = True
    for seed in range(30):
        left = _load(root, scm, base, "QCBO", seed)
        right = _load(root, scm, cond, "QCBO", seed)
        sup = max(sup, float(np.max(np.abs(
            left.best_y.to_numpy(float) - right.best_y.to_numpy(float)))))
        decisions_equal &= left.arm.equals(right.arm)
        decisions_equal &= left.x_values.equals(right.x_values)
    return {"sup_incumbent_deviation": sup,
            "decisions_equal": bool(decisions_equal)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/minimal_exact")
    parser.add_argument("--out", default="artifacts/summaries/minimal_exact.json")
    args = parser.parse_args()
    root = Path(args.dir)

    y_star = {mb.PP_NAME: 0.0, mb.FD_NAME: 0.0,
              mb.MC_NAME: mb.ORACLE[mb.MC_NAME]["y_star"]}
    summary: dict = {
        "protocol": {"seeds": 30, "trials": 50,
                     "mediated_chain_trials": 60,
                     "initial_points_per_arm": 3,
                     "population_targets": "exact"},
        "groups": {},
        "paired": {},
    }

    for scm, cond, method in GROUPS:
        frames = [_load(root, scm, cond, method, seed) for seed in range(30)]
        finals = np.asarray([frame.best_y.iloc[-1] for frame in frames], float)
        regrets = np.asarray([
            cumulative_regret(frame.best_y.to_numpy(float), y_star[scm])
            for frame in frames
        ], float)
        key = f"{scm}_{cond}_{method}"
        summary["groups"][key] = {
            "n": 30,
            "final_mean": float(finals.mean()),
            "final_se": _se(finals),
            "cumulative_regret_mean": float(regrets.mean()),
            "cumulative_regret_se": _se(regrets),
        }

    for scm, base, cond in (
        (mb.PP_NAME, "A0", "A1"),
        (mb.PP_NAME, "A0", "A2"),
        (mb.PP_NAME, "A0", "A3"),
        (mb.FD_NAME, "B0", "B1"),
    ):
        summary["paired"][f"{scm}_{base}_vs_{cond}_QCBO"] = \
            _paired_deviation(root, scm, base, cond)

    fd_delta = []
    for seed in range(30):
        correct = _load(root, mb.FD_NAME, "B0", "CBO", seed)
        wrong = _load(root, mb.FD_NAME, "B1", "CBO", seed)
        fd_delta.append(
            cumulative_regret(wrong.best_y.to_numpy(float), 0.0)
            - cumulative_regret(correct.best_y.to_numpy(float), 0.0))
    fd_delta_array = np.asarray(fd_delta)
    summary["paired"]["FrontDoor_B1_minus_B0_CBO_cumulative_regret"] = {
        "mean": float(fd_delta_array.mean()),
        "se": _se(fd_delta_array),
    }

    info_path = root / "refine_info.json"
    info = json.loads(info_path.read_text())
    triggers = np.asarray([
        record["trigger_step"] for record in info.values()
        if record.get("trigger_step") is not None
    ], float)
    summary["refinement"] = {
        "n": int(len(triggers)),
        "trigger_mean": float(triggers.mean()),
        "trigger_se": _se(triggers),
        "all_splits_accepted": all(not record.get("refused", False)
                                     for record in info.values()),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
