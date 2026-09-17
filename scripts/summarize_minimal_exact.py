"""Summarize the MinimalBench rerun (engine v3, all arms).

Validates the complete 30-seed trace grid and emits the numerical aggregates
used by the controlled-experiment figures and manuscript: finals and
cumulative incumbent regret per (SCM, condition, arm); paired invariance
deviations for QCBO *and* QCBONP; BO / BOS byte-identity across conditions;
the FrontDoor paired regret increase for CBO and CBONP; the refinement
records of HQCBO and HQCBO-GF and their paired contrast (95 % paired-t CI).
Row zero is the shared initial incumbent and is excluded from cumulative
incumbent regret, consistently with :mod:`ccbo.metrics`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ccbo import minibench as mb
from ccbo.metrics import cumulative_regret

from scripts.publication_methods import (PLAIN, PRIMARY_GROUPS, HISTORICAL_GROUPS,
                                         publication_name, required_files)
GROUPS = HISTORICAL_GROUPS
SEEDS = 30


def _se(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _ci(values: np.ndarray, level: float = 0.95) -> list[float]:
    m, se = float(values.mean()), _se(values)
    t = stats.t.ppf(0.5 + level / 2, df=len(values) - 1)
    return [m - t * se, m + t * se]


def _load(root: Path, scm: str, cond: str, method: str, seed: int) -> pd.DataFrame:
    path = root / f"{scm}_{cond}_{method}_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _paired_deviation(root: Path, scm: str, base: str, cond: str, arm: str) -> dict:
    sup, decisions_equal, bytes_equal = 0.0, True, True
    for seed in range(SEEDS):
        left = _load(root, scm, base, arm, seed)
        right = _load(root, scm, cond, arm, seed)
        sup = max(sup, float(np.max(np.abs(
            left.best_y.to_numpy(float) - right.best_y.to_numpy(float)))))
        decisions_equal &= left.arm.equals(right.arm)
        decisions_equal &= left.x_values.equals(right.x_values)
        bytes_equal &= ((root / f"{scm}_{base}_{arm}_seed{seed}.csv").read_bytes()
                        == (root / f"{scm}_{cond}_{arm}_seed{seed}.csv").read_bytes())
    return {"sup_incumbent_deviation": sup,
            "decisions_equal": bool(decisions_equal),
            "bytes_equal": bool(bytes_equal)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/v3/minimal")
    parser.add_argument("--out", default="artifacts/summaries/minimal_exact.json")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--inventory", choices=("historical", "primary"), default="historical",
                        help="Historical validates both refinement variants; primary requires only HQCBOGF.")
    args = parser.parse_args()
    root = Path(args.dir)
    global SEEDS
    SEEDS = args.seeds

    y_star = {mb.PP_NAME: 0.0, mb.FD_NAME: 0.0,
              mb.MC_NAME: mb.ORACLE[mb.MC_NAME]["y_star"]}
    summary: dict = {
        "protocol": {"inventory": args.inventory, "seeds": SEEDS, "trials": 50,
                     "mediated_chain_trials": 60,
                     "initial_points_per_arm": 3,
                     "population_targets": "exact",
                     "cost_axis": "row 0 = initial-design cost; refinement "
                                  "arms add the split design at the trigger",
                     "engine": "v3"},
        "groups": {},
        "paired": {},
    }

    per_seed_cache: dict = {}
    groups = PRIMARY_GROUPS if args.inventory == "primary" else HISTORICAL_GROUPS
    for scm, cond, method in groups:
        required_files(root, f"{scm}_{cond}_{method}", SEEDS)
        frames = [_load(root, scm, cond, method, seed) for seed in range(SEEDS)]
        expected_rows = 61 if scm == mb.MC_NAME else 51
        for seed, frame in enumerate(frames):
            if len(frame) != expected_rows or not np.isfinite(frame.best_y.to_numpy(float)).all():
                raise ValueError(f"{scm}_{cond}_{method}_seed{seed}: invalid trajectory length or incumbent")
        finals = np.asarray([f.best_y.iloc[-1] for f in frames], float)
        regrets = np.asarray([
            cumulative_regret(f.best_y.to_numpy(float), y_star[scm]) for f in frames
        ], float)
        per_seed_cache[(scm, cond, method)] = (finals, regrets)
        summary["groups"][f"{scm}_{cond}_{method}"] = {
            "archive_key": method, "publication_name": publication_name(method),
            "n": SEEDS,
            "final_mean": float(finals.mean()), "final_se": _se(finals),
            "cumulative_regret_mean": float(regrets.mean()),
            "cumulative_regret_se": _se(regrets),
            "final_cost_mean": float(np.mean([f.cum_cost.iloc[-1] for f in frames])),
            "init_cost": float(frames[0].cum_cost.iloc[0]),
        }

    for scm, base, cond in ((mb.PP_NAME, "A0", "A1"), (mb.PP_NAME, "A0", "A2"),
                            (mb.PP_NAME, "A0", "A3"), (mb.FD_NAME, "B0", "B1")):
        for arm in ("QCBO", "QCBONP", "BO", "BOS"):
            summary["paired"][f"{scm}_{base}_vs_{cond}_{arm}"] = \
                _paired_deviation(root, scm, base, cond, arm)

    for arm in ("CBO", "CBONP"):
        delta = []
        for seed in range(SEEDS):
            correct = _load(root, mb.FD_NAME, "B0", arm, seed)
            wrong = _load(root, mb.FD_NAME, "B1", arm, seed)
            delta.append(cumulative_regret(wrong.best_y.to_numpy(float), 0.0)
                         - cumulative_regret(correct.best_y.to_numpy(float), 0.0))
        d = np.asarray(delta)
        summary["paired"][f"FrontDoor_B1_minus_B0_{arm}_cumulative_regret"] = {
            "mean": float(d.mean()), "se": _se(d), "ci95": _ci(d)}

    info = json.loads((root / "refine_info.json").read_text())
    summary["refinement"] = {}
    for arm in (("HQCBOGF",) if args.inventory == "primary" else ("HQCBO", "HQCBOGF")):
        recs = {k: v for k, v in info.items()
                if k.startswith(f"{mb.MC_NAME}_C0_{arm}_seed")}
        expected = {f"{mb.MC_NAME}_C0_{arm}_seed{i}" for i in range(SEEDS)}
        if set(recs) != expected:
            raise ValueError(f"Incomplete refinement metadata for {arm}: expected {SEEDS} seeds")
        trig = np.asarray([r["trigger_step"] for r in recs.values()
                           if r.get("trigger_step") is not None], float)
        summary["refinement"][arm] = {
            "n": int(len(trig)), "n_seeds": len(recs), "split_count": int(len(trig)),
            "trigger_min": float(trig.min()) if len(trig) else None,
            "trigger_max": float(trig.max()) if len(trig) else None,
            "trigger_mean": float(trig.mean()) if len(trig) else None,
            "trigger_se": _se(trig) if len(trig) > 1 else None,
            "all_splits_accepted": all(not r.get("refused", False) for r in recs.values()),
            "split_init_cost": float(np.mean([r.get("split_init_cost", 0.0)
                                              for r in recs.values()])) if recs else None,
        }
    if args.inventory == "historical":
        fin_g, reg_g = per_seed_cache[(mb.MC_NAME, "C0", "HQCBO")]
        fin_f, reg_f = per_seed_cache[(mb.MC_NAME, "C0", "HQCBOGF")]
        summary["refinement"]["HQCBOGF_minus_HQCBO"] = {
            "final_mean": float((fin_f - fin_g).mean()), "final_ci95": _ci(fin_f - fin_g),
            "regret_mean": float((reg_f - reg_g).mean()), "regret_ci95": _ci(reg_f - reg_g),
            "n": SEEDS,
        }
    for arm in ("CBONP", "QCBONP", "BOS"):
        base = "CBO" if arm == "CBONP" else "QCBO" if arm == "QCBONP" else "CBO"
        summary["paired"][f"MediatedChain_C0_{arm}_minus_{base}_regret"] = {
            "mean": float((per_seed_cache[(mb.MC_NAME, "C0", arm)][1]
                           - per_seed_cache[(mb.MC_NAME, "C0", base)][1]).mean()),
            "ci95": _ci(per_seed_cache[(mb.MC_NAME, "C0", arm)][1]
                        - per_seed_cache[(mb.MC_NAME, "C0", base)][1])}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("paired", "refinement")}, indent=1, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
