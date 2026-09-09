"""Run DCBO or QDCBO on DCBO's own paper setups (one unit = one process).

Mirrors ccbo/qmcbo/runner.py: (i) dispatches QDCBO by quotienting the model's
graph view, (ii) supports E2 misspecification injection (an intra-slice,
intra-cluster edge edit of the ASSUMED graph only; the SEM/objective — the
fixed objective within this family — is untouched), (iii) is deterministic
per seed (numpy + python RNGs seeded; the process re-execs itself with
PYTHONHASHSEED=0 because the stock ``_post_optimisation_assignments``
enumerates a *set* of exploration-set members, which is hash-order
dependent), and (iv) writes the same best-so-far CSV schema as the vendored
``run_dcbo.py`` into ``baselines/DCBO/results/_qdcbo/``.

The vendored ``run_dcbo.select_setup`` is NOT reused: its ``stat`` branch is
broken (a function-local ``from ... import StationaryDependentSEM`` in the
``synthetic_2`` branch makes the name local to the whole function, so the
``stat`` branch dies with UnboundLocalError). The setups are rebuilt here
from the same primitives (graph topology, SEM class, exploration sets and
domains exactly as in ``dcbo/examples/example_setups.py``), skipping only the
ground-truth precomputation, which DCBO uses for debug plotting alone.

Usage:
  PYTHONPATH=. ~/venvs/ccbo/bin/python -m ccbo.qdcbo.runner \
      --setup stat --algo QDCBO --T 3 --trials 10 --seed 0 [--misspec e2]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import random
import sys
import time
import warnings
from copy import deepcopy

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"GPy\..*")

import numpy as np

from ccbo.qdcbo.quotient_dbn import (ensure_dcbo_on_path, _REPO_ROOT,
                                     perturb_temporal_edges)
from ccbo.qdcbo.qdcbo_base import QDCBO

ensure_dcbo_on_path()

# Partitions over manipulable base variables; the target (and any variable
# not listed) stays atomic. All three paper setups share the {X, Z} cluster.
PARTITIONS = {
    "stat":    [["X", "Z"]],
    "ind":     [["X", "Z"]],
    "nonstat": [["X", "Z"]],
}

# The intra-slice, intra-cluster edit each setup's E2 applies to the ASSUMED
# graph (model view only). stat/nonstat carry a within-slice X->Z edge to
# delete; ind has none inside {X, Z}, so E2 adds a spurious one.
E2_OPS = {
    "stat":    [("del", "X", "Z")],
    "ind":     [("add", "X", "Z")],
    "nonstat": [("del", "X", "Z")],
}

FINEST = [["X"], ["Z"]]

DEFAULT_OUTDIR = os.path.join(_REPO_ROOT, "results", "qdcbo")


def get_setup(name: str, T: int):
    """(sem_class, G, exploration_sets, intervention_domain, base_target,
    change_points) — identical to dcbo/examples/example_setups.py minus the
    ground-truth sweep."""
    from ccbo.qdcbo.quotient_dbn import make_temporal_graph
    from dcbo.utils.utilities import powerset
    from dcbo.utils.sem_utils.toy_sems import (
        StationaryDependentSEM, StationaryIndependentSEM,
        NonStationaryDependentSEM)

    exploration_sets = list(powerset(["X", "Z"]))
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    change_points = None
    if name == "stat":
        G = make_temporal_graph(0, T - 1, topology="dependent",
                                nodes=["X", "Z", "Y"])
        sem_class = StationaryDependentSEM
    elif name == "ind":
        G = make_temporal_graph(0, T - 1, topology="independent",
                                nodes=["X", "Z", "Y"], target_node="Y")
        sem_class = StationaryIndependentSEM
    elif name == "nonstat":
        assert T >= 2, "nonstat needs a change point"
        G = make_temporal_graph(0, T - 1, topology="dependent",
                                nodes=["X", "Z", "Y"])
        sem_class = NonStationaryDependentSEM
        change_points = [False, True] + [False] * (T - 2)
    else:
        raise ValueError(f"unknown setup {name!r}")
    return sem_class, G, exploration_sets, intervention_domain, "Y", \
        change_points


def sample_observations(sem_class, change_points, T: int, n_obs: int,
                        seed: int) -> dict:
    """{base var: (n_obs, T)}: same construction (and seeding) as the stock
    ``run_methods_replicates``."""
    from dcbo.utils.sequential_sampling import sequentially_sample_model
    true_sem = sem_class() if change_points is None else \
        sem_class(change_points.index(True))
    np.random.seed(seed)
    return sequentially_sample_model(
        true_sem.static(), true_sem.dynamic(), total_timesteps=T,
        sample_count=n_obs)


def run_unit(setup: str, algo: str, T: int, trials: int, seed: int,
             misspec: str = "", n_obs: int = 100,
             num_anchor_points: int = 100, partition=None,
             stock_quirks: bool = False):
    """One (setup, algo, seed) run; returns (model, wall_seconds).

    Both algorithms receive byte-identical observational data and RNG states;
    ``--misspec e2`` perturbs ONLY the graph the model reasons with.
    ``stock_quirks=False`` (engine v3 default) runs BOTH the stock DCBO
    baseline and QDCBO with corrected transition fitting and ``is not None``
    intervention clamping (:mod:`ccbo.qdcbo.stock_fixes`); ``True`` is the
    historical-reproduction mode.
    """
    assert algo in ("DCBO", "QDCBO"), algo
    from ccbo.qdcbo import stock_fixes
    stock_fixes.apply(stock_quirks)
    from dcbo.methods.dcbo import DCBO
    from dcbo.utils.sem_utils.sem_estimate import build_sem_hat

    sem_class, G, exploration_sets, intervention_domain, base_target, \
        change_points = get_setup(setup, T)

    observation_samples = sample_observations(
        sem_class, change_points, T, n_obs, seed)

    G_model = G
    if misspec:
        assert misspec == "e2", misspec
        G_model = perturb_temporal_edges(G, E2_OPS[setup])

    common = {
        "G": G_model,
        "sem": sem_class,
        "make_sem_estimator": build_sem_hat,
        "observation_samples": deepcopy(observation_samples),
        "intervention_domain": intervention_domain,
        "intervention_samples": None,
        "exploration_sets": exploration_sets,
        "number_of_trials": trials,
        "base_target_variable": base_target,
        "task": "min",
        "estimate_sem": True,
        "cost_type": 1,
        "ground_truth": None,
        "n_restart": 1,
        "use_mc": False,
        "debug_mode": False,
        "online": False,
        "optimal_assigned_blankets": None,
        "use_di": False,
        "transfer_hp_o": False,
        "transfer_hp_i": False,
        "hp_i_prior": True,
        "n_obs_t": None,
        "num_anchor_points": num_anchor_points,
        "seed": seed,
        "sample_anchor_points": False,
        "seed_anchor_points": 1,   # parity with run_methods_replicates rep 0
        "args_sem": None,
        "manipulative_variables": None,
        "change_points": change_points,
    }

    np.random.seed(seed)
    random.seed(seed)   # Root.__init__ draws the initial ES via random.choice
    t0 = time.time()
    if algo == "DCBO":
        model = DCBO(**common)
    else:
        model = QDCBO(partition=partition or PARTITIONS[setup],
                      stock_quirks=stock_quirks, **common)
    model.run()
    wall = time.time() - t0
    return model, wall


def trajectory(model) -> list:
    """best-so-far per (time_index, trial_index), the run_dcbo.py contract."""
    return [list(map(float, model.optimal_outcome_values_during_trials[t]))
            for t in range(model.T)]


def _plain(x):
    """JSON-safe copy (ndarray -> list, nan -> None, tuple -> list)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if f != f else f
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def decisions_payload(model, algo: str, setup: str, T: int, trials: int,
                      seed: int, misspec: str, stock_quirks: bool) -> dict:
    """Full-precision per-(time slice, trial) decision record (engine v3):
    chosen intervention set, its level, the outcome, the best-so-far value,
    the per-trial cost and the blanket the slice was closed with."""
    per_t = []
    for t in range(model.T):
        levels = {}
        for es in model.exploration_sets:
            lv = model.optimal_intervention_levels[t][es]
            levels["+".join(es)] = _plain(list(lv)) if lv is not None else None
        per_t.append({
            "t": t,
            "chosen_sets": _plain([list(es) for es in
                                   model.sequence_of_interventions_during_trials[t]]),
            "levels_by_set": levels,
            "outcome_values": _plain(list(model.outcome_values[t])),
            "best_so_far": _plain(list(model.optimal_outcome_values_during_trials[t])),
            "per_trial_cost": _plain(list(model.per_trial_cost[t])),
            "optimal_intervention_set": _plain(
                list(model.optimal_intervention_sets[t]) if model.optimal_intervention_sets[t] else None),
        })
    return {
        "schema": 1,
        "unit": {"suite": "qdcbo", "setup": setup, "algo": algo, "T": T,
                 "trials": trials, "seed": seed, "misspec": misspec,
                 "stock_quirks": bool(stock_quirks),
                 "partition": _plain(getattr(getattr(model, "_spec", None), "clusters", None)),
                 "engine_sha": os.environ.get("CCBO_GIT_SHA")},
        "exploration_sets": [list(es) for es in model.exploration_sets],
        "assigned_blanket": _plain(model.assigned_blanket),
        "per_t": per_t,
    }


def write_csv(model, algo: str, setup: str, T: int, trials: int, seed: int,
              misspec: str, outdir: str) -> str:
    """Same schema as run_dcbo.py: method, replicate, time_index,
    trial_index, best_so_far_value, y."""
    os.makedirs(outdir, exist_ok=True)
    tag = f"_{misspec}" if misspec else ""
    method = f"{algo}_{misspec}" if misspec else algo
    path = os.path.join(
        outdir,
        f"{algo.lower()}_{setup}{tag}_best_so_far_seed{seed}_T{T}"
        f"_trials{trials}_reps1.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "replicate", "time_index", "trial_index",
                         "best_so_far_value", "y"])
        for t_idx in range(model.T):
            bsf_list = model.optimal_outcome_values_during_trials[t_idx]
            for trial_idx, val in enumerate(bsf_list):
                if trial_idx == 0:
                    y_val = ""
                else:
                    y_seq = model.outcome_values[t_idx]
                    y_val = y_seq[trial_idx + 1] \
                        if len(y_seq) > trial_idx + 1 else ""
                writer.writerow([method, 0, t_idx, trial_idx, val, y_val])
    return path


def main():
    # The stock _post_optimisation_assignments iterates a *set* of ES member
    # strings; pin the hash seed so trajectories are process-independent.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=sorted(PARTITIONS))
    ap.add_argument("--algo", required=True, choices=["DCBO", "QDCBO"])
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--misspec", default="", choices=["", "e2"],
                    help="e2: intra-slice, intra-cluster edit of the ASSUMED "
                         "graph (objective unchanged)")
    ap.add_argument("--n-obs", type=int, default=100)
    ap.add_argument("--num-anchor-points", type=int, default=100)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--stock-quirks", action="store_true",
                    help="historical mode: reproduce the stock backend's "
                         "transition-slice and truthiness-clamp semantics")
    args = ap.parse_args()

    model, wall = run_unit(args.setup, args.algo, args.T, args.trials,
                           args.seed, args.misspec, args.n_obs,
                           args.num_anchor_points,
                           stock_quirks=args.stock_quirks)
    path = write_csv(model, args.algo, args.setup, args.T, args.trials,
                     args.seed, args.misspec, args.outdir)
    payload = decisions_payload(model, args.algo, args.setup, args.T,
                                args.trials, args.seed, args.misspec,
                                args.stock_quirks)
    with open(path.replace(".csv", ".decisions.json"), "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    finals = [tr[-1] for tr in trajectory(model)]
    info = {"setup": args.setup, "algo": args.algo, "T": args.T,
            "trials": args.trials, "seed": args.seed,
            "misspec": args.misspec, "n_obs": args.n_obs,
            "stock_quirks": bool(args.stock_quirks),
            "engine_sha": os.environ.get("CCBO_GIT_SHA"),
            "final_best_so_far_per_t": finals, "secs": wall, "csv": path}
    with open(path.replace(".csv", "_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"DONE {args.algo:6s} {args.setup:8s} seed{args.seed} "
          f"misspec={args.misspec or '-'} finals={finals} "
          f"({wall:.0f}s) -> {path}", flush=True)


if __name__ == "__main__":
    main()
