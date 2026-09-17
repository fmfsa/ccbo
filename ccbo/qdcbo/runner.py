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
    "nonstat_regularized": [["X", "Z"]],
}

# The intra-slice, intra-cluster edit each setup's E2 applies to the ASSUMED
# graph (model view only). stat/nonstat carry a within-slice X->Z edge to
# delete; ind has none inside {X, Z}, so E2 adds a spurious one.
E2_OPS = {
    "stat":    [("del", "X", "Z")],
    "ind":     [("add", "X", "Z")],
    "nonstat": [("del", "X", "Z")],
    "nonstat_regularized": [("del", "X", "Z")],
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
    elif name in ("nonstat", "nonstat_regularized"):
        assert T >= 2, "nonstat needs a change point"
        G = make_temporal_graph(0, T - 1, topology="dependent",
                                nodes=["X", "Z", "Y"])
        sem_class = NonStationaryDependentSEM
        change_points = [False, True] + [False] * (T - 2)
        if name == 'nonstat_regularized':
            from ccbo.qdcbo.population import RegularizedNonStationarySEM, regularized_graph
            sem_class = RegularizedNonStationarySEM
            G = regularized_graph(G, change_point=1)
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
             stock_quirks: bool = False, mechanism_protocol: str = "legacy",
             predictive_samples: int = 256, action_menu: str = "native",
             objective_protocol: str = 'zero-disturbance', feedback_samples: int = 2048,
             score_samples: int = 100000):
    """One (setup, algo, seed) run; returns (model, wall_seconds).

    Both algorithms receive byte-identical observational data and RNG states;
    ``--misspec e2`` perturbs ONLY the graph the model reasons with.
    ``stock_quirks=False`` (engine v3 default) runs BOTH the stock DCBO
    baseline and QDCBO with corrected transition fitting and ``is not None``
    intervention clamping (:mod:`ccbo.qdcbo.stock_fixes`); ``True`` is the
    historical-reproduction mode.
    """
    assert algo in ("DCBO", "QDCBO"), algo
    if objective_protocol == 'population-mc':
        if mechanism_protocol != 'coherent':
            raise ValueError('Population response requires coherent mechanism protocol')
        if setup == 'nonstat':
            raise ValueError('Original nonstat has singular/nonintegrable regimes; use separately named nonstat_regularized')
    elif objective_protocol != 'zero-disturbance':
        raise ValueError('Unknown objective protocol')
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

    if action_menu == 'coarse':
        from ccbo.qdcbo.quotient_dbn import build_quotient_dbn
        menu_spec = build_quotient_dbn(G_model, PARTITIONS[setup], base_target)
        exploration_sets = menu_spec.lift_exploration_sets(exploration_sets)
    elif action_menu != 'native':
        raise ValueError(f'Unknown action menu {action_menu}')

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
    if mechanism_protocol == "coherent":
        from ccbo.qdcbo.coherent import CoherentDCBO, CoherentQDCBO
        if stock_quirks:
            raise ValueError('Coherent protocol does not support --stock-quirks')
        coherent_args = dict(predictive_samples=predictive_samples,
                             predictive_seed=seed, objective_protocol=objective_protocol,
                             feedback_samples=feedback_samples, population_seed=seed)
        if algo == "DCBO":
            model = CoherentDCBO(**common, **coherent_args)
        else:
            model = CoherentQDCBO(partition=partition or PARTITIONS[setup],
                                  **common, **coherent_args)
    elif mechanism_protocol != "legacy":
        raise ValueError(f'Unknown mechanism protocol {mechanism_protocol}')
    elif algo == "DCBO":
        model = DCBO(**common)
    else:
        model = QDCBO(partition=partition or PARTITIONS[setup],
                      stock_quirks=stock_quirks, **common)
    model.action_menu = action_menu
    # This runner passes intervention_samples=None. Mark the lack of an
    # initial executed recommendation explicitly rather than scoring blank_val.
    model.runner_has_initial_interventions = False
    model.run()
    model.training_wall_seconds = time.time() - t0
    if objective_protocol == 'population-mc':
        from ccbo.qdcbo.population import score_events
        score_events(model, samples=score_samples)
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
            "trial_types": list(model.trial_type[t]),
            "eligible_recommendation": _eligible_recommendations(model, t),
            "recommended_values": [value if eligible else None for value, eligible in zip(
                _plain(list(model.optimal_outcome_values_during_trials[t])),
                _eligible_recommendations(model, t))],
            "optimal_intervention_set": _plain(
                list(model.optimal_intervention_sets[t]) if model.optimal_intervention_sets[t] else None),
        })
    return {
        "schema": 1,
        "unit": {"suite": "qdcbo", "setup": setup, "algo": algo, "T": T,
                 "trials": trials, "seed": seed, "misspec": misspec,
                 "stock_quirks": bool(stock_quirks),
                 "mechanism_protocol": getattr(model, 'mechanism_protocol', 'legacy'),
                 "predictive_samples": getattr(model, 'predictive_samples', None),
                 "action_menu": getattr(model, 'action_menu', 'native'),
                 "objective_protocol": getattr(model, 'objective_protocol', 'zero-disturbance'),
                 "feedback_samples": getattr(model, 'feedback_samples', None),
                 "partition": _plain(getattr(getattr(model, "_spec", None), "clusters", None)),
                 "engine_sha": os.environ.get("CCBO_GIT_SHA")},
        "exploration_sets": [list(es) for es in model.exploration_sets],
        "assigned_blanket": _plain(model.assigned_blanket),
        "per_t": per_t,
        "population_events": _plain(getattr(model, 'population_events', None)),
        "policy_commits": _plain(getattr(model, 'policy_commits', None)),
    }


def _eligible_recommendations(model, t):
    """Explicit event eligibility; observations alone cannot create an arm."""
    eligible = bool(getattr(model, 'runner_has_initial_interventions', False))
    flags = []
    for trial_type in model.trial_type[t]:
        eligible = eligible or trial_type == 'i'
        flags.append(eligible)
    return flags


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
    if getattr(model, 'objective_protocol', '') == 'population-mc':
        events = {(e['t'], e['trial']): e for e in model.population_events}
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['method', 'seed', 'time_index', 'trial_index',
                'eligible_recommendation', 'population_recommendation_mean',
                'population_mc_se', 'training_incumbent', 'cost', 'recommendation_event'])
            for t in range(model.T):
                for trial in range(trials):
                    event = events.get((t, trial))
                    if event is None:
                        writer.writerow([method, seed, t, trial, False, '', '', '', 0, ''])
                    else:
                        recommended = model.population_events[event['recommendation_event']]
                        writer.writerow([method, seed, t, trial, True,
                            event['recommendation_population_mean'], event['recommendation_population_mc_se'],
                            recommended['training_mean'], event['cost'], event['recommendation_event']])
        return path
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "replicate", "time_index", "trial_index",
                         "best_so_far_value", "y"])
        for t_idx in range(model.T):
            bsf_list = model.optimal_outcome_values_during_trials[t_idx]
            eligible = _eligible_recommendations(model, t_idx)
            for trial_idx, val in enumerate(bsf_list):
                if getattr(model, 'mechanism_protocol', '').startswith('coherent') and not eligible[trial_idx]:
                    val = ''  # no executed recommendation, never a 1e7 score
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
    ap.add_argument('--mechanism-protocol', choices=['legacy', 'coherent'], default='legacy',
                    help='opt-in union-parent conditional model with ancestral integration')
    ap.add_argument('--predictive-samples', type=int, default=256,
                    help='fixed integration particles for coherent protocol')
    ap.add_argument('--action-menu', choices=['native', 'coarse'], default='native',
                    help='coarse restricts fine DCBO to the quotient action menu')
    ap.add_argument('--objective-protocol', choices=['zero-disturbance', 'population-mc'],
                    default='zero-disturbance')
    ap.add_argument('--feedback-samples', type=int, default=2048)
    ap.add_argument('--score-samples', type=int, default=100000)
    args = ap.parse_args()

    model, wall = run_unit(args.setup, args.algo, args.T, args.trials,
                           args.seed, args.misspec, args.n_obs,
                           args.num_anchor_points,
                           stock_quirks=args.stock_quirks,
                           mechanism_protocol=args.mechanism_protocol,
                           predictive_samples=args.predictive_samples,
                           action_menu=args.action_menu,
                           objective_protocol=args.objective_protocol,
                           feedback_samples=args.feedback_samples,
                           score_samples=args.score_samples)
    output_algo = args.algo + ('-coarse-menu' if args.action_menu == 'coarse' else '')
    if args.mechanism_protocol != 'legacy':
        output_algo += '-coherent'
    if args.objective_protocol == 'population-mc':
        output_algo += '-population'
    path = write_csv(model, output_algo, args.setup, args.T, args.trials,
                     args.seed, args.misspec, args.outdir)
    payload = decisions_payload(model, args.algo, args.setup, args.T,
                                args.trials, args.seed, args.misspec,
                                args.stock_quirks)
    with open(path.replace(".csv", ".decisions.json"), "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    finals = [tr[-1] for tr in trajectory(model)]
    if args.mechanism_protocol == 'coherent':
        finals = [value if _eligible_recommendations(model, t)[-1] else None
                  for t, value in enumerate(finals)]
    if args.objective_protocol == 'population-mc':
        finals = [commit['population_mean'] for commit in model.policy_commits]
    info = {"setup": args.setup, "algo": args.algo, "T": args.T,
            "trials": args.trials, "seed": args.seed,
            "misspec": args.misspec, "n_obs": args.n_obs,
            "stock_quirks": bool(args.stock_quirks),
            "mechanism_protocol": args.mechanism_protocol,
            "predictive_samples": args.predictive_samples if args.mechanism_protocol == 'coherent' else None,
            "action_menu": args.action_menu,
            "objective_protocol": args.objective_protocol,
            "feedback_samples": args.feedback_samples if args.objective_protocol == 'population-mc' else None,
            "score_samples": args.score_samples if args.objective_protocol == 'population-mc' else None,
            "engine_sha": os.environ.get("CCBO_GIT_SHA"),
            "final_best_so_far_per_t": finals, "secs": wall, "csv": path}
    with open(path.replace(".csv", "_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"DONE {args.algo:6s} {args.setup:8s} seed{args.seed} "
          f"misspec={args.misspec or '-'} finals={finals} "
          f"({wall:.0f}s) -> {path}", flush=True)


if __name__ == "__main__":
    main()
