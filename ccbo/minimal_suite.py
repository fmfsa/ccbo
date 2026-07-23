"""Unit runners for the MinimalBench suite (Experiments A-C).

One module holds the plain CBO/QCBO unit AND the native two-phase HQCBO
(refinement) unit so that HQCBO's phase 1 is *the same code path* as the
QCBO unit — the byte-identical-prefix property is enforced by construction,
not by copying.

Why a native HQCBO: ``ccbo.benchmark.run_qcbo_refine_benchmark`` is coupled
to the retired external-baselines loader (``load_graph``) and passes a
``csv_log_file`` kwarg the vendored ``ccbo/cbo/cbo.py::CBO`` does not
accept. This module reuses only its standalone, tested helpers
(``_plateau``, ``_split_partition``, ``_union_and_carry``) on top of the
family-runner pipeline (CoarsenedGraph -> generate_interventional_data ->
define_initial_data_CBO -> CBO), the same one used by
``scripts/run_cbo_family.py``.

Row convention (shared with run_cbo_family): each unit produces
``trials + 1`` aligned rows — row 0 is the incumbent of the initial
interventional design at cost 0, then exactly one row per trial (observe
trials repeat the previous best/cost). Rows additionally carry the selected
arm and intervention values ('' for observe/init rows), reconstructed from
the intervention callback + the unit-cost increments.

Protocol invariants: one shared observations.pkl per SCM (fixed n_obs
across conditions/methods/seeds); the misspecified conditions inject
``assumed_graph_name`` only — SEM, objective, and data always come from the
true graph (the fixed-objective seam).
"""

import os

import numpy as np
import pandas as pd

from ccbo import minibench as mb
from ccbo.benchmark import _plateau, _split_partition, _union_and_carry

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLATEAU_K = 5
PLATEAU_DELTA = 1e-3


# ---------------------------------------------------------------------------
# Shared pipeline pieces
# ---------------------------------------------------------------------------

def load_scm(scm, initial_num_obs_samples):
    """Observational data + true fine-graph instance for a MinimalBench SCM."""
    from ccbo.run_experiment import get_original_graph

    data_path = os.path.join(_REPO, "ccbo", "cbo", "data", scm,
                             "observations.pkl")
    full_obs = pd.read_pickle(data_path)
    obs = full_obs[:initial_num_obs_samples]
    graph = get_original_graph(scm, obs)
    return graph, obs, full_obs


def _setup(scm, cond, partition, seed, num_interventions,
           initial_num_obs_samples, type_cost):
    """Deterministic pre-loop construction, order copied from
    run_cbo_family.run_unit (models fitted BEFORE the init-data re-seed)."""
    from ccbo.coarsened_graph import CoarsenedGraph
    from ccbo.data_generation import generate_interventional_data
    from ccbo.cbo.utils import compute_coverage, define_initial_data_CBO

    mb.register_variants()
    np.random.seed(seed)
    graph, obs, full_obs = load_scm(scm, initial_num_obs_samples)

    cg = CoarsenedGraph(graph, partition, scm, obs, num_mc_samples=2000,
                        assumed_graph_name=mb.variant_name(cond))
    ES, _, manip_vars = cg.get_sets()

    functions = cg.fit_all_models()
    dict_ranges = cg.get_interventional_ranges()
    costs = cg.get_cost_structure(type_cost)

    np.random.seed(seed)
    int_data = generate_interventional_data(
        cg.define_SEM, ES, dict_ranges,
        num_points=20, num_mc_samples=2000, seed=seed)

    _, _, coverage = compute_coverage(obs, manip_vars, dict_ranges)
    x_list, y_list, best_x, opt_y, best_var = define_initial_data_CBO(
        int_data, num_interventions, ES, 0, "min")

    return dict(graph=graph, obs=obs, full_obs=full_obs, cg=cg, ES=ES,
                manip_vars=manip_vars, functions=functions,
                dict_ranges=dict_ranges, costs=costs, coverage=coverage,
                x_list=x_list, y_list=y_list, best_x=best_x, opt_y=opt_y,
                best_var=best_var)


def _run_cbo(ctx, trials, num_interventions, initial_num_obs_samples):
    """One vendored-CBO call over a prepared context; returns
    (global_opt, current_cost, arm_log, out) with arm_log the per-intervention
    (vars, x) sequence captured via the callback."""
    from ccbo.cbo.cbo import CBO

    arm_log = []

    def _cb(int_vars, x_new, _y_new, _sem):
        arm_log.append((list(int_vars), np.ravel(x_new).tolist()))

    out = CBO(
        trials, ctx["ES"], ctx["manip_vars"], ctx["x_list"], ctx["y_list"],
        ctx["best_x"], ctx["opt_y"], ctx["best_var"], ctx["dict_ranges"],
        ctx["functions"], ctx["obs"], ctx["coverage"], ctx["cg"],
        20, ctx["costs"], ctx["full_obs"], "min",
        initial_num_obs_samples + 50, initial_num_obs_samples,
        num_interventions, Causal_prior=True, intervention_callback=_cb)
    current_cost, _, _, global_opt, _, _ = out
    return global_opt, current_cost, arm_log, out


def _rows_from(global_opt, current_cost, arm_log, start_trial=0,
               cost_offset=0.0, drop_first=False):
    """Aligned (trial, best_y, cum_cost, arm, x_values) rows.

    Intervene rows are the ones where cumulative cost strictly increases
    (unit costs); they consume ``arm_log`` in order. ``drop_first`` drops the
    repeated entry-incumbent row of a warm-started phase-2 call.
    """
    rows, k = [], 0
    for i, (y, c) in enumerate(zip(global_opt, current_cost)):
        if i == 0:
            if drop_first:
                continue
            rows.append((start_trial, float(y), cost_offset, "init", ""))
            continue
        if c > current_cost[i - 1]:          # intervene trial
            vars_, xs = arm_log[k]
            k += 1
            arm = "+".join(vars_)
            xv = ";".join(f"{v:.6g}" for v in xs)
        else:                                 # observe trial
            arm, xv = "", ""
        rows.append((start_trial + i - (1 if drop_first else 0),
                     float(y), float(c) + cost_offset, arm, xv))
    assert k == len(arm_log), "arm log misaligned with cost increments"
    return rows


def run_plain_unit(scm, cond, arm, seed, trials, num_interventions=10,
                   initial_num_obs_samples=100, type_cost=1):
    """One (scm, condition, CBO|QCBO, seed) unit. Returns (rows, info)."""
    partition = (mb.fine_partition(scm) if arm == "CBO"
                 else mb.coarse_partition(scm))
    ctx = _setup(scm, cond, partition, seed, num_interventions,
                 initial_num_obs_samples, type_cost)
    global_opt, current_cost, arm_log, _ = _run_cbo(
        ctx, trials, num_interventions, initial_num_obs_samples)
    assert len(global_opt) == len(current_cost) == trials + 1, (
        f"trajectory length mismatch: {len(global_opt)} vs {trials + 1}")
    rows = _rows_from(global_opt, current_cost, arm_log)
    uninformative = sorted(
        "+".join(k) for k, ok in ctx["cg"]._arm_identifiable.items() if not ok)
    info = dict(final_y=float(global_opt[-1]),
                final_cost=float(current_cost[-1]),
                es=[list(e) for e in ctx["ES"]],
                uninformative_arms=uninformative)
    return rows, info


# ---------------------------------------------------------------------------
# Native HQCBO (two-phase refine)
# ---------------------------------------------------------------------------

def run_refine_unit(scm, cond, seed, trials, num_interventions=10,
                    initial_num_obs_samples=100, type_cost=1,
                    plateau_k=PLATEAU_K, plateau_delta=PLATEAU_DELTA,
                    coarse_traj=None):
    """HQCBO unit: coarse QCBO until plateau, split the incumbent's cluster
    per ``minibench``'s refine map, warm-start the remaining budget.

    ``coarse_traj``: the QCBO unit's best-so-far list for this seed (rows
    0..trials). When None it is computed here by running the coarse
    configuration once (simulation compute only; the method's budget stays
    ``trials`` steps, exactly as ``run_qcbo_refine_benchmark`` does).

    Phase 1 is byte-identical to the QCBO unit's prefix: same ``_setup``,
    same ``_run_cbo``, same seed, fewer trials. The phase boundary costs one
    forced observation trial (the vendored CBO observes at step 0 of every
    fresh call) — a bias against refinement, kept for fidelity.
    """
    from ccbo.coarsened_graph import CoarsenedGraph
    from ccbo.data_generation import generate_interventional_data
    from ccbo.cbo.utils import compute_coverage, define_initial_data_CBO

    refine_map = {mb.MC_NAME: mb.MC_REFINE_MAP}.get(scm)
    assert refine_map is not None, f"no refine map declared for {scm}"

    info = {"trigger_step": None, "split_clusters": [], "refused": False}

    # ---- Trigger time from the coarse trajectory --------------------------
    if coarse_traj is None:
        rows, _ = run_plain_unit(scm, cond, "QCBO", seed, trials,
                                 num_interventions, initial_num_obs_samples,
                                 type_cost)
        coarse_traj = [r[1] for r in rows]
    t_fire = next((t for t in range(len(coarse_traj))
                   if _plateau(coarse_traj[:t + 1], plateau_k,
                               plateau_delta, "min")), None)
    if t_fire is not None and not (0 < t_fire < trials):
        t_fire = None

    partition = mb.coarse_partition(scm)
    ctx = _setup(scm, cond, partition, seed, num_interventions,
                 initial_num_obs_samples, type_cost)

    if t_fire is None:                       # never refines: this IS coarse
        global_opt, current_cost, arm_log, _ = _run_cbo(
            ctx, trials, num_interventions, initial_num_obs_samples)
        return _rows_from(global_opt, current_cost, arm_log), info

    # ---- Phase 1: coarse prefix up to the trigger -------------------------
    go1, cc1, log1, _ = _run_cbo(ctx, t_fire, num_interventions,
                                 initial_num_obs_samples)
    rows = _rows_from(go1, cc1, log1)

    # ---- Split the incumbent's cluster ------------------------------------
    ES, x_list, y_list = ctx["ES"], ctx["x_list"], ctx["y_list"]
    per_arm = [float(np.min(Y)) for Y in y_list]
    incumbent = tuple(ES[int(np.argmin(per_arm))])
    new_partition, split = _split_partition(partition, refine_map, incumbent)

    new_cg = None
    if split:
        try:
            new_cg = CoarsenedGraph(
                ctx["graph"], new_partition, scm, ctx["obs"],
                num_mc_samples=2000, assumed_graph_name=mb.variant_name(cond))
        except ValueError:                   # invalid split: refuse, stay coarse
            info["refused"] = True

    if new_cg is not None:
        new_ES = [list(e) for e in new_cg.get_sets()[0]]

        def _sampler(entries):
            int_data = generate_interventional_data(
                new_cg.define_SEM, entries, ctx["dict_ranges"],
                num_points=20, num_mc_samples=2000, seed=seed + 1000)
            fx, fy, _, _, _ = define_initial_data_CBO(
                int_data, num_interventions, entries, 0, "min")
            return fx, fy

        ES, x_list, y_list = _union_and_carry(ES, x_list, y_list, new_ES,
                                              _sampler)
        # Register unioned phase-1 arms with the refined graph before its
        # memoized get_all_do first runs, so they keep C-DAG priors too.
        for e in ES:
            if list(e) not in new_cg._exploration_set:
                new_cg._exploration_set.append(list(e))
        info["trigger_step"] = t_fire
        info["split_clusters"] = [sorted(c) for c in split]

        ctx["cg"] = new_cg
        ctx["functions"] = new_cg.fit_all_models()
        ctx["costs"] = new_cg.get_cost_structure(type_cost)
        ctx["manip_vars"] = new_cg.get_sets()[2]
        _, _, ctx["coverage"] = compute_coverage(
            ctx["obs"], ctx["manip_vars"], ctx["dict_ranges"])

    # ---- Phase 2 entry incumbents from the carried data -------------------
    s = int(np.argmin([float(np.min(Y)) for Y in y_list]))
    i_best = int(np.argmin(y_list[s]))
    ctx.update(ES=ES, x_list=x_list, y_list=y_list,
               best_var="".join(ES[s]),
               opt_y=float(y_list[s][i_best]),
               best_x=np.asarray(x_list[s][i_best], dtype=float))

    # ---- Phase 2: the remaining budget in one warm-started call -----------
    go2, cc2, log2, _ = _run_cbo(ctx, trials - t_fire, num_interventions,
                                 initial_num_obs_samples)
    rows += _rows_from(go2, cc2, log2, start_trial=t_fire + 1,
                       cost_offset=rows[-1][2], drop_first=True)

    # Best-so-far must be monotone across the boundary; enforce defensively.
    out, best = [], np.inf
    for (t, y, c, a, xv) in rows:
        best = min(best, y)
        out.append((t, best, c, a, xv))
    assert len(out) == trials + 1, (
        f"refine rows mismatch: {len(out)} vs {trials + 1}")
    return out, info
