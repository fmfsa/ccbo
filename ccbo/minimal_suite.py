"""Unit runners for the MinimalBench suite (Experiments A-C), engine v3.

One module holds the plain CBO/QCBO units, the ablation arms, the joint BO
baseline AND the native two-phase HQCBO (refinement) units so that HQCBO's
phase 1 is *the same code path* as the QCBO unit — the byte-identical-prefix
property is enforced by construction, not by copying.

Arms
----
``CBO``     fine (identity) partition, causal priors, observations on
``QCBO``    coarse partition, causal priors, observations on
``CBONP``   fine partition, **plain RBF GPs** (no causal prior), observations on
``QCBONP``  coarse partition, plain RBF GPs, observations on
``BOS``     every nonempty subset of the manipulables as an arm, plain RBF
            GPs, **no observation actions** (structure-free arm menu)
``BO``      one joint arm over all manipulables, plain GP, no observations
``HQCBO``   graph-informed refinement: the refined quotient (MIS + C-DAG
            priors) is read from the fine hypothesis graph after the split
``HQCBOGF`` graph-free refinement: the declared hierarchy exposes *all*
            nonempty unions of the split cluster's sub-clusters as new
            arms with plain GPs; historical arms are retained; nothing
            after the split consults the fine graph (no MIS pruning, no
            causal priors, no validity check)

Row convention (shared with run_cbo_family): each unit produces
``trials + 1`` aligned rows ``(trial, best_y, cum_cost, arm, x_values)`` —
row 0 is the incumbent of the initial interventional design at
``cum_cost = init_cost`` (engine v3 charges the initial design: ``n_init``
points per arm at per-variable unit cost), then exactly one row per trial
(observe trials repeat the previous best/cost).  HQCBO / HQCBOGF add the
split-design cost at the trigger row.  Intervention values are written at
full ``repr`` precision.  Every unit also produces a decision-log payload
(``info['decision_log']``, see :mod:`ccbo.decision_log`).

Protocol invariants: one shared observations.pkl per SCM (fixed n_obs
across conditions/methods/seeds); the misspecified conditions inject
``assumed_graph_name`` only — SEM, objective, and data always come from the
true graph (the fixed-objective seam).
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd

from ccbo import minibench as mb
from ccbo import decision_log as dlog
from ccbo.benchmark import _plateau, _split_partition, _union_and_carry

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLATEAU_K = 5
PLATEAU_DELTA = 1e-3

PLAIN_ARMS = ("CBO", "QCBO", "CBONP", "QCBONP", "BOS")
REFINE_ARMS = ("HQCBO", "HQCBOGF")
REFINE_MODE = {"HQCBO": "graph", "HQCBOGF": "free"}


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


def all_subsets(manip_vars):
    """Every nonempty subset of ``manip_vars`` in canonical order (by size,
    then lexicographically) — the BOS arm menu and the graph-free
    refinement's exposed sub-arms."""
    manip = sorted(manip_vars)
    out = []
    for k in range(1, len(manip) + 1):
        for c in combinations(manip, k):
            out.append(list(c))
    return out


def init_design_cost(ES, x_list, costs):
    """Cost of the initial interventional design under the run's cost
    structure: every stored design point of every arm is charged as if it
    had been executed (per-variable unit costs -> ``n_init * sum |arm|``)."""
    from ccbo.cbo.utils.utils import get_new_dict_x
    from ccbo.cbo.utils.cost_functions import total_cost

    total = 0.0
    for arm, X in zip(ES, x_list):
        arm = list(arm)
        for row in np.atleast_2d(np.asarray(X, dtype=float)):
            total += float(total_cost(arm, costs, get_new_dict_x(row[None, :], arm)))
    return float(total)


def _setup(scm, cond, partition, seed, num_interventions,
           initial_num_obs_samples, type_cost, joint_arm=False,
           subset_arms=False):
    """Deterministic pre-loop construction, order copied from
    run_cbo_family.run_unit (models fitted BEFORE the init-data re-seed).

    ``joint_arm``: replace the exploration set with the single joint arm
    over every manipulable variable (the standard-BO baseline).
    ``subset_arms``: replace it with every nonempty subset (the BOS arm).
    In both cases the partition is structural bookkeeping only."""
    from ccbo.coarsened_graph import CoarsenedGraph
    from ccbo.data_generation import generate_interventional_data
    from ccbo.cbo.utils import compute_coverage, define_initial_data_CBO

    mb.register_variants()
    np.random.seed(seed)
    graph, obs, full_obs = load_scm(scm, initial_num_obs_samples)

    cg = CoarsenedGraph(graph, partition, scm, obs, num_mc_samples=2000,
                        assumed_graph_name=mb.variant_name(cond))
    ES, _, manip_vars = cg.get_sets()
    if joint_arm:
        ES = [list(manip_vars)]
    elif subset_arms:
        ES = all_subsets(manip_vars)

    functions = cg.fit_all_models()
    dict_ranges = cg.get_interventional_ranges()
    costs = cg.get_cost_structure(type_cost)

    np.random.seed(seed)
    int_data = generate_interventional_data(
        cg.define_SEM, ES, dict_ranges,
        num_points=20, num_mc_samples=2000, seed=seed,
        target_evaluator=mb.population_evaluator(scm))

    _, _, coverage = compute_coverage(obs, manip_vars, dict_ranges)
    x_list, y_list, best_x, opt_y, best_var = define_initial_data_CBO(
        int_data, num_interventions, ES, 0, "min")

    return dict(scm=scm, cond=cond, seed=seed, partition=list(partition),
                graph=graph, obs=obs, full_obs=full_obs, cg=cg, ES=ES,
                manip_vars=manip_vars, functions=functions,
                dict_ranges=dict_ranges, costs=costs, coverage=coverage,
                x_list=x_list, y_list=y_list, best_x=best_x, opt_y=opt_y,
                best_var=best_var, num_interventions=num_interventions,
                initial_num_obs_samples=initial_num_obs_samples,
                type_cost=type_cost,
                init_cost=init_design_cost(ES, x_list, costs))


def _run_cbo(ctx, trials, num_interventions, initial_num_obs_samples,
             causal_prior=True, max_N=None):
    """One vendored-CBO call over a prepared context; returns
    (global_opt, current_cost, arm_log, out, state).  ``arm_log`` is the
    per-intervention (vars, x) sequence captured via the callback; ``state``
    is the CBO state dict (observation cursor, full-precision trial log,
    prior mask).  ``ctx['prior_mask']`` (per-arm) overrides ``causal_prior``.
    ``max_N`` defaults to the protocol's ``n_obs + 50``; passing
    ``initial_num_obs_samples`` makes observation impossible (BOS)."""
    from ccbo.cbo.cbo import CBO

    arm_log = []

    def _cb(int_vars, x_new, _y_new, _sem):
        arm_log.append((list(int_vars), np.ravel(x_new).tolist()))

    if max_N is None:
        max_N = initial_num_obs_samples + 50
    prior = ctx.get("prior_mask", causal_prior)

    out, state = CBO(
        trials, ctx["ES"], ctx["manip_vars"], ctx["x_list"], ctx["y_list"],
        ctx["best_x"], ctx["opt_y"], ctx["best_var"], ctx["dict_ranges"],
        ctx["functions"], ctx["obs"], ctx["coverage"], ctx["cg"],
        20, ctx["costs"], ctx["full_obs"], "min",
        max_N, initial_num_obs_samples,
        num_interventions, Causal_prior=prior, intervention_callback=_cb,
        target_evaluator=mb.population_evaluator(ctx["scm"]),
        observation_cursor=ctx.get("obs_cursor"), return_state=True)
    current_cost, _, _, global_opt, _, _ = out
    return global_opt, current_cost, arm_log, out, state


def _rows_from(global_opt, current_cost, arm_log, start_trial=0,
               cost_offset=0.0, drop_first=False):
    """Aligned (trial, best_y, cum_cost, arm, x_values) rows.

    Intervene rows are the ones where cumulative cost strictly increases
    (unit costs); they consume ``arm_log`` in order. ``drop_first`` drops the
    repeated entry-incumbent row of a warm-started phase-2 call.
    ``cost_offset`` carries the initial-design (and split-design) cost.
    """
    rows, k = [], 0
    for i, (y, c) in enumerate(zip(global_opt, current_cost)):
        if i == 0:
            if drop_first:
                continue
            rows.append((start_trial, float(y), float(cost_offset), "init", ""))
            continue
        if c > current_cost[i - 1]:          # intervene trial
            vars_, xs = arm_log[k]
            k += 1
            arm = "+".join(vars_)
            xv = ";".join(repr(float(v)) for v in xs)
        else:                                 # observe trial
            arm, xv = "", ""
        rows.append((start_trial + i - (1 if drop_first else 0),
                     float(y), float(c) + float(cost_offset), arm, xv))
    assert k == len(arm_log), "arm log misaligned with cost increments"
    return rows


def _unit_meta(ctx, arm, trials, max_N=None, suite="minimal"):
    return dict(suite=suite, scm=ctx["scm"], cond=ctx["cond"], arm=arm,
                seed=int(ctx["seed"]), trials=int(trials),
                n_init=int(ctx["num_interventions"]),
                n_obs=int(ctx["initial_num_obs_samples"]),
                type_cost=int(ctx["type_cost"]),
                max_N=int(max_N if max_N is not None
                          else ctx["initial_num_obs_samples"] + 50))


def _init_meta(ctx, prior_mask):
    cg = ctx["cg"]
    gate = {"+".join(k): v for k, v in getattr(cg, "_arm_gate_status", {}).items()}
    return dict(partition=[sorted(p) for p in ctx["partition"]],
                es=[list(e) for e in ctx["ES"]],
                prior_mask=[bool(b) for b in prior_mask],
                x_list=[np.asarray(x, dtype=float).tolist() for x in ctx["x_list"]],
                y_list=[np.asarray(y, dtype=float).ravel().tolist() for y in ctx["y_list"]],
                init_cost=float(ctx["init_cost"]),
                incumbent=float(ctx["opt_y"]),
                gate_status=gate)


def _uninformative(cg):
    return sorted("+".join(k) for k, ok in cg._arm_identifiable.items() if not ok)


# ---------------------------------------------------------------------------
# Plain units: CBO / QCBO / CBONP / QCBONP / BOS
# ---------------------------------------------------------------------------

def run_plain_unit(scm, cond, arm, seed, trials, num_interventions=10,
                   initial_num_obs_samples=100, type_cost=1):
    """One (scm, condition, arm, seed) unit for a plain arm. Returns
    ``(rows, info)``; ``info['decision_log']`` holds the sidecar payload."""
    assert arm in PLAIN_ARMS, f"unknown plain arm {arm!r}"
    fine = arm in ("CBO", "CBONP")
    causal = arm in ("CBO", "QCBO")
    partition = mb.fine_partition(scm) if fine else mb.coarse_partition(scm)
    ctx = _setup(scm, cond, partition, seed, num_interventions,
                 initial_num_obs_samples, type_cost,
                 subset_arms=(arm == "BOS"))
    max_N = initial_num_obs_samples if arm == "BOS" else None   # BOS never observes
    global_opt, current_cost, arm_log, _, state = _run_cbo(
        ctx, trials, num_interventions, initial_num_obs_samples,
        causal_prior=causal, max_N=max_N)
    assert len(global_opt) == len(current_cost) == trials + 1, (
        f"trajectory length mismatch: {len(global_opt)} vs {trials + 1}")
    rows = _rows_from(global_opt, current_cost, arm_log,
                      cost_offset=ctx["init_cost"])
    prior_mask = state["prior_mask"]
    payload = dlog.build_payload(
        unit=_unit_meta(ctx, arm, trials, max_N),
        init=_init_meta(ctx, prior_mask),
        phases=[dlog.phase_record("main", ctx["ES"], prior_mask, state["trial_log"])],
        rows=rows)
    info = dict(final_y=float(rows[-1][1]), final_cost=float(rows[-1][2]),
                es=[list(e) for e in ctx["ES"]],
                uninformative_arms=_uninformative(ctx["cg"]) if causal else [],
                init_cost=float(ctx["init_cost"]),
                gate_status=payload["init"]["gate_status"],
                observed=int(state["observed"]),
                decision_log=payload)
    return rows, info


# ---------------------------------------------------------------------------
# Joint BO baseline
# ---------------------------------------------------------------------------

def run_bo_unit(scm, cond, seed, trials, num_interventions=10,
                initial_num_obs_samples=100, type_cost=1):
    """One (scm, condition, BO, seed) unit. Returns ``(rows, info)``.

    Standard Bayesian optimization over a single joint arm on every
    manipulable variable: no causal prior, no do-calculus, no observation
    actions.  BO reads no graph structure, so its trajectory is *identical*
    across every misspecification condition of a given SCM; under
    per-variable unit costs it pays ``len(manip_vars)`` per trial.
    Targets are the exact population expectations used by the other arms.
    """
    from ccbo.cbo.bo import NonCausal_BO

    partition = mb.coarse_partition(scm)
    ctx = _setup(scm, cond, partition, seed, num_interventions,
                 initial_num_obs_samples, type_cost, joint_arm=True)

    manip = list(ctx["manip_vars"])
    arm_log = []

    def _cb(int_vars, x_new, _y_new, _sem):
        arm_log.append((list(int_vars), np.ravel(x_new).tolist()))

    current_cost, _, current_best_y, _, trial_log = NonCausal_BO(
        trials, ctx["cg"], ctx["dict_ranges"], ctx["x_list"][0],
        ctx["y_list"][0], ctx["costs"], ctx["obs"], ctx["functions"],
        ctx["best_x"], ctx["opt_y"], manip, Causal_prior=False,
        target_evaluator=mb.population_evaluator(scm), task="min",
        intervention_callback=_cb, return_log=True)

    best_y = np.asarray(current_best_y, dtype=float).ravel()
    cum_cost = np.asarray(current_cost, dtype=float).ravel()
    assert len(best_y) == len(cum_cost) == trials + 1, (
        f"BO trajectory length mismatch: {len(best_y)} vs {trials + 1}")
    assert len(arm_log) == trials, (
        f"BO arm log misaligned: {len(arm_log)} vs {trials}")

    init_cost = float(ctx["init_cost"])
    arm = "+".join(manip)
    rows = [(0, float(best_y[0]), init_cost, "init", "")]
    for j in range(trials):
        xs = ";".join(repr(float(v)) for v in arm_log[j][1])
        rows.append((j + 1, float(best_y[j + 1]),
                     float(cum_cost[j + 1]) + init_cost, arm, xs))
    for e in trial_log:                      # charge the init cost in the log too
        e["cum_cost"] = float(e["cum_cost"]) + init_cost

    payload = dlog.build_payload(
        unit=_unit_meta(ctx, "BO", trials, initial_num_obs_samples),
        init=_init_meta(ctx, [False]),
        phases=[dlog.phase_record("main", [manip], [False], trial_log)],
        rows=rows)
    info = dict(final_y=float(rows[-1][1]), final_cost=float(rows[-1][2]),
                es=[manip], uninformative_arms=[], init_cost=init_cost,
                gate_status={}, observed=0, decision_log=payload)
    return rows, info


# ---------------------------------------------------------------------------
# Native HQCBO (two-phase refine): graph-informed and graph-free
# ---------------------------------------------------------------------------

def _exposed_sub_arms(split_clusters, refine_map):
    """Graph-free refinement: every nonempty union of the sub-clusters of
    each split cluster (canonical order), as the arms the hierarchy exposes."""
    rmap = {frozenset(c): [frozenset(s) for s in subs]
            for c, subs in refine_map.items()}
    arms = []
    for part in split_clusters:
        subs = rmap[frozenset(part)]
        for k in range(1, len(subs) + 1):
            for combo in combinations(range(len(subs)), k):
                arm = sorted(set().union(*[subs[i] for i in combo]))
                if arm not in arms:
                    arms.append(arm)
    arms.sort(key=lambda a: (len(a), a))
    return arms


def run_refine_unit(scm, cond, seed, trials, num_interventions=10,
                    initial_num_obs_samples=100, type_cost=1,
                    plateau_k=PLATEAU_K, plateau_delta=PLATEAU_DELTA,
                    coarse_traj=None, mode="graph"):
    """HQCBO unit: coarse QCBO until plateau, split the incumbent's cluster
    per ``minibench``'s refine map, warm-start the remaining budget.

    ``mode="graph"`` (arm ``HQCBO``): the refined quotient is built from the
    fine hypothesis graph (``assumed_graph_name``); new arms are the refined
    MIS with C-DAG priors, historical arms retained and re-registered with
    C-DAG priors.  A cyclic refined quotient is refused (stays coarse).

    ``mode="free"`` (arm ``HQCBOGF``): the hierarchy alone exposes every
    nonempty union of the split cluster's sub-clusters; new arms get plain
    RBF GPs (prior mask ``False``); historical arms keep their coarse
    priors; **no** object derived from the fine graph is constructed after
    the split and no validity check runs (the coarse quotient is the only
    graph input).

    Both modes charge the split design (``num_interventions`` exact-oracle
    points per new arm) at the trigger row.  ``coarse_traj`` is the QCBO
    unit's best-so-far list for this seed; when None it is recomputed here.
    Phase 1 is byte-identical to the QCBO unit's prefix (same ``_setup``,
    same ``_run_cbo``, same seed).  Following the vendored protocol the
    refined phase begins with one forced observation whenever observational
    budget remains.
    """
    from ccbo.coarsened_graph import CoarsenedGraph
    from ccbo.data_generation import generate_interventional_data
    from ccbo.cbo.utils import compute_coverage, define_initial_data_CBO

    assert mode in ("graph", "free"), mode
    arm_name = "HQCBO" if mode == "graph" else "HQCBOGF"
    refine_map = {mb.MC_NAME: mb.MC_REFINE_MAP}.get(scm)
    assert refine_map is not None, f"no refine map declared for {scm}"

    info = {"trigger_step": None, "split_clusters": [], "split_arms": [],
            "split_init_cost": 0.0, "refused": False, "mode": mode}

    # ---- Trigger time from the coarse trajectory --------------------------
    if coarse_traj is None:
        rows0, _ = run_plain_unit(scm, cond, "QCBO", seed, trials,
                                  num_interventions, initial_num_obs_samples,
                                  type_cost)
        coarse_traj = [r[1] for r in rows0]
    t_fire = next((t for t in range(len(coarse_traj))
                   if _plateau(coarse_traj[:t + 1], plateau_k,
                               plateau_delta, "min")), None)
    if t_fire is not None and not (0 < t_fire < trials):
        t_fire = None

    partition = mb.coarse_partition(scm)
    ctx = _setup(scm, cond, partition, seed, num_interventions,
                 initial_num_obs_samples, type_cost)
    init_cost = float(ctx["init_cost"])
    info["init_cost"] = init_cost
    unit_meta = _unit_meta(ctx, arm_name, trials)

    if t_fire is None:                       # never refines: this IS coarse
        global_opt, current_cost, arm_log, _, state = _run_cbo(
            ctx, trials, num_interventions, initial_num_obs_samples)
        rows = _rows_from(global_opt, current_cost, arm_log, cost_offset=init_cost)
        payload = dlog.build_payload(
            unit=unit_meta, init=_init_meta(ctx, state["prior_mask"]),
            phases=[dlog.phase_record("coarse", ctx["ES"], state["prior_mask"],
                                      state["trial_log"])],
            rows=rows, refine=dict(info))
        info.update(final_y=float(rows[-1][1]), final_cost=float(rows[-1][2]),
                    observed=int(state["observed"]), decision_log=payload)
        return rows, info

    # ---- Phase 1: coarse prefix up to the trigger -------------------------
    go1, cc1, log1, _, state1 = _run_cbo(ctx, t_fire, num_interventions,
                                         initial_num_obs_samples)
    rows = _rows_from(go1, cc1, log1, cost_offset=init_cost)
    phase1 = dlog.phase_record("coarse", ctx["ES"], state1["prior_mask"],
                               state1["trial_log"])
    init_meta = _init_meta(ctx, state1["prior_mask"])

    # Persistent run state crosses the phase boundary: the observational rows
    # revealed in phase 1 and the cursor into the pool.  Phase 2 is still a
    # *fresh optimizer phase* (new arms, new models, forced entry
    # observation) -- it just may not re-reveal rows it has already seen.
    latest_obs = state1["observational_samples"]
    obs_cursor = state1["obs_cursor"]

    # ---- Split the incumbent's cluster (hierarchy only) --------------------
    ES, x_list, y_list = ctx["ES"], ctx["x_list"], ctx["y_list"]
    old_keys = {tuple(e) for e in ES}
    per_arm = [float(np.min(Y)) for Y in y_list]
    incumbent = tuple(ES[int(np.argmin(per_arm))])
    new_partition, split = _split_partition(partition, refine_map, incumbent)

    def _sampler(entries):
        int_data = generate_interventional_data(
            ctx["cg"].define_SEM, entries, ctx["dict_ranges"],
            num_points=20, num_mc_samples=2000, seed=seed + 1000,
            target_evaluator=mb.population_evaluator(scm))
        fx, fy, _, _, _ = define_initial_data_CBO(
            int_data, num_interventions, entries, 0, "min")
        return fx, fy

    fresh_init = {}
    if split and mode == "graph":
        new_cg = None
        try:
            new_cg = CoarsenedGraph(
                ctx["graph"], new_partition, scm, latest_obs,
                num_mc_samples=2000, assumed_graph_name=mb.variant_name(cond))
        except ValueError:                   # invalid split: refuse, stay coarse
            info["refused"] = True
        if new_cg is not None:
            new_ES = [list(e) for e in new_cg.get_sets()[0]]
            ES, x_list, y_list = _union_and_carry(ES, x_list, y_list, new_ES,
                                                  _sampler)
            # Register unioned phase-1 arms with the refined graph before its
            # memoized get_all_do first runs, so they keep C-DAG priors too.
            for e in ES:
                if list(e) not in new_cg._exploration_set:
                    new_cg._exploration_set.append(list(e))
            ctx["cg"] = new_cg
            # refit_models, not fit_all_models: the latter delegates to the
            # fine graph's cached column attributes, which would revert the
            # refined phase to the initial observational rows.
            ctx["functions"] = new_cg.refit_models(latest_obs)
            ctx["costs"] = new_cg.get_cost_structure(type_cost)
            ctx["manip_vars"] = new_cg.get_sets()[2]
            ctx["prior_mask"] = [True] * len(ES)
            ctx["partition"] = list(new_partition)
    elif split and mode == "free":
        exposed = _exposed_sub_arms(split, refine_map)
        new_arms = [a for a in exposed if a not in [list(e) for e in ES]]
        ES, x_list, y_list = _union_and_carry(ES, x_list, y_list,
                                              [list(e) for e in ES] + new_arms,
                                              _sampler)
        # The coarse quotient stays the only graph input: same ``cg``, same
        # costs and manipulables; new arms run on plain GPs.
        ctx["functions"] = ctx["cg"].refit_models(latest_obs)
        ctx["prior_mask"] = [tuple(e) in old_keys for e in ES]
        ctx["partition"] = list(new_partition)

    if split and not info["refused"]:
        # New arms are identified by key, not position: _union_and_carry
        # orders the refined MIS first in graph mode and the old arms first
        # in free mode.
        new_idx = [i for i, e in enumerate(ES) if tuple(e) not in old_keys]
        for i in new_idx:
            fresh_init["+".join(ES[i])] = dict(
                x=np.asarray(x_list[i], dtype=float).tolist(),
                y=np.asarray(y_list[i], dtype=float).ravel().tolist())
        split_cost = init_design_cost([list(ES[i]) for i in new_idx],
                                      [x_list[i] for i in new_idx], ctx["costs"])
        info.update(trigger_step=t_fire,
                    split_clusters=[sorted(c) for c in split],
                    split_arms=["+".join(ES[i]) for i in new_idx],
                    split_init_cost=float(split_cost))
        # Charge the split design at the trigger row (engine v3 cost axis).
        t, y, c, a, xv = rows[-1]
        rows[-1] = (t, y, float(c) + float(split_cost), a, xv)
        _, _, ctx["coverage"] = compute_coverage(
            latest_obs, ctx["manip_vars"], ctx["dict_ranges"])

    # Carry the observational pool state into phase 2 regardless of whether
    # the split was accepted.
    ctx["obs"], ctx["obs_cursor"] = latest_obs, obs_cursor

    # ---- Phase 2 entry incumbents from the carried data -------------------
    s = int(np.argmin([float(np.min(Y)) for Y in y_list]))
    i_best = int(np.argmin(y_list[s]))
    ctx.update(ES=ES, x_list=x_list, y_list=y_list,
               best_var="".join(ES[s]),
               opt_y=float(y_list[s][i_best]),
               best_x=np.asarray(x_list[s][i_best], dtype=float))

    # ---- Phase 2: the remaining budget in one warm-started call -----------
    go2, cc2, log2, _, state2 = _run_cbo(ctx, trials - t_fire, num_interventions,
                                         initial_num_obs_samples)
    rows += _rows_from(go2, cc2, log2, start_trial=t_fire + 1,
                       cost_offset=rows[-1][2], drop_first=True)
    phase2_log = [dict(e) for e in state2["trial_log"]]
    for e in phase2_log:                     # phase-2 costs restart at 0
        e["cum_cost"] = float(e["cum_cost"]) + float(rows[t_fire][2])
    phase2 = dlog.phase_record("refined", ES, state2["prior_mask"], phase2_log,
                               global_offset=t_fire + 1)

    # Best-so-far must be monotone across the boundary; enforce defensively.
    out, best = [], np.inf
    for (t, y, c, a, xv) in rows:
        best = min(best, y)
        out.append((t, best, c, a, xv))
    assert len(out) == trials + 1, (
        f"refine rows mismatch: {len(out)} vs {trials + 1}")

    payload = dlog.build_payload(
        unit=unit_meta, init=init_meta, phases=[phase1, phase2], rows=out,
        refine=dict(info, fresh_init=fresh_init,
                    refined_es=[list(e) for e in ES],
                    refined_prior_mask=[bool(b) for b in state2["prior_mask"]]))
    info.update(final_y=float(out[-1][1]), final_cost=float(out[-1][2]),
                observed=int(state1["observed"]) + int(state2["observed"]),
                decision_log=payload)
    return out, info
