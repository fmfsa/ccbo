"""CEO (authors' stack) on MinimalBench: the graph-uncertain comparator.

Answers the reviewer question "why commit to a quotient instead of learning
structural uncertainty online?" on exactly the paper's misspecification
benchmark: same SCMs, same conditions, same seeds (0-29), same trial budget
(T=50, 60 on MediatedChain), same n_init=3 points per arm, same shared
observational dataset (first 100 rows), same objective.

Protocol (prespecified before any result was seen)
--------------------------------------------------
* Objective seam: every method optimises E[Y | do(x)] of the TRUE SCM. The
  vendored CBO/QCBO arms estimate it with a common-random-numbers Monte
  Carlo mean (``Intervention_function``, 100k samples); CEO's NOISELESS
  path (trajectory/incumbent scoring) receives the same surface EXACTLY
  via MinimalBench's closed forms patched in as its per-arm noiseless
  target functions. ``seam_gate`` verifies the closed forms against a
  fresh Monte-Carlo estimate of the true latent SEM at random levels.
* Learning seam: CEO's NOISY path — the samples that train its GPs and
  update its graph posterior — consists of genuine stochastic draws from
  the TRUE latent SEM (latent confounders included, then discarded),
  drawn through CEO's own stateful per-seed RandomState. This is CEO's
  native regime; the fine methods instead receive CRN-MC mean
  evaluations, an asymmetry disclosed in the paper's protocol appendix.
* CEO's candidate DAGs are over OBSERVABLES only (its machinery has no
  bidirected edges), so conditions that only add a latent confounder are
  observationally identical to their base condition and are ALIASED:
  A3 -> A0, B1 -> B0. This inability to represent the confounder error
  class is itself a documented finding, mirroring the CB10 protocol's P7.
* Pool ("hedge"): the asserted condition's observable DAG first (CEO's
  acquisition shortcut keys off graphs[0]), then every observable-distinct
  MinimalBench variant of the same SCM, uniform prior. On ParallelParent
  that is {asserted, A0, A1, A2} deduplicated; FrontDoor and MediatedChain
  have no observable-distinct variants, so their pools are singletons
  (CEO runs in committed / pure-optimisation mode -- documented).
* Exploration set: all non-empty subsets of the manipulable variables
  (CEO's own convention; on these 2-manipulable SCMs that is 3 arms,
  a superset of the fine MIS where the joint arm is non-minimal).
* Metrics (prespecified): final best-so-far E[Y|do]; full trajectory
  (trial_number, current_optimal; row 0 = initial incumbent); per-seed
  paired differences vs the archived CBO/QCBO trajectories on the same
  seeds; for multi-graph pools, CEO's final posterior mass on the baseline
  observable DAG (the truth's directed edges with bidirected edges dropped).

Run one unit:   PYTHONPATH=. python scripts/run_ceo_minimal.py \
                    --scm ParallelParent --cond A1 --seed 0
Local sweep:    PYTHONPATH=. python scripts/run_ceo_minimal.py --all --jobs 8
LSF array:      bash scripts/lsf/submit_ceo_minimal.sh
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import json
import time
import random
import argparse
from copy import deepcopy
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, REPO)

from ccbo import minibench as mb
import ceo_adapter  # reused for ensure_ceo() stack patches only

OUTDIR_DEFAULT = os.path.join(REPO, "results", "ceo_minimal")

# ---------------------------------------------------------------------------
# SCM registry: observable DAG, collapsed SEM, closed-form oracle
# ---------------------------------------------------------------------------
# Collapsed SEMs preserve every interventional marginal the arms can reach
# (each arm clamps at least one of the two upstream manipulables, so only
# marginal natural-response laws matter); latent variance is folded into the
# nearest observable noise. Evaluations are noise-free forward passes whose
# Y is REPLACED by the closed-form E[Y | do(blanket)], so the objective is
# exact regardless of the collapse.

PP_SD = float(np.sqrt(mb.PP_SIGMA_U ** 2 + mb.PP_SIGMA_IN ** 2))   # 0.5
FD_SD_X1 = PP_SD
FD_SD_Y = float(np.sqrt(mb.FD_SIGMA_U ** 2 + mb.SIGMA_Y ** 2))


def _pp_oracle(iv):
    if set(iv) == {"X1", "X2"}:
        return mb.pp_do_joint(iv["X1"], iv["X2"])
    if set(iv) == {"X1"}:
        return mb.pp_do_x1(iv["X1"])
    if set(iv) == {"X2"}:
        return mb.pp_do_x2(iv["X2"])
    raise ValueError(f"unexpected PP intervention {iv}")


def _fd_oracle(iv):
    if "M" in iv:                       # do(M) or do(X1, M): M clamped
        return mb.fd_do_m(iv["M"])
    if set(iv) == {"X1"}:
        return mb.fd_do_x1(iv["X1"])
    raise ValueError(f"unexpected FD intervention {iv}")


def _mc_oracle(iv):
    if "X2" in iv:                      # do(X2) or do(X1, X2): X2 clamped
        return mb.mc_do_x2(iv["X2"])
    if set(iv) == {"X1"}:
        return mb.mc_do_x1(iv["X1"])
    raise ValueError(f"unexpected MC intervention {iv}")


SCMS = {
    mb.PP_NAME: dict(
        nodes=["X1", "X2", "Y"], manip=["X1", "X2"],
        domain={"X1": list(mb.DOMAIN), "X2": list(mb.DOMAIN)},
        conds=["A0", "A1", "A2"], aliases={"A3": "A0"}, true_cond="A0",
        trials=50, oracle=_pp_oracle,
        static=OrderedDict([
            ("X1", lambda noise, t, sample: PP_SD * noise),
            ("X2", lambda noise, t, sample: PP_SD * noise),
            ("Y", lambda noise, t, sample:
                mb.PP_LAM * (sample["X1"][t] - mb.PP_A) ** 2
                + (sample["X2"][t] - mb.PP_B) ** 2 + mb.SIGMA_Y * noise),
        ]),
    ),
    mb.FD_NAME: dict(
        nodes=["X1", "M", "Y"], manip=["X1", "M"],
        domain={"X1": list(mb.DOMAIN), "M": list(mb.DOMAIN)},
        conds=["B0"], aliases={"B1": "B0"}, true_cond="B0",
        trials=50, oracle=_fd_oracle,
        static=OrderedDict([
            ("X1", lambda noise, t, sample: FD_SD_X1 * noise),
            ("M", lambda noise, t, sample:
                mb.FD_B * sample["X1"][t] + mb.FD_SIGMA_M * noise),
            ("Y", lambda noise, t, sample:
                mb.FD_AMP * (1.0 - np.exp(
                    -(sample["M"][t] - mb.FD_C) ** 2 / (2 * mb.FD_W ** 2)))
                + FD_SD_Y * noise),
        ]),
    ),
    mb.MC_NAME: dict(
        nodes=["X1", "X2", "Y"], manip=["X1", "X2"],
        domain={"X1": list(mb.DOMAIN), "X2": list(mb.MC_X2_BOX)},
        conds=["C0"], aliases={}, true_cond="C0",
        trials=60, oracle=_mc_oracle,
        static=OrderedDict([
            ("X1", lambda noise, t, sample: mb.MC_SIGMA_1 * noise),
            ("X2", lambda noise, t, sample:
                mb.MC_BETA * sample["X1"][t] + mb.MC_SIGMA_2 * noise),
            ("Y", lambda noise, t, sample:
                (sample["X2"][t] - mb.MC_C) ** 2 + mb.SIGMA_Y * noise),
        ]),
    ),
}

RUN_UNITS = [(scm, cond) for scm, spec in SCMS.items() for cond in spec["conds"]]


def observable_edges(scm, cond):
    """Directed observable edges of a condition's assumed graph."""
    base = {mb.PP_NAME: mb.PP_TRUE_EDGES, mb.FD_NAME: mb.FD_TRUE_EDGES,
            mb.MC_NAME: mb.MC_TRUE_EDGES}[scm]
    pert = next(p for p in mb.PERTURBATIONS
                if p["scm"] == scm and p["id"] == cond)
    edges = list(base)
    for op, u, v in pert.get("ops", []):
        if op == "del":
            edges.remove((u, v))
        elif op == "add":
            edges.append((u, v))
        elif op == "rev":
            edges.remove((u, v)); edges.append((v, u))
    return edges


def variant_graph(scm, cond):
    import networkx as nx
    spec = SCMS[scm]
    gr = nx.MultiDiGraph()
    for n in spec["nodes"]:                      # causal order
        gr.add_node(f"{n}_0")
    for u, v in observable_edges(scm, cond):
        gr.add_edge(f"{u}_0", f"{v}_0")
    return gr


def build_pool(scm, cond):
    """Asserted graph first, then every observable-distinct variant."""
    spec = SCMS[scm]
    pids, seen = [], set()
    for p in [cond] + [c for c in spec["conds"] if c != cond]:
        key = frozenset(observable_edges(scm, p))
        if key not in seen:
            seen.add(key)
            pids.append(p)
    graphs = [variant_graph(scm, p) for p in pids]
    return graphs, [1.0 / len(graphs)] * len(graphs), pids


def exploration_sets(scm):
    import itertools
    manip = SCMS[scm]["manip"]
    return sorted((tuple(c) for r in range(1, len(manip) + 1)
                   for c in itertools.combinations(manip, r)),
                  key=lambda t: (len(t), t))


def load_observations(scm, n_obs=100):
    df = pd.read_pickle(os.path.join(
        REPO, "ccbo", "cbo", "data", scm, "observations.pkl"))[:n_obs]
    return {v: df[v].to_numpy()[:, None] for v in SCMS[scm]["nodes"]}


# ---------------------------------------------------------------------------
# True-SEM sampling (noisy path) and oracle targets (noiseless path)
# ---------------------------------------------------------------------------

def sample_true(scm, iv, rng):
    """One stochastic draw of the TRUE latent SEM under clamps ``iv``.

    Latent confounders (U on ParallelParent/FrontDoor) are drawn and then
    discarded; all observables are returned. Consumes exactly one
    ``rng.randn(4)`` per call (stateful: consecutive calls differ).
    """
    z = rng.randn(4)
    if scm == mb.PP_NAME:
        U = mb.PP_SIGMA_U * z[0]
        X1 = iv.get("X1", U + mb.PP_SIGMA_IN * z[1])
        X2 = iv.get("X2", U + mb.PP_SIGMA_IN * z[2])
        Y = (mb.PP_LAM * (X1 - mb.PP_A) ** 2 + (X2 - mb.PP_B) ** 2
             + mb.SIGMA_Y * z[3])
        return {"X1": float(X1), "X2": float(X2), "Y": float(Y)}
    if scm == mb.FD_NAME:
        U = mb.FD_SIGMA_U * z[0]
        X1 = iv.get("X1", U + mb.PP_SIGMA_IN * z[1])
        M = iv.get("M", mb.FD_B * X1 + mb.FD_SIGMA_M * z[2])
        Y = (mb.FD_AMP * (1.0 - np.exp(
            -(M - mb.FD_C) ** 2 / (2 * mb.FD_W ** 2)))
            + mb.FD_G * U + mb.SIGMA_Y * z[3])
        return {"X1": float(X1), "M": float(M), "Y": float(Y)}
    if scm == mb.MC_NAME:
        X1 = iv.get("X1", mb.MC_SIGMA_1 * z[1])
        X2 = iv.get("X2", mb.MC_BETA * X1 + mb.MC_SIGMA_2 * z[2])
        Y = (X2 - mb.MC_C) ** 2 + mb.SIGMA_Y * z[3]
        return {"X1": float(X1), "X2": float(X2), "Y": float(Y)}
    raise ValueError(scm)


def forward_mean(scm, iv):
    """Noise-free forward pass with clamps ``iv``; Y replaced by the oracle."""
    spec = SCMS[scm]
    sample = {}
    for node, fn in spec["static"].items():
        val = iv[node] if node in iv else fn(0.0, 0, {k: [v] for k, v in sample.items()})
        sample[node] = float(val)
    sample["Y"] = float(spec["oracle"](iv)) if iv else sample["Y"]
    return sample


def make_target_factory(scm):
    """Same signature/contract as ceo_utils.evaluate_target_function_all_for_ceo.

    ``noisy=True``  -> genuine stochastic draw of the TRUE latent SEM through
                       the shared stateful ``random_state`` (feeds CEO's GP
                       targets and graph-posterior evidence).
    ``noisy=False`` -> deterministic mean profile with the closed-form
                       E[Y|do] oracle; consumes NO randomness (feeds only the
                       trajectory/incumbent scoring).
    """

    def factory(noisy, random_state, initial_structural_equation_model,
                structural_equation_model, graph, exploration_set, all_vars, T):
        def target(current_target, intervention_levels, assigned_blanket):
            levels = np.atleast_1d(np.asarray(intervention_levels, dtype=float))
            iv = {v: float(levels[j]) for j, v in enumerate(exploration_set)}
            sample = (sample_true(scm, iv, random_state) if noisy
                      else forward_mean(scm, iv))
            return {k: np.array([v]) for k, v in sample.items()}
        return target

    return factory


def seam_gate(scm, rng, n=6, mc=200_000, sigmas=5.0):
    """Closed-form oracle == MC mean of the TRUE latent SEM, per arm."""
    spec = SCMS[scm]
    for es in exploration_sets(scm):
        for _ in range(n):
            iv = {v: float(rng.uniform(*spec["domain"][v])) for v in es}
            ys = np.array([sample_true(scm, iv, rng)["Y"] for _ in range(mc)])
            mean, se = float(np.mean(ys)), float(np.std(ys) / np.sqrt(mc))
            oracle = float(spec["oracle"](iv))
            assert abs(mean - oracle) <= sigmas * se + 1e-6, (
                f"SEAM FAIL {scm} {iv}: oracle {oracle:.5f} vs MC "
                f"{mean:.5f} +- {se:.5f}")
    return True


def initial_interventional_data(scm, es_list, ninit, seed):
    """ninit uniform points per arm at SHARED levels: a noisy/noiseless pair.

    Returns ``(D_noisy, D_noiseless)``: the same intervention levels per arm,
    with noisy responses drawn from the true latent SEM (feeds CEO's GP and
    graph posterior) and noiseless responses from the mean profile with the
    closed-form oracle Y (feeds CEO's initial-incumbent selection).
    """
    spec = SCMS[scm]
    rng = np.random.RandomState(seed)
    keys = spec["nodes"]
    D, Dn = {}, {}
    for es in es_list:
        rows = {k: [] for k in keys}
        rows_n = {k: [] for k in keys}
        for _ in range(ninit):
            iv = {v: float(rng.uniform(*spec["domain"][v])) for v in es}
            noisy = sample_true(scm, iv, rng)
            clean = forward_mean(scm, iv)
            for k in keys:
                rows[k].append(noisy[k])
                rows_n[k].append(clean[k])
        D[es] = {k: np.array(rows[k])[:, None] for k in keys}
        Dn[es] = {k: np.array(rows_n[k])[:, None] for k in keys}
    return D, Dn


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def csv_path(outdir, scm, cond, seed):
    return os.path.join(outdir, f"CEO_{scm}_{cond}_seed{seed}.csv")


def run_unit(scm, cond, seed, outdir, num_anchor_points=35):
    spec = SCMS[scm]
    trials = spec["trials"]
    out_csv = csv_path(outdir, scm, cond, seed)
    if os.path.exists(out_csv.replace(".csv", ".meta.json")):
        print(f"skip {scm} {cond} seed{seed} (done)")
        return

    ceo_adapter.ensure_ceo()
    import src.bases.ceo_base as ceo_base
    ceo_base.evaluate_target_function_all_for_ceo = make_target_factory(scm)
    from src.methods.ceo import CEO
    from src.utils.sem_utils.sem_estimate import build_sem_hat
    from src.utils.utilities import normalize_log

    np.random.seed(seed)
    random.seed(seed)
    rs = np.random.RandomState(seed)

    seam_gate(scm, np.random.RandomState(10_000 + seed))

    graphs, init_posterior, pool_pids = build_pool(scm, cond)
    es_list = exploration_sets(scm)
    d_obs = load_observations(scm)
    d_int, d_int_nl = initial_interventional_data(scm, es_list, ninit=3,
                                                  seed=seed)

    class SEM:
        @staticmethod
        def static():
            return spec["static"]

        @staticmethod
        def dynamic():
            return None

    t0 = time.time()
    ceo = CEO(
        graphs=graphs,
        init_posterior=list(init_posterior),
        sem=SEM,
        make_sem_estimator=build_sem_hat,
        observation_samples=d_obs,
        intervention_domain={k: list(v) for k, v in spec["domain"].items()},
        intervention_samples=deepcopy(d_int),
        intervention_samples_noiseless=deepcopy(d_int_nl),
        exploration_sets=es_list,
        number_of_trials=trials + 1,
        base_target_variable="Y",
        task="min",
        estimate_sem=True,
        num_anchor_points=num_anchor_points,
        sample_anchor_points=True,
        seed_anchor_points=seed + 1,
        seed=seed,
        manipulative_variables=list(spec["manip"]),
        random_state=rs,
    )
    blanket = set(ceo.empty_intervention_blanket)
    assert blanket == set(spec["nodes"]), (
        f"blanket canary failed: {blanket} != {set(spec['nodes'])}")
    ceo.run()
    wall = time.time() - t0

    traj = [float(v) for v in ceo.optimal_outcome_values_during_trials[0]]
    assert len(traj) == trials + 1, (len(traj), trials + 1)

    # Per-trial history for initialization-inclusive cost accounting:
    # per_trial_cost[0] == [0.0] + [len(es) per intervention trial] under
    # cost_type=1 (1.0 per intervened variable = per-variable unit costs).
    per_trial_cost = [float(c) for c in ceo.per_trial_cost[0]]
    seq_es = [list(es) for es in
              ceo.sequence_of_interventions_during_trials[0]]
    assert len(per_trial_cost) == trials + 1, (len(per_trial_cost), trials + 1)
    assert per_trial_cost[1:] == [float(len(e)) for e in seq_es], \
        "per-trial cost does not match selected exploration sets"

    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame({"trial_number": range(len(traj)),
                  "current_optimal": traj}).to_csv(out_csv, index=False)
    meta = {
        "scm": scm, "cond": cond, "seed": seed, "trials": trials, "ninit": 3,
        "n_obs": 100, "pool_pids": pool_pids, "es": [list(e) for e in es_list],
        # The true PP/FD graphs carry bidirected edges and are in no pool.
        # This names the baseline observable DAG: the pool member with the
        # truth's directed edges and its bidirected edges dropped.
        "baseline_observable_cond": spec["true_cond"],
        "posterior_final": [float(p) for p in
                            normalize_log(deepcopy(ceo.posterior))],
        "per_trial_cost": per_trial_cost,
        "sequence_of_interventions": seq_es,
        "init_cost": 3 * sum(len(e) for e in es_list),
        "wall_seconds": wall, "num_anchor_points": num_anchor_points,
        "objective": "noiseless path: closed-form E[Y|do] (minibench "
                     "oracles), seam-gated against 200k-sample MC of the "
                     "true latent SEM; noisy path (GP + posterior "
                     "evidence): stochastic draws of the true latent SEM",
    }
    with open(out_csv.replace(".csv", ".meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"done {scm} {cond} seed{seed}: final {traj[-1]:.4f} "
          f"posterior {meta['posterior_final']} ({wall:.0f}s)")


def materialize_aliases(outdir):
    """A3 <- A0, B1 <- B0 (observable-identical for CEO)."""
    import shutil
    for scm, spec in SCMS.items():
        for alias, base in spec["aliases"].items():
            for seed in range(30):
                src_csv = csv_path(outdir, scm, base, seed)
                dst_csv = csv_path(outdir, scm, alias, seed)
                if os.path.exists(src_csv) and not os.path.exists(dst_csv):
                    shutil.copyfile(src_csv, dst_csv)
                    meta_src = src_csv.replace(".csv", ".meta.json")
                    if os.path.exists(meta_src):
                        meta = json.load(open(meta_src))
                        meta["cond"] = alias
                        meta["alias_of"] = base
                        meta["note"] = ("observable edge set identical to "
                                        f"{base}; CEO cannot represent the "
                                        "assumed-confounder error class")
                        with open(dst_csv.replace(".csv", ".meta.json"),
                                  "w") as fh:
                            json.dump(meta, fh, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scm", choices=list(SCMS))
    ap.add_argument("--cond")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    ap.add_argument("--aliases", action="store_true",
                    help="only materialize alias CSVs and exit")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.aliases:
        materialize_aliases(args.outdir)
        return

    if args.all:
        lo, hi = (args.seeds.split("-") + [args.seeds])[:2]
        seeds = range(int(lo), int(hi) + 1)
        units = [(s, c, sd) for (s, c) in RUN_UNITS for sd in seeds]
        if args.jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = [ex.submit(run_unit, s, c, sd, args.outdir)
                        for s, c, sd in units]
                for f in futs:
                    f.result()
        else:
            for s, c, sd in units:
                run_unit(s, c, sd, args.outdir)
        materialize_aliases(args.outdir)
    else:
        assert args.scm and args.cond is not None and args.seed is not None
        run_unit(args.scm, args.cond, args.seed, args.outdir)


if __name__ == "__main__":
    main()
