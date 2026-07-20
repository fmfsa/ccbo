"""Adapter: run the CEO authors' stack on ClusterBench10 (misspec protocol).

Bridges third_party/CEO (Branchini et al. 2023, pinned by scripts/fetch_ceo.sh)
to the ClusterBench10 misspecification study. The hard invariant is the same as
misspec_inject's: every method optimises the TRUE SCM's noise-free objective;
only the structure CEO reasons with (its candidate-graph pool) tracks the
perturbation id. The SEM class handed to CEO is built from ``cb.TRUE_EDGES``
and the generator's coefficients only, so the objective cannot depend on the
pool by construction; ``seam_ok_ceo`` checks it against the benchmark oracle
anyway.

Protocol notes (documented in the paper's fixed-objective appendix):
  * CEO candidate DAGs are over OBSERVABLES only -- its DCBO-style machinery
    has no bidirected edges. P7 (forget U12) therefore has the same observable
    edge set as P0 and is aliased to P0 by the runner.
  * ``--pool hedge``: {asserted variant, P0, distractors P1/P3/P5} dedup'd by
    edge set, uniform prior. The ASSERTED graph is always graphs[0]: CEO's
    acquisition shortcut (posterior on graphs[0] >= 0.9 -> pure optimization
    CES) and its blanket construction key off index 0.
  * ``--pool committed``: {asserted variant} only; the posterior is degenerate
    and CES runs in pure-optimization mode -- the strict analog of what the
    CBO arm receives.
  * Evaluations are noise-free by default (noise_scale=0): the benchmark's
    arms fit GPs on single-draw noise-free evaluations, and CEO hard-wires its
    noisy ``y_new`` into its GP data, so a zero noise scale is the only way to
    give CEO the same evaluation oracle without touching its code.
"""

import os
import sys
import time
import types
import random
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import pandas as pd

from ccbo import benchmark, clusterbench10 as cb

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
CEO_ROOT = os.path.join(REPO_ROOT, "third_party", "CEO")

# Distractor variants for the hedge pool: one intra add (P1), one intra
# reversal (P3), one inter deletion (P5) -- fixed across conditions so the
# pool composition itself never varies with the condition being tested.
DISTRACTORS = ("P0", "P1", "P3", "P5")

_gen = None
_ceo_ready = False


def gen_module():
    """scripts/generate_clusterbench10.py (SCM coefficients; import, don't copy)."""
    global _gen
    if _gen is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_clusterbench10",
            os.path.join(_HERE, "generate_clusterbench10.py"))
        _gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_gen)
    return _gen


def ensure_ceo():
    """Put third_party/CEO on sys.path and patch it -- BEFORE any src.* import.

    Patches (must precede the imports because CEO modules import these
    functions by name at module import time):
      1. ``notebooks.ceo_display_epidem`` stub: ceo_base imports it but the
         released repo ships no ``notebooks/`` directory.
      2. ``make_sequential_intervention_dictionary``: upstream joins node
         names and keeps single alphabetic characters, silently mangling
         multi-character names like ``X1_0`` into {M,X,Y,Z}.
      3. ``get_interventional_grids``: upstream builds a size^dim meshgrid
         whose built-in reduction only triggers at size>=100, so 35^5 rows
         for a 5-variable arm; cap the per-dimension resolution instead.
      4. ``GPy.priors.InverseGamma.from_EV``: an unimplemented stub in
         GPy>=1.13 (CEO's ``fit_gp`` lengthscale prior calls it); restore the
         moment-matching construction (mean b/(a-1), var b^2/((a-1)^2(a-2))).
    """
    global _ceo_ready
    if _ceo_ready:
        return
    if not os.path.isdir(CEO_ROOT):
        raise RuntimeError(f"CEO stack not found at {CEO_ROOT}; "
                           "run scripts/fetch_ceo.sh first.")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if CEO_ROOT not in sys.path:
        sys.path.insert(0, CEO_ROOT)

    nb = types.ModuleType("notebooks")
    disp = types.ModuleType("notebooks.ceo_display_epidem")
    disp.set_plotting = lambda *a, **k: None
    nb.ceo_display_epidem = disp
    sys.modules.setdefault("notebooks", nb)
    sys.modules["notebooks.ceo_display_epidem"] = disp

    import src.utils.sequential_intervention_functions as sif

    def make_blanket(graph, time_series_length):
        variables = sorted({n.split("_")[0] for n in graph.nodes})
        return {v: time_series_length * [None] for v in variables}

    def grids(exploration_set, intervention_limits, size_intervention_grid=100):
        out = {}
        for es in exploration_set:
            if len(es) == 1:
                out[es] = sif.create_n_dimensional_intervention_grid(
                    intervention_limits[es[0]], size_intervention_grid)
            else:
                per_dim = {2: 15, 3: 7}.get(len(es), 4)
                out[es] = sif.create_n_dimensional_intervention_grid(
                    [intervention_limits[j] for j in es], per_dim)
            assert out[es].shape[0] <= 10_000, (es, out[es].shape)
        return out

    sif.make_sequential_intervention_dictionary = make_blanket
    sif.get_interventional_grids = grids

    from GPy.core.parameterization.priors import InverseGamma

    def _inverse_gamma_from_EV(E, V):
        a = np.square(E) / V + 2.0
        b = E * (a - 1.0)
        return InverseGamma(a, b)

    try:
        InverseGamma.from_EV(1.0, 1.0)
    except NotImplementedError:
        InverseGamma.from_EV = staticmethod(_inverse_gamma_from_EV)
    _ceo_ready = True


# ---------------------------------------------------------------------------
# Problem construction (SEM, graphs, exploration sets, data)
# ---------------------------------------------------------------------------

def make_sem_class(noise_scale=0.0):
    """CEO-format SEM class for the TRUE ClusterBench10 SCM.

    ``static()`` returns an OrderedDict of ``lambda noise, t, sample`` in the
    true graph's topological order (CEO's samplers iterate it in key order);
    ``dynamic()`` is None (single time slice). Latents are absent: they are
    zero in the noise-free objective, exactly the benchmark's own convention.
    """
    g = gen_module()
    order = g._causal_order(cb.TRUE_EDGES)
    parents = g._parents(cb.TRUE_EDGES)

    def node_fn(node):
        if node == "Y":
            def fy(noise, t, sample, _p=tuple(parents["Y"])):
                y = g.Y_INTERCEPT
                for p in _p:
                    y = y + g.YA[p] * (sample[p][t] - g.YC[p]) ** 2
                return y + noise_scale * g.NOISE_STD["Y"] * noise
            return fy
        std = g.ROOT_STD.get(node, g.NOISE_STD[node])
        def f(noise, t, sample, _n=node, _p=tuple(parents[node]), _s=std):
            mean = 0.0
            for p in _p:
                mean = mean + g.LIN_COEF[(p, _n)] * sample[p][t]
            return mean + noise_scale * _s * noise
        return f

    fns = OrderedDict((n, node_fn(n)) for n in order)

    class ClusterBench10SEM:
        @staticmethod
        def static():
            return fns

        @staticmethod
        def dynamic():
            return None

    return ClusterBench10SEM


def variant_graph(pid):
    """MultiDiGraph over ``{v}_0`` nodes for perturbation ``pid``.

    Nodes are inserted in a topological order of the variant (CEO's SEMHat
    iterates ``G.nodes`` assuming causal order). Observables only.
    """
    import networkx as nx
    edges = cb.variant_edges(pid)
    order = gen_module()._causal_order(edges)
    gr = nx.MultiDiGraph()
    for n in order:
        gr.add_node(f"{n}_0")
    for u, v in edges:
        gr.add_edge(f"{u}_0", f"{v}_0")
    return gr


def build_pool(pid, mode):
    """Candidate graphs + uniform init posterior. Asserted graph is graphs[0]."""
    if mode == "committed":
        pids = [pid]
    elif mode == "hedge":
        pids, seen = [], set()
        for p in (pid,) + DISTRACTORS:
            key = frozenset(cb.variant_edges(p))
            if key not in seen:
                seen.add(key)
                pids.append(p)
    else:
        raise ValueError(mode)
    graphs = [variant_graph(p) for p in pids]
    return graphs, [1.0 / len(graphs)] * len(graphs), pids


def exploration_sets(mode):
    """ES over the manipulable set, matching what the misspec CBO arm sees.

    ``full``    all 31 non-empty subsets (strict parity with CBO at cap=5);
    ``cluster`` 5 singletons + the two cluster arms + their union (contains
                the global-optimum arm do(C1); CEO's own experiments used <=3
                arms, so this is the tractable fallback);
    ``cap2``    subsets up to size 2 (ablation only).
    """
    import itertools
    manip = list(cb.MANIPULATIVE)
    if mode == "full":
        sizes = range(1, len(manip) + 1)
        es = [tuple(c) for r in sizes for c in itertools.combinations(manip, r)]
    elif mode == "cap2":
        es = [tuple(c) for r in (1, 2) for c in itertools.combinations(manip, r)]
    elif mode == "cluster":
        clusters = [tuple(v for v in manip if v in set(c))
                    for c in cb.COARSE_CLUSTERS]
        es = [(v,) for v in manip] + clusters + [tuple(manip)]
    else:
        raise ValueError(mode)
    return sorted(es, key=lambda t: (len(t), t))


def load_observations():
    """Benchmark observational CSV -> {var: (n, 1)} (base-named, byte-identical
    to what every other method in the study sees)."""
    df = pd.read_csv(benchmark._paths(cb.NAME)["obs"])
    return {v: df[v].to_numpy()[:, None] for v in cb.NODES}


def load_domain():
    cfg = benchmark.load_config(cb.NAME)
    return {v: [float(lo), float(hi)]
            for v, (lo, hi) in cfg["interventional_domain"].items()}


def _forward(static, blanket, rng, noise_scale):
    """One draw of the SEM under ``blanket`` (zero epsilon iff noise-free)."""
    from src.utils.sequential_sampling import sequential_sample_from_true_SEM
    eps = None if noise_scale > 0 else {k: np.zeros(1) for k in static}
    return sequential_sample_from_true_SEM(
        random_state=rng, static_sem=static, dynamic_sem=None, timesteps=1,
        interventions=blanket, epsilon=eps)


def initial_interventional_data(es_list, ninit, seed, sem_cls, domain,
                                noise_scale=0.0):
    """ninit uniform points per arm, mirrored on benchmark._initial_interventional_data.

    Returns ``(D_I, D_I_noiseless)``: per ES entry a dict over ALL observables
    with (ninit, 1) arrays (CEO's initialiser and its posterior update consume
    whole-sample dicts).
    """
    static = sem_cls().static()
    keys = list(static.keys())
    rng = np.random.RandomState(seed)
    D, Dn = {}, {}
    for es in es_list:
        xs = np.column_stack([rng.uniform(domain[v][0], domain[v][1], ninit)
                              for v in es])
        rows, rows_n = {k: [] for k in keys}, {k: [] for k in keys}
        for i in range(ninit):
            blanket = {k: [None] for k in keys}
            for j, v in enumerate(es):
                blanket[v][0] = float(xs[i, j])
            smp_n = _forward(static, blanket, rng, 0.0)
            smp = smp_n if noise_scale == 0 else _forward(
                static, blanket, rng, noise_scale)
            for k in keys:
                rows[k].append(float(smp[k][0]))
                rows_n[k].append(float(smp_n[k][0]))
        D[es] = {k: np.array(rows[k])[:, None] for k in keys}
        Dn[es] = {k: np.array(rows_n[k])[:, None] for k in keys}
    return D, Dn


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_ceo_once(pid, seed, trials, ninit, pool="hedge", es_mode="cluster",
                 noise_scale=0.0, num_anchor_points=35):
    """One CEO run under perturbation ``pid``. Returns (trajectory, meta).

    trajectory has ``trials + 1`` entries (entry 0 = initial incumbent),
    the study's ``current_optimal`` contract.
    """
    ensure_ceo()
    from src.methods.ceo import CEO
    from src.utils.sem_utils.sem_estimate import build_sem_hat

    np.random.seed(seed)
    random.seed(seed)
    rs = np.random.RandomState(seed)

    sem_cls = make_sem_class(noise_scale)
    graphs, init_posterior, pool_pids = build_pool(pid, pool)
    es_list = exploration_sets(es_mode)
    domain = load_domain()
    d_obs = load_observations()
    d_int, d_int_nl = initial_interventional_data(
        es_list, ninit, seed, sem_cls, domain, noise_scale)

    ceo = CEO(
        graphs=graphs,
        init_posterior=init_posterior,
        sem=sem_cls,
        make_sem_estimator=build_sem_hat,
        observation_samples=d_obs,
        intervention_domain=domain,
        intervention_samples=d_int,
        intervention_samples_noiseless=d_int_nl,
        exploration_sets=es_list,
        number_of_trials=trials + 1,
        base_target_variable="Y",
        task=benchmark.load_config(cb.NAME)["task"],
        estimate_sem=True,
        num_anchor_points=num_anchor_points,
        sample_anchor_points=True,
        seed_anchor_points=seed + 1,
        seed=seed,
        manipulative_variables=list(cb.MANIPULATIVE),
        random_state=rs,
    )
    blanket_keys = set(ceo.empty_intervention_blanket)
    assert blanket_keys == set(cb.NODES), (
        f"blanket canary failed (single-char patch not applied?): {blanket_keys}")

    t0 = time.time()
    ceo.run()
    wall = time.time() - t0

    traj = [float(v) for v in ceo.optimal_outcome_values_during_trials[0]]
    assert len(traj) == trials + 1, (len(traj), trials + 1)

    from src.utils.utilities import normalize_log
    meta = {
        "pid": pid, "seed": seed, "trials": trials, "ninit": ninit,
        "pool": pool, "pool_pids": pool_pids, "es_mode": es_mode,
        "n_es": len(es_list), "noise_scale": noise_scale,
        "num_anchor_points": num_anchor_points, "wall_seconds": wall,
        "final_posterior": [float(p) for p in
                            normalize_log(deepcopy(ceo.posterior))],
        "n_posterior_snapshots": len(ceo.all_posteriors),
    }
    return traj, meta


# ---------------------------------------------------------------------------
# Seam check
# ---------------------------------------------------------------------------

def seam_ok_ceo(n=8, tol=1e-9, seed=0):
    """Objective seam: CEO's noiseless target == benchmark oracle.

    Evaluates the SEM handed to CEO (zero epsilon) against the benchmark's
    ``intervention_function`` on every singleton arm plus the cluster arms at
    random levels. The SEM is built from cb.TRUE_EDGES only, so it cannot vary
    with pid/pool; this guards the coefficients and the sampler contract.
    """
    ensure_ceo()
    sys.path.insert(0, _HERE)
    import misspec_inject

    g0, _, _, config = misspec_inject.build_cbo_graph(cb.NAME, seed, "P0")
    static = make_sem_class(0.0)().static()
    keys = list(static.keys())
    dom = config["interventional_domain"]
    manip = list(cb.MANIPULATIVE)
    arms = [[v] for v in manip] + \
           [[v for v in manip if v in set(c)] for c in cb.COARSE_CLUSTERS] + \
           [manip]
    rng = np.random.RandomState(0)
    for arm in arms:
        idx_map = {v: i for i, v in enumerate(arm)}
        tfn, _ = g0.intervention_function(idx_map)
        X = np.column_stack([rng.uniform(dom[v][0], dom[v][1], n) for v in arm])
        for i in range(n):
            a = float(tfn(X[i:i + 1])[0, 0])
            blanket = {k: [None] for k in keys}
            for j, v in enumerate(arm):
                blanket[v][0] = float(X[i, j])
            b = float(_forward(static, blanket, rng, 0.0)["Y"][0])
            if abs(a - b) > tol:
                print(f"seam FAIL arm={arm} x={X[i]}: bench={a} ceo={b}")
                return False
    return True


if __name__ == "__main__":
    print(f"CEO seam check on {cb.NAME}: "
          f"{'OK' if seam_ok_ceo() else 'FAILED'}")
