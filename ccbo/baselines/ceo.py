"""CEO graph declarations and compatibility fixes for the paper experiments."""
from collections import OrderedDict
from pathlib import Path
import os
import sys
import types
import numpy as np
from ccbo import minibench as mb

CEO_ROOT = str(Path(__file__).resolve().parents[2] / "third_party" / "CEO")
_ceo_ready = False
PP_SD = float(np.sqrt(mb.PP_SIGMA_U ** 2 + mb.PP_SIGMA_IN ** 2))
FD_SD_X1 = PP_SD
FD_SD_Y = float(np.sqrt(mb.FD_SIGMA_U ** 2 + mb.SIGMA_Y ** 2))

SCMS = {
    mb.PP_NAME: dict(
        nodes=['X1', 'X2', 'Y'],
        manip=['X1', 'X2'],
        aliases={'A3': 'A0'},
        static=OrderedDict([
            ('X1', lambda noise, t, sample: PP_SD * noise),
            ('X2', lambda noise, t, sample: PP_SD * noise),
            ('Y', lambda noise, t, sample: mb.PP_LAM * (sample['X1'][t] - mb.PP_A) ** 2 + (sample['X2'][t] - mb.PP_B) ** 2 + mb.SIGMA_Y * noise),
        ]),
    ),
    mb.FD_NAME: dict(
        nodes=['X1', 'M', 'Y'],
        manip=['X1', 'M'],
        aliases={'B1': 'B0'},
        static=OrderedDict([
            ('X1', lambda noise, t, sample: FD_SD_X1 * noise),
            ('M', lambda noise, t, sample: mb.FD_B * sample['X1'][t] + mb.FD_SIGMA_M * noise),
            ('Y', lambda noise, t, sample: mb.FD_AMP * (1.0 - np.exp(-(sample['M'][t] - mb.FD_C) ** 2 / (2 * mb.FD_W ** 2))) + FD_SD_Y * noise),
        ]),
    ),
    mb.MC_NAME: dict(
        nodes=['X1', 'X2', 'Y'],
        manip=['X1', 'X2'],
        aliases={},
        static=OrderedDict([
            ('X1', lambda noise, t, sample: mb.MC_SIGMA_1 * noise),
            ('X2', lambda noise, t, sample: mb.MC_BETA * sample['X1'][t] + mb.MC_SIGMA_2 * noise),
            ('Y', lambda noise, t, sample: (sample['X2'][t] - mb.MC_C) ** 2 + mb.SIGMA_Y * noise),
        ]),
    ),
}

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

def ensure_ceo():
    """Load pinned CEO and repair node names, grids, priors and GP bookkeeping."""
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

    import src.utils.gp_utils as gp_utils
    _builtin_str = str

    def _memo_key(o):
        return o.tobytes() if isinstance(o, np.ndarray) else _builtin_str(o)

    gp_utils.str = _memo_key

    import src.utils.ceo_utils as ceo_utils

    def update_posterior_interventional(graphs, posterior, intervened_var,
                                        all_emission_fncs,
                                        interventional_samples,
                                        total_timesteps=1, it=0, lr=0.05):
        for graph_idx, emission_fncs in enumerate(all_emission_fncs):
            for temporal_index in range(total_timesteps):
                for pa in emission_fncs[temporal_index]:
                    xx, yy, inputs, output = ceo_utils.get_sem_emit_obs(
                        G=graphs[graph_idx], sem_emit_fncs=emission_fncs,
                        observational_samples=interventional_samples,
                        t=temporal_index, pa=pa, t_index_data=None)
                    if isinstance(output, list):
                        assert len(output) == 1
                        output = output[0]
                    if output in intervened_var:
                        continue
                    posterior[graph_idx] += lr * ceo_utils.log_likelihood(
                        emission_fncs[temporal_index][pa], xx, yy, graph_idx,
                        inputs, output, it)
        return posterior

    ceo_utils.update_posterior_interventional = update_posterior_interventional
    _ceo_ready = True
