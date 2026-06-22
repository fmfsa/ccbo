"""Generate fine-DAG + C-DAG appendix figures, one pair per benchmark dataset.

For each curated CausalBO dataset (plus the bespoke ConfoundedCluster bow) we
extract the ground-truth structure *from code* — never transcribed by hand:

    benchmark.register_dataset(ds)        -> manipulable variables
    coarsening.project_out_hidden(ds)     -> fine ADMG  {vertices, di, bi}
    coarsening.build_coarsened_admg(...)  -> C-DAG       {vertices(frozensets), di, bi}

and render two matplotlib panels:

    paper/figures/<Dataset>_dag.pdf    the fine ADMG (directed + latent bidirected)
    paper/figures/<Dataset>_cdag.pdf   the quotient under the coarse partition

Visual encoding: solid arrow = directed cause; dashed red double arrow = latent
confounder (bidirected); gold = target Y; orange = manipulable; blue = other
observable; box = a quotient cluster of >1 variable.

Run:  PYTHONPATH=. python scripts/generate_dataset_dags.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

from ccbo import benchmark, coarsening
from ccbo.coarsening import get_dag_edges_from_sem
from ccbo.visualize import NODE_COLOR, MANIP_COLOR, TARGET_COLOR, EDGE_COLOR

FIGURES_DIR = os.path.join('paper', 'figures')
CONF_COLOR = '#b3202c'   # dashed latent-confounder (bidirected) edges

# Benchmark datasets + their coarse partition (besides identity).
# Mirrors scripts/validate_benchmark_qcbo.py:PARTITIONS. 'Y' is the target.
PARTITIONS = {
    'toyGraph':     [['X', 'Z']],
    'synthetic':    [['B'], ['D', 'E']],
    'healthcare':   [['Aspirin', 'Statin']],
    'epidemiology': [['L', 'B']],
    'ecology':      [['C', 'N', 'O'], ['D', 'T']],
}

# (file stem, dataset key, partition clusters, manip override).  toyGraph and
# synthetic_2 are structurally identical (X->Z->Y), so one figure covers both.
DATASETS = [
    ('ToyGraph',          'toyGraph',         PARTITIONS['toyGraph'],     None),
    ('Synthetic',         'synthetic',        PARTITIONS['synthetic'],    None),
    ('Healthcare',        'healthcare',       PARTITIONS['healthcare'],   None),
    ('Epidemiology',      'epidemiology',     PARTITIONS['epidemiology'], None),
    ('Ecology',           'ecology',          PARTITIONS['ecology'],      None),
    ('ConfoundedCluster', 'ConfoundedCluster', [['A'], ['B', 'C']],       None),
]


# --------------------------------------------------------------------------
# Structure extraction
# --------------------------------------------------------------------------

def _extract(ds_key, clusters):
    """Return (fine_admg, c_admg, manip_set) for a dataset, read from code."""
    if ds_key == 'ConfoundedCluster':
        _, _, _, manip = get_dag_edges_from_sem('ConfoundedCluster')
    else:
        _, _, _, manip = benchmark.register_dataset(ds_key)
    fine = coarsening.project_out_hidden(ds_key)
    partition = [frozenset(c) for c in clusters] + [frozenset({'Y'})]
    c_admg = coarsening.build_coarsened_admg(partition, fine)
    return fine, c_admg, set(manip)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def _key(node):
    """Stable, deterministic sort key for str or frozenset nodes."""
    return tuple(sorted(node)) if not isinstance(node, str) else (node,)


def _layered_pos(vertices, di):
    """Top-down layered layout from topological generations, with a light
    barycenter ordering within each layer to reduce edge crossings."""
    DG = nx.DiGraph()
    DG.add_nodes_from(vertices)
    DG.add_edges_from(di)
    pos = {}
    for i, layer in enumerate(nx.topological_generations(DG)):
        if i == 0:
            order = sorted(layer, key=_key)
        else:
            def bary(nd):
                xs = [pos[p][0] for p in DG.predecessors(nd) if p in pos]
                return float(np.mean(xs)) if xs else 0.0
            order = sorted(layer, key=lambda nd: (bary(nd), _key(nd)))
        n = len(order)
        for j, node in enumerate(order):
            pos[node] = ((j - (n - 1) / 2.0) * 2.4, -i * 1.9)
    return pos


def _label(n):
    if isinstance(n, str):
        return n
    return next(iter(n)) if len(n) == 1 else '{' + ','.join(sorted(n)) + '}'


def _draw(ax, admg, manip, pos):
    vertices, di, bi = admg['vertices'], admg['di'], admg['bi']

    G = nx.DiGraph()
    G.add_nodes_from(vertices)
    G.add_edges_from(di)

    # directed causal edges (drawn first, behind the node boxes). Skip-level
    # edges bow outward in proportion to the layers they span, so they fan out
    # instead of overlapping on a single column (e.g. the Healthcare chain).
    def _layer(n):
        return int(round(-pos[n][1] / 1.9))
    for u, v in di:
        skip = abs(_layer(v) - _layer(u))
        rad = 0.06 if skip <= 1 else min(0.55, 0.18 * skip)
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)], node_size=2300,
            edge_color=EDGE_COLOR, width=1.6, arrows=True, arrowstyle='-|>',
            arrowsize=16, connectionstyle=f'arc3,rad={rad}',
            min_source_margin=2, min_target_margin=2)

    # latent confounders -> dashed red double-headed curved arcs
    if bi:
        B = nx.DiGraph()
        B.add_nodes_from(vertices)
        bi_edges = [tuple(e) for e in bi]
        B.add_edges_from(bi_edges)
        nx.draw_networkx_edges(
            B, pos, ax=ax, edgelist=bi_edges, node_size=2300,
            edge_color=CONF_COLOR, width=1.6, style='dashed', arrows=True,
            arrowstyle='<|-|>', arrowsize=13,
            connectionstyle='arc3,rad=0.28')

    # nodes as text boxes on top (covers arrow tips for a clean stop at border)
    for n in vertices:
        if ('Y' in n) if not isinstance(n, str) else (n == 'Y'):
            fc = TARGET_COLOR
        elif (n in manip) if isinstance(n, str) else bool(set(n) & manip):
            fc = MANIP_COLOR
        else:
            fc = NODE_COLOR
        is_cluster = (not isinstance(n, str)) and len(n) > 1
        boxstyle = 'round,pad=0.4' if is_cluster else 'circle,pad=0.32'
        x, y = pos[n]
        ax.text(x, y, _label(n), ha='center', va='center', zorder=5,
                fontsize=10, fontweight='bold', color='#102a3c',
                bbox=dict(boxstyle=boxstyle, facecolor=fc,
                          edgecolor=EDGE_COLOR, linewidth=1.4))

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.6, max(xs) + 1.6)
    ax.set_ylim(min(ys) - 1.2, max(ys) + 1.2)
    ax.axis('off')


def _render(stem, admg, manip, suffix):
    pos = _layered_pos(admg['vertices'], admg['di'])
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    figsize = (max(3.4, 0.95 * (max(xs) - min(xs) + 2.4)),
               max(2.6, 0.95 * (max(ys) - min(ys) + 2.0)))
    fig, ax = plt.subplots(figsize=figsize)
    _draw(ax, admg, manip, pos)
    out = os.path.join(FIGURES_DIR, f'{stem}_{suffix}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for stem, ds_key, clusters, _ in DATASETS:
        fine, c_admg, manip = _extract(ds_key, clusters)
        nbi_fine = len(fine['bi'])
        nbi_c = len(c_admg['bi'])
        print(f"== {stem}: fine |V|={len(fine['vertices'])} bi={nbi_fine}; "
              f"C-DAG |V|={len(c_admg['vertices'])} bi={nbi_c}")
        _render(stem, fine, manip, 'dag')
        _render(stem, c_admg, manip, 'cdag')


if __name__ == '__main__':
    main()
