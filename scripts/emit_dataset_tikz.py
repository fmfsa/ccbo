"""Emit TikZ DAG + C-DAG figures for the 6 standardized CausalBO datasets.

For each dataset we read its registered structure (directed edges + latent
confounders + manipulable set) and build the quotient C-DAG under the coarse
partition used in the paper, then lay both out with a longest-path layered
layout and emit a TikZ `figure` per dataset into paper/figures/standardized_dags.tex
(same drawing convention as Fig. clusterbench10). The appendix \inputs that file.

Run:  PYTHONPATH=. python scripts/emit_dataset_tikz.py
"""

import os
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

from ccbo import benchmark, coarsening
from ccbo.coarsened_graph import CoarsenedGraph

OUT = "paper/figures/standardized_dags.tex"

# dataset -> (coarse manipulable clusters, pretty name)
COARSE = {
    "toyGraph":     ([["X", "Z"]], "ToyGraph"),
    "synthetic_2":  ([["X", "Z"]], "Synthetic-2"),
    "synthetic":    ([["B"], ["D", "E"]], "Synthetic"),
    "healthcare":   ([["Aspirin", "Statin"]], "Healthcare"),
    "epidemiology": ([["L", "B"]], "Epidemiology"),
    "ecology":      ([["C", "N", "O"], ["D", "T"]], "Ecology"),
}


def _safe(name):
    """TikZ-safe node id (letters/digits only)."""
    return "n" + "".join(ch for ch in name if ch.isalnum())


def _label(name):
    """Readable node label."""
    return name.replace("_", "")


def layered_pos(nodes, edges, sinks, dx=None, dy=None, clusters=()):
    """Longest-path layering with barycenter crossing reduction.

    Sinks (e.g. Y) are pinned to the last layer; within-layer order is then
    refined by a few barycenter sweeps (median-free Sugiyama step), which
    removes most edge crossings on the denser datasets (Healthcare, Ecology).
    Spacing adapts to the graph: wide layers pack slightly tighter vertically,
    shallow graphs spread slightly wider horizontally.

    ``clusters`` makes the layout partition-aware: source members of a cluster
    are pulled to the cluster's earliest layer (so an isolated member like
    Ecology's T sits next to its cluster-mates instead of drifting), and each
    layer is finally stable-sorted so cluster members are vertically adjacent
    at the top --- the fitted cluster boxes then stay tight and cannot engulf
    non-members.
    """
    succ, pred = defaultdict(list), defaultdict(list)
    indeg = {n: 0 for n in nodes}
    for u, v in edges:
        if u in indeg and v in indeg:
            succ[u].append(v)
            pred[v].append(u)
            indeg[v] += 1
    # Kahn topological order
    q = [n for n in nodes if indeg[n] == 0]
    order, ind = [], dict(indeg)
    while q:
        n = q.pop(0)
        order.append(n)
        for v in succ[n]:
            ind[v] -= 1
            if ind[v] == 0:
                q.append(v)
    layer = {n: 0 for n in nodes}
    for n in order:
        for v in succ[n]:
            layer[v] = max(layer[v], layer[n] + 1)
    # pin sinks (Y) strictly to the rightmost layer
    if sinks:
        mx = max(layer.values(), default=0)
        for s in sinks:
            if s in layer:
                layer[s] = mx
    # pull source members of each cluster to the cluster's earliest layer, so
    # weakly-connected members sit beside their cluster-mates
    for cl in clusters:
        members = [m for m in cl if m in layer]
        if len(members) >= 2:
            lmin = min(layer[m] for m in members)
            for m in members:
                if indeg.get(m, 0) == 0:
                    layer[m] = lmin
    bylayer = defaultdict(list)
    for n in nodes:
        bylayer[layer[n]].append(n)
    orders = {L: sorted(ns) for L, ns in bylayer.items()}
    layers = sorted(orders)

    # ---- barycenter sweeps: order each layer by the mean index of its
    #      neighbours in the previously-ordered adjacent layer ----
    for sweep in range(4):
        forward = sweep % 2 == 0
        seq = layers[1:] if forward else layers[-2::-1]
        for L in seq:
            refL = L - 1 if forward else L + 1
            if refL not in orders:
                continue
            idx = {n: i for i, n in enumerate(orders[refL])}
            nbrs = pred if forward else succ
            cur = {n: i for i, n in enumerate(orders[L])}

            def bary(n):
                ns = [idx[m] for m in nbrs[n] if m in idx]
                return sum(ns) / len(ns) if ns else cur[n]
            orders[L] = sorted(orders[L], key=bary)

    # ---- adaptive spacing ----
    widest = max((len(ns) for ns in orders.values()), default=1)
    if dx is None:
        dx = 2.4 if len(layers) <= 3 else 2.1
    if dy is None:
        dy = 1.35 if widest <= 4 else 1.1

    # ---- global row bands: each multi-member cluster gets dedicated top rows
    #      (the same rows in every layer), non-members stack below. The fitted
    #      cluster boxes then occupy disjoint horizontal strips and can neither
    #      engulf a non-member nor overlap each other, no matter how far apart
    #      a cluster's members sit across layers. ----
    multi = [cl for cl in clusters
             if len([m for m in cl if m in layer]) >= 2]
    row_of_cluster, cursor = {}, 0
    for k, cl in enumerate(multi):
        percount = defaultdict(int)
        for m in cl:
            if m in layer:
                percount[layer[m]] += 1
        need = max(percount.values(), default=1)
        row_of_cluster[k] = cursor
        cursor += need
    reserved = cursor

    def cluster_idx(n):
        for k, cl in enumerate(multi):
            if n in cl:
                return k
        return None

    pos = {}
    for L in layers:
        ns = orders[L]
        used = defaultdict(int)   # cluster -> members already placed this layer
        free = reserved           # next free row for non-members
        for n in ns:              # barycenter order preserved within groups
            k = cluster_idx(n)
            if k is None:
                row = free
                free += 1
            else:
                row = row_of_cluster[k] + used[k]
                used[k] += 1
            pos[n] = (L * dx, -row * dy)
    return pos


def _cluster_of(n, clusters):
    for k, cl in enumerate(clusters):
        if n in cl:
            return k
    return None


def tikz_graph(nodes, di, bi, manip, target, clusters, pos):
    """Emit one tikzpicture for a (di, bi) graph with cluster boxes.

    Uses the shared style vocabulary defined in the paper preamble (obsnode /
    mannode / tgtnode / clbox / diredge / biintra / bicross) so the appendix
    figures match Fig. 1 exactly. Bidirected edges are drawn intra- vs
    cross-cluster with distinct styles, with bends computed from the endpoint
    distance (closer pairs arc more) and staggered when several confounders
    share an endpoint, so parallel arcs cannot coincide.
    """
    man = set(manip)
    L = [r"\begin{tikzpicture}"]
    for n in nodes:
        x, y = pos[n]
        sty = "tgtnode" if n == target else ("mannode" if n in man else "obsnode")
        L.append(f"  \\node[{sty}] ({_safe(n)}) at ({x:.2f},{y:.2f}) {{{_label(n)}}};")
    for u, v in di:
        if u in pos and v in pos:
            L.append(f"  \\draw[diredge] ({_safe(u)}) -- ({_safe(v)});")
    seen_at = defaultdict(int)  # endpoint -> #bidirected arcs so far (stagger)
    for u, v in bi:
        if u not in pos or v not in pos:
            continue
        cu, cv = _cluster_of(u, clusters), _cluster_of(v, clusters)
        sty = "biintra" if (cu is not None and cu == cv) else "bicross"
        (x1, y1), (x2, y2) = pos[u], pos[v]
        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        bend = max(14, int(34 - 4 * dist)) + 8 * max(seen_at[u], seen_at[v])
        seen_at[u] += 1
        seen_at[v] += 1
        L.append(f"  \\draw[{sty}] ({_safe(u)}) to[bend left={bend}] ({_safe(v)});")
    # cluster boxes (only multi-member manipulable clusters)
    for k, cl in enumerate(clusters):
        members = [c for c in cl if c in pos]
        if len(members) >= 2:
            fit = "".join(f"({_safe(c)})" for c in members)
            L.append(r"  \begin{scope}[on background layer]")
            L.append(f"    \\node[clbox,fit={fit},"
                     f"label={{[label distance=0.5mm]above:{{\\scriptsize$C_{{{k+1}}}$}}}}] {{}};")
            L.append(r"  \end{scope}")
    L.append(r"\end{tikzpicture}")
    return "\n".join(L)


def cdag_struct(cg):
    """Extract C-DAG (di, bi, vertices) from a CoarsenedGraph as label strings."""
    admg = cg._coarsened_admg

    def lab(fs):
        return ",".join(sorted(fs))
    verts = [lab(v) for v in admg.get("vertices", [])]
    di = [(lab(u), lab(v)) for (u, v) in admg.get("di", [])]
    bi = []
    for e in admg.get("bi", []):
        e = list(e)
        if len(e) == 2:
            bi.append((lab(e[0]), lab(e[1])))
    return verts, di, bi


def emit_dataset(ds, clusters, pretty, fobj):
    edges, nodes, confounders, manip = benchmark.register_dataset(ds)
    dg, obs, config = benchmark.load_graph(ds, seed=0)
    target = "Y"
    # ---- fine DAG bidirected edges from latent confounders (>=2 obs) ----
    fine_bi = []
    for lat, vs in confounders:
        vs = [v for v in vs if v in nodes]
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                fine_bi.append((vs[i], vs[j]))
    print(f"\n[{ds}] manip={manip} nonmanip="
          f"{[n for n in nodes if n not in manip and n != target]} "
          f"edges={len(edges)} confounders={confounders}")
    # ---- C-DAG under the coarse partition ----
    partition = [frozenset(c) for c in clusters] + [frozenset({"Y"})]
    cg = CoarsenedGraph(dg, partition, ds, obs, max_intervention_size=len(manip),
                        num_mc_samples=50, assumed_graph_name=ds)
    cverts, cdi, cbi = cdag_struct(cg)
    print(f"   C-DAG verts={cverts}\n   C-DAG di={cdi}\n   C-DAG bi={cbi}")

    # layouts
    fpos = layered_pos(list(nodes), edges, sinks=[target], clusters=clusters)
    cpos = layered_pos(cverts, cdi, sinks=["Y"])
    cmanip = [",".join(sorted(frozenset(c))) for c in clusters]  # cluster labels in C-DAG

    dag = tikz_graph(list(nodes), edges, fine_bi, manip, target, clusters, fpos)
    cdag = tikz_graph(cverts, cdi, cbi, cmanip, "Y", [], cpos)

    fobj.write(r"\begin{figure}[t]\centering" + "\n")
    # shrink-only (never enlarge): a 2-node C-DAG stretched to half a page
    # would dwarf its fine-DAG panel; adjustbox caps width and leaves small
    # pictures at natural size.
    fobj.write(r"\adjustbox{max width=0.49\linewidth,valign=c}{" + dag + "}\hfill\n")
    fobj.write(r"\adjustbox{max width=0.49\linewidth,valign=c}{" + cdag + "}\n")
    fobj.write(
        r"\caption{\textbf{%s.} Fine DAG with coarse partition (left) and quotient "
        r"C-DAG (right), in the convention of Fig.~\ref{fig:clusterbench10}: "
        r"manipulable variables orange, target blue, latent confounding dashed "
        r"(intra-cluster) or dash-dotted (cross-cluster); dashed boxes are the "
        r"manipulable clusters.}"
        % pretty + "\n")
    fobj.write(r"\label{fig:dag-%s}" % ds + "\n")
    fobj.write(r"\end{figure}" + "\n\n")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("% Auto-generated by scripts/emit_dataset_tikz.py -- do not edit by hand.\n\n")
        for ds, (clusters, pretty) in COARSE.items():
            emit_dataset(ds, clusters, pretty, f)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
