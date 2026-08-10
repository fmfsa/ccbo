"""Emit TikZ DAG + C-DAG figures for every dataset used in the paper.

Sources are self-contained (no third_party needed):

* CBO family (ToyGraph, CompleteGraph, SimplifiedCoralGraph) and
  ClusterBench10: the structure registry in ``ccbo.coarsening`` /
  ``ccbo.clusterbench10``; the C-DAG is the Lee-2019 latent projection +
  quotient, exactly what ``CoarsenedGraph`` builds.
* DCBO setups (stat / ind / nonstat share slice topology): the three-slice
  temporal adjacency, hardcoded below (matching
  ``dcbo.utils.dag_utils.graph_functions.make_graphical_model``).
* MCBO envs (ToyGraph-M, PSAGraph): the parent lists hardcoded in
  ``ccbo.qmcbo.runner``.

Outputs:
  paper/figures/dataset_dags.tex  -- appendix gallery, one figure per dataset
  paper/figures/toy_bow.tex       -- main-text panel: the ToyGraph bow

Run:  PYTHONPATH=. python scripts/emit_dataset_tikz.py
"""

import os
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

from ccbo import clusterbench10 as cb
from ccbo import minibench as mb
from ccbo.coarsening import (get_dag_edges_from_sem, project_out_hidden,
                             build_coarsened_admg)

OUT_GALLERY = "paper/figures/dataset_dags.tex"
OUT_TOYBOW = "paper/figures/toy_bow.tex"


def _safe(name):
    """TikZ-safe node id (letters/digits only)."""
    return "n" + "".join(ch for ch in name if ch.isalnum())


def _label(name):
    """Readable node label."""
    return name.replace("_", r"\_")


def layered_pos(nodes, edges, sinks, dx=None, dy=None, clusters=()):
    """Longest-path layering with barycenter crossing reduction.

    Sinks (e.g. Y) are pinned to the last layer; within-layer order is then
    refined by a few barycenter sweeps. ``clusters`` makes the layout
    partition-aware: each multi-member cluster gets dedicated top rows so the
    fitted cluster boxes stay tight and never engulf non-members.
    """
    succ, pred = defaultdict(list), defaultdict(list)
    indeg = {n: 0 for n in nodes}
    for u, v in edges:
        if u in indeg and v in indeg:
            succ[u].append(v)
            pred[v].append(u)
            indeg[v] += 1
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
    if sinks:
        mx = max(layer.values(), default=0)
        for s in sinks:
            if s in layer:
                layer[s] = mx
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

    widest = max((len(ns) for ns in orders.values()), default=1)
    if dx is None:
        dx = 2.4 if len(layers) <= 3 else 2.1
    if dy is None:
        dy = 1.35 if widest <= 4 else 1.1

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
        used = defaultdict(int)
        free = reserved
        for n in ns:
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


def tikz_graph(nodes, di, bi, manip, target, clusters, pos, targets=None):
    """Emit one tikzpicture for a (di, bi) graph with cluster boxes.

    Uses the shared style vocabulary defined in the paper preamble (obsnode /
    mannode / tgtnode / clbox / diredge / biintra / bicross) so the figures
    match Fig.~1 exactly.
    """
    man = set(manip)
    tgt = set(targets or [target])
    L = [r"\begin{tikzpicture}"]
    for n in nodes:
        x, y = pos[n]
        sty = "tgtnode" if n in tgt else ("mannode" if n in man else "obsnode")
        L.append(f"  \\node[{sty}] ({_safe(n)}) at ({x:.2f},{y:.2f}) {{{_label(n)}}};")
    for u, v in di:
        if u in pos and v in pos:
            L.append(f"  \\draw[diredge] ({_safe(u)}) -- ({_safe(v)});")
    seen_at = defaultdict(int)
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


# ---------------------------------------------------------------------------
# Structure sources
# ---------------------------------------------------------------------------

def registry_structs(name, clusters):
    """(fine + quotient structure) for a graph in the coarsening registry."""
    edges, nodes, hidden, manip = get_dag_edges_from_sem(name)
    hidden = set(hidden or [])
    obs_nodes = [n for n in nodes if n not in hidden]
    proj = project_out_hidden(name)
    fine_bi = []
    for e in proj.get("bi", []):
        e = sorted(e)
        if len(e) == 2:
            fine_bi.append((e[0], e[1]))
    fine_bi = sorted(set(fine_bi))   # byte-stable across PYTHONHASHSEED
    nonmanip = set(obs_nodes) - set(manip) - {"Y"}
    partition = [frozenset(c) for c in clusters] + [frozenset({"Y"})]
    admg = build_coarsened_admg(partition, proj, atomic_vertices=nonmanip)

    def lab(v):
        return ",".join(sorted(v)) if isinstance(v, frozenset) else str(v)
    # build_coarsened_admg returns sets; sort everything so the emitted TikZ
    # is byte-stable across runs (set iteration follows PYTHONHASHSEED).
    cverts = sorted(lab(v) for v in admg.get("vertices", []))
    cdi = sorted((lab(u), lab(v)) for (u, v) in admg.get("di", []))
    cbi = []
    for e in admg.get("bi", []):
        e = list(e)
        if len(e) == 2:
            cbi.append(tuple(sorted((lab(e[0]), lab(e[1])))))
    cbi = sorted(set(cbi))
    obs_edges = [(u, v) for (u, v) in edges
                 if u not in hidden and v not in hidden]
    return dict(nodes=obs_nodes, di=obs_edges, bi=fine_bi, manip=list(manip),
                cverts=cverts, cdi=cdi, cbi=cbi)


def quotient_literal(nodes, di, manip, clusters, targets):
    """Quotient a latent-free literal graph (temporal / MCBO cases)."""
    cl_of = {}
    for c in clusters:
        name = ",".join(sorted(c))
        for m in c:
            cl_of[m] = name
    for n in nodes:
        cl_of.setdefault(n, n)
    cverts, seen = [], set()
    for n in nodes:
        v = cl_of[n]
        if v not in seen:
            seen.add(v)
            cverts.append(v)
    cdi, eseen = [], set()
    for u, v in di:
        cu, cv = cl_of[u], cl_of[v]
        if cu != cv and (cu, cv) not in eseen:
            eseen.add((cu, cv))
            cdi.append((cu, cv))
    cmanip = sorted({cl_of[m] for m in manip})
    ctargets = sorted({cl_of[t] for t in targets})
    return cverts, cdi, cmanip, ctargets


def temporal_struct(topology, T=3):
    """DCBO slice topology over X, Z, Y (matches make_graphical_model)."""
    nodes, di = [], []
    for t in range(T):
        nodes += [f"X_{t}", f"Z_{t}", f"Y_{t}"]
        if topology == "dependent":
            di += [(f"X_{t}", f"Z_{t}"), (f"Z_{t}", f"Y_{t}")]
        else:
            di += [(f"X_{t}", f"Y_{t}"), (f"Z_{t}", f"Y_{t}")]
    for t in range(T - 1):
        for v in ("X", "Z", "Y"):
            di.append((f"{v}_{t}", f"{v}_{t+1}"))
    manip = [f"{v}_{t}" for t in range(T) for v in ("X", "Z")]
    clusters = [[f"X_{t}", f"Z_{t}"] for t in range(T)]
    targets = [f"Y_{t}" for t in range(T)]
    return nodes, di, manip, clusters, targets


MCBO_GRAPHS = {
    # env -> (node names in topological order, parent lists, cluster)
    "ToyGraph-M": (["X0", "X1", "Y"], [[], [0], [1]], [["X0", "X1"]]),
    "PSAGraph": (["age", "bmi", "A", "S", "ca", "Y"],
                 [[], [0], [0, 1], [0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4]],
                 [["A", "S"]]),
}


# ---------------------------------------------------------------------------
# Figure emission
# ---------------------------------------------------------------------------

def fig_block(fobj, key, pretty, fine_pic, cdag_pic, caption_extra=""):
    fobj.write(r"\begin{figure}[t]\centering" + "\n")
    fobj.write(r"\adjustbox{max width=0.58\linewidth,valign=c}{" + fine_pic
               + "}\hfill\n")
    fobj.write(r"\adjustbox{max width=0.38\linewidth,valign=c}{" + cdag_pic
               + "}\n")
    fobj.write(
        r"\caption{\textbf{%s.} Fine DAG with coarse partition (left) and "
        r"quotient C-DAG (right); drawing conventions as stated at the start "
        r"of this appendix.%s}" % (pretty, caption_extra) + "\n")
    fobj.write(r"\label{fig:dag-%s}" % key + "\n")
    fobj.write(r"\end{figure}" + "\n\n")


def emit_registry(fobj, name, clusters, pretty, key, caption_extra=""):
    st = registry_structs(name, clusters)
    fpos = layered_pos(st["nodes"], st["di"], sinks=["Y"], clusters=clusters)
    cpos = layered_pos(st["cverts"], st["cdi"], sinks=["Y"])
    cmanip = [",".join(sorted(frozenset(c))) for c in clusters] + [
        v for v in st["cverts"]
        if v != "Y" and all(v != ",".join(sorted(frozenset(c)))
                            for c in clusters)
        and v in {m for m in st["manip"]}]
    fine = tikz_graph(st["nodes"], st["di"], st["bi"], st["manip"], "Y",
                      clusters, fpos)
    cdag = tikz_graph(st["cverts"], st["cdi"], st["cbi"], cmanip, "Y", [],
                      cpos)
    fig_block(fobj, key, pretty, fine, cdag, caption_extra)
    print(f"[{pretty}] C-DAG di={st['cdi']} bi={st['cbi']}")
    return st


def emit_literal(fobj, key, pretty, nodes, di, manip, clusters, targets,
                 caption_extra=""):
    cverts, cdi, cmanip, ctargets = quotient_literal(nodes, di, manip,
                                                     clusters, targets)
    fpos = layered_pos(nodes, di, sinks=[targets[-1]], clusters=clusters)
    cpos = layered_pos(cverts, cdi, sinks=[ctargets[-1]])
    fine = tikz_graph(nodes, di, [], manip, None, clusters, fpos,
                      targets=targets)
    cdag = tikz_graph(cverts, cdi, [], cmanip, None, [], cpos,
                      targets=ctargets)
    fig_block(fobj, key, pretty, fine, cdag, caption_extra)


def emit_toy_bow():
    """Main-text panel: ToyGraph fine DAG vs its coarse C-DAG (the bow)."""
    clusters = [["X", "Z"]]
    st = registry_structs("ToyGraph", clusters)
    fpos = layered_pos(st["nodes"], st["di"], sinks=["Y"], clusters=clusters)
    cpos = {"X,Z": (0.0, 0.0), "Y": (2.2, 0.0)}
    fine = tikz_graph(st["nodes"], st["di"], st["bi"], st["manip"], "Y",
                      clusters, fpos)
    cdag = tikz_graph(st["cverts"], st["cdi"], st["cbi"], ["X,Z"], "Y", [],
                      cpos)
    with open(OUT_TOYBOW, "w") as f:
        f.write("% Auto-generated by scripts/emit_dataset_tikz.py"
                " -- do not edit by hand.\n")
        f.write(r"\adjustbox{max width=0.46\linewidth,valign=c}{" + fine
                + "}\hfill\n")
        f.write(r"\adjustbox{max width=0.50\linewidth,valign=c}{" + cdag
                + "}\n")
    print(f"wrote {OUT_TOYBOW}")


def main():
    os.makedirs(os.path.dirname(OUT_GALLERY), exist_ok=True)
    cb.register_variants()
    mb.register_variants()
    with open(OUT_GALLERY, "w") as f:
        f.write("% Auto-generated by scripts/emit_dataset_tikz.py"
                " -- do not edit by hand.\n\n")
        emit_registry(
            f, mb.PP_NAME, [list(c) for c in mb.PP_COARSE_CLUSTERS],
            "ParallelParent (MinimalBench, Exp.\\ A)", "parallelparent",
            caption_extra=" The intra-cluster $X_1\\leftrightarrow X_2$ "
                          "disappears in the quotient; deleting "
                          "$X_1\\to Y$ leaves $C_1\\to Y$ alive through "
                          "$X_2\\to Y$ (quotient-redundant).")
        emit_registry(
            f, mb.FD_NAME, [list(c) for c in mb.FD_COARSE_CLUSTERS],
            "FrontDoor (MinimalBench, Exp.\\ B)", "frontdoor",
            caption_extra=" The quotient is a bow ($C_1\\to Y$, "
                          "$C_1\\leftrightarrow Y$): the cluster arm runs "
                          "on the uninformative prior tier in every "
                          "condition, so the assumed intra-cluster "
                          "$X_1\\leftrightarrow M$ cannot change it.")
        emit_registry(
            f, mb.MC_NAME, [list(c) for c in mb.MC_COARSE_CLUSTERS],
            "MediatedChain (MinimalBench, Exp.\\ C)", "mediatedchain",
            caption_extra=" The policy box restricts $D(X_2)$ to "
                          "$[-2,2]$; the fine optimum $\\doo(X_1{=}2)$ "
                          "drives $X_2$ to $4$, outside the box.")
        emit_registry(
            f, "ToyGraph", [["X", "Z"]], "ToyGraph (CBO family)", "toy",
            caption_extra=" The quotient is a bow ($C_1\\to Y$, "
                          "$C_1\\leftrightarrow Y$): $\\doo(C_1)$ is not "
                          "identifiable from the C-DAG, so the arm runs on "
                          "the uninformative prior tier.")
        emit_registry(f, "CompleteGraph", [["B"], ["D", "E"]],
                      "CompleteGraph (CBO family)", "complete")
        emit_registry(f, "SimplifiedCoralGraph",
                      [["C", "N", "O"], ["D", "T"]],
                      "SimplifiedCoralGraph (CBO family)", "coral")
        n, di, m, cl, tg = temporal_struct("dependent")
        emit_literal(f, "dcbo-stat", "DCBO stat / nonstat (QDCBO family)",
                     n, di, m, cl, tg,
                     caption_extra=" Slices $t=0,1,2$; the nonstat setup "
                                   "shares this topology with a change "
                                   "point in the SEM.")
        n, di, m, cl, tg = temporal_struct("independent")
        emit_literal(f, "dcbo-ind", "DCBO ind (QDCBO family)",
                     n, di, m, cl, tg)
        for env, (names, parents, cls) in MCBO_GRAPHS.items():
            di = [(names[p], names[i]) for i, ps in enumerate(parents)
                  for p in ps]
            manip = [nm for nm in names if nm != "Y"]
            emit_literal(f, f"mcbo-{env.lower()}",
                         f"{env} (QMCBO family)",
                         names, di, manip, cls, ["Y"])
    print(f"wrote {OUT_GALLERY}")
    emit_toy_bow()


if __name__ == "__main__":
    main()
