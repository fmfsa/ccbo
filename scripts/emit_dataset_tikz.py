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


def layered_pos(nodes, edges, sinks, dx=2.1, dy=1.35):
    """Longest-path layering; sinks (e.g. Y) pinned to the last layer."""
    succ = defaultdict(list)
    indeg = {n: 0 for n in nodes}
    for u, v in edges:
        if u in indeg and v in indeg:
            succ[u].append(v)
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
    maxL = max(layer.values()) if layer else 0
    for s in sinks:
        if s in layer:
            layer[s] = maxL if maxL > max((layer[x] for x in nodes if x not in sinks),
                                          default=0) else max(layer.values(), default=0)
    # re-pin Y strictly to the rightmost layer
    if sinks:
        mx = max(layer.values())
        for s in sinks:
            layer[s] = mx
    bylayer = defaultdict(list)
    for n in nodes:
        bylayer[layer[n]].append(n)
    pos = {}
    for L, ns in sorted(bylayer.items()):
        ns = sorted(ns)
        for i, n in enumerate(ns):
            pos[n] = (L * dx, (i - (len(ns) - 1) / 2.0) * dy)
    return pos


def tikz_graph(nodes, di, bi, manip, target, clusters, pos):
    """Emit one tikzpicture for a (di, bi) graph with cluster boxes."""
    man = set(manip)
    L = [r"\begin{tikzpicture}[>=Stealth, scale=0.95, every node/.style={transform shape},",
         r"  obs/.style={circle,draw,minimum size=6mm,inner sep=0.5pt,font=\scriptsize},",
         r"  man/.style={obs,fill=orange!25}, tgt/.style={obs,fill=yellow!55},",
         r"  cl/.style={draw,dashed,rounded corners,inner sep=2.6mm},",
         r"  bi/.style={<->,dashed,red!70!black}]"]
    for n in nodes:
        x, y = pos[n]
        sty = "tgt" if n == target else ("man" if n in man else "obs")
        L.append(f"  \\node[{sty}] ({_safe(n)}) at ({x:.2f},{y:.2f}) {{{_label(n)}}};")
    for u, v in di:
        if u in pos and v in pos:
            L.append(f"  \\draw[->] ({_safe(u)}) -- ({_safe(v)});")
    for u, v in bi:
        if u in pos and v in pos:
            L.append(f"  \\draw[bi,bend left=22] ({_safe(u)}) to ({_safe(v)});")
    # cluster boxes (only multi-member manipulable clusters)
    for k, cl in enumerate(clusters):
        members = [c for c in cl if c in pos]
        if len(members) >= 2:
            fit = "".join(f"({_safe(c)})" for c in members)
            L.append(f"  \\begin{{scope}}[on background layer]")
            L.append(f"    \\node[cl,fit={fit},label=above:{{\\scriptsize$C_{{{k+1}}}$}}] {{}};")
            L.append(f"  \\end{{scope}}")
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
    fpos = layered_pos(list(nodes), edges, sinks=[target])
    cpos = layered_pos(cverts, cdi, sinks=["Y"])
    cmanip = [",".join(sorted(frozenset(c))) for c in clusters]  # cluster labels in C-DAG

    dag = tikz_graph(list(nodes), edges, fine_bi, manip, target, clusters, fpos)
    cdag = tikz_graph(cverts, cdi, cbi, cmanip, "Y", [], cpos)

    fobj.write(r"\begin{figure}[t]\centering" + "\n")
    fobj.write(r"\resizebox{0.49\linewidth}{!}{" + dag + "}\hfill\n")
    fobj.write(r"\resizebox{0.49\linewidth}{!}{" + cdag + "}\n")
    fobj.write(
        r"\caption{\textbf{%s.} Fine DAG with coarse partition (left) and quotient "
        r"C-DAG (right). Manipulable variables in orange, target in yellow, dashed "
        r"red edges are latent confounding; dashed boxes are manipulable clusters.}"
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
