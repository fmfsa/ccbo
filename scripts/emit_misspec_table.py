"""Emit the ClusterBench10 misspecification LaTeX tables from the driver JSON.

Reads results/clusterbench10_misspec.json (produced by run_misspec_parallel.py /
run_misspec_fullfield.py) and writes three self-contained tabulars into
paper/tables/:

  clusterbench10_headline.tex  -- T1: the head-to-head. Per method, sample
                                  efficiency under the CORRECT DAG (absolute
                                  GAP@50, GAP@100) and robustness under the intra
                                  bow P1 (dGAP@100, byte-identity).
  clusterbench10_taxonomy.tex  -- T2: per-perturbation dGAP@100 for every method
                                  (the conditions characterization).
  clusterbench10_fulltable.tex -- appendix: every method x perturbation, all GAP
                                  budgets + paired deltas.

GAP is in [0,1], higher = faster convergence (more sample-efficient). dGAP =
GAP(misspecified) - GAP(correct); 0 = unaffected, negative = hurt.

Run:  PYTHONPATH=. python scripts/emit_misspec_table.py
"""

import os
import json

IN = "results/clusterbench10_misspec.json"
OUTDIR = "paper/tables"

# Field: BO (non-causal), CBO (benchmark non-gating), QCBO-finest (gating
# do-calculus CBO, Prop. 1), QCBO-coarse. CEO/CoCaBO are out of scope.
METHOD_ORDER = ["BO", "CBO", "QCBO-finest", "QCBO-coarse"]
PERT_ORDER = ["P1", "P2", "P3", "Pic", "P5", "P6", "S1", "S2", "S3"]
PERT_LABEL = {
    "P1": r"add $X_1\!\to\!X_2$ (bow)", "P2": r"del $X_3\!\to\!X_2$",
    "P3": r"rev $X_3\!\to\!X_2$", "Pic": r"add $X_1\!\to\!X_5$ (bow)",
    "P5": r"del $X_3\!\to\!M_1$", "P6": r"del $X_1\!\to\!M_2$",
    "S1": "sweep $k{=}1$", "S2": "sweep $k{=}2$", "S3": "sweep $k{=}3$",
}
LOCUS_TEX = {"intra": "intra", "inter": "inter", "inter-redundant": "inter (red.)"}
MLABEL = {"QCBO-finest": r"\QCBO-finest", "QCBO-coarse": r"\QCBO-coarse",
          "BO": r"\BO", "CBO": r"\CBO"}


def _ms(pair, fmt="%+.2f"):
    m, s = pair
    return (fmt % m) + r"{\scriptstyle\,\pm\,}" + ("%.2f" % s)


def _mp(pair, fmt="%.2f"):
    """mean$\\pm$sem, unsigned (for absolute GAP)."""
    m, s = pair
    return (fmt % m) + r"{\scriptstyle\,\pm\,}" + ("%.2f" % s)


def _present(data):
    return [m for m in METHOD_ORDER if m in data["methods"]]


def emit_headline(data):
    methods = _present(data)
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"\textbf{Method} & \textbf{GAP@50} & \textbf{GAP@100} & "
             r"$\Delta$\textbf{GAP@100} & \textbf{byte-id.} \\",
             r"\multicolumn{1}{c}{} & \multicolumn{2}{c}{\footnotesize correct DAG"
             r" ($\uparrow$ efficient)} & \multicolumn{1}{c}{\footnotesize bow P1}"
             r" & \\",
             r"\midrule"]
    for m in methods:
        c = data["methods"][m]
        p0, p1 = c.get("P0"), c.get("P1")
        if p0 is None or p1 is None:
            continue
        bid = r"\checkmark" if p1["byte_identical_to_P0"] else "--"
        g50, g100 = _mp(p0["GAP50"]), _mp(p0["GAP100"])
        dg = _ms(p1["dGAP"], fmt="%+.3f")
        if m == "QCBO-coarse":
            row = (r"\textbf{" + MLABEL[m] + r"} & $" + g50 + r"$ & $" + g100
                   + r"$ & $\mathbf{" + dg + r"}$ & " + bid)
        else:
            row = f"{MLABEL.get(m, m)} & ${g50}$ & ${g100}$ & ${dg}$ & {bid}"
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_taxonomy(data):
    methods = _present(data)
    col = "l l " + " ".join(["c"] * len(methods))
    head = (r"\textbf{Perturbation} & \textbf{Locus} & "
            + " & ".join(r"\textbf{%s}" % m.replace("QCBO-", r"Q-") for m in methods)
            + r" \\")
    lines = [r"\begin{tabular}{@{}" + col + r"@{}}", r"\toprule", head,
             r"\multicolumn{2}{c}{} & \multicolumn{%d}{c}{\footnotesize "
             r"$\Delta$GAP@100 (0 = unaffected)} \\" % len(methods), r"\midrule"]
    for pid in PERT_ORDER:
        ref = next((data["methods"][m][pid] for m in methods
                    if pid in data["methods"][m]), None)
        if ref is None:
            continue
        locus = LOCUS_TEX.get(ref.get("locus"), ref.get("locus", ""))
        cells = []
        for m in methods:
            c = data["methods"][m].get(pid)
            cells.append(("$" + _ms(c["dGAP"], fmt="%+.3f") + "$") if c else "--")
        lines.append(f"{PERT_LABEL.get(pid, pid)} & {locus} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_fulltable(data):
    methods = _present(data)
    perts = ["P0"] + PERT_ORDER
    lines = [r"\begin{tabular}{@{}ll ccc cc c@{}}", r"\toprule",
             r"\textbf{Method} & \textbf{Pert.} & \textbf{GAP@20} & \textbf{GAP@50} "
             r"& \textbf{GAP@100} & $\Delta$\textbf{GAP@100} & $\Delta$\textbf{Final}$Y$"
             r" & \textbf{b-id} \\", r"\midrule"]
    for m in methods:
        first = True
        for pid in perts:
            c = data["methods"][m].get(pid)
            if c is None:
                continue
            name = MLABEL.get(m, m) if first else ""
            first = False
            bid = r"\checkmark" if c["byte_identical_to_P0"] else "--"
            dg = "--" if pid == "P0" else "$" + _ms(c["dGAP"], fmt="%+.3f") + "$"
            df = "--" if pid == "P0" else "$" + _ms(c["dFinalY"]) + "$"
            lines.append(
                f"{name} & {pid} & ${_mp(c['GAP20'])}$ & ${_mp(c['GAP50'])}$ & "
                f"${_mp(c['GAP100'])}$ & {dg} & {df} & {bid} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    with open(IN) as f:
        data = json.load(f)
    os.makedirs(OUTDIR, exist_ok=True)
    for name, fn in [("clusterbench10_headline", emit_headline),
                     ("clusterbench10_taxonomy", emit_taxonomy),
                     ("clusterbench10_fulltable", emit_fulltable)]:
        path = os.path.join(OUTDIR, name + ".tex")
        with open(path, "w") as f:
            f.write(fn(data))
        print(f"wrote {path}")
    print(f"methods: {_present(data)}  cap={data.get('cap')} "
          f"seeds={data.get('seeds')} T={data.get('trials')}")


if __name__ == "__main__":
    main()
