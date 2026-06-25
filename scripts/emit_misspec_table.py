"""Emit the ClusterBench10 misspecification LaTeX tables from the driver JSON.

Reads results/clusterbench10_misspec.json (produced by run_misspec_fullfield.py)
and writes two self-contained tabulars into paper/tables/:

  clusterbench10_headline.tex  -- T1: paired degradation under the intra bow (P1),
                                  one row per method (Final Y under the correct
                                  DAG, dFinalY, dGAP, byte-identity).
  clusterbench10_taxonomy.tex  -- T2: per-perturbation dFinalY for every method
                                  (the conditions characterization).

Run:  PYTHONPATH=. python scripts/emit_misspec_table.py
"""

import os
import json

IN = "results/clusterbench10_misspec.json"
OUTDIR = "paper/tables"

# Display order: field first, ours last; only those present are emitted.
METHOD_ORDER = ["BO", "CBO", "CEO", "CoCaBO", "QCBO-finest", "QCBO-coarse"]
# Perturbation rows for the taxonomy (intra first, then inter, then sweep).
PERT_ORDER = ["P1", "P2", "P3", "Pic", "P5", "P6", "S1", "S2", "S3"]
PERT_LABEL = {
    "P1": r"add $X_1\!\to\!X_2$ (bow)", "P2": r"del $X_3\!\to\!X_2$",
    "P3": r"rev $X_3\!\to\!X_2$", "Pic": r"add $X_1\!\to\!X_5$ (bow)",
    "P5": r"del $X_3\!\to\!M_1$", "P6": r"del $X_1\!\to\!M_2$",
    "S1": "sweep $k{=}1$", "S2": "sweep $k{=}2$", "S3": "sweep $k{=}3$",
}
LOCUS_TEX = {"intra": "intra", "inter": "inter", "inter-redundant": "inter (red.)"}


def _ms(pair, fmt="%+.2f"):
    """mean$\\pm$sem from a [mean, sem] pair."""
    m, s = pair
    return (fmt % m) + r"{\scriptstyle\,\pm\,}" + ("%.2f" % s)


def _present_methods(data):
    return [m for m in METHOD_ORDER if m in data["methods"]]


def emit_headline(data):
    methods = _present_methods(data)
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"\textbf{Method} & \textbf{Final $Y$} & $\Delta$\textbf{Final $Y$} & "
             r"$\Delta$\textbf{GAP} & \textbf{byte-id.} \\",
             r"\midrule"]
    for m in methods:
        cells = data["methods"][m]
        p0, p1 = cells.get("P0"), cells.get("P1")
        if p0 is None or p1 is None:
            continue
        bid = r"\checkmark" if p1["byte_identical_to_P0"] else "--"
        row = (f"{m} & ${p0['finalY'][0]:.2f}$ & ${_ms(p1['dFinalY'])}$ & "
               f"${_ms(p1['dGAP'], fmt='%+.3f')}$ & {bid}")
        if m == "QCBO-coarse":
            row = r"\textbf{" + m + r"} & $" + f"{p0['finalY'][0]:.2f}" + r"$ & $\mathbf{" \
                  + _ms(p1["dFinalY"]) + r"}$ & $\mathbf{" + _ms(p1["dGAP"], fmt="%+.3f") \
                  + r"}$ & " + bid
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_taxonomy(data):
    methods = _present_methods(data)
    col = "l l " + " ".join(["c"] * len(methods))
    head = (r"\textbf{Perturbation} & \textbf{Locus} & "
            + " & ".join(r"\textbf{%s}" % m.replace("QCBO-", r"Q-") for m in methods)
            + r" \\")
    lines = [r"\begin{tabular}{@{}" + col + r"@{}}", r"\toprule", head, r"\midrule"]
    for pid in PERT_ORDER:
        # use the first method that has this perturbation to read locus
        ref = next((data["methods"][m][pid] for m in methods
                    if pid in data["methods"][m]), None)
        if ref is None:
            continue
        locus = LOCUS_TEX.get(ref.get("locus"), ref.get("locus", ""))
        cells = []
        for m in methods:
            c = data["methods"][m].get(pid)
            cells.append(("$" + _ms(c["dFinalY"]) + "$") if c else "--")
        lines.append(f"{PERT_LABEL.get(pid, pid)} & {locus} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    with open(IN) as f:
        data = json.load(f)
    os.makedirs(OUTDIR, exist_ok=True)
    for name, fn in [("clusterbench10_headline", emit_headline),
                     ("clusterbench10_taxonomy", emit_taxonomy)]:
        path = os.path.join(OUTDIR, name + ".tex")
        with open(path, "w") as f:
            f.write(fn(data))
        print(f"wrote {path}")
    print(f"methods: {_present_methods(data)}  cap={data.get('cap')} "
          f"seeds={data.get('seeds')} T={data.get('trials')}")


if __name__ == "__main__":
    main()
