"""Emit the cross-dataset QCBO-refine comparison table.

paper/tables/refine_suite.tex -- BO / CBO / QCBO-coarse / QCBO-refine on the
two ClusterBench variants and the survey datasets refine was run on
(synthetic_2, ecology, healthcare), scored by final Y, GAP@100, and cumulative
incumbent regret R_T. Dataset row-groups; best per (group, metric) in bold,
respecting task direction (final Y = closest to y*; GAP higher; R_T lower).

Run:  PYTHONPATH=. python scripts/emit_refine_table.py
"""

import os
import json

SRC = "results/clusterbench10_refine.json"
OUT = "paper/tables/refine_suite.tex"

# (json dataset key, display label, lossy?) in presentation order.
DATASETS = [
    ("ClusterBench10", r"\textsc{ClusterBench10}", "lossless, $y^\\star{=}0$"),
    ("ClusterBench10L", r"\textsc{ClusterBench10L}", "lossy, $y^\\star{=}0.10$"),
]
ROWS = ["BO", "CBO", "QCBO-coarse", "QCBO-refine"]
MLABEL = {"BO": r"\BO", "CBO": r"\CBO", "QCBO-coarse": r"\QCBO{}",
          "QCBO-refine": r"\textbf{\HQCBO{}}"}


def _mp(pair, fmt="%.3f"):
    return (fmt % pair[0]) + r"{\scriptstyle\,\pm\,}" + (fmt % pair[1])


def _best(entry, metric, ystar, task):
    """Return the set of method keys tied for best on this metric."""
    vals = {}
    for m in ROWS:
        c = entry["methods"].get(m)
        if c is None:
            continue
        v = c[metric][0]
        if metric == "finalY":
            vals[m] = abs(v - ystar)          # closest to y*
        elif metric == "GAP100":
            vals[m] = -v                       # higher better
        else:                                  # cumRegret
            vals[m] = v                        # lower better
    if not vals:
        return set()
    lo = min(vals.values())
    return {m for m, x in vals.items() if abs(x - lo) < 1e-9}


def main():
    with open(SRC) as f:
        d = json.load(f)["datasets"]

    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"\textbf{Dataset} & \textbf{Method} & \textbf{Final $Y$} & "
             r"\textbf{GAP@100} & $R_T$ \\",
             r" & & {\footnotesize $\to y^\star$} & {\footnotesize $\uparrow$} "
             r"& {\footnotesize $\downarrow$} \\",
             r"\midrule"]

    for i, (key, label, note) in enumerate(DATASETS):
        if key not in d:
            continue
        e = d[key]
        ystar, task = e["y_star"], e["task"]
        best = {m: _best(e, m, ystar, task)
                for m in ("finalY", "GAP100", "cumRegret")}
        lines.append(
            rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{label}}} "
            rf"{{\footnotesize({note})}}}} \\")
        for m in ROWS:
            c = e["methods"].get(m)
            if c is None:
                continue

            def cell(metric, fmt="%.3f"):
                s = _mp(c[metric], fmt)
                return r"$\mathbf{" + s + r"}$" if m in best[metric] \
                    else "$" + s + "$"

            lines.append(
                rf"\quad & {MLABEL[m]} & {cell('finalY')} & "
                rf"{cell('GAP100')} & {cell('cumRegret', '%.1f')} \\")
        if i < len(DATASETS) - 1:
            lines.append(r"\addlinespace[2pt]")

    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
