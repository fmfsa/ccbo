"""Emit the two review-response LaTeX tables (kept separate from the 4-method
misspec tables so those don't change):

  clusterbench10_wrongpi.tex   -- cost of a MISSPECIFIED PARTITION on ClusterBench10
                                  (correct objective). QCBO holds a fixed, acyclic,
                                  but WRONG C-DAG (base structure quotiented under a
                                  wrong partition). Shows the price of getting the
                                  clustering wrong vs the correct-partition runs, and
                                  that it still beats non-causal BO (graceful
                                  degradation, not collapse).
  clusterbench10L_price.tex    -- realized PRICE OF COARSENING on ClusterBench10L
                                  (the lossy companion: the optimum splits a cluster).
                                  QCBO-coarse plateaus above y* by the realized price;
                                  CBO = QCBO-finest reaches y*. Reported next to the
                                  oracle price (Prop. 5).

Run:  PYTHONPATH=. python scripts/emit_review_tables.py
"""

import os
import json

MISSPEC = "results/clusterbench10_misspec.json"
LOSSY = "results/clusterbench10L.json"
OUTDIR = "paper/tables"

MLABEL = {"BO": r"\BO", "CBO": r"\CBO", "QCBO-finest": r"\QCBO-finest",
          "QCBO-coarse": r"\QCBO-coarse", "QCBO-wrongpi": r"\QCBO-wrong$\pi$"}


def _mp(pair, fmt="%.3f", fmt_s="%.3f"):
    m, s = pair
    return (fmt % m) + r"{\scriptstyle\,\pm\,}" + (fmt_s % s)


def emit_wrongpi(data):
    """Cost of a wrong partition: 4-way at P0 (correct objective).

    Rows: BO, CBO (=QCBO-finest, Prop.1), QCBO-coarse (correct pi), QCBO-wrong-pi.
    Cols: Final Y (lower better), GAP@100, PA-GAP@100 (higher better).
    """
    rows = ["BO", "CBO", "QCBO-coarse", "QCBO-wrongpi"]
    note = {"QCBO-coarse": r"\;{\footnotesize(correct $\pi$)}",
            "QCBO-wrongpi": r"\;{\footnotesize(wrong $\pi$)}"}
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"\textbf{Method} & \textbf{Final $Y$} & \textbf{GAP@100} & "
             r"\textbf{PA-GAP@100} \\",
             r"\multicolumn{1}{c}{} & {\footnotesize $\downarrow$, $y^\star{=}0$} & "
             r"\multicolumn{2}{c}{\footnotesize $\uparrow$ closer to optimum} \\",
             r"\midrule"]
    for m in rows:
        c = data["methods"].get(m, {}).get("P0")
        if c is None:
            continue
        label = MLABEL.get(m, m) + note.get(m, "")
        if m == "QCBO-coarse":
            label = r"\textbf{" + MLABEL[m] + r"}" + note.get(m, "")
        lines.append(
            f"{label} & ${_mp(c['finalY'])}$ & ${_mp(c['GAP100'])}$ & "
            f"${_mp(c['PAGAP100'])}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_lossy(data):
    """Realized price of coarsening on ClusterBench10L.

    Rows: BO, CBO (=QCBO-finest), QCBO-coarse. Cols: Final Y, gap-to-y*
    (= Final Y - y*), GAP@100. Caption carries the oracle price.
    """
    ystar = data["y_star"]
    rows = ["BO", "CBO", "QCBO-coarse"]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"\textbf{Method} & \textbf{Final $Y$} & "
             r"\textbf{Final $Y-y^\star$} & \textbf{GAP@100} \\",
             r"\multicolumn{1}{c}{} & {\footnotesize $\downarrow$} & "
             r"{\footnotesize price$\,\downarrow$} & {\footnotesize $\uparrow$} \\",
             r"\midrule"]
    for m in rows:
        c = data["methods"].get(m)
        if c is None:
            continue
        gap_to_star = c["finalY"][0] - ystar
        label = (r"\textbf{" + MLABEL[m] + r"}" if m == "QCBO-coarse"
                 else MLABEL.get(m, m))
        lines.append(
            f"{label} & ${_mp(c['finalY'])}$ & ${gap_to_star:+.3f}$ & "
            f"${_mp(c['GAP100'])}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    with open(MISSPEC) as f:
        data = json.load(f)
    p = os.path.join(OUTDIR, "clusterbench10_wrongpi.tex")
    with open(p, "w") as f:
        f.write(emit_wrongpi(data))
    print(f"wrote {p}")

    if os.path.exists(LOSSY):
        with open(LOSSY) as f:
            ld = json.load(f)
        p = os.path.join(OUTDIR, "clusterbench10L_price.tex")
        with open(p, "w") as f:
            f.write(emit_lossy(ld))
        realized = ld.get("price_of_coarsening_realized")
        oracle = ld.get("price_of_coarsening_oracle")
        print(f"wrote {p}  (realized price={realized}, oracle={oracle})")
    else:
        print(f"SKIP lossy table: {LOSSY} not found (run scripts/run_clusterbench10L.py)")


if __name__ == "__main__":
    main()
