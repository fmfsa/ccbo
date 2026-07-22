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
DAMAGE = "results/clusterbench10D.json"
OUTDIR = "paper/tables"

MLABEL = {"BO": r"\BO", "CBO": r"\CBO", "QCBO-finest": r"\CBO{}",
          "QCBO-coarse": r"\QCBO{}", "QCBO-wrongpi": r"\QCBO{}-wrong$\Pi$",
          "QCBO-refine": r"\HQCBO{}"}


def _mp(pair, fmt="%.3f", fmt_s="%.3f"):
    m, s = pair
    return (fmt % m) + r"{\scriptstyle\,\pm\,}" + (fmt_s % s)


def emit_wrongpi(data):
    """Cost of a wrong partition: 4-way at P0 (correct objective).

    Rows: BO, CBO (=QCBO-finest, Prop.1), QCBO-coarse (correct pi), QCBO-wrong-pi.
    Cols: Final Y and r_T (lower better, primary), GAP@100 (secondary).
    """
    rows = ["BO", "CBO", "QCBO-coarse", "QCBO-wrongpi"]
    note = {"QCBO-coarse": r"\;{\footnotesize(correct $\Pi$)}",
            "QCBO-wrongpi": r"\;{\footnotesize(wrong $\Pi$)}"}
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"\textbf{Method} & \textbf{Final $Y$} & $r_T$ & "
             r"\textbf{GAP@100} \\",
             r"\multicolumn{1}{c}{} & \multicolumn{2}{c}{\footnotesize "
             r"$\downarrow$, $y^\star{=}0$} & "
             r"{\footnotesize $\uparrow$ secondary} \\",
             r"\midrule"]
    for m in rows:
        c = data["methods"].get(m, {}).get("P0")
        if c is None:
            continue
        label = MLABEL.get(m, m) + note.get(m, "")
        if m == "QCBO-coarse":
            label = r"\textbf{" + MLABEL[m] + r"}" + note.get(m, "")
        rt = ("$" + _mp(c["regretT"]) + "$") if "regretT" in c else "--"
        lines.append(
            f"{label} & ${_mp(c['finalY'])}$ & {rt} & "
            f"${_mp(c['GAP100'])}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_lossy(data):
    """Realized price of coarsening on ClusterBench10L.

    Rows: BO, CBO (=QCBO-finest), QCBO-coarse, QCBO-refine. Cols: Final Y,
    gap-to-y* (= Final Y - y*), cumulative incumbent regret R_T, GAP@100.
    R_T is the framing metric: fixed-coarse pays the price linearly in T
    (its R_T keeps growing), refine's is a bounded transient. Caption
    carries the oracle price.
    """
    ystar = data["y_star"]
    rows = ["BO", "CBO", "QCBO-coarse", "QCBO-refine"]
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"\textbf{Method} & \textbf{Final $Y$} & "
             r"\textbf{Final $Y-y^\star$} & $R_T$ & \textbf{GAP@100} \\",
             r"\multicolumn{1}{c}{} & {\footnotesize $\downarrow$} & "
             r"{\footnotesize price$\,\downarrow$} & {\footnotesize $\downarrow$} "
             r"& {\footnotesize $\uparrow$} \\",
             r"\midrule"]
    for m in rows:
        c = data["methods"].get(m)
        if c is None:
            continue
        gap_to_star = c["finalY"][0] - ystar
        label = (r"\textbf{" + MLABEL[m] + r"}"
                 if m in ("QCBO-coarse", "QCBO-refine") else MLABEL.get(m, m))
        rt = (r" & $" + _mp(c["cumRegret"], fmt="%.1f", fmt_s="%.1f") + r"$"
              if "cumRegret" in c else r" & --")
        lines.append(
            f"{label} & ${_mp(c['finalY'])}$ & ${gap_to_star:+.3f}$"
            f"{rt} & ${_mp(c['GAP100'])}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_damage(data):
    """Deleted-optimal-arm damage on ClusterBench10D.

    Rows: QCBO-finest (=CBO) under the correct DAG and the intra bow P1, then
    QCBO-coarse (identical across both). Cols: Final Y, cumulative
    incumbent regret R_T, GAP@100 -- R_T is the metric that registers the
    slowdown; GAP saturates and misses it.
    """
    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"\textbf{Method} & \textbf{Assumed DAG} & \textbf{Final $Y$} & "
             r"$R_T$ & \textbf{GAP@100} \\",
             r"\multicolumn{2}{c}{} & {\footnotesize $\downarrow$, $y^\star{=}0$} & "
             r"{\footnotesize $\downarrow$} & {\footnotesize $\uparrow$} \\",
             r"\midrule"]
    fin = data["methods"]["QCBO-finest"]
    for pid, tag in [("P0", "correct"), ("P1", r"intra bow $P_1$")]:
        c = fin[pid]
        lines.append(
            rf"\CBO$\,\equiv\,$\QCBO-finest & {tag} & ${_mp(c['finalY'])}$ & "
            rf"${_mp(c['cumRegret'], fmt='%.1f', fmt_s='%.1f')}$ & ${_mp(c['GAP100'])}$ \\")
    c = data["methods"]["QCBO-coarse"]["P0"]
    lines.append(
        rf"\textbf{{\QCBO-coarse}} & either {{\footnotesize(identical)}} & "
        rf"${_mp(c['finalY'])}$ & ${_mp(c['cumRegret'], fmt='%.1f', fmt_s='%.1f')}$ & "
        rf"${_mp(c['GAP100'])}$ \\")
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

    if os.path.exists(DAMAGE):
        with open(DAMAGE) as f:
            dd = json.load(f)
        p = os.path.join(OUTDIR, "clusterbench10D_damage.tex")
        with open(p, "w") as f:
            f.write(emit_damage(dd))
        print(f"wrote {p}")
    else:
        print(f"SKIP damage table: {DAMAGE} not found (run scripts/run_clusterbench10D.py)")


if __name__ == "__main__":
    main()
