"""Emit the ClusterBench10 misspecification LaTeX tables from the driver JSON.

Reads results/clusterbench10_misspec.json and writes three tabulars into
paper/tables/:

  clusterbench10_headline.tex  -- T1 (sample efficiency, CORRECT DAG): per method,
                                  absolute Final Y and GAP@{20,50,100}. Non-causal
                                  BO is far worse; the causal methods exploit the
                                  prior.
  clusterbench10_taxonomy.tex  -- T2 (invariance under misspecification): per
                                  perturbation, the paired dGAP@100 and dPA-GAP@100
                                  for QCBO-finest vs QCBO-coarse. Coarse is exactly
                                  0 on quotient-invisible perturbations (byte-id);
                                  finest is not -- a misspecified edge perturbs its
                                  GAP/PA-GAP even when the final value converges.
  clusterbench10_fulltable.tex -- appendix: every method x perturbation, all metrics.
  clusterbench10_regret.tex    -- appendix: simple regret @T and cumulative
                                  (incumbent) regret per method x perturbation.

GAP/PA-GAP in [0,1], higher = faster convergence. Final Y: lower = better (y*=0).
Regret: lower = better (0 = oracle optimum found).

Run:  PYTHONPATH=. python scripts/emit_misspec_table.py
"""

import os
import json

IN = "results/clusterbench10_misspec.json"
OUTDIR = "paper/tables"

METHOD_ORDER = ["BO", "CBO", "QCBO-finest", "QCBO-coarse"]
PERT_ORDER = ["P1", "P2", "P3", "S1", "S2", "S3", "P6", "P5", "Pic"]
PERT_LABEL = {
    "P1": r"add $X_1\!\to\!X_2$ (bow)", "P2": r"del $X_3\!\to\!X_2$",
    "P3": r"rev $X_3\!\to\!X_2$", "Pic": r"add $X_1\!\to\!X_5$ (bow)",
    "P5": r"del $X_3\!\to\!M_1$", "P6": r"del $X_1\!\to\!M_2$",
    "S1": "sweep $k{=}1$", "S2": "sweep $k{=}2$", "S3": "sweep $k{=}3$",
}
LOCUS_TEX = {"intra": "intra", "inter": "inter", "inter-redundant": "inter (red.)"}
MLABEL = {"QCBO-finest": r"\QCBO-finest", "QCBO-coarse": r"\QCBO-coarse",
          "BO": r"\BO", "CBO": r"\CBO"}


def _ms(pair, fmt="%+.3f"):
    m, s = pair
    return (fmt % m) + r"{\scriptstyle\,\pm\,}" + ("%.3f" % s)


def _mp(pair, fmt="%.2f", fmt_s="%.2f"):
    m, s = pair
    return (fmt % m) + r"{\scriptstyle\,\pm\,}" + (fmt_s % s)


def _present(data):
    return [m for m in METHOD_ORDER if m in data["methods"]]


def emit_headline(data):
    """T1: sample efficiency on the correct DAG (P0)."""
    methods = _present(data)
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"\textbf{Method} & \textbf{Final $Y$} & \textbf{GAP@20} & "
             r"\textbf{GAP@50} & \textbf{GAP@100} & $r_T$ & $R_T$ \\",
             r"\multicolumn{1}{c}{} & {\footnotesize $\downarrow$, $y^\star{=}0$} & "
             r"\multicolumn{3}{c}{\footnotesize $\uparrow$ more sample-efficient} & "
             r"\multicolumn{2}{c}{\footnotesize $\downarrow$ regret} \\",
             r"\midrule"]
    for m in methods:
        p0 = data["methods"][m].get("P0")
        if p0 is None:
            continue
        fy = _mp(p0["finalY"], fmt="%.3f")
        g20, g50, g100 = _mp(p0["GAP20"]), _mp(p0["GAP50"]), _mp(p0["GAP100"])
        rT = "$" + _mp(p0["regretT"]) + "$" if "regretT" in p0 else "--"
        cr = ("$" + _mp(p0["cumRegret"], fmt="%.1f", fmt_s="%.1f") + "$"
              if "cumRegret" in p0 else "--")
        name = (r"\textbf{" + MLABEL[m] + r"}") if m == "QCBO-coarse" else MLABEL.get(m, m)
        lines.append(f"{name} & ${fy}$ & ${g20}$ & ${g50}$ & ${g100}$ & {rT} & {cr} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_taxonomy(data):
    """T2: paired dGAP@100 and dPA-GAP@100 for finest vs coarse per perturbation."""
    pair_methods = [m for m in ("QCBO-finest", "QCBO-coarse") if m in data["methods"]]
    lines = [r"\begin{tabular}{@{}ll cc cc@{}}", r"\toprule",
             r"\textbf{Perturbation} & \textbf{Locus} & "
             r"\multicolumn{2}{c}{\textbf{\QCBO-finest}} & "
             r"\multicolumn{2}{c}{\textbf{\QCBO-coarse}} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r" & & $\Delta$GAP & $\Delta$PA-GAP & $\Delta$GAP & $\Delta$PA-GAP \\",
             r"\midrule"]
    for pid in PERT_ORDER:
        ref = next((data["methods"][m][pid] for m in pair_methods
                    if pid in data["methods"][m]), None)
        if ref is None:
            continue
        locus = LOCUS_TEX.get(ref.get("locus"), ref.get("locus", ""))
        cells = []
        for m in pair_methods:
            c = data["methods"][m].get(pid)
            if c is None:
                cells += ["--", "--"]
            elif c.get("byte_identical_to_P0"):
                cells += [r"$\mathbf{0}$", r"$\mathbf{0}$"]   # exact byte-identity
            else:
                cells += ["$" + _ms(c["dGAP"]) + "$",
                          "$" + _ms(c.get("dPAGAP", [0.0, 0.0])) + "$"]
        lines.append(f"{PERT_LABEL.get(pid, pid)} & {locus} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_fulltable(data):
    methods = _present(data)
    perts = ["P0"] + PERT_ORDER
    lines = [r"\begin{tabular}{@{}ll ccc c cc c@{}}", r"\toprule",
             r"\textbf{Method} & \textbf{Pert.} & \textbf{GAP@20} & \textbf{GAP@50} "
             r"& \textbf{GAP@100} & \textbf{PA-GAP} & $\Delta$\textbf{GAP} & "
             r"$\Delta$\textbf{PA-GAP} & \textbf{b-id} \\", r"\midrule"]
    for m in methods:
        first = True
        for pid in perts:
            c = data["methods"][m].get(pid)
            if c is None:
                continue
            name = MLABEL.get(m, m) if first else ""
            first = False
            bid = r"\checkmark" if c["byte_identical_to_P0"] else "--"
            dg = "--" if pid == "P0" else "$" + _ms(c["dGAP"]) + "$"
            dp = "--" if pid == "P0" else "$" + _ms(c.get("dPAGAP", [0.0, 0.0])) + "$"
            lines.append(
                f"{name} & {pid} & ${_mp(c['GAP20'])}$ & ${_mp(c['GAP50'])}$ & "
                f"${_mp(c['GAP100'])}$ & ${_mp(c.get('PAGAP100', [0.0, 0.0]))}$ & "
                f"{dg} & {dp} & {bid} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def emit_regret(data):
    """Appendix: simple regret r_T and cumulative (incumbent) regret R_T,
    per perturbation (rows) x method (column pairs)."""
    methods = _present(data)
    perts = ["P0"] + PERT_ORDER
    header = " & ".join(r"\multicolumn{2}{c}{\textbf{" + MLABEL.get(m, m) + "}}"
                        for m in methods)
    cmid = "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i)
                   for i in range(len(methods)))
    sub = " & ".join([r"$r_T$ & $R_T$"] * len(methods))
    lines = [r"\begin{tabular}{@{}l" + "cc" * len(methods) + r"@{}}",
             r"\toprule",
             r"\textbf{Pert.} & " + header + r" \\",
             cmid,
             r" & " + sub + r" \\",
             r"\midrule"]
    for pid in perts:
        cells = []
        for m in methods:
            c = data["methods"][m].get(pid)
            if c is None or "regretT" not in c:
                cells += ["--", "--"]
            else:
                cells += ["$" + _mp(c["regretT"]) + "$",
                          "$" + _mp(c["cumRegret"], fmt="%.1f", fmt_s="%.1f") + "$"]
        lines.append(f"{pid} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    with open(IN) as f:
        data = json.load(f)
    os.makedirs(OUTDIR, exist_ok=True)
    for name, fn in [("clusterbench10_headline", emit_headline),
                     ("clusterbench10_taxonomy", emit_taxonomy),
                     ("clusterbench10_fulltable", emit_fulltable),
                     ("clusterbench10_regret", emit_regret)]:
        path = os.path.join(OUTDIR, name + ".tex")
        with open(path, "w") as f:
            f.write(fn(data))
        print(f"wrote {path}")
    print(f"methods: {_present(data)}  cap={data.get('cap')} "
          f"seeds={data.get('seeds')} T={data.get('trials')}")


if __name__ == "__main__":
    main()
