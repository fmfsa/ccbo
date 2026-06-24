"""Emit the QCBO-across-coarsenings vs field comparison table.

Reads:
  * results/qcbo_coarsening_sweep.json   (QCBO best/worst/avg over valid
    coarsenings; produced by scripts/run_qcbo_coarsening_sweep.py)
  * results/survey_table4.json           (the field: the survey's published
    Table 4, 8 methods, 20 seeds -- TRANSCRIBED, verify before publishing)

NOTE (fairness): QCBO here is our own 5-seed run through the benchmark harness,
while the field is the survey's separately-run 20-seed Table 4. This is the
"use Table 4 for now" comparison; a fully-fair same-environment re-run of every
method is deferred. The caption must state this.

Writes paper/tables/benchmark_coarsening_sweep.tex (one tabular per metric/budget)
and prints a plaintext summary.

Run:  PYTHONPATH=. python scripts/emit_coarsening_sweep_table.py
"""

import os, json
import numpy as np

SWEEP = 'results/qcbo_coarsening_sweep.json'
FIELD_JSON = 'results/survey_table4.json'
TABLES = os.path.join('paper', 'tables')
OUT = os.path.join(TABLES, 'benchmark_coarsening_sweep.tex')

# Display order of datasets.
DS_ORDER = ['toyGraph', 'synthetic', 'synthetic_2', 'chain',
            'ecology', 'protein', 'healthcare', 'epidemiology']
DS_LABEL = {'toyGraph': 'ToyGraph', 'synthetic': 'Synthetic',
            'synthetic_2': 'Synth-2', 'chain': 'Chain', 'ecology': 'Ecology',
            'protein': 'Protein', 'healthcare': 'Health', 'epidemiology': 'Epi'}


def _fmt(x):
    return '---' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def _bold_max(col_vals):
    """Index of the max finite value in a column (for bolding); None if empty."""
    finite = [(i, v) for i, v in enumerate(col_vals) if v is not None and not np.isnan(v)]
    return max(finite, key=lambda t: t[1])[0] if finite else None


def emit_metric(sweep, field, metric, budget):
    """One tabular: rows = field methods + QCBO best/avg/worst, cols = datasets."""
    key = f'{metric}{budget}'
    field_methods = field['methods']
    field_block = field.get(metric, {}).get(str(budget), {})
    # Datasets present in the field block for this metric/budget.
    datasets = [d for d in DS_ORDER if d in field_block]

    def _field_val(d, mi):
        vals = field_block.get(d)
        return vals[mi] if vals is not None and mi < len(vals) else None

    rows = []   # (label, [values per dataset])
    for mi, m in enumerate(field_methods):
        rows.append((m, [_field_val(d, mi) for d in datasets]))
    for agg_kind in ('best', 'avg', 'worst'):
        rows.append((f'\\textbf{{QCBO-{agg_kind}}}',
                     [sweep.get(d, {}).get('aggregate', {}).get(key, {}).get(agg_kind)
                      for d in datasets]))

    # Per-column bold index (over ALL rows incl. QCBO).
    bold = []
    for j in range(len(datasets)):
        bold.append(_bold_max([r[1][j] for r in rows]))

    metric_disp = 'GAP' if metric == 'gap' else 'PA-GAP'
    lines = [f"% {metric_disp}@{budget}: QCBO over valid coarsenings vs field "
             f"(5 seeds, same env)",
             "\\begin{tabular}{@{}l" + "c" * len(datasets) + "@{}}",
             "\\toprule",
             "\\textbf{Method} & " + " & ".join(DS_LABEL[d] for d in datasets) + " \\\\",
             "\\midrule"]
    for i, (label, vals) in enumerate(rows):
        if label.startswith('\\textbf{QCBO-best'):
            lines.append("\\midrule")
        cells = []
        for j, v in enumerate(vals):
            s = _fmt(v)
            if bold[j] == i and s != '---':
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return datasets, rows, '\n'.join(lines)


def main():
    sweep = json.load(open(SWEEP))
    field = json.load(open(FIELD_JSON))
    os.makedirs(TABLES, exist_ok=True)
    if field.get('_status') == 'UNVERIFIED_TRANSCRIPTION':
        print("WARNING: field numbers are an UNVERIFIED transcription of survey "
              "Table 4 -- verify before publishing.\n")

    blocks = []
    for metric in ('gap', 'pagap'):
        for budget in (100, 50, 20):
            if str(budget) not in field.get(metric, {}):
                continue   # budget not transcribed yet
            datasets, rows, tab = emit_metric(sweep, field, metric, budget)
            blocks.append(f"% ==== {metric.upper()}@{budget} ====\n{tab}\n")
            if budget == 100:
                disp = 'GAP' if metric == 'gap' else 'PA-GAP'
                print(f"\n=== {disp}@100 (5 seeds, same env) ===")
                hdr = "Method        " + "".join(f"{DS_LABEL[d]:>9s}" for d in datasets)
                print(hdr)
                for label, vals in rows:
                    lab = label.replace('\\textbf{', '').replace('}', '')
                    print(f"{lab:13s} " + "".join(
                        f"{_fmt(v):>9s}" for v in vals))

    with open(OUT, 'w') as f:
        f.write('\n'.join(blocks))
    print(f"\nSaved {OUT}")


if __name__ == '__main__':
    main()
