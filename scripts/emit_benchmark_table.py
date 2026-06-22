"""Emit paper/tables/benchmark_gap.tex from results/qcbo_benchmark_results.json.

Rows: Dataset & Method & Final Y & GAP & PA-GAP (mean +/- s.e.), scored by the
CausalBO benchmark's own GAP / PA-GAP at 100 trials. QCBO-finest is labelled
"CBO (=QCBO-finest)" per Prop. 1.

Run (after the suite finishes):
    PYTHONPATH=. python scripts/emit_benchmark_table.py
"""

import json
import os

RESULTS = 'results/qcbo_benchmark_results.json'
OUT = 'paper/tables/benchmark_gap.tex'

# Display order + pretty dataset names.
ORDER = [('toyGraph', 'ToyGraph'), ('synthetic_2', 'Synthetic-2'),
         ('synthetic', 'Synthetic'), ('healthcare', 'Healthcare'),
         ('epidemiology', 'Epidemiology'), ('ecology', 'Ecology')]
METHOD_ROWS = [('QCBO-finest', 'CBO ($=$\\QCBO{}-finest)'),
               ('QCBO-coarse', '\\QCBO{}-coarse')]


def cell(pair):
    return f"${pair[0]:.3f} \\pm {pair[1]:.3f}$"


def main():
    data = json.load(open(RESULTS))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    lines = []
    present = [(k, name) for k, name in ORDER if k in data]
    for di, (key, name) in enumerate(present):
        methods = data[key]['methods']
        for ri, (m, label) in enumerate(METHOD_ROWS):
            if m not in methods:
                continue
            s = methods[m]
            ds_cell = f"\\multirow{{2}}{{*}}{{{name}}}" if ri == 0 else ""
            lines.append(
                f"{ds_cell} & {label} & {cell(s['final'])} & "
                f"{cell(s['gap100'])} & {cell(s['pagap100'])} \\\\")
        if di < len(present) - 1:
            lines.append("\\midrule")
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n\\bottomrule\n')
    print(f"Wrote {OUT} ({len(present)} datasets)")
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
