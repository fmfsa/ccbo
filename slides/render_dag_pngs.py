"""Render fine-DAG + quotient-C-DAG PNGs for the slide deck.

Reuses the figure logic in scripts/generate_dataset_dags.py (structure read from
code via the coarsening pipeline), but writes PNGs (white bg) into slides/figures/
for the benchmark datasets QCBO is evaluated on.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.generate_dataset_dags import _extract, _layered_pos, _draw

OUT = 'slides/figures'
os.makedirs(OUT, exist_ok=True)

# (stem, dataset key, coarse partition clusters used for the quotient)
DATASETS = [
    ('ToyGraph',     'toyGraph',     [['X', 'Z']]),
    ('Synthetic',    'synthetic',    [['B'], ['D', 'E']]),
    ('Chain',        'chain',        [['Z', 'W']]),
    ('Healthcare',   'healthcare',   [['Aspirin', 'Statin']]),
    ('Epidemiology', 'epidemiology', [['L', 'B']]),
    ('Ecology',      'ecology',      [['C', 'N', 'O'], ['D', 'T']]),
    ('Protein',      'protein',      [['Akt', 'Mek'], ['PKA', 'PKC']]),
]


def render(stem, admg, manip, suffix):
    pos = _layered_pos(admg['vertices'], admg['di'])
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    figsize = (max(3.4, 0.95 * (max(xs) - min(xs) + 2.4)),
               max(2.6, 0.95 * (max(ys) - min(ys) + 2.0)))
    fig, ax = plt.subplots(figsize=figsize)
    _draw(ax, admg, manip, pos)
    out = os.path.join(OUT, f'{stem}_{suffix}.png')
    fig.savefig(out, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close(fig)
    w, h = figsize
    print(f"saved {out}  ({w:.1f}x{h:.1f} in)")


def main():
    for stem, key, clusters in DATASETS:
        try:
            fine, c, manip = _extract(key, clusters)
        except Exception as e:
            print(f"SKIP {stem}: {type(e).__name__}: {e}")
            continue
        print(f"== {stem}: fine |V|={len(fine['vertices'])} bi={len(fine['bi'])}; "
              f"C-DAG |V|={len(c['vertices'])} bi={len(c['bi'])}")
        render(stem, fine, manip, 'dag')
        render(stem, c, manip, 'cdag')


if __name__ == '__main__':
    main()
