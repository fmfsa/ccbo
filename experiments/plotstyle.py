"""Shared plotting style for the paper's figures.

One registry maps every method label to a fixed colour, dash pattern and role,
so a method looks the same in every figure. Okabe-Ito hues mark the methods the
paper argues about; generic baselines (BO, BO-S) are grey so they recede.
Figures are drawn at their final size (column or text width) and included in
LaTeX without rescaling.
"""
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLUMN = 3.25   # in, aistats2027 \columnwidth
TEXT = 6.75     # in, aistats2027 \textwidth

# Okabe-Ito
BLUE, SKY, GREEN, VERMILION, PURPLE = '#0072B2', '#56B4E9', '#009E73', '#D55E00', '#CC79A7'
GREY, LIGHTGREY = '#7a7a7a', '#a8a8a8'
SOLID, DASHED, DOTTED, DASHDOT = '-', (0, (3.2, 1.6)), (0, (1, 1.4)), (0, (4, 1.4, 1, 1.4))

# label -> (colour, linestyle, role)
#   'key'       : the comparison a figure is about -- full weight + 95% band
#   'secondary' : ablations/controls -- own colour, thinner, no band
#   'base'      : generic baselines -- grey, thin, no band, drawn underneath
METHODS = {
    'CBO':                ( BLUE,      SOLID,   'key'),
    'DCBO':               ( BLUE,      SOLID,   'key'),
    'MCBO':               ( BLUE,      SOLID,   'key'),
    'CBO (misspecified)': ( SKY,       DASHED,  'key'),
    'DCBO (matched exploration set)': (SKY, DASHDOT, 'key'),
    'MCBO (matched exploration set)': (SKY, DASHDOT, 'key'),
    'QCBO':               ( VERMILION, SOLID,   'key'),
    'QDCBO':              ( VERMILION, SOLID,   'key'),
    'QMCBO':              ( VERMILION, SOLID,   'key'),
    'HQCBO':              ( GREEN,     SOLID,   'key'),
    'CBO-NP':             ( PURPLE,    DASHDOT, 'secondary'),
    'BO':                 ( GREY,      SOLID,   'base'),
    'BO-S':               ( LIGHTGREY, DASHED,  'base'),
}
LW = {'key': 1.15, 'secondary': 0.95, 'base': 0.8}
Z = {'key': 3, 'secondary': 2.5, 'base': 2}
BAND_ALPHA = 0.13


def use():
    """Activate the paper style."""
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    plt.style.use(HERE / 'ccbo_paper.mplstyle')


def size(width='column', aspect=0.62):
    """(width, height) in inches; width is 'column', 'text' or a number."""
    w = {'column': COLUMN, 'text': TEXT}.get(width, width)
    return (w, w * aspect)


def method_style(label, role=None):
    colour, ls, default = METHODS[label]
    role = role or default
    return dict(color=colour, ls=ls, lw=LW[role], zorder=Z[role]), role


def step_band(ax, x, mean, half=None, label=None, role=None, band=None, steps=True):
    """Mean curve (step-post by default: the incumbent is piecewise constant)
    with a light 95% band for key methods."""
    kw, role = method_style(label, role)
    band = (role == 'key') if band is None else band
    draw = ax.step if steps else ax.plot
    extra = dict(where='post') if steps else {}
    (line,) = draw(x, mean, label=label, solid_capstyle='butt', **kw, **extra)
    if band and half is not None and not np.all(np.isnan(half)):
        ax.fill_between(x, mean - half, mean + half, color=kw['color'], alpha=BAND_ALPHA, lw=0,
                        step='post' if steps else None, zorder=1)
    return line


def panel_title(ax, letter, text):
    ax.set_title(f'({letter}) {text}', loc='left')


def legend_handles(labels):
    """Proxy artists for a shared figure legend in registry style."""
    from matplotlib.lines import Line2D
    out = []
    for label in labels:
        kw, _ = method_style(label)
        kw.pop('zorder')
        out.append(Line2D([], [], **kw))
    return out
