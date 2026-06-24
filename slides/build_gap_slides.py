"""Build the QCBO explainer deck: DAG->quotient + GAP/PA-GAP (python-pptx)."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE

# ---- palette ----------------------------------------------------------------
NAVY   = RGBColor(0x16, 0x2A, 0x4A)   # titles, key text
INK    = RGBColor(0x24, 0x32, 0x42)   # body
MUTED  = RGBColor(0x5B, 0x6B, 0x7B)   # captions / secondary
TEAL   = RGBColor(0x0E, 0x83, 0x88)   # GAP accent
TEALLT = RGBColor(0xE6, 0xF2, 0xF2)   # GAP tint
CORAL  = RGBColor(0xC2, 0x57, 0x3A)   # PA-GAP accent
CORLT  = RGBColor(0xF7, 0xEC, 0xE7)   # PA-GAP tint
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
BORDER = RGBColor(0xD8, 0xDE, 0xE4)
GRID   = RGBColor(0xE6, 0xEA, 0xEE)

SERIF = "Cambria"      # headers
SANS  = "Calibri"      # body
MONO  = "Courier New"  # formulas

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def _set_shadow(shape):
    """Subtle outer drop shadow via raw OOXML (schema-correct: effectLst after ln)."""
    spPr = shape._element.spPr
    effx = spPr.find(qn('a:effectLst'))
    if effx is None:
        effx = spPr.makeelement(qn('a:effectLst'), {})
        spPr.append(effx)
    sh = effx.makeelement(qn('a:outerShdw'),
                          {'blurRad': '90000', 'dist': '38000', 'dir': '5400000',
                           'rotWithShape': '0'})
    clr = sh.makeelement(qn('a:srgbClr'), {'val': '9AA6B2'})
    a = clr.makeelement(qn('a:alpha'), {'val': '26000'})
    clr.append(a); sh.append(clr); effx.append(sh)


def textbox(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            space_after=4, line_spacing=1.05):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'):
        setattr(tf, m, 0)
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.space_before = Pt(0)
        p.line_spacing = line_spacing
        for (txt, size, color, bold, font, *rest) in para:
            r = p.add_run(); r.text = txt
            r.font.size = Pt(size); r.font.bold = bold
            r.font.color.rgb = color; r.font.name = font
            if rest and rest[0] == 'italic':
                r.font.italic = True
    return tb


def card(slide, x, y, w, h, fill, line=None, radius=0.10, shadow=True):
    sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(1)
    try:
        sp.adjustments[0] = radius
    except Exception:
        pass
    sp.shadow.inherit = False
    if shadow:
        _set_shadow(sp)
    return sp


def dot(slide, x, y, d, color):
    o = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    o.fill.solid(); o.fill.fore_color.rgb = color
    o.line.fill.background(); o.shadow.inherit = False
    return o


def rt_curve():
    cats = [str(t) for t in range(0, 101, 10)]
    vals = [0.0, 0.35, 0.60, 0.73, 0.79, 0.82, 0.835, 0.845, 0.85, 0.85, 0.85]
    return cats, vals


def add_curve(slide, x, y, w, h, kind, accent, fill_tint):
    cats, vals = rt_curve()
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("R_t", vals)
    ctype = XL_CHART_TYPE.LINE if kind == "line" else XL_CHART_TYPE.AREA
    gf = slide.shapes.add_chart(ctype, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = False
    plot = ch.plots[0]
    s = plot.series[0]
    if kind == "line":
        s.smooth = True
        s.format.line.color.rgb = accent
        s.format.line.width = Pt(3.25)
    else:
        s.format.fill.solid(); s.format.fill.fore_color.rgb = fill_tint
        s.format.line.color.rgb = accent; s.format.line.width = Pt(2.5)
    va = ch.value_axis
    va.minimum_scale = 0.0; va.maximum_scale = 1.0
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = GRID
    va.major_gridlines.format.line.width = Pt(0.5)
    va.tick_labels.font.size = Pt(10); va.tick_labels.font.color.rgb = MUTED
    va.tick_labels.font.name = SANS
    va.format.line.color.rgb = BORDER
    ca = ch.category_axis
    ca.has_major_gridlines = False
    ca.tick_labels.font.size = Pt(10); ca.tick_labels.font.color.rgb = MUTED
    ca.tick_labels.font.name = SANS
    ca.format.line.color.rgb = BORDER
    return gf


# =============================================================================
def build_slide(spec):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = WHITE
    accent, tint = spec['accent'], spec['tint']

    # Title + one-line definition
    textbox(s, 0.6, 0.42, 12.1, 0.8,
            [[(spec['title'], 33, NAVY, True, SERIF)]])
    textbox(s, 0.62, 1.24, 12.1, 0.7,
            [[(spec['subtitle'], 16.5, MUTED, False, SANS)]], line_spacing=1.08)

    # ---- LEFT column: what it measures + plain idea ----
    lx, lw = 0.6, 5.75
    dot(s, lx, 2.18, 0.16, accent)
    textbox(s, lx + 0.28, 2.07, lw - 0.28, 0.45,
            [[("What it measures", 19, NAVY, True, SERIF)]])
    textbox(s, lx + 0.28, 2.62, lw - 0.28, 1.25, spec['measures'],
            space_after=7, line_spacing=1.06)

    cy = 4.05
    card(s, lx, cy, lw, 2.55, tint, radius=0.07)
    textbox(s, lx + 0.32, cy + 0.26, lw - 0.62, 0.4,
            [[("Plain idea", 15, accent, True, SANS)]])
    textbox(s, lx + 0.32, cy + 0.72, lw - 0.62, 1.7,
            [[(spec['idea'], 15.5, INK, False, SANS)]], line_spacing=1.12)

    # ---- RIGHT column: formula card + curve ----
    rx, rw = 6.75, 6.0
    fy, fh = 2.07, spec.get('formula_h', 2.35)
    card(s, rx, fy, rw, fh, WHITE, line=BORDER, radius=0.06)
    textbox(s, rx + 0.34, fy + 0.22, rw - 0.68, 0.4,
            [[(spec['formula_head'], 14.5, accent, True, SANS)]])
    textbox(s, rx + 0.34, fy + 0.70, rw - 0.68, fh - 1.30, spec['formula'],
            space_after=6, line_spacing=1.0)
    textbox(s, rx + 0.34, fy + fh - 0.62, rw - 0.68, 0.6,
            [[(spec['legend'], 10.5, MUTED, False, SANS)]], line_spacing=1.04)

    chy = fy + fh + 0.2
    add_curve(s, rx + 0.05, chy, rw - 0.1, 6.67 - chy, spec['chart'], accent, tint)
    textbox(s, rx + 0.05, 6.72, rw - 0.1, 0.55,
            [[(spec['caption'], 12, MUTED, False, SANS, 'italic')]],
            line_spacing=1.05)


SLIDE1 = {
    'accent': TEAL, 'tint': TEALLT, 'chart': 'line',
    'title': "GAP — how close did you get, and how fast?",
    'subtitle': "One score in [0, 1] (higher is better): did the optimizer reach "
                "a value near the best possible, and did it get there early?",
    'measures': [
        [("Quality — ", 16, INK, True, SANS),
         ("how close the best result got to the best possible value y*.", 16, INK, False, SANS)],
        [("Speed — ", 16, INK, True, SANS),
         ("how early in the run that best value was first reached.", 16, INK, False, SANS)],
    ],
    'idea': "It rewards an optimizer that jumps to a near-best answer EARLY. "
            "GAP looks only at the single best value and the moment it first "
            "appeared — not the ups and downs along the way.",
    'formula_head': "Formula  (higher = better, 0–1)",
    'formula': [
        [("R     = clip( (y0 - y_best) / (y0 - y*),  0, 1)", 13, INK, False, MONO)],
        [("speed = (T - t*) / T", 13, INK, False, MONO)],
        [("GAP   = ( R + speed ) / ( 1 + (T-1)/T )", 13.5, NAVY, True, MONO)],
    ],
    'legend': "y0 = starting best   ·   y_best = best found   ·   y* = best "
              "possible (oracle)   ·   t* = trial the best was first reached   ·   "
              "T = number of optimization trials (the budget)",
    'caption': "The curve = how far you've improved over time. GAP reads its final "
               "height (≈0.85) and how early it got there — a snapshot, not the path.",
}

SLIDE2 = {
    'accent': CORAL, 'tint': CORLT, 'chart': 'area',
    'title': "PA-GAP — was progress steady the whole way?",
    'subtitle': "Path-Aware GAP scores the WHOLE journey, not just the final value: "
                "it averages how much you'd improved at every trial.",
    'measures': [
        [("Whole path — ", 16, INK, True, SANS),
         ("it sums up improvement at every step, not only the end.", 16, INK, False, SANS)],
        [("Early counts more — ", 16, INK, True, SANS),
         ("each trial is weighted, with earlier progress worth more.", 16, INK, False, SANS)],
    ],
    'formula_h': 2.72,
    'idea': "Two methods can reach the same final value but score differently: "
            "the one that improves EARLY and holds it wins. Because early trials "
            "weigh most, even a perfect run caps near 1/2 (its maximum is "
            "(T+1)/(2T)) — so PA-GAP scores look smaller than GAP's 0–1.",
    'formula_head': "Formula  (higher = better)",
    'formula': [
        [("R_t = clip( (y0 - best_t)/(y0 - y*), 0, 1)", 12, INK, False, MONO)],
        [("w_t = (T - (t-1)) / T      (early weight)", 12, INK, False, MONO)],
        [("PA-GAP = (1/T) · Σ_t  R_t · w_t", 12.5, NAVY, True, MONO)],
        [("0  <=  PA-GAP  <=  (T+1)/(2T)  ~  0.5", 12, CORAL, True, MONO)],
    ],
    'legend': "best_t = best-so-far at trial t   ·   R_t = improvement so far "
              "(0–1)   ·   T = number of optimization trials (the budget)",
    'caption': "PA-GAP adds up the shaded area under the whole curve (early trials "
               "weighted more) — it rewards the journey, not just the endpoint.",
}

# =============================================================================
# DAG -> quotient slides
# =============================================================================
FIGDIR = "slides/figures"
# pixel dims of the rendered PNGs (from slides/render_dag_pngs.py)
DIMS = {
    'ToyGraph_dag.png': (567, 888), 'ToyGraph_cdag.png': (567, 610),
    'Synthetic_dag.png': (746, 1166), 'Synthetic_cdag.png': (746, 1166),
    'Chain_dag.png': (746, 888), 'Chain_cdag.png': (567, 888),
    'Healthcare_dag.png': (746, 1444), 'Healthcare_cdag.png': (567, 1444),
    'Epidemiology_dag.png': (746, 1166), 'Epidemiology_cdag.png': (567, 1166),
    'Ecology_dag.png': (1453, 1166), 'Ecology_cdag.png': (1100, 1444),
    'Protein_dag.png': (1453, 1444), 'Protein_cdag.png': (1100, 1166),
}
# node-encoding colours (match the matplotlib figures)
C_MANIP = RGBColor(0xE0, 0x76, 0x3A)
C_OTHER = RGBColor(0x7F, 0xB2, 0xD4)
C_TARGET = RGBColor(0xCD, 0xE0, 0x5A)
C_CONF = RGBColor(0xB3, 0x20, 0x2C)


def add_image_fit(slide, name, bx, by, bw, bh):
    """Place an image scaled to fit inside (bx,by,bw,bh), centred, aspect kept."""
    w, h = DIMS[name]; ar = w / h
    if bw / bh > ar:
        ih = bh; iw = bh * ar
    else:
        iw = bw; ih = bw / ar
    ix = bx + (bw - iw) / 2; iy = by + (bh - ih) / 2
    slide.shapes.add_picture(os.path.join(FIGDIR, name),
                             Inches(ix), Inches(iy), width=Inches(iw), height=Inches(ih))


def legend_row(s, y):
    items = [('oval', C_MANIP, "manipulable"), ('oval', C_OTHER, "other variable"),
             ('oval', C_TARGET, "target Y"), ('rrect', C_MANIP, "cluster (quotient)"),
             ('dash', C_CONF, "latent confounder")]
    x = 1.05
    for kind, color, label in items:
        if kind == 'oval':
            sh = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(0.22), Inches(0.22))
        elif kind == 'rrect':
            sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y + 0.02), Inches(0.34), Inches(0.18))
        else:
            sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.09), Inches(0.30), Inches(0.045))
        sh.fill.solid(); sh.fill.fore_color.rgb = color
        sh.line.fill.background(); sh.shadow.inherit = False
        wsw = 0.34 if kind == 'rrect' else (0.30 if kind == 'dash' else 0.22)
        wlab = 0.14 + 0.082 * len(label)
        textbox(s, x + wsw + 0.12, y - 0.05, wlab, 0.32,
                [[(label, 11.5, INK, False, SANS)]], anchor=MSO_ANCHOR.MIDDLE)
        x += wsw + 0.12 + wlab + 0.34


def build_concept():
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = WHITE
    textbox(s, 0.6, 0.42, 12.1, 0.8,
            [[("What QCBO operates on: from DAG to quotient", 31, NAVY, True, SERIF)]])
    textbox(s, 0.62, 1.22, 12.1, 0.7,
            [[("QCBO groups the manipulable variables into clusters and reasons over the ", 16, MUTED, False, SANS),
              ("quotient", 16, TEAL, True, SANS),
              (" graph — the cluster-level causal graph — instead of the full DAG.", 16, MUTED, False, SANS)]],
            line_spacing=1.08)
    add_image_fit(s, 'Synthetic_dag.png', 0.9, 2.05, 4.1, 3.95)
    textbox(s, 0.7, 6.02, 4.5, 0.4,
            [[("Full causal DAG  (Synthetic / CompleteGraph)", 13, MUTED, False, SANS)]],
            align=PP_ALIGN.CENTER)
    textbox(s, 5.15, 3.05, 3.0, 0.8, [[("→", 44, TEAL, True, SANS)]],
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, 5.15, 3.95, 3.0, 0.5, [[("merge  {D, E}", 15, NAVY, True, SANS)]],
            align=PP_ALIGN.CENTER)
    add_image_fit(s, 'Synthetic_cdag.png', 8.3, 2.05, 4.1, 3.95)
    textbox(s, 8.1, 6.02, 4.5, 0.4,
            [[("Quotient C-DAG  (what QCBO uses)", 13, TEAL, True, SANS)]],
            align=PP_ALIGN.CENTER)
    legend_row(s, 6.72)


def build_gallery(title, rows):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = WHITE
    textbox(s, 0.6, 0.42, 12.1, 0.8, [[(title, 28, NAVY, True, SERIF)]])
    top, bot = 1.5, 7.15
    rowh = (bot - top) / len(rows)
    for i, (stem, name, part) in enumerate(rows):
        ry = top + i * rowh
        textbox(s, 0.55, ry, 2.55, rowh,
                [[(name, 17, NAVY, True, SANS)], [(part, 12, MUTED, False, SANS)]],
                anchor=MSO_ANCHOR.MIDDLE, space_after=3, line_spacing=1.05)
        ih = rowh - 0.34
        add_image_fit(s, f'{stem}_dag.png', 3.15, ry + 0.1, 3.5, ih)
        textbox(s, 6.7, ry, 0.7, rowh, [[("→", 26, TEAL, True, SANS)]],
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        add_image_fit(s, f'{stem}_cdag.png', 7.5, ry + 0.1, 3.5, ih)
    textbox(s, 0.6, 7.12, 12.1, 0.32,
            [[("fine DAG  →  quotient C-DAG   (orange = manipulable, box = cluster, "
               "dashed red = latent confounder)", 11, MUTED, False, SANS, 'italic')]],
            align=PP_ALIGN.CENTER)


build_slide(SLIDE1)
build_slide(SLIDE2)
build_concept()
build_gallery("Benchmark structures and their quotients  (1 / 3)", [
    ('ToyGraph',     "ToyGraph / Synth-2", "cluster {X, Z}"),
    ('Chain',        "Chain-hard",         "cluster {Z, W}"),
])
build_gallery("Benchmark structures and their quotients  (2 / 3)", [
    ('Epidemiology', "Epidemiology", "cluster {L, B}"),
    ('Healthcare',   "Healthcare",   "cluster {Aspirin, Statin}"),
])
build_gallery("Benchmark structures and their quotients  (3 / 3)", [
    ('Ecology', "Ecology", "clusters {C,N,O}, {D,T}"),
    ('Protein', "Protein", "clusters {Akt,Mek}, {PKA,PKC}"),
])

# Speaker notes
prs.slides[0].notes_slide.notes_text_frame.text = (
    "GAP = quality + speed in one number. R measures how far you closed the gap to "
    "the oracle optimum y*; the speed term rewards reaching that best value early. "
    "Key point: GAP only looks at the single best value and when it first appeared.")
prs.slides[1].notes_slide.notes_text_frame.text = (
    "PA-GAP is path-aware: it averages the improvement ratio over every trial, "
    "weighting earlier trials more. It rewards steady, early progress. GAP and "
    "PA-GAP can disagree (a big early jump scores high on GAP; gradual progress "
    "scores higher on PA-GAP) — which is why the benchmark reports both.")

out = "slides/GAP_PA-GAP_explainer.pptx"
prs.save(out)
print("saved", out)
