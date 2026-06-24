"""Build a 2-slide explainer for GAP and PA-GAP (python-pptx)."""
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
    fy, fh = 2.07, 2.35
    card(s, rx, fy, rw, fh, WHITE, line=BORDER, radius=0.06)
    textbox(s, rx + 0.34, fy + 0.22, rw - 0.68, 0.4,
            [[(spec['formula_head'], 14.5, accent, True, SANS)]])
    textbox(s, rx + 0.34, fy + 0.70, rw - 0.68, 1.15, spec['formula'],
            space_after=6, line_spacing=1.0)
    textbox(s, rx + 0.34, fy + fh - 0.62, rw - 0.68, 0.6,
            [[(spec['legend'], 10.5, MUTED, False, SANS)]], line_spacing=1.04)

    add_curve(s, rx + 0.05, 4.62, rw - 0.1, 2.05, spec['chart'], accent, tint)
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
              "possible (oracle)   ·   t* = trial best first reached   ·   T = total trials",
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
    'idea': "Two methods can reach the same final value but score differently: "
            "the one that improves EARLY and holds it wins. PA-GAP = the "
            "(early-weighted) area under the improvement curve.",
    'formula_head': "Formula  (higher = better)",
    'formula': [
        [("R_t = clip( (y0 - best_t) / (y0 - y*),  0, 1)", 13, INK, False, MONO)],
        [("w_t = (T - (t-1)) / T          (early-trial weight)", 12.5, INK, False, MONO)],
        [("PA-GAP = (1/T) · Σ_t  R_t · w_t", 13.5, NAVY, True, MONO)],
    ],
    'legend': "best_t = best-so-far at trial t   ·   R_t = how far you've improved "
              "(0–1)   ·   w_t = weight that fades over the run   ·   T = total trials",
    'caption': "PA-GAP adds up the shaded area under the whole curve (early trials "
               "weighted more) — it rewards the journey, not just the endpoint.",
}

build_slide(SLIDE1)
build_slide(SLIDE2)

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
