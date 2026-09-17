"""Protect intervention meaning in generated and curated graph figures."""
import io
import re
from pathlib import Path

import pytest
from scripts import emit_dataset_tikz as figures

ROOT = Path(__file__).resolve().parents[2]


def orange_nodes(text):
    return set(re.findall(r"\\node\[mannode(?:,[^\]]*)?\] \(([^)]+)\)", text))


def test_psa_target_metadata_and_generated_panels():
    nodes, parents, clusters, manip = figures.MCBO_GRAPHS["PSAGraph"]
    assert set(manip) == {"A", "S"}
    edges = [(nodes[p], nodes[i]) for i, ps in enumerate(parents) for p in ps]
    _, _, quotient_manip, _ = figures.quotient_literal(
        nodes, edges, manip, clusters, ["Y"])
    assert quotient_manip == ["A,S"]
    out = io.StringIO()
    figures.emit_literal(out, "psa", "PSAGraph", nodes, edges, manip,
                         clusters, ["Y"], figures.PSA_CAPTION_EXTRA)
    panels = out.getvalue().split(r"\begin{tikzpicture}")[1:]
    assert orange_nodes(panels[0]) == {"nA", "nS"}
    assert orange_nodes(panels[1]) == {"nAS"}
    assert r"reward $-\mathrm{PSA}$" in out.getvalue()
    assert set(figures.MCBO_GRAPHS["ToyGraph-M"][3]) == {"X0", "X1"}


def test_curated_psa_panels_and_confounded_captions():
    text = (ROOT / "paper/figures/dataset_dags.tex").read_text()
    blocks = text.split(r"\begin{figure}")
    psa = next(b for b in blocks if r"\label{fig:dag-mcbo-psagraph}" in b)
    panels = psa.split(r"\begin{tikzpicture}")[1:]
    assert orange_nodes(panels[0]) == {"nA", "nS"}
    assert orange_nodes(panels[1]) == {"nAS"}
    assert r"reward $-\mathrm{PSA}$" in psa
    for key in ("parallelparent", "complete"):
        block = next(b for b in blocks if f"\\label{{fig:dag-{key}}}" in b)
        assert "Fine ADMG" in block


@pytest.mark.parametrize("name,clusters,kind", [
    (figures.mb.PP_NAME, [["X1", "X2"]], "ADMG"),
    ("CompleteGraph", [["B"], ["D", "E"]], "ADMG"),
    ("ToyGraph", [["X", "Z"]], "ADMG"),
    (figures.mb.MC_NAME, [["X1", "X2"]], "DAG"),
])
def test_registry_captions_follow_bidirected_edges(name, clusters, kind):
    figures.mb.register_variants()
    out = io.StringIO()
    structure = figures.emit_registry(out, name, clusters, name, "test")
    assert bool(structure["bi"]) == (kind == "ADMG")
    assert f"Fine {kind} with coarse partition" in out.getvalue()
