#!/usr/bin/env bash
# One-command regeneration: archived raw traces -> every figure/table -> PDF.
# Engine v3 (2026-09): archives carry per-unit .decisions.json sidecars and
# charged initial designs; verify_traces.py gates both.
#
# Usage:
#   bash scripts/reproduce_paper.sh [WORKDIR]
#
# WORKDIR is where the trace archives are extracted and verified (default: a
# fresh mktemp -d). This script NEVER reads from, writes to, or deletes the
# repository's results/ working directory -- every emitter is pointed at
# WORKDIR explicitly. Outputs land in paper/figures, paper/tables,
# artifacts/summaries, and paper/qcbo_aistats.pdf, exactly as tracked.
#
# Requires the ccbo environment (see envs/README.md):
#   PYTHON=~/venvs/ccbo/bin/python bash scripts/reproduce_paper.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/venvs/ccbo/bin/python}"
WORKDIR="${1:-$(mktemp -d)}"
mkdir -p "$WORKDIR"

export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONHASHSEED=0

echo "== repo:    $REPO"
echo "== python:  $PYTHON ($("$PYTHON" -V 2>&1))"
echo "== workdir: $WORKDIR"

echo "== 1/5 extract + checksum trace archives"
for suite in minimal family_cbo qdcbo qmcbo ceo_minimal; do
    tar -C "$WORKDIR" -xzf "$REPO/artifacts/traces/$suite.tar.gz"
    (cd "$WORKDIR" && sha256sum --check --quiet \
        "$REPO/artifacts/traces/$suite.sha256")
    echo "   $suite: extracted, checksums OK"
done

echo "== 2/5 verify traces against published aggregates"
"$PYTHON" "$REPO/scripts/verify_traces.py" --results-dir "$WORKDIR"

echo "== 3/5 regenerate figures and tables"
cd "$REPO"
"$PYTHON" scripts/emit_minimal_taxonomy.py --dir "$WORKDIR/minimal"
"$PYTHON" scripts/plot_minimal_misspec.py  --dir "$WORKDIR/minimal"
"$PYTHON" scripts/plot_minimal_refine.py   --dir "$WORKDIR/minimal"
"$PYTHON" scripts/plot_minimal_ablations.py --dir "$WORKDIR/minimal"
"$PYTHON" scripts/plot_cost_indexed.py     --dir "$WORKDIR/minimal"
"$PYTHON" scripts/summarize_minimal_exact.py --dir "$WORKDIR/minimal"
"$PYTHON" scripts/emit_minimal_ablation_table.py
"$PYTHON" scripts/plot_family_suite.py     --results-dir "$WORKDIR"
"$PYTHON" scripts/emit_qmcbo_tables.py     --results-dir "$WORKDIR" \
                                           --summary "$WORKDIR/qmcbo_pilot.json"
"$PYTHON" scripts/emit_paired_effects.py   --results-dir "$WORKDIR"
"$PYTHON" scripts/emit_cost_analysis.py    --results-dir "$WORKDIR" \
                                           --ceo-dir "$WORKDIR/ceo_minimal"
"$PYTHON" scripts/emit_ceo_comparison.py   --ceo-dir "$WORKDIR/ceo_minimal" \
                                           --minimal-dir "$WORKDIR/minimal"
"$PYTHON" scripts/emit_dataset_tikz.py
if ! diff -q "$WORKDIR/qmcbo_pilot.json" results/qmcbo_pilot.json >/dev/null 2>&1; then
    echo "   NOTE: regenerated qmcbo_pilot.json differs from the tracked copy" >&2
else
    echo "   qmcbo_pilot.json reproduced byte-identical to the tracked copy"
fi

echo "== 4/5 build the paper"
(cd paper && latexmk -pdf -interaction=nonstopmode qcbo_aistats.tex >/dev/null)

echo "== 5/5 grep gates"
fail=0
gate() {  # gate <description> <pattern> <files...>
    local desc="$1" pat="$2"; shift 2
    if grep -rn "$pat" "$@" >/dev/null 2>&1; then
        echo "   FAIL: $desc"; grep -rn "$pat" "$@" | head -3; fail=1
    else
        echo "   ok:   $desc"
    fi
}
gate "no stale claim language" \
    "PA-GAP\|byte-for-byte\|byte-identical\|uncapped\|cluster budget\|QMCBO-ind\|QMCBOJ\|ten-dimensional\|quarter of the wall" \
    paper/qcbo_aistats.tex paper/tables/
gate "no 'six of eight'" "six of eight\|six of the eight" paper/qcbo_aistats.tex
gate "no 'identifiable cluster unions'" "identifiable cluster unions" \
    paper/qcbo_aistats.tex paper/tables/
gate "no TODO markers" "TODO" paper/qcbo_aistats.tex
gate "thm:contract fully renamed" "thm:contract" paper/qcbo_aistats.tex
gate "family_price fully retired" "family_price" paper/qcbo_aistats.tex
if pdftotext -q paper/qcbo_aistats.pdf - | \
        grep -q "AISTATS 2026\|Under review by AISTATS"; then
    echo "   FAIL: venue branding still renders"; fail=1
else
    echo "   ok:   no AISTATS 2026 branding in the rendered PDF"
fi

if [ "$fail" -ne 0 ]; then
    echo "== GATES FAILED"; exit 1
fi
echo "== done: all artifacts regenerated from $WORKDIR"
