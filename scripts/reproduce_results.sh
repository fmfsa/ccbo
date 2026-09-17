#!/usr/bin/env bash
# Verify archived traces and regenerate figures/tables for Overleaf; no LaTeX needed.
# Usage: [PYTHON=/path/to/python] bash scripts/reproduce_results.sh [WORKDIR]
# WORKDIR holds extracted traces. Outputs: paper/{figures,tables}, artifacts/summaries.
# Extracts archived evidence into WORKDIR; does not modify existing raw runs in results/.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO/.venv/bin/python" ]]; then
        PYTHON="$REPO/.venv/bin/python"
    else
        PYTHON="python3"
    fi
fi
WORKDIR="${1:-$(mktemp -d)}"
mkdir -p "$WORKDIR"
WORKDIR="$(cd "$WORKDIR" && pwd)"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WORKDIR/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$REPO/paper/figures" "$REPO/paper/tables" "$REPO/artifacts/summaries"
echo "Verifying archive SHA256 checksums"
(cd "$REPO/artifacts/traces" && shasum -a 256 --check --quiet SHA256SUMS)
echo "== 1/3 extract + checksum trace archives"
for suite in minimal family_cbo qdcbo qmcbo ceo_minimal; do
    tar -C "$WORKDIR" -xzf "$REPO/artifacts/traces/$suite.tar.gz"
    (cd "$WORKDIR" && shasum -a 256 --check --quiet \
        "$REPO/artifacts/traces/$suite.sha256")
    echo "   $suite: extracted, checksums OK"
done

echo "== 2/3 verify traces against published aggregates"
"$PYTHON" "$REPO/scripts/verify_traces.py" --results-dir "$WORKDIR"

echo "== 3/3 regenerate figures and tables"
cd "$REPO"
"$PYTHON" scripts/emit_minimal_taxonomy.py --dir "$WORKDIR/minimal"
"$PYTHON" scripts/plot_minimal_misspec.py  --dir "$WORKDIR/minimal"
"$PYTHON" scripts/plot_minimal_refine.py   --dir "$WORKDIR/minimal"
"$PYTHON" scripts/plot_minimal_ablations.py --dir "$WORKDIR/minimal"
"$PYTHON" scripts/plot_cost_indexed.py     --dir "$WORKDIR/minimal"
"$PYTHON" scripts/summarize_minimal_exact.py --dir "$WORKDIR/minimal"
"$PYTHON" scripts/summarize_minimal_exact.py --dir "$WORKDIR/minimal" --inventory primary --out artifacts/summaries/minimal_primary.json
"$PYTHON" scripts/emit_minimal_ablation_table.py
"$PYTHON" scripts/plot_family_suite.py     --results-dir "$WORKDIR"
"$PYTHON" scripts/emit_qmcbo_tables.py     --results-dir "$WORKDIR" \
                                           --summary "$WORKDIR/qmcbo_pilot.json"
"$PYTHON" scripts/emit_paired_effects.py   --results-dir "$WORKDIR"
"$PYTHON" scripts/emit_protocol_manifest.py --results-dir "$WORKDIR"
"$PYTHON" scripts/emit_cost_analysis.py    --results-dir "$WORKDIR" \
                                           --ceo-dir "$WORKDIR/ceo_minimal"
"$PYTHON" scripts/emit_ceo_comparison.py   --ceo-dir "$WORKDIR/ceo_minimal" \
                                           --minimal-dir "$WORKDIR/minimal"
"$PYTHON" scripts/emit_dataset_tikz.py
# The revised QMCBO summary separates final/trajectory/decision equality.
# Historical summary schemas are retained as provenance; scientific numbers
# are checked above by verify_traces.py rather than comparing JSON schemas.
echo "   QMCBO summary regenerated with explicit equality metrics: $WORKDIR/qmcbo_pilot.json"
cp "$WORKDIR/qmcbo_pilot.json" artifacts/summaries/qmcbo_revision.json
"$PYTHON" scripts/emit_revision_manifest.py

echo "Done: verified result assets regenerated for Overleaf. Extracted traces: $WORKDIR"
