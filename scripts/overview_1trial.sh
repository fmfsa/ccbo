#!/bin/bash
# End-to-end overview at TRIALS=1 (fast) to validate the full pipeline before
# committing to the 100-trial runs: benchmark suite + both wrong-edge
# conditions + all paper tables.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="conda run -n ccbo python"
SEEDS=2

echo "===== [1/6] benchmark suite (trials=1, seeds=$SEEDS) ====="
$PY scripts/run_qcbo_benchmark_suite.py --trials 1 --seeds $SEEDS || echo "SUITE FAILED"

echo "===== [2/6] ConfoundedCluster wrong_edge (Tier-2) ====="
$PY run_experiments.py --benchmark ConfoundedCluster --condition wrong_edge --seeds $SEEDS --trials 1 || echo "BOW FAILED"

echo "===== [3/6] CompleteGraph wrong_edge (Tier-1) ====="
$PY run_experiments.py --benchmark CompleteGraph --condition wrong_edge --seeds $SEEDS --trials 1 || echo "CG FAILED"

echo "===== [4/6] emit benchmark GAP table ====="
$PY scripts/emit_benchmark_table.py || echo "EMIT BENCH FAILED"

echo "===== [5/6] emit ConfoundedCluster wrong_edge table ====="
$PY generate_paper_figures.py --benchmark ConfoundedCluster --seeds $SEEDS --only wrong_edge || echo "EMIT BOW TABLE FAILED"

echo "===== [6/6] emit CompleteGraph wrong_edge table ====="
$PY generate_paper_figures.py --benchmark CompleteGraph --seeds $SEEDS --only wrong_edge || echo "EMIT CG TABLE FAILED"

echo "===== OVERVIEW DONE ====="
