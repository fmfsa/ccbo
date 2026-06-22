#!/bin/bash
# Full paper run: 100 trials x 5 seeds. Benchmark suite + both wrong-edge
# conditions + regenerate all paper tables. Heavy (~1-2h); checkpointed per
# dataset (suite writes JSON incrementally; wrong_edge checkpoints per seed).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="conda run -n ccbo python"
SEEDS=5
TRIALS=100

echo "===== [1/6] benchmark suite (trials=$TRIALS, seeds=$SEEDS) ====="
$PY scripts/run_qcbo_benchmark_suite.py --trials $TRIALS --seeds $SEEDS || echo "SUITE FAILED"

echo "===== [2/6] ConfoundedCluster wrong_edge (Tier-2) ====="
$PY run_experiments.py --benchmark ConfoundedCluster --condition wrong_edge --seeds $SEEDS --trials $TRIALS || echo "BOW FAILED"

echo "===== [3/6] CompleteGraph wrong_edge (Tier-1) ====="
$PY run_experiments.py --benchmark CompleteGraph --condition wrong_edge --seeds $SEEDS --trials $TRIALS || echo "CG FAILED"

echo "===== [4/6] emit benchmark GAP table ====="
$PY scripts/emit_benchmark_table.py || echo "EMIT BENCH FAILED"

echo "===== [5/6] emit ConfoundedCluster wrong_edge table ====="
$PY generate_paper_figures.py --benchmark ConfoundedCluster --seeds $SEEDS --only wrong_edge || echo "EMIT BOW TABLE FAILED"

echo "===== [6/6] emit CompleteGraph wrong_edge table ====="
$PY generate_paper_figures.py --benchmark CompleteGraph --seeds $SEEDS --only wrong_edge || echo "EMIT CG TABLE FAILED"

echo "===== FULL RUN DONE ====="
