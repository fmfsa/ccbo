#!/bin/bash
# Full 10-seed experiment suite for the paper. Sequential to avoid
# oversubscribing the box (each run already parallelizes GP fits).
# Checkpointed per seed inside each condition, so interruptions only lose
# the current seed of the current condition.
set -uo pipefail
PY=/home/fmfsa/.conda/envs/ccbo/bin/python
LOG=results/full_suite.log
mkdir -p results
{
echo "=== suite started $(date) ==="
$PY run_experiments.py --benchmark LightTunnel          --condition wrong_edge --seeds 10 --trials 40
$PY run_experiments.py --benchmark CompleteGraph        --condition wrong_edge --seeds 10 --trials 40
$PY run_experiments.py --benchmark CompleteGraph        --condition main       --seeds 10 --trials 40
$PY run_experiments.py --benchmark CompleteGraph        --condition sweep      --seeds 10 --trials 40
$PY run_experiments.py --benchmark SimplifiedCoralGraph --condition main       --seeds 10 --trials 40
$PY run_experiments.py --benchmark SimplifiedCoralGraph --condition sweep      --seeds 10 --trials 40
echo "=== suite finished $(date) ==="
} 2>&1 | grep -vE "DeprecationWarning|RuntimeWarning|UserWarning|paramz|GP:" >> $LOG
