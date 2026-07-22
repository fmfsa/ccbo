#!/usr/bin/env bash
# LSF job for the MCBO vs QMCBO suite (RUN_TODO step 2c; the longest —
# PSAGraph units are hours each). Needs ~/venvs/mcbo (era botorch stack).
# Usage: bash scripts/lsf/submit_qmcbo.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$REPO/logs"

bsub <<EOF
#BSUB -J qmcbo_suite
#BSUB -q hpc
#BSUB -n 10
#BSUB -W 72:00
#BSUB -R "rusage[mem=4GB] span[hosts=1]"
#BSUB -o $REPO/logs/qmcbo_%J.out
#BSUB -e $REPO/logs/qmcbo_%J.err

cd "$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=. PY="\$HOME/venvs/mcbo/bin/python" \
  bash scripts/run_qmcbo_pilot.sh results/qmcbo 10
EOF
