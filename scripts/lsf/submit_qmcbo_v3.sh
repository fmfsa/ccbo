#!/usr/bin/env bash
# LSF job: engine-v3 rerun of the MCBO vs QMCBO suite (20 seeds, E1+E2) with
# the zero-range normalisation guard and per-iteration decision logs.
# Needs ~/venvs/mcbo (era botorch stack) and third_party/mcbo
# (scripts/fetch_mcbo.sh). PSAGraph units take hours each.
# Usage: bash scripts/lsf/submit_qmcbo_v3.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$REPO/logs"
SHA="$(git -C "$REPO" rev-parse HEAD)"

bsub <<EOF2
#BSUB -J qmcbo_v3
#BSUB -q hpc
#BSUB -n 16
#BSUB -W 72:00
#BSUB -R "rusage[mem=4GB] span[hosts=1]"
#BSUB -o $REPO/logs/qmcbo_v3_%J.out
#BSUB -e $REPO/logs/qmcbo_v3_%J.err

cd "$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CCBO_GIT_SHA="$SHA"
PYTHONPATH=. PY="\$HOME/venvs/mcbo/bin/python" \\
  SEEDS="\$(seq -s ' ' 0 19)" \\
  bash scripts/run_qmcbo_pilot.sh results/v3/qmcbo 16
EOF2
