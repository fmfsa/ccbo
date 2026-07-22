#!/usr/bin/env bash
# LSF job for the DCBO vs QDCBO suite (RUN_TODO step 2b; 20 seeds, E1+E2).
# Usage: bash scripts/lsf/submit_qdcbo.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$REPO/logs"

bsub <<EOF
#BSUB -J qdcbo_suite
#BSUB -q hpc
#BSUB -n 16
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB] span[hosts=1]"
#BSUB -o $REPO/logs/qdcbo_%J.out
#BSUB -e $REPO/logs/qdcbo_%J.err

cd "$REPO"
PYTHONPATH=. PY="\$HOME/venvs/ccbo/bin/python" \
  bash scripts/run_qdcbo_pilot.sh results/qdcbo 16
EOF
