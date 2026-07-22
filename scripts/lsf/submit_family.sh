#!/usr/bin/env bash
# LSF job for the CBO vs QCBO family suite (RUN_TODO step 2a).
# Usage: bash scripts/lsf/submit_family.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$REPO/logs"

bsub <<EOF
#BSUB -J qcbo_family
#BSUB -q hpc
#BSUB -n 16
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB] span[hosts=1]"
#BSUB -o $REPO/logs/family_%J.out
#BSUB -e $REPO/logs/family_%J.err

source "\$HOME/venvs/ccbo/bin/activate"
cd "$REPO"
PYTHONPATH=. python scripts/run_cbo_family.py \
  --seeds 10 --trials 40 --outdir results/family_cbo --jobs 16
EOF
