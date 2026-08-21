#!/usr/bin/env bash
# LSF job: CBO vs QCBO family suite rerun after the online-observation fix,
# plus the new BO baseline arm (see RERUN_ONLINE_OBS.md).
#
# Writes to results/v2/family_cbo -- a FRESH directory on purpose; see the
# note in submit_minimal_v2.sh.
#
# Usage: bash scripts/lsf/submit_family_v2.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="${OUTDIR:-$REPO/results/v2/family_cbo}"
mkdir -p "$REPO/logs"

bsub <<EOF2
#BSUB -J qcbo_family_v2
#BSUB -q hpc
#BSUB -n 16
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB] span[hosts=1]"
#BSUB -o $REPO/logs/family_v2_%J.out
#BSUB -e $REPO/logs/family_v2_%J.err

source "\$HOME/venvs/ccbo/bin/activate"
cd "$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=. python scripts/run_cbo_family.py \
  --seeds 10 --trials 40 --outdir "$OUTDIR" --jobs 16
EOF2
