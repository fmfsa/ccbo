#!/usr/bin/env bash
# LSF job: MinimalBench A-C rerun after the online-observation fix, plus the
# new BO baseline arm (see RERUN_ONLINE_OBS.md).
#
# Writes to results/v2/minimal -- a FRESH directory on purpose. The runner
# skips any unit whose CSV already exists, so reusing results/minimal would
# silently keep the old, wrong results and still report success.
#
# Usage: bash scripts/lsf/submit_minimal_v2.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="${OUTDIR:-$REPO/results/v2/minimal}"
mkdir -p "$REPO/logs"

bsub <<EOF2
#BSUB -J qcbo_minimal_v2
#BSUB -q hpc
#BSUB -n 16
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB] span[hosts=1]"
#BSUB -o $REPO/logs/minimal_v2_%J.out
#BSUB -e $REPO/logs/minimal_v2_%J.err

source "\$HOME/venvs/ccbo/bin/activate"
cd "$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=. python scripts/run_minimal_suite.py \
  --seeds 30 --trials 50 --mediated-trials 60 \
  --outdir "$OUTDIR" --jobs 16
EOF2
