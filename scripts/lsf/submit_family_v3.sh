#!/usr/bin/env bash
# LSF job: engine-v3 rerun of the family suite (see RERUN_ENGINE_V3.md):
# corrected front-door / g-computation integration, causal-prior gradients,
# fixed GP noise, charged initial design, decision-log sidecars, new arms.
#
# Writes to results/v3/family_cbo -- a FRESH directory on purpose. The runner skips any
# unit whose CSV+sidecar already exist, so reusing results/v2 would
# silently keep the old results and still report success.
#
# Usage: bash scripts/lsf/submit_family_v3.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="${OUTDIR:-$REPO/results/v3/family_cbo}"
mkdir -p "$REPO/logs"
SHA="$(git -C "$REPO" rev-parse HEAD)"

bsub <<EOF2
#BSUB -J qcbo_family_v3
#BSUB -q hpc
#BSUB -n 16
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB] span[hosts=1]"
#BSUB -o $REPO/logs/family_v3_%J.out
#BSUB -e $REPO/logs/family_v3_%J.err

source "\$HOME/venvs/ccbo/bin/activate"
cd "$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CCBO_GIT_SHA="$SHA"
PYTHONPATH=. python scripts/run_cbo_family.py --seeds 10 --trials 40 \\
  --outdir "$OUTDIR" --jobs 16
EOF2
