#!/usr/bin/env bash
# LSF job: engine-v3 rerun of the DCBO vs QDCBO suite (20 seeds, E1+E2) with
# the corrected stock semantics applied to BOTH baseline and quotient
# (ccbo/qdcbo/stock_fixes.py: transition fits on the parent slice, is-not-None
# clamping) and per-unit decision logs. Fresh output dir on purpose.
# Usage: bash scripts/lsf/submit_qdcbo_v3.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$REPO/logs"
SHA="$(git -C "$REPO" rev-parse HEAD)"

bsub <<EOF2
#BSUB -J qdcbo_v3
#BSUB -q hpc
#BSUB -n 16
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB] span[hosts=1]"
#BSUB -o $REPO/logs/qdcbo_v3_%J.out
#BSUB -e $REPO/logs/qdcbo_v3_%J.err

cd "$REPO"
export CCBO_GIT_SHA="$SHA"
PYTHONPATH=. PY="\$HOME/venvs/ccbo/bin/python" \\
  bash scripts/run_qdcbo_pilot.sh results/v3/qdcbo 16
EOF2
