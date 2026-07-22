#!/usr/bin/env bash
# LSF job for the MIS/two-tier exploration-set audit (RUN_TODO step 3).
# Cheap to launch, slow to finish; output goes to results/es_audit.txt.
# Usage: bash scripts/lsf/submit_es_audit.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$REPO/logs"

bsub <<EOF
#BSUB -J es_audit
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 72:00
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -o $REPO/logs/es_audit_%J.out
#BSUB -e $REPO/logs/es_audit_%J.err

source "\$HOME/venvs/ccbo/bin/activate"
cd "$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=. python scripts/audit_es_change.py | tee results/es_audit.txt
EOF
