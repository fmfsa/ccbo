#!/usr/bin/env bash
# LSF array for the CEO-on-MinimalBench comparator: 5 runnable
# (SCM, condition) units x 30 seeds = 150 jobs.
#   units: ParallelParent x {A0, A1, A2}, FrontDoor x B0, MediatedChain x C0
#   (A3 and B1 are observable-identical aliases, materialised afterwards by
#    PYTHONPATH=. python scripts/run_ceo_minimal.py --aliases)
# Pilot timing: ~70 s/trial with the 3-graph ParallelParent pool (T=50),
# singleton pools far faster; -W 6:00 is a comfortable ceiling.
# Usage: bash scripts/lsf/submit_ceo_minimal.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$REPO/logs"

bsub <<EOF
#BSUB -J ceo_minimal[1-150]
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 6:00
#BSUB -R "rusage[mem=6GB] span[hosts=1]"
#BSUB -o $REPO/logs/ceo_min_%J_%I.out
#BSUB -e $REPO/logs/ceo_min_%J_%I.err

SCMS=(ParallelParent ParallelParent ParallelParent FrontDoor MediatedChain)
CONDS=(A0 A1 A2 B0 C0)
i=\$((LSB_JOBINDEX-1))
u=\$((i/30))
seed=\$((i%30))

cd "$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=. "\$HOME/venvs/ccbo/bin/python" scripts/run_ceo_minimal.py \
    --scm "\${SCMS[\$u]}" --cond "\${CONDS[\$u]}" --seed "\$seed" \
    --outdir results/ceo_minimal
EOF
