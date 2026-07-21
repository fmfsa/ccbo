#!/usr/bin/env bash
# LSF array for the CEO misspecification sweep: 9 runnable pids x 10 seeds.
# (P7 and S1 are aliases -- see scripts/run_ceo_misspec.py.)
#
# Usage:  bash scripts/lsf/submit_ceo_misspec.sh [extra run_ceo_misspec args]
# After completion, per-seed P7 aliases are materialised by:
#   PYTHONPATH=. python scripts/run_ceo_misspec.py --pid P0 --seeds 0-9
# (the alias step is a no-op copy once P0 CSVs exist).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$REPO/logs"

bsub <<EOF
#BSUB -J ceo_misspec[1-90]
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 72:00
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -o $REPO/logs/ceo_%J_%I.out
#BSUB -e $REPO/logs/ceo_%J_%I.err

PIDS=(P0 P1 P2 P3 Pic P5 P6 S2 S3)
i=\$((LSB_JOBINDEX-1))
pid=\${PIDS[\$((i/10))]}
seed=\$((i%10))

source "$HOME/venvs/ccbo/bin/activate"
cd "$REPO"
PYTHONPATH=. python scripts/run_ceo_misspec.py \
  --pid "\$pid" --seed "\$seed" --trials 100 --ninit 5 \
  --pool hedge --es cluster $@
EOF
