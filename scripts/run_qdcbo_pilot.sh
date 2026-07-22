#!/usr/bin/env bash
# QDCBO suite: E1 (DCBO vs QDCBO on DCBO's own setups, 20 seeds, T=3) and
# E2 (intra-cluster misspecification of the ASSUMED graph: correct vs
# perturbed model view; objective fixed).
# Requires third_party/DCBO (scripts/fetch_dcbo.sh).
# Usage: PYTHONPATH=. bash scripts/run_qdcbo_pilot.sh [outroot] [jobs]
set -u
OUTROOT=${1:-results/qdcbo}
JOBS=${2:-6}
SEEDS=$(seq 0 19)
SETUPS="stat ind nonstat"
T=3
TRIALS=10
PY=${PY:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUTROOT" "$OUTROOT/_e2"

sem() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

# ---- E1: correct model view --------------------------------------------
for setup in $SETUPS; do for algo in DCBO QDCBO; do for s in $SEEDS; do
  lname=$(echo "$algo" | tr '[:upper:]' '[:lower:]')
  csv="$OUTROOT/${lname}_${setup}_best_so_far_seed${s}_T${T}_trials${TRIALS}_reps1.csv"
  [ -f "${csv%.csv}_info.json" ] && { echo "skip $algo $setup $s"; continue; }
  sem
  PYTHONPATH=. "$PY" -m ccbo.qdcbo.runner --setup "$setup" --algo "$algo" \
    --T $T --trials $TRIALS --seed "$s" --outdir "$OUTROOT" \
    > "$OUTROOT/log_${algo}_${setup}_${s}.txt" 2>&1 &
done; done; done

# ---- E2: intra-cluster perturbation of the model's DAG view ------------
for setup in $SETUPS; do for algo in DCBO QDCBO; do for s in $SEEDS; do
  lname=$(echo "$algo" | tr '[:upper:]' '[:lower:]')
  csv="$OUTROOT/_e2/${lname}_${setup}_e2_best_so_far_seed${s}_T${T}_trials${TRIALS}_reps1.csv"
  [ -f "${csv%.csv}_info.json" ] && { echo "skip e2 $algo $setup $s"; continue; }
  sem
  PYTHONPATH=. "$PY" -m ccbo.qdcbo.runner --setup "$setup" --algo "$algo" \
    --T $T --trials $TRIALS --seed "$s" --misspec e2 --outdir "$OUTROOT/_e2" \
    > "$OUTROOT/_e2/log_${algo}_${setup}_${s}.txt" 2>&1 &
done; done; done

wait
echo "QDCBO SUITE DONE"
