#!/usr/bin/env bash
# QMCBO pilot: E1 (MCBO vs QMCBO, 5 seeds, T=100, three envs) and
# E2 (intra-cluster misspecification: correct vs perturbed model view).
# Usage: PYTHONPATH=. bash scripts/run_qmcbo_pilot.sh [outroot] [jobs]
set -u
OUTROOT=${1:-third_party/CausalBO_Benchmark/results/_qmcbo}
JOBS=${2:-8}
SEEDS="0 1 2 3 4"
ENVS="ToyGraph PSAGraph"
T=100
mkdir -p "$OUTROOT" "$OUTROOT/_e2"

sem() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

# ---- E1: correct model view --------------------------------------------
for env in $ENVS; do for spec in "MCBO|MCBO|" "QMCBO|QMCBOJ|--mechanism joint" "QMCBO|QMCBO|--mechanism ind"; do for s in $SEEDS; do
  algo=${spec%%|*}; rest=${spec#*|}; label=${rest%%|*}; mech=${rest#*|}
  csv="$OUTROOT/trial_results_${label}_${env}_${s}.csv"
  [ -f "${csv%.csv}_info.json" ] && { echo "skip $label $env $s"; continue; }
  sem
  "$HOME/venvs/ccbo/bin/python" -m ccbo.qmcbo.runner --env "$env" --algo "$algo" $mech \
    --seed "$s" --num_trials $T --outdir "$OUTROOT" \
    > "$OUTROOT/log_${label}_${env}_${s}.txt" 2>&1 &
done; done; done

# ---- E2: canonical intra-cluster perturbation of the model view --------
for env in $ENVS; do for spec in "MCBO|MCBO|" "QMCBO|QMCBOJ|--mechanism joint" "QMCBO|QMCBO|--mechanism ind"; do for s in $SEEDS; do
  algo=${spec%%|*}; rest=${spec#*|}; label=${rest%%|*}; mech=${rest#*|}
  csv="$OUTROOT/_e2/trial_results_${label}_${env}_${s}.csv"
  [ -f "${csv%.csv}_info.json" ] && { echo "skip e2 $label $env $s"; continue; }
  sem
  "$HOME/venvs/ccbo/bin/python" -m ccbo.qmcbo.runner --env "$env" --algo "$algo" $mech \
    --seed "$s" --num_trials $T --misspec e2 --outdir "$OUTROOT/_e2" \
    > "$OUTROOT/_e2/log_${label}_${env}_${s}.txt" 2>&1 &
done; done; done

wait
echo "QMCBO PILOT DONE"
