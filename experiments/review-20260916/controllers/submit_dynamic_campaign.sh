#!/bin/bash
# Root/operator invocation only; file preparation submits nothing.
set -euo pipefail
BASE=/zhome/15/b/215295/Repositories/ccbo-experiments-20260916
PYTHON=/zhome/15/b/215295/venvs/ccbo/bin/python
RELEASE="$BASE/releases/dynamic-population-final-v2"
RESULTS="$BASE/results/dynamic-final-campaign"
AFTER_JOB=''
if (( $# )); then
    [[ $# == 2 && "$1" == '--after-job' && "$2" =~ ^[0-9]+$ ]] || { echo 'Usage: submit_dynamic_campaign.sh [--after-job PREFLIGHT_ID]' >&2; exit 2; }
    AFTER_JOB=$2
fi
[[ ! -e "$RESULTS/STOP.json" ]] || { echo 'Persistent STOP; inspect before any launch' >&2; exit 2; }
# In dependency mode, the strict gate runs only after all nine pilots END,
# including failures. Scientific tasks require successful gate completion.
if [[ -z "$AFTER_JOB" ]]; then
    gate=$("$PYTHON" "$BASE/server-launch/check_dynamic_launch_gate.py" --release "$RELEASE" --pilot-results "$BASE/results/dynamic-final-preflight")
fi
mkdir -p "$BASE/launch-records" "$BASE/logs" "$RESULTS"
RECORD="$BASE/launch-records/dynamic-population-final-v2"
mkdir "$RECORD" || { echo 'Submission previously attempted; no automatic retry' >&2; exit 2; }
job_id() {
    "$PYTHON" -c 'import re,sys; m=re.search(r"Job <([0-9]+)>",sys.argv[1]); assert m,"Unrecognized scheduler response"; print(m.group(1))' "$1"
}
if [[ -n "$AFTER_JOB" ]]; then
    gate_dependency="numended($AFTER_JOB, == 9)"
    printf '%s\n' "$gate_dependency" > "$RECORD/gate-dependency.txt"
    gate_submission=$(bsub -w "$gate_dependency" < "$BASE/server-launch/dynamic_launch_gate.lsf")
    printf '%s\n' "$gate_submission" > "$RECORD/gate-submission.txt"
    gate_id=$(job_id "$gate_submission")
    printf '%s\n' "$gate_id" > "$RECORD/gate-job-id.txt"
    submission=$(bsub -w "done($gate_id)" < "$BASE/server-launch/dynamic_campaign.lsf")
else
    printf '%s\n' "$gate" > "$RECORD/launch-gate.json"
    submission=$(bsub < "$BASE/server-launch/dynamic_campaign.lsf")
fi
printf '%s\n' "$submission" > "$RECORD/array-submission.txt"
array_id=$(job_id "$submission")
printf '%s\n' "$array_id" > "$RECORD/array-job-id.txt"
dependency="numended($array_id, == 180)"
printf '%s\n' "$dependency" > "$RECORD/analysis-dependency.txt"
bsub -w "$dependency" < "$BASE/server-launch/dynamic_final_analysis.lsf" > "$RECORD/analysis-submission.txt"
cat "$RECORD/array-submission.txt" "$RECORD/analysis-submission.txt"
