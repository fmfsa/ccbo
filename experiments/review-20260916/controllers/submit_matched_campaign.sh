#!/bin/bash
# Root/operator invocation only. Preparing this file does not submit jobs.
set -euo pipefail
BASE=/zhome/15/b/215295/Repositories/ccbo-experiments-20260916
PYTHON=/zhome/15/b/215295/venvs/ccbo/bin/python
RELEASE="$BASE/releases/matched-preflight-v3"
CAMPAIGN=989629b3691bc2b0a015eff4beee55e5dbb24a287692401acf66f87605bc23d4
after_job=""
while (( $# )); do
    case "$1" in
        --after-job)
            [[ $# -ge 2 && $2 =~ ^[1-9][0-9]*$ ]] || { echo 'Expected positive job ID after --after-job' >&2; exit 2; }
            after_job=$2; shift 2 ;;
        --help)
            echo 'Usage: submit_matched_campaign.sh [--after-job PREFLIGHT_ARRAY_JOB_ID]'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "$after_job" ]]; then
    "$PYTHON" "$BASE/server-launch/check_matched_launch_gate.py" --release "$RELEASE" --preflight-results "$BASE/results/matched-preflight" --diagnostic-results "$BASE/results/matched-hq-diagnostic"
fi
if [[ -e "$BASE/results/matched-replication/$CAMPAIGN/STOP.json" ]]; then
    echo 'Campaign STOP marker exists; refusing submission' >&2
    exit 2
fi
# Atomic duplicate-submission guard. On any later failure this directory remains
# for operator investigation; there is intentionally no automatic retry/cleanup.
mkdir -p "$BASE/launch-records" "$BASE/logs"
RECORD="$BASE/launch-records/$CAMPAIGN"
mkdir "$RECORD" || { echo 'Campaign submission already attempted; inspect preserved record' >&2; exit 2; }
gate_args=()
if [[ -n "$after_job" ]]; then
    gate_args=(-w "numended($after_job, == 5)")
    printf '%s\n' "$after_job" > "$RECORD/preflight-dependency-job-id.txt"
    printf '%s\n' 'Gate waits for all five preflight elements including failures. A failed pilot produces a blocked gate record; science remains dependent on gate success. Investigate without automatic retry or method/seed changes.' > "$RECORD/pending-dependency-policy.txt"
fi
gate_submission=$(bsub "${gate_args[@]}" < "$BASE/server-launch/matched_launch_gate.lsf")
printf '%s\n' "$gate_submission" > "$RECORD/gate-submission.txt"
gate_id=$("$PYTHON" -c 'import re,sys; m=re.search(r"Job <([0-9]+)>",sys.argv[1]); assert m,"Unrecognized bsub response"; print(m.group(1))' "$gate_submission")
printf '%s\n' "$gate_id" > "$RECORD/gate-job-id.txt"
submission=$(bsub -w "done($gate_id)" < "$BASE/server-launch/matched_replication.lsf")
printf '%s\n' "$submission" > "$RECORD/array-submission.txt"
array_id=$("$PYTHON" -c 'import re,sys; m=re.search(r"Job <([0-9]+)>",sys.argv[1]); assert m,"Unrecognized bsub response"; print(m.group(1))' "$submission")
printf '%s\n' "$array_id" > "$RECORD/array-job-id.txt"
# numended counts DONE and EXIT, so failed units lead to a strict blocked audit,
# not an analysis job waiting forever for every element to succeed.
dependency="numended($array_id, == 720)"
printf '%s\n' "$dependency" > "$RECORD/analysis-dependency.txt"
bsub -w "$dependency" < "$BASE/server-launch/matched_final_analysis.lsf" > "$RECORD/analysis-submission.txt"
cat "$RECORD/array-submission.txt" "$RECORD/analysis-submission.txt"
