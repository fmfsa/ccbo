# Frozen static matched campaign launch

These files are prepared for the root/operator. They have not submitted any jobs. Deploy this directory to `/zhome/15/b/215295/Repositories/ccbo-experiments-20260916/server-launch`; preserve the immutable `releases/matched-preflight-v3` directory.

The independent audit of the four completed nominal pilots is in `reports/matched-preflight-independent-audit.json`. Every measured row and all100 observational rows were independently reconstructed, all635 scalar acquisition parameter snapshots were checked, and all four unit ledgers/hashes/scores passed the frozen strict analyzer. HQCBO split naturally at the first eligible plateau (purchase7), retained the exact QCBO prefix and paid six split-design rows. CEO was still running at that audit; the real launch guard refused launch. No comparative objective ranking was produced.

## Dependency chain and submission

After all five pilots finish, the operator may run:

```
bash server-launch/submit_matched_campaign.sh
```

To queue the gates while the five-element preflight array is still running:

```
bash server-launch/submit_matched_campaign.sh --after-job 29424824
```

The optional job ID must identify the frozen five-element preflight array. The chain is:

1. Launch gate depends on `numended(PREFLIGHT_ID, == 5)` when supplied. It runs even if a preflight element failed and writes a clear blocked gate record.
2. Scientific array `[1-720]%4` depends on `done(GATE_ID)`. No scientific unit can execute if the gate fails. Each task independently rechecks the same gate before invoking the frozen runner.
3. Final analysis depends on `numended(SCIENCE_ARRAY_ID, == 720)`, counting both DONE and EXIT. It reports failures instead of waiting forever for all-success. LSF's array completion-count semantics are documented in [IBM's whole-array dependency guide](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=arrays-setting-whole-array-dependency).

The submission helper stores scheduler responses and IDs beneath `launch-records/989629b3691bc2b0a015eff4beee55e5dbb24a287692401acf66f87605bc23d4`. Atomic directory creation prevents accidentally submitting the same campaign twice. If a submission fails partway through, the record remains; inspect it manually rather than rerunning blindly. The root may instead submit these three LSF files manually with the same dependencies.

## Required launch gate

`check_matched_launch_gate.py` validates exact preflight/final manifest IDs, every frozen source hash, all five completed nominal-budget unit results through the frozen strict analyzer, the complete720-unit matrix, and the natural HQ split (or separately hashed conditional diagnostic if necessary). Missing, running, failed, altered or invalid units cause exit2. It never uses objective rankings to decide readiness.

It also checks Python3.10.x (HPC3.10.18 and local3.10.20 are both permitted) and these exact principal package versions: GPy1.13.2, NumPy1.26.4, SciPy1.12.0, pandas1.5.3, Emukit0.5.1, NetworkX3.4.2, scikit-learn1.7.2, statsmodels0.13.5. All nominal timings must be positive and finite.

The engineering resource projection is `630*max(four scalar nominal seconds) + 90*CEO nominal seconds`. Launch is blocked if this exceeds500 CPU-hours. This is a conservative phase-planning gate, not a guarantee of actual compute cost or a user-specified spending budget. Exceeding it requires resourcing review; do not remove methods, seeds or difficult outcomes. The gate JSON records the timing inputs, projection, cap, package/Python versions and queue limits. The gate job writes `launch-records/<manifest-id>/gate-job-<LSF-job-id>.json`, including a blocked record on validation/resource failure.

## Scientific resources and failure behavior

Every scientific task requests1 CPU,6GB and8 hours. This array reserves at most4 slots, leaving4 for the separately managed MCBO2 plus dynamic2 slots; the combined reservation is8. The final analyzer runs only after the scientific array has ended.

All outputs are written through the frozen runner to `results/matched-replication/989629b3691bc2b0a015eff4beee55e5dbb24a287692401acf66f87605bc23d4/<unit-id>/result`. LSF indices1–720 map exactly to manifest indices0–719.

The first scientific task failure atomically creates `<campaign>/STOP.json` containing index, job ID, exit code, command, signal and timestamp. Later queued tasks refuse to execute, with an immediate second check before the scientific subprocess. At most the other three already running units can finish. The marker is never overwritten; no automatic retries or cleanup occur. ERR/EXIT/TERM/INT traps cover normal process failures and catchable scheduler termination. SIGKILL, node loss, or a filesystem outage can prevent shell traps; root monitoring must still inspect scheduler failures. No shell mechanism can guarantee a marker after uncatchable termination.

Final analysis has a STOP veto even if files appear otherwise complete. It emits no inference unless all720 units pass and there is no persistent campaign STOP. Failed/incomplete directories and gate/analysis artifacts remain available for investigation. Existing completed unit outputs are hash-checked by the runner and never overwritten.

## Local validation

`python server-launch/test_launch_scripts.py -v` passed six tests: shell syntax/exact indices,20 concurrent atomic marker writers, a failing scientific shim followed by a refused queued task, resource-cap behavior, version/finite-timing checks, and a STOP veto on apparent complete analysis. The real incomplete-preflight guard also exited2 with four valid units and CEO running. No scheduler submission was made by these tests, and the frozen release source was not modified.
