# Dynamic final launch bundle

Release: dynamic-population-final-v2 (scientific sources frozen).
Pilot manifest: 7099d1ec0e3d5215da0c431636f438b5b8842b80f115308998870bf6599a80c5.
Final manifest: d4dd9d45227cd0f36a3061af210fdcf1c9f2765a0e1bb1d363c8e2f9d8579275.

Chosen-settings pilot job29424849 is owned/submitted by root; its exact result root is `$BASE/results/dynamic-final-preflight`.
The prepared `submit_dynamic_campaign.sh --after-job 29424849` queues a strict gate after numended==9; the180-unit array requires gate success and runs at most2 jobs. Final analysis waits for numended==180. Files were prepared only; this agent submitted no jobs.
Final results and persistent STOP live under `$BASE/results/dynamic-final-campaign`. Launch records are exclusive under `$BASE/launch-records/dynamic-population-final-v2`; repeated submission attempts fail.

Every campaign task reruns the strict pilot/source/runtime/numerical gate and scheduler peer check, then rechecks STOP immediately before execution. Atomic first failure is retained. All nine pilot units must pass; projection uses the maximum complete pilot walltime times180 and must not exceed100CPUh. Final tasks each use1CPU/8GB/2h. Current STOP does not kill a peer already running, but prevents queued jobs starting science; final analysis vetoes inference.

Six local scheduler-shim tests passed: unverified-analysis import prevention, shell syntax/resources, array boundaries, first-failure persistence, scheduler peer veto, and deferred submission dependencies/duplicate suppression. No real scheduler was called by tests.
