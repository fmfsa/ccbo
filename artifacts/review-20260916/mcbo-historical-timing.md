# Historical MCBO runtime check (17 September 2026)

## Confirmed per-run wall times

The 80 main archived `*_info.json` files in `work/revision-run/reproduced-traces/qmcbo/` record `secs`, 100 trials, seeds 0–19, and engine SHA `9f25b415ff90502533cf3b013c0a10fd14571781`. These are measured runner wall times, not estimates from file timestamps. The archived runner starts `time.time()` after major imports and records elapsed seconds after trial execution and decision-file serialization, immediately before writing the info file (`work/revision-run/repo/ccbo/qmcbo/runner.py`, lines 89 and 295).

| Environment | Method | Runs | Minimum min | Median min | Maximum min | Mean min |
|---|---|---:|---:|---:|---:|---:|
| PSAGraph | MCBO | 20 | 109.24 | 113.46 | 123.18 | 113.62 |
| PSAGraph | QMCBO | 20 | 35.67 | 37.47 | 41.37 | 37.57 |
| ToyGraph | MCBO | 20 | 16.72 | 22.22 | 37.02 | 23.23 |
| ToyGraph | QMCBO | 20 | 17.68 | 22.42 | 32.56 | 23.23 |

Thus historical PSA MCBO was already approximately 1.9 hours per 100-round run; historical PSA QMCBO was approximately 37 minutes. The current Toy timings supplied by the campaign owner (42.2 min full, 19.8 min coarse-menu, 40.7 min quotient) put full/quotient approximately 1.90/1.82 times their respective historical medians. They are different seeds and corrected algorithms, not a controlled runtime benchmark. The current PSA figures (45.8/13.8/13.7 minutes) concern only 20 rounds and must not be directly compared with historical 100-round totals or multiplied linearly into an asserted final runtime: GP fitting and acquisition workloads change with data size.

## Historical batch resources and the 80 versus 200 question

Read-only verification of the actual completed LSF report `/zhome/15/b/215295/Repositories/ccbo/logs/qmcbo_v3_29365483.out` confirms:

- Submitted 9 September 2026 15:03:13; executed 15:04:23; terminated 23:43:36.
- Runtime 31,153 seconds = 8 h 39 min 13 s; CPU time 465,379.84 seconds = 129.27 CPU hours.
- 16 allocated CPUs, one host `n-62-31-23`, queue `hpc`, requested memory 4 GB per allocated slot (64 GB total), maximum recorded memory 8,488 MB.
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`; command used `~/venvs/mcbo/bin/python`, seeds 0–19, and `scripts/run_qmcbo_pilot.sh results/v3/qmcbo 16`.

The suite script has two loops: E1 is 2 environments × 2 methods × 20 seeds = 80 main runs; E2 repeats those 80 with `--misspec e2`, writing to `_e2`. The old batch therefore covered 160 planned units, of which 80 were the main method comparison. The script supports skipping existing completed units; the completed log contains no skip messages, only the completion banner. This supports, rather than merely assumes, a full batch execution.

The new frozen `server-launch/mcbo_campaign.lsf` requests 1 CPU per unit and caps the array at 2 simultaneous units. Its 200 units are 120 main runs (2 environments × 3 methods × 20 seeds) plus 80 protected-perturbation runs. The extra 40 main runs are the fine mechanism model constrained to the coarse intervention menu, which separates action-menu effects from mechanism coarsening. This is 200 versus 160 for the full suites, or 120 versus 80 for main comparisons. Calling all 200 an expansion from only 80 conflates the protected controls.

The historical 8.65-hour batch time cannot be used as the current elapsed ETA: it used eight times the current concurrency. This does not mean each historical fit had 16 CPUs; the script launched 16 separate single-thread units. This check does not alter campaign scope or scheduling.

## Environment and scoring comparability

At the recorded historical engine SHA, `envs/requirements-mcbo-py39.txt` and `envs/README.md` describe the archived-run environment: Python 3.9.18, torch 1.13.1, BoTorch 0.7.2, GPyTorch 1.9.0, NumPy 1.26.4, SciPy 1.13.1, pandas 2.3.3, wandb 0.26.1. These match the current frozen `releases/mcbo-campaign-v3/envs/mcbo-runtime-provenance.json`. Both identify MCBO vendor revision `0d0650e5c3c104235ae942762eea0f5b36e96b0f`. Historical versions are a committed environment snapshot and documentation, not a per-unit import-attestation record; current units have stronger runtime validation. Processor model/clock and identical binary builds are not established by these records.

The pinned upstream `third_party/mcbo/mcbo/mcbo_trial.py`, function `obj_mean`, already evaluates `function_network(X.repeat(100000, 1, 1))`. Historical code invokes this for initialization and new sampled interventions. Consequently 100,000-draw population scoring is not a newly added workload that explains the slowdown by itself. The corrected scorer has changed RNG separation, scale/target handling and recommendation reporting, so precise equal computational work is not asserted.

The historical search did produce candidate interventions and scored best visited values. That is useful historical evidence, but does not certify that the corrected noisy recommendation protocol has found an optimum. The historical incumbent is population-scored best visited; the new recommendation is chosen from learner-visible measurements and independently scored afterward. Corrected intervention mapping, noise handling and controls also change the experiment. Determining how much each repair costs requires code-level profiling, outside this bounded historical-record check.

## Limits and decision

Confirmed: historical main per-run times, historical total suite count/resources/timestamps, matching recorded key package versions and vendor revision, historical 100k scoring, and lower present concurrency. Not confirmed: a controlled causal attribution of the per-run slowdown, exact future PSA 100-round timing, equivalent processor speed, or optimality of historical recommendations. No jobs were launched and no experimental sources or campaign parameters were changed.
