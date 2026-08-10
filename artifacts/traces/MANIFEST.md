# Raw trace archives behind every number in the paper

Each `<suite>.tar.gz` holds the complete per-seed raw traces for one
experimental suite exactly as produced on the DTU HPC cluster (LSF `hpc`
queue). `<suite>.sha256` lists the SHA256 of every archived file;
`SHA256SUMS` covers the tarballs and checksum lists themselves.

Verification: `scripts/verify_traces.py` re-checks the full expected file
grid, recomputes every published aggregate at the paper's displayed
precision, and re-tests the exact-invariance pairs predicted by the quotient
contract. Last run 2026-08-10 on both the original directories and a fresh
extraction of these archives: **38 checks, 0 failures**.

```bash
mkdir -p /tmp/qcbo-traces && for t in artifacts/traces/*.tar.gz; do tar -C /tmp/qcbo-traces -xzf "$t"; done
PYTHONPATH=. python scripts/verify_traces.py --results-dir /tmp/qcbo-traces
```

## minimal.tar.gz — MinimalBench (Experiments A–C; Figs. 2–3, Table 1)

- 450 CSVs (`{scm}_{cond}_{arm}_seed{0..29}.csv`, columns
  `trial,best_y,cum_cost,arm,x_values`) + `refine_info.json` (30 records,
  `trigger_step`/`split_clusters`/`refused`).
- Grid: ParallelParent × {A0,A1,A2,A3} × {CBO,QCBO}; FrontDoor × {B0,B1} ×
  {CBO,QCBO}; MediatedChain × C0 × {CBO,QCBO,HQCBO}; 30 seeds.
- Generator: `scripts/run_minimal_suite.py` (defaults: seeds 0–29, T=50
  trials — 60 on MediatedChain, 3 initial interventional points per arm,
  n_obs=100 from one shared dataset per SCM, per-variable unit costs;
  HQCBO plateau trigger k=5, δ=1e-3).
- Source: worktree `quotient-causal-optimization-paper-660283` @ `b02762b`,
  file mtimes 2026-07-23.
- Note: row 0 of each CSV is the incumbent of the initial interventional
  design **at cost 0** — `cum_cost` excludes initialization
  (see `ccbo/minimal_suite.py` docstring).

## family_cbo.tar.gz — CBO vs QCBO on the CBO family (Fig. 4 top, Table 2)

- 60 CSVs (`{ToyGraph,CompleteGraph,SimplifiedCoralGraph}_{CBO,QCBO}_seed{0..9}.csv`,
  columns `trial,best_y,cum_cost`).
- Generator: `scripts/run_cbo_family.py` (10 seeds, 40 trials, 10 initial
  interventional points per arm, 100 obs, unit cost, minimization) via
  `scripts/lsf/submit_family.sh`.
- Source: worktree `benchmark-suite-paper-refresh-9b33cd` @ `9d3c543`,
  file mtimes 2026-07-22.

## qdcbo.tar.gz — DCBO vs QDCBO (Fig. 4 middle, Table 2)

- 720 files: 120 E1 CSVs + sidecar `_info.json` + run logs, and the same
  for E2 under `_e2/`
  (`{dcbo,qdcbo}_{stat,ind,nonstat}[_e2]_best_so_far_seed{0..19}_T3_trials10_reps1.csv`,
  columns `method,replicate,time_index,trial_index,best_so_far_value,y`).
- E2 = intra-slice perturbation of the model's DAG view, objective fixed.
- Generator: `scripts/run_qdcbo_pilot.sh` → `ccbo.qdcbo.runner` over
  `third_party/DCBO` @ `85a9bdf`, via `scripts/lsf/submit_qdcbo.sh`.
- Source: worktree `benchmark-suite-paper-refresh-9b33cd` @ `9d3c543`,
  file mtimes 2026-07-22.

## qmcbo.tar.gz — MCBO vs QMCBO (Fig. 4 bottom, Tables 2/3/5)

- 120 files: E1 and `_e2/` each holding
  `trial_results_{MCBO,QMCBO}_{ToyGraph,PSAGraph}_{0..4}.csv` (columns
  `trial_number,current_optimal`) + `_info.json` sidecars + run logs.
- E2 ops: ToyGraph `del 0→1`, PSAGraph `add 2→3` (model's graph view only).
- Generator: `scripts/run_qmcbo_pilot.sh` → `ccbo.qmcbo.runner` over
  `third_party/mcbo` @ `0d0650e`, T=100, β=10, zero observation noise, on
  the era-pinned MCBO stack (Python 3.9, torch 1.13.1, gpytorch 1.9.0,
  botorch 0.7.2), via `scripts/lsf/submit_qmcbo.sh`.
- Source: worktree `benchmark-suite-paper-refresh-9b33cd` @ `9d3c543`,
  file mtimes 2026-07-22.
- `scripts/verify_qmcbo_release.py` additionally validates the five
  ToyGraph MCBO records against the published mean
  1.1433154226418354 ± 0.27128671442865404.

## Packing

Tarballs created 2026-08-10 with
`tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='2026-07-23 00:00Z' -czf`
for deterministic re-packing. Apparent raw size 4.41 MB (directory `du`
is dominated by filesystem block overhead on many small files).

## ceo_minimal.tar.gz — CEO comparator on MinimalBench (added 2026-08-10)

- 210 CSVs (`CEO_{scm}_{cond}_seed{0..29}.csv`, columns
  `trial_number,current_optimal`) + `.meta.json` sidecars (pool, final
  graph posterior, wall time). 150 executed units (ParallelParent ×
  {A0,A1,A2}, FrontDoor × B0, MediatedChain × C0) + 60 documented
  observable-identical aliases (A3←A0, B1←B0).
- Generator: `scripts/run_ceo_minimal.py` over `third_party/CEO` @
  `35dd277` (prespecified protocol: seeds 0–29, T=50/60, 3 init points
  per arm, n_obs=100 shared dataset, closed-form E[Y|do] objective
  seam-gated per run against 200k-sample MC of the true latent SEMs),
  via `scripts/lsf/submit_ceo_minimal.sh`, LSF job 29078320, 150/150
  elements completed with zero failures, 2026-08-10.
- Aggregation: `scripts/emit_ceo_comparison.py` →
  `paper/tables/ceo_minimal.tex` + `artifacts/summaries/ceo_comparison.json`.
