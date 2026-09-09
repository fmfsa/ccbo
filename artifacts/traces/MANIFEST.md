# Raw trace archives behind every number in the paper

Each `<suite>.tar.gz` holds the complete per-seed raw traces for one
experimental suite exactly as produced on the DTU HPC cluster (LSF `hpc`
queue). `<suite>.sha256` lists the SHA256 of every archived file;
`SHA256SUMS` covers the tarballs and checksum lists themselves.

Verification: `scripts/verify_traces.py` re-checks the full expected file
grid, recomputes every published aggregate at the paper's displayed
precision, re-tests the exact-invariance pairs predicted by the quotient
contract, and validates the CEO comparator archive (grid, trajectory
lengths, alias identity and metadata, protocol constants, finals).
Last run 2026-08-21 on a fresh extraction of these archives:
**61 checks, 0 failures**.

```bash
mkdir -p /tmp/qcbo-traces && for t in artifacts/traces/*.tar.gz; do tar -C /tmp/qcbo-traces -xzf "$t"; done
PYTHONPATH=. python scripts/verify_traces.py --results-dir /tmp/qcbo-traces
```

## minimal.tar.gz — MinimalBench (Experiments A–C; Figs. 2–3, Table 1, ablations)

- **Engine v3 (2026-09-09)**: 1320 CSVs (`{scm}_{cond}_{arm}_seed{0..29}.csv`,
  columns `trial,best_y,cum_cost,arm,x_values`) + 1320 `.decisions.json`
  sidecars (full-precision per-trial decision logs, see `ccbo/decision_log.py`)
  + `refine_info.json` (60 records keyed `{scm}_{cond}_{arm}_seed{N}`).
- Grid: ParallelParent × {A0,A1,A2,A3} × {BO,BOS,CBO,CBONP,QCBO,QCBONP};
  FrontDoor × {B0,B1} × same six arms; MediatedChain × C0 × the six arms +
  {HQCBO, HQCBOGF}; 30 seeds.
- Row convention: row 0 is the incumbent of the initial interventional design
  at `cum_cost` = **charged** initial-design cost (3 points per arm, unit
  per-variable costs); HQCBO / HQCBOGF add the split design (3 points per new
  arm) at the trigger row; `x_values` at `repr` precision.
- Generator: `scripts/run_minimal_suite.py` (seeds 0–29, T=50 / 60 on
  MediatedChain, 3 initial points per arm, n_obs=100 from one shared pool,
  batch 20, N_max=150, plateau k=5, δ=1e-3) via
  `scripts/lsf/submit_minimal_v3.sh`; LSF job 29365450, 1320/1320 units OK,
  engine `f2a98e0` (branch `fix/engine-v3`; see `RERUN_ENGINE_V3.md`).
- Engine changes vs the 2026-08-21 archive: integrated front-door /
  g-computation priors, corrected causal-prior gradients, CBO-faithful prior
  variance, GP noise fixed at 1e-10, exact cache keys, tri-state ID gate,
  charged cost axis, graph-free refinement arm, no-prior and all-subsets
  ablation arms. Observational samples remain stochastic; intervention
  targets are exact population expectations.
- Invariance: QCBO and QCBONP identical at full decision-log precision under
  A1, A2, B1 (QCBONP also under A3); BO and BOS byte-identical across every
  condition of an SCM.

## family_cbo.tar.gz — CBO vs QCBO on the CBO family (Fig. 4 top, Table 2)

- 90 CSVs (`{ToyGraph,CompleteGraph,SimplifiedCoralGraph}_{BO,CBO,QCBO}_seed{0..9}.csv`,
  columns `trial,best_y,cum_cost`; observe trials are the rows with zero
  `cum_cost` increment).
- Generator: `scripts/run_cbo_family.py` (10 seeds, 40 trials, 10 initial
  interventional points per arm, 100 obs, unit cost, minimization) via
  `scripts/lsf/submit_family_v2.sh`.
- Source: rerun after the online-observation protocol fix
  (see `RERUN_ONLINE_OBS.md`), plus the new plain-BO baseline arm
  (one joint arm over all manipulable variables, no causal prior, no
  observation actions). LSF job 29163302 completed 2026-08-21,
  90/90 units OK, repo @ `d23022e`.

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

- **Extended 5 → 20 seeds on 2026-08-11** (LSF job 29078186, 120 new
  units, zero failures; identical protocol, verified field-by-field
  against the original `_info.json` sidecars; the original five seeds
  are byte-identical to the 2026-07-22 archive and still pass
  `verify_qmcbo_release.py`).
- E1 and `_e2/` each holding
  `trial_results_{MCBO,QMCBO}_{ToyGraph,PSAGraph}_{0..19}.csv` (columns
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

Tarballs packed with
`tar --sort=name --owner=0 --group=0 --numeric-owner --mtime=<pinned per suite> -czf`
for deterministic re-packing (qdcbo packed 2026-08-10, qmcbo extended
2026-08-11, ceo_minimal corrected 2026-08-11; minimal and family_cbo
repacked 2026-08-21 from the online-observation rerun with
`--mtime='2026-08-21 00:00Z'`). All archives are covered by both
per-file and archive checksums.

## ceo_minimal.tar.gz — CEO comparator on MinimalBench (corrected 2026-08-11)

- **Supersedes the 2026-08-10 run (LSF job 29078320), which was
  invalidated by PR-review finding 1**: the adapter's target factory
  ignored CEO's `noisy` flag, so the graph posterior and GP targets
  learned from exact mean pseudo-observations instead of valid
  stochastic interventional samples. The corrected adapter draws the
  noisy path from the true latent SEM (guarded by
  `ccbo/tests/test_ceo_minimal_protocol.py`); rerun as LSF job
  29090857, 150/150 elements, zero failures, 2026-08-11. The flawed
  archive remains in git history.
- 210 CSVs (`CEO_{scm}_{cond}_seed{0..29}.csv`, columns
  `trial_number,current_optimal`) + `.meta.json` sidecars (pool, final
  graph posterior, per-trial intervention costs and exploration sets,
  wall time). 150 executed units (ParallelParent × {A0,A1,A2},
  FrontDoor × B0, MediatedChain × C0) + 60 documented
  observable-identical aliases (A3←A0, B1←B0, alias `cond` set to the
  alias name with `alias_of` preserving the source).
- Generator: `scripts/run_ceo_minimal.py` over `third_party/CEO` @
  `35dd277`: seeds 0–29, T=50/60, 3 init points per arm, n_obs=100
  shared dataset; noiseless objective = closed-form E[Y|do] seam-gated
  per run against 200k-sample MC of the true latent SEMs; noisy
  learning path = stochastic true-latent-SEM draws, via
  `scripts/lsf/submit_ceo_minimal.sh`.
- Aggregation: `scripts/emit_ceo_comparison.py` →
  `paper/tables/ceo_minimal.tex` + `artifacts/summaries/ceo_comparison.json`.
