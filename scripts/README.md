# `scripts/` — QCBO experiment pipeline

Run everything from the repo root with the `ccbo` conda env and the repo on the path:

```bash
PYTHONPATH=$PWD conda run -n ccbo python scripts/<script>.py ...
```

`third_party/CausalBO_Benchmark/` and `results/` are gitignored; in a worktree, symlink the
benchmark from the main checkout: `ln -s /home/fmfsa/Repositories/ccbo/third_party third_party`.

## DAG-misspecification stress test (ClusterBench10)

The robustness experiment: the field (CBO/CEO/CoCaBO) degrades under a misspecified edge while
QCBO-coarse is byte-identical, with the objective held fixed (only the *structure* each method
reasons with is perturbed). Structural spec: [`ccbo/clusterbench10.py`](../ccbo/clusterbench10.py).

Run in order:

1. **`generate_clusterbench10.py`** — build the 10-variable benchmark dataset (SEM equations,
   adjacency, observational CSV, feature params, theoretical-best, BO/CBO interventional pkls)
   into `third_party/CausalBO_Benchmark/`, and run the Monte-Carlo **oracle checks** (do(X1) is the
   strongest singleton; global optimum is a union of clusters ⇒ coarse partition lossless).
2. **`misspec_inject.py`** — structure-injection shim: perturbs `adjacency_matrix` for the external
   baselines while leaving the objective untouched. `python scripts/misspec_inject.py` runs the
   `seam_ok` guard (objective must be byte-identical under every perturbation).
3. **`run_misspec_fullfield.py`** — the driver. Runs each method (QCBO-finest/coarse, CBO, …) under
   the correct DAG and every perturbation, computes paired-by-seed Δ (final-Y, GAP), asserts
   QCBO-coarse byte-identity on the protected (quotient-invisible) perturbations, and writes
   `results/clusterbench10_misspec.json`.
   Example: `... run_misspec_fullfield.py --seeds 10 --trials 100 --cap 1` (Tier-2 headline);
   add `--cap 5` for the uncapped field comparison.
4. **`emit_misspec_table.py`** — score + emit the paired-Δ head-to-head table and the conditions
   taxonomy into `paper/tables/`. *(pending)*

Structural premise (no dataset needed): `pytest ccbo/tests/test_clusterbench10_structure.py`.

## Standardized CausalBO benchmark (sample-efficiency / main comparison)

- `run_qcbo_benchmark_suite.py` / `run_qcbo_benchmark_parallel.py` — QCBO on the 6 curated datasets.
- `run_baselines_suite.py` — the benchmark's own baselines (BO/CBO/CEO/CoCaBO/DCBO/MCBO).
- `emit_baseline_table.py` / `emit_benchmark_table.py` — score + emit `paper/tables/benchmark_*.tex`.
- `run_qcbo_coarsening_sweep.py` / `emit_coarsening_sweep_table.py` — sweep the coarsening lattice.
