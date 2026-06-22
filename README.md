# QCBO — Quotient Causal Bayesian Optimization

Quotient CBO (QCBO) drops CBO's requirement of a fully specified DAG over the
manipulable variables. The practitioner supplies only a **partition** of the
manipulable variables into clusters plus the cluster-level graph (a C-DAG);
identification, exploration-set construction, and the causal GP prior are all
derived from the **quotient** structure via Lee-2019 latent projection + the
complete ID algorithm (`ananke`). Headline guarantee: any DAG misspecification
**within** a cluster is provably invisible to QCBO, because it never appears in
the quotient.

> The Python package is named `ccbo/` and the wrapper class is `CoarsenedGraph`
> for historical reasons; these are internal names — the method is QCBO.

We evaluate QCBO on a curated subset of the standardized **CausalBO benchmark**
(`anonymous.4open.science/r/CausalBO_Benchmark`), scored with its own GAP /
PA-GAP metrics, plus a purpose-built **confounded-cluster** benchmark that
isolates the Tier-2 (arm-deletion) damage regime.

RCCBO (Recursive QCBO — online partition discovery via RePaRe) ships in
`ccbo/rccbo/` but is **excluded from the paper experiment matrix**; see
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Repository layout

```
ccbo/
├── adjustment.py             # Backdoor / frontdoor / g-computation on C-DAGs
├── coarsened_graph.py        # CoarsenedGraph wrapper (implements QCBO; plugs into CBO)
├── coarsening.py             # Lee-2019 latent projection + POMIS + graph registry
├── metrics.py                # GAP / PA-GAP (match the CausalBO benchmark scorer)
├── benchmark.py              # Adapter: run QCBO on CausalBO_Benchmark datasets
├── data_generation.py        # Interventional sampling utility
├── generic_do.py             # MC-based do-effect for arbitrary SEMs
├── cbo/                      # Vendored CBO (Aglietti 2020) — loop, graphs, utils, data
│   └── graphs/
│       ├── CompleteGraph.py
│       └── ConfoundedCluster*.py   # Tier-2 "bow" benchmark (this repo)
├── rccbo/                    # Recursive QCBO (excluded from paper; see KNOWN_ISSUES.md)
└── tests/
    ├── test_pomis_lb18.py              # POMIS vs Lee-Bareinboim 2018 ground truth
    ├── test_seed_handling.py           # Target fn must not touch global RNG
    ├── test_bow_invariance.py          # Tier-2 structural claim (bow benchmark)
    └── test_bow_do_effect.py           # Tier-2 numerical claim [slow]

third_party/CausalBO_Benchmark/   # downloaded benchmark (datasets, SEMs, scorer)
scripts/
├── confoundedcluster_arm_values.py     # ground-truth arm values for the bow benchmark
├── validate_benchmark_qcbo.py          # builds CoarsenedGraph on each benchmark dataset
└── run_qcbo_benchmark_suite.py         # runs + scores QCBO on the curated subset
run_experiments.py            # bespoke conditions (CompleteGraph Tier-1; bow Tier-2)
generate_paper_figures.py     # builds paper figures/tables from results
```

## Quickstart

### 1. Environment
```bash
conda create -n ccbo python=3.10 && conda activate ccbo
pip install -e .
```

### 2. Get the benchmark
Download `CausalBO_Benchmark` into `third_party/` (it uses the same
GPy/emukit stack; no extra runtime deps are needed for the curated subset).

### 3. Tests
```bash
pytest             # fast: POMIS vs LB18, bow-benchmark C-DAG invariance, seed contract
pytest -m slow     # adds the numerical Tier-2 do-effect demonstration
```

### 4. QCBO on the CausalBO benchmark
```bash
# Build the C-DAG + gated exploration set for every curated dataset (sanity):
PYTHONPATH=. python scripts/validate_benchmark_qcbo.py
# Run QCBO (finest = CBO; and coarse) and score with the benchmark's GAP/PA-GAP:
PYTHONPATH=. python scripts/run_qcbo_benchmark_suite.py --trials 100 --seeds 5
# -> results/qcbo_benchmark_results.json
```
Curated subset: `toyGraph`, `synthetic` (=CompleteGraph), `synthetic_2`,
`healthcare`, `epidemiology`, `ecology`. Each gets the identity partition
(QCBO-finest ≡ CBO, Prop. 1) and a domain-meaningful coarse partition.

### 5. Bespoke robustness conditions + figures
```bash
python run_experiments.py --benchmark CompleteGraph     --condition wrong_edge  # Tier-1 (prior bias)
python run_experiments.py --benchmark ConfoundedCluster --condition wrong_edge  # Tier-2 (arm deletion)
python generate_paper_figures.py --benchmark CompleteGraph --only tables
```

## Headline result (Tier-2, confounded-cluster benchmark)

Manipulable `{A, B, C}`, target `Y`, latent `U → {B, C}` (so `B ↔ C` with
`corr(B,C) ≈ 0.70`); the strongest lever is `do(B)`. The misspecified variant
`ConfoundedCluster_WrongBC` adds a *spurious* intra-cluster edge `B → C`.
Together with `B ↔ C` this is a **bow**, so `P(Y | do(B))` becomes
non-identifiable and the fine-grained method **loses its best target `{B}`**
(ground-truth arms: `{B}`=−3.75, best survivor `{C}`=−2.75 → +1.0 unrecoverable
gap). Under the coarse partition `{A} | {B,C} | {Y}` the bow is intra-cluster:
Lee-2019 projection drops `B → C`, the C-DAG is byte-identical under both DAGs,
and QCBO's trajectories coincide exactly. Proven structurally and numerically
in `tests/test_bow_*.py`.

## Companion paper
`paper/ccbo_paper.tex` is the manuscript (title: *Quotient Causal Bayesian
Optimization*). `paper/MFACBO_comparison.md` positions this work against
Zeitler 2025 — QCBO coarsens the DAG's *inputs*; MFACBO coarsens the *outcome*.

## License
Research code; no license file shipped. Ask before reuse.
