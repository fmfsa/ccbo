# QCBO — Quotient Causal Bayesian Optimization

Causal Bayesian optimization with **coarsening as the input contract**: the
practitioner supplies a partition of the manipulable variables and the
cluster-level (quotient) graph; every graph-dependent component of the
optimizer — exploration set, identification, priors — is computed from
the quotient.

> The Python package is named `ccbo/` and the wrapper class is `CoarsenedGraph`
> for historical reasons; these are internal names — the method is QCBO.

The exploration set is the **MIS of the (C-)DAG** (Lee & Bareinboim 2018) —
the same rule CBO applies to the full DAG, so at the identity partition QCBO
*is* CBO. Identifiability from the quotient decides each arm's **prior
tier**, never its membership: identifiable arms get do-calculus priors,
non-identifiable arms share a common uninformative prior (observational
mean/variance of Y) and are learned from experimental data alone.

Three instantiations, each validated against its base method's own
experiments (original authors' stacks):

| Family | Base stack | Quotient layer |
|---|---|---|
| QCBO (arm-based) | vendored `ccbo/cbo` (Aglietti et al. 2020) | `ccbo/coarsened_graph.py` |
| QDCBO (dynamic) | `third_party/DCBO` (github.com/neildhir/DCBO) | `ccbo/qdcbo/` |
| QMCBO (model-based) | `third_party/mcbo` (github.com/ssethz/mcbo) | `ccbo/qmcbo/` |

RCCBO (online partition discovery via RePaRe) ships in `ccbo/rccbo/` but is
**excluded from the paper experiment matrix**; see
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Repository layout

```
ccbo/
├── adjustment.py             # Backdoor / frontdoor / g-computation on C-DAGs
├── coarsened_graph.py        # CoarsenedGraph wrapper (implements QCBO; plugs into CBO)
├── coarsening.py             # Lee-2019 latent projection + MIS/POMIS + graph registry
├── metrics.py                # Final Y / regret (primary) + GAP (secondary)
├── benchmark.py              # ClusterBench10 dataset adapter (see KNOWN_ISSUES TODO)
├── clusterbench10.py         # ClusterBench10 structural spec + perturbation taxonomy
├── cbo/                      # Vendored CBO (Aglietti 2020) — loop, graphs, utils, data
├── qdcbo/                    # QDCBO: quotient layer over the DCBO authors' stack
├── qmcbo/                    # QMCBO: quotient layer over the MCBO authors' stack
├── rccbo/                    # Recursive QCBO (excluded from paper)
└── tests/                    # pytest suite (see below)

third_party/                  # gitignored; fetched by scripts/fetch_*.sh
├── DCBO/                     # neildhir/DCBO  (scripts/fetch_dcbo.sh, pinned)
├── mcbo/                     # ssethz/mcbo    (scripts/fetch_mcbo.sh, pinned)
└── CEO/                      # nicola144/CEO  (scripts/fetch_ceo.sh, pinned)

scripts/                      # runners + table/figure emitters (see scripts/README.md)
run_experiments.py            # bespoke robustness conditions
paper/qcbo_aistats.tex        # the manuscript
```

## Quickstart

### 1. Environment
```bash
conda create -n ccbo python=3.10 && conda activate ccbo
pip install -e .
```
MCBO/QMCBO additionally need a botorch contemporary with the authors' stack
(`FixedNoiseGP`, `botorch.sampling.samplers`); use a second env, e.g.:
```bash
conda create -n mcbo python=3.9
conda run -n mcbo pip install "numpy<2" "torch==1.13.1" "gpytorch==1.9.0" \
    "botorch==0.7.2" pandas scipy matplotlib wandb
```

### 2. Fetch the base stacks
```bash
bash scripts/fetch_dcbo.sh    # -> third_party/DCBO   (QDCBO / DCBO)
bash scripts/fetch_mcbo.sh    # -> third_party/mcbo   (QMCBO / MCBO)
```
The CBO stack is vendored in-repo (`ccbo/cbo`); nothing to fetch.

### 3. Tests
```bash
pytest             # MIS identity anchor, invariance, quotient structure, seeds
pytest -m slow     # adds the numerical prior-tier demonstration
```
Without `third_party/DCBO` / `third_party/mcbo` the QDCBO tests and the
QMCBO joint-mechanism tests **skip** with a pointer to the fetch scripts.

### 4. Family suite (each family on its base method's own experiments)
```bash
# CBO vs QCBO (ToyGraph, CompleteGraph, SimplifiedCoralGraph):
PYTHONPATH=. python scripts/run_cbo_family.py --seeds 10 --trials 40 --jobs 8
# DCBO vs QDCBO (stat / ind / nonstat, 20 seeds):
PYTHONPATH=. bash scripts/run_qdcbo_pilot.sh results/qdcbo 6
# MCBO vs QMCBO (ToyGraph / PSAGraph, 5 seeds; era-pinned env):
PYTHONPATH=. PY="$(conda run -n mcbo which python)" bash scripts/run_qmcbo_pilot.sh results/qmcbo 8
# Aggregate figure + price table:
PYTHONPATH=. python scripts/plot_family_suite.py
PYTHONPATH=. python scripts/emit_family_price_table.py
```

### 5. Robustness conditions + figures
```bash
python run_experiments.py --benchmark CompleteGraph     --condition wrong_edge
python run_experiments.py --benchmark ConfoundedCluster --condition wrong_edge
```

## Headline robustness example (confounded cluster)

Manipulable `{A, B, C}`, target `Y`, latent `U → {B, C}` (so `B ↔ C` with
`corr(B,C) ≈ 0.70`); the strongest lever is `do(B)`. The misspecified variant
`ConfoundedCluster_WrongBC` adds a *spurious* intra-cluster edge `B → C`.
Together with `B ↔ C` this is a **bow**, so `P(Y | do(B))` becomes
non-identifiable and the fine-grained method's best arm `{B}` falls to the
uninformative prior tier. Under the coarse partition `{A} | {B,C} | {Y}` the
bow is intra-cluster: Lee-2019 projection drops `B → C`, the C-DAG is
identical under both DAGs, and QCBO's trajectories coincide exactly. Proven
structurally and numerically in `tests/test_bow_*.py`.

## Companion paper
`paper/qcbo_aistats.tex` is the manuscript (*Quotient Causal Bayesian
Optimization*).

## License
Research code; no license file shipped. Ask before reuse.
