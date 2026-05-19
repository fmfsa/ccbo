# CCBO — Coarsened Causal Bayesian Optimization

Implementation of CCBO (Coarsened CBO) and RCCBO (Recursive CCBO), plus a
light-tunnel benchmark built on top of the
[`causalchamber`](https://github.com/juangamella/causal-chamber-package)
simulators that demonstrates the headline claim: under a coarsening of the
manipulable variables, intra-cluster DAG misspecifications are **provably
invisible** to the do-calculus identification CCBO uses.

## Repository layout

```
ccbo/
├── adjustment.py             # Backdoor / frontdoor / g-computation on C-DAGs
├── coarsened_graph.py        # CoarsenedGraph wrapper (plugs into CBO)
├── coarsening.py             # Lee-2019 latent projection + POMIS + DAG registry
├── data_generation.py        # Interventional sampling utility
├── generic_do.py             # MC-based do-effect for arbitrary SEMs
├── visualize.py              # DAG / convergence / wrong-edge plots
├── cbo/                      # Vendored CBO (Aglietti 2020) — graphs, utils, data
│   ├── cbo.py                # Main CBO loop
│   ├── graphs/               # GraphStructure subclasses
│   │   ├── CompleteGraph.py
│   │   ├── SimplifiedCoralGraph.py
│   │   └── LightTunnel*.py   # Light-tunnel benchmark (this repo)
│   └── data/
│       └── LightTunnel/
│           ├── generate_observations.py
│           └── observations.pkl
├── rccbo/                    # Recursive CCBO (BO -> RePaRe -> CCBO)
│   ├── rccbo.py
│   ├── repare_bridge.py
│   ├── state_manager.py
│   └── partition_ops.py
├── repare_lib/               # Vendored RePaRe partition-discovery routine
└── tests/
    ├── test_do_accuracy.py
    ├── test_rccbo_determinism.py        # Locks in the seed fixes
    ├── test_lighttunnel_invariance.py   # Structural claim (PASSES)
    └── test_lighttunnel_do_effect.py    # Numerical claim (PASSES)

paper/
├── ccbo_paper.tex
├── ccbo_paper.pdf
├── MFACBO_comparison.md      # Positioning vs Zeitler 2025 (CAR @ UAI)
└── figures/                  # CompleteGraph and LightTunnel figures

run_experiments.py            # CLI entry point for all conditions
generate_paper_figures.py     # Builds paper figures from result pickles
```

## Quickstart

### 1. Environment

```bash
conda create -n ccbo python=3.10
conda activate ccbo
pip install -e .
pip install causalchamber  # for the light-tunnel simulator
```

### 2. Tests (the publishable claims)

```bash
PYTHONPATH=. python ccbo/tests/test_lighttunnel_invariance.py
# Asserts: under the coarse partition, LightTunnel and LightTunnel_WrongRG
# produce byte-identical C-DAGs; under the finest, they differ.

PYTHONPATH=. python ccbo/tests/test_lighttunnel_do_effect.py
# Surfaces the misspec gap numerically: shows which exploration-set entries
# the WrongRG DAG loses and verifies coarse adjustments are byte-identical.

PYTHONPATH=. python ccbo/tests/test_rccbo_determinism.py
# Locks in the seed-handling fixes (A1 audit).
```

### 3. Regenerate observational data (optional)

```bash
PYTHONPATH=. python -m ccbo.cbo.data.LightTunnel.generate_observations
# Writes ccbo/cbo/data/LightTunnel/observations.pkl and asserts corr(R,G) > 0.4.
```

### 4. Multi-seed BO experiments

```bash
# CompleteGraph (Aglietti 2020 toy): main comparison + wrong-edge robustness
python run_experiments.py --benchmark CompleteGraph --condition main      --seeds 5 --trials 40
python run_experiments.py --benchmark CompleteGraph --condition wrong_edge --seeds 5 --trials 40

# Light tunnel — wrong-edge condition with WrongRG misspec
python run_experiments.py --benchmark LightTunnel    --condition wrong_edge --seeds 5 --trials 30
```

Results land in `results/` (per-benchmark pickles). Each CBO trial is heavy
(GPy + emukit gradient-based acquisition); plan for ~minutes per seed at
this budget on a multi-core box.

### 5. Figures

```bash
python generate_paper_figures.py --benchmark CompleteGraph --seeds 5
python generate_paper_figures.py --benchmark LightTunnel   --seeds 5 --only wrong_edge
```

## What's the headline result?

The light-tunnel benchmark uses `causalchamber.simulators.lt.Deterministic`
with manipulable inputs `{R, G, B, P1, P2}` and target `Y = vis_3` (the
sensor reading behind both polarisers). A latent confounder `U_color`
on R and G yields observational `corr(R, G) ≈ 0.68`.

Two DAGs are compared:

* `LightTunnel` — true fine DAG (`R, G, B, P1, P2 → Y`; `U_color → R, G`).
* `LightTunnel_WrongRG` — same plus a *spurious* intra-cluster edge `R → G`.

Under the coarse partition `{R, G, B} | {P1, P2} | {Y}`:

* Lee-2019 latent projection drops the spurious `R → G` edge (both
  endpoints sit inside cluster `{R, G, B}`).
* The resulting C-DAG is *byte-identical* under both fine DAGs.
* CCBO's adjustment formulas are therefore identical → identical GP priors
  → identical BO trajectories under the same seed.

The two tests above prove this both structurally (graph signatures) and
numerically (do-effect values). The multi-seed `wrong_edge` run produces
the convergence / final-Y figures for the paper.

## Companion paper

`paper/ccbo_paper.tex` is the manuscript draft. `paper/MFACBO_comparison.md`
positions this work against Zeitler 2025 (UAI 2025 CAR workshop) — CCBO
coarsens the *inputs* of the DAG; MFACBO coarsens the *outcome* through
multi-fidelity measurements. The two axes are orthogonal.

## License

Research code; no license file shipped. Ask before reuse.
