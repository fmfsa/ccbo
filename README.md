# CCBO — Coarsened Causal Bayesian Optimization

Implementation of CCBO (Coarsened CBO), plus a light-tunnel benchmark built
on top of the
[`causalchamber`](https://github.com/juangamella/causal-chamber-package)
simulators that demonstrates the headline claim: under a coarsening of the
manipulable variables, intra-cluster DAG misspecifications are **provably
invisible** to the do-calculus identification CCBO uses.

RCCBO (Recursive CCBO — online partition discovery via RePaRe) ships in
`ccbo/rccbo/` but is **excluded from the paper experiment matrix**; see
[KNOWN_ISSUES.md](KNOWN_ISSUES.md) for the backlog it must clear first.

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
├── rccbo/                    # Recursive CCBO (excluded from paper; see KNOWN_ISSUES.md)
│   ├── rccbo.py
│   ├── repare_bridge.py
│   ├── state_manager.py
│   └── partition_ops.py
├── repare_lib/               # Vendored RePaRe partition-discovery routine
└── tests/                    # pytest suite (slow tests behind -m slow)
    ├── test_pomis_lb18.py               # POMIS vs LB18 hand-derived truth
    ├── test_seed_handling.py            # Target fn must not touch global RNG
    ├── test_do_accuracy.py              # Do-function RMSE vs true SEM [slow]
    ├── test_rccbo_determinism.py        # RCCBO seed reproducibility [slow]
    ├── test_lighttunnel_invariance.py   # Structural claim
    └── test_lighttunnel_do_effect.py    # Numerical claim [slow]

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
pytest             # fast suite: POMIS vs Lee-Bareinboim 2018 ground truth,
                   # C-DAG invariance under intra-cluster misspecification,
                   # seed-handling contract
pytest -m slow     # adds the heavy validations: numerical misspec gap,
                   # do-function RMSE vs true SEM, RCCBO determinism
```

### 3. Regenerate observational data (optional)

```bash
PYTHONPATH=. python -m ccbo.cbo.data.LightTunnel.generate_observations
# Writes ccbo/cbo/data/LightTunnel/observations.pkl and asserts corr(R,G) > 0.4.
```

### 4. Multi-seed BO experiments

```bash
# Full paper suite (10 seeds x 40 trials, sequential, checkpointed per seed)
./run_full_suite.sh

# Or individual conditions:
python run_experiments.py --benchmark CompleteGraph --condition main        # BO/CBO/GACBO/CCBO
python run_experiments.py --benchmark CompleteGraph --condition wrong_edge  # tier-1 misspec damage
python run_experiments.py --benchmark CompleteGraph --condition sweep       # price-of-coarsening lattice
python run_experiments.py --benchmark LightTunnel   --condition wrong_edge  # tier-2 (arm) damage + invariance
```

Results land in `results/` (per-benchmark pickles); a fairness check
asserts identical trial counts and seed coverage across methods. Each CBO
trial is heavy (GPy + emukit gradient-based acquisition); plan for
~minutes per seed per method.

### 5. Figures and LaTeX tables

```bash
python generate_paper_figures.py --benchmark CompleteGraph
python generate_paper_figures.py --benchmark LightTunnel --only wrong_edge
python generate_paper_figures.py --benchmark CompleteGraph --only tables
# Tables land in paper/tables/ and are \input by paper/ccbo_paper.tex.
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
