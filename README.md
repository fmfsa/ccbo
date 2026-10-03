# Coarsened Causal Bayesian Optimization

Models and experiments for *Coarse Causal Graphs for Bayesian Optimization*.

- `ccbo/`: QCBO, HQCBO, QMCBO and QDCBO, their CBO backends, and model tests.
- `experiments/`: paper configurations, one runner, result analysis and the
  figure/table report.
- `scripts/`: backend entry points and pinned external-code fetchers.

## Install

Run from this repository's root. Static and dynamic experiments use Python 3.10;
MCBO uses a separate Python 3.9 environment because its upstream GP APIs differ.
Graphviz and its development headers are required for `pygraphviz` (for example,
`apt install graphviz libgraphviz-dev` on Debian/Ubuntu or `brew install graphviz`
on macOS).

```sh
python3.10 -m venv .venv-static
.venv-static/bin/pip install -r requirements-static.txt
bash scripts/fetch_dcbo.sh

python3.9 -m venv .venv-mcbo
.venv-mcbo/bin/pip install -r requirements-mcbo.txt
bash scripts/fetch_mcbo.sh
```

On macOS with Homebrew, `pygraphviz` may need the Graphviz paths:
`CFLAGS="-I$(brew --prefix graphviz)/include" LDFLAGS="-L$(brew --prefix graphviz)/lib"`
before the first `pip install`.

The requirements pin the numerical packages used by the experiments, rather than
an entire development machine. All experiments run on CPUs; no GPU or HPC
scheduler is required. Use the corresponding environment's Python for each
command below.

## Run

```sh
# Inspect settings without importing a model.
.venv-static/bin/python experiments/run.py --suite static --list

# Short wiring check, saved separately from paper results.
.venv-static/bin/python experiments/run.py --suite static --smoke --index 0 --outdir results/check

# Full paper experiments; choose --jobs to match available CPUs and memory.
.venv-static/bin/python experiments/run.py --suite static --jobs 8 --outdir results/reproduction
.venv-mcbo/bin/python experiments/run.py --suite mcbo --jobs 8 --outdir results/reproduction
.venv-static/bin/python experiments/run.py --suite dynamic --jobs 8 --outdir results/reproduction
```

| Suite | Configurations × seeds | Budget |
|---|---:|---|
| Static | 21 × 30 = 630 | Paid intervention cost 100 or 120 |
| MCBO | 10 × 20 = 200 | 100 optimization rounds |
| Dynamic | 9 × 20 = 180 | Three slices, 10 trials per slice |

`experiments/configs/*.json` contains the complete settings. The MCBO suite
includes the matched action-menu control and four graph-edit configurations (IDs ending
in `-e2`). The graph-edit units are not plotted; the analysis uses them to check
that QMCBO's executed trace is unchanged when the fine graph is edited inside a
cluster. Dynamic experiments use 1,024 predictive particles, 2,048 feedback
draws, and 100,000 scoring draws; the regularized nonstationary SCM is
explicitly named.

The static suite uses protocol `matched-controlled-cbo-v4`, which
follows the CBO reference implementation (Aglietti et al., 2020):
- Every measurement returns the exact population value E[Y | do(x)]. The reference implementation averages 100,000 SCM draws; we use closed forms.
- Each arm starts with 3 uniformly drawn initial interventions that are given, not charged.
- Every trial either observes or intervenes, using CBO's coverage-based ε rule: 100 initial observational rows, 20 more per observe trial, capped at 150. Observing is free.

Runs stop at a fixed intervention-cost budget, counted from 0. Trial indices are logged for a trial-indexed view. The recommendation is the executed intervention with the smallest measured target, or no intervention when the observational mean of the target over the initial rows is strictly smaller. Exposed refinement arms start from a plain prior with no data.

Use `--seed 2000` for one replicate across configurations, or `--index 0` for one
unit in the listed matrix. Each worker uses one CPU. Completed runs resume only
when settings, code identity and output hashes match; incomplete runs are
preserved. Independent processes must use disjoint indices. Smoke runs use pilot
seeds and fewer decisions and are not paper evidence.

## Analyze

```sh
.venv-static/bin/python experiments/analyze.py --suite static --results results/reproduction --out results/static.json
.venv-mcbo/bin/python experiments/analyze.py --suite mcbo --results results/reproduction --out results/mcbo.json
.venv-static/bin/python experiments/analyze.py --suite dynamic --results results/reproduction --out results/dynamic.json
```

Analysis checks measurements, costs, recommendations, independent scoring and
protected comparisons, then writes JSON and a seed-level CSV. Paired confidence
intervals require the complete declared matrix. Subsets and `--smoke` outputs
receive descriptive summaries only.

## Reproducing the paper

After the three suites have finished, one command audits every suite and
regenerates all figures and tables:

```sh
.venv-static/bin/python experiments/report.py --results results/reproduction --out results/paper_outputs
```

The report refuses to write paper outputs unless every suite passes the audit
and its declared matrix is complete. Output file names match the manuscript
sources:

| Paper | Output | Suite | Content |
|---|---|---|---|
| Figure 2 | `figures/cost_indexed.pdf` | static | ParallelParent and FrontDoor incumbent objective against cost |
| Appendix | `figures/trial_indexed.pdf` | static | The same, against optimizer trials (observe or intervene) |
| Figure 3 | `figures/minimal_refine.pdf` | static | MediatedChain incumbent objective and cumulative regret |
| Figure 4 | `figures/family_suite.pdf` | dynamic, mcbo | Base, matched-exploration-set and quotient methods |
| Table 1 | `tables/family_effects.tex` | dynamic, mcbo | Final dynamic and model-based outcomes |
| Table 2 | `tables/minimal_taxonomy.tex` | static | Controlled conditions; written only if protected QCBO traces match |
| Table 3 | `tables/minimal_ablations.tex` | static | Final and cumulative incumbent regret |
| Table 4 | `tables/cost_accounting.tex` | static | Free initial points, observe trials, sequential, refinement and unspent budget |

Cumulative regret sums the recommendation's regret at every integer cost from 0
to the budget. All intervals are two-sided 95%
Student-t intervals over paired seeds. Use `--suites static` to
report the controlled suite alone, and `--smoke` to check the
pipeline on pilot results (for example, `--results results/check`); smoke outputs
are not paper evidence.

## Test

```sh
.venv-static/bin/python -m pytest
.venv-mcbo/bin/python -m pytest ccbo/tests/test_qmcbo.py ccbo/tests/test_qmcbo_corrected.py experiments/tests
# Exercise actual static integration after fetching the backends:
MATCHED_INTEGRATION=1 .venv-static/bin/python -m pytest ccbo/tests/test_matched_*.py
```

Unavailable optional backends are reported as skips. Additional expensive checks
are marked `slow` and can be selected with `-m slow`.

## Citation

```bibtex
@inproceedings{anonymous2027coarse,
  title     = {Coarse Causal Graphs for Bayesian Optimization},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2027}
}
```

## License

The code written for this project is released under the MIT License (see
`LICENSE`).

`ccbo/cbo/` is adapted from the CBO reference implementation of Aglietti et al.
(2020), <https://github.com/VirgiAgl/CausalBayesianOptimization>. That repository
does not include a license, so the MIT License does not cover this directory;
see `ccbo/cbo/NOTICE.md`.

The DCBO (<https://github.com/neildhir/DCBO>) and MCBO
(<https://github.com/ssethz/mcbo>) reference implementations are released under
the MIT License. They are not redistributed here: `scripts/fetch_dcbo.sh` and
`scripts/fetch_mcbo.sh` clone them at pinned commits, together with their
license and attribution files.
