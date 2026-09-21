# Coarsened Causal Bayesian Optimization

Models and experiments for *Coarse Causal Graphs for Bayesian Optimization*.

- `ccbo/`: QCBO, QMCBO, QDCBO, their CBO backends, and model tests.
- `experiments/`: paper configurations, one runner, and result analysis.
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
bash scripts/fetch_ceo.sh
bash scripts/fetch_dcbo.sh

python3.9 -m venv .venv-mcbo
.venv-mcbo/bin/pip install -r requirements-mcbo.txt
bash scripts/fetch_mcbo.sh
```

The requirements pin the numerical packages used by the experiments, rather than
an entire development machine. No GPU or HPC scheduler is required. Use the
corresponding environment's Python for each command below.

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
| Static | 24 × 30 = 720 | Paid intervention cost 100 or 120 |
| MCBO | 10 × 20 = 200 | 100 optimization rounds |
| Dynamic | 9 × 20 = 180 | Three slices, 10 trials per slice |

Static measurements return the exact population expectation E[Y | do(X = x)]
of the true SCM, the evaluation convention of the CBO reference implementation;
observational data are sampled. `experiments/configs/*.json` contains the complete settings. MCBO includes the
matched action-menu control and protected graph edits. Dynamic experiments use
1,024 predictive particles, 2,048 feedback draws, and 100,000 scoring draws; the
regularized nonstationary SCM is explicitly named.

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

## Test

```sh
.venv-static/bin/python -m pytest
.venv-mcbo/bin/python -m pytest ccbo/tests/test_qmcbo.py ccbo/tests/test_qmcbo_corrected.py experiments/tests
# Exercise actual static/CEO integration after fetching the backends:
MATCHED_INTEGRATION=1 .venv-static/bin/python -m pytest ccbo/tests/test_matched_*.py
```

Unavailable optional backends are reported as skips. Additional expensive checks
are marked `slow` and can be selected with `-m slow`.

The CBO implementation derives from Aglietti et al. (2020). External
[CEO](https://github.com/nicola144/CEO), [DCBO](https://github.com/neildhir/DCBO),
and [MCBO](https://github.com/ssethz/mcbo) code is fetched at the revisions in the
fetch scripts; retain those projects' license and attribution files.
