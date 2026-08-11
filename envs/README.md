# Environment snapshots for the paper's results

The two requirement files are **`pip freeze` snapshots of the exact
environments that produced and verified the archived traces** — a record of
a working configuration, not a portable lock file. Resolving them on another
platform may pick different builds; treat them as the authoritative version
record and reconstruct as below.

| File | Environment | Python | Role |
|---|---|---|---|
| `requirements-ccbo-py310.txt` | `~/venvs/ccbo` | 3.10.18 | main stack: QCBO/CBO/QDCBO runners, MinimalBench, emitters, tests |
| `requirements-mcbo-py39.txt` | `~/venvs/mcbo` | 3.9.18 | era-pinned MCBO stack (torch 1.13.1, gpytorch 1.9.0, botorch 0.7.2) for MCBO/QMCBO |

Platform of record (where all archived traces were produced and verified):
DTU HPC login/compute nodes, Linux 6.1.180-1.el9.elrepo.x86_64 x86_64,
glibc 2.34, LSF `hpc` queue. LaTeX: TeX Live 2024, latexmk 4.85.

## Key pins and why

- `numpy<2` (pyproject): `third_party/CEO` uses `numpy.core.multiarray` /
  `np.PINF`. Snapshot has numpy 1.26.4.
- `jax`/`jaxlib` 0.4.38 arrive **transitively via `ananke-causal` 0.5.0**,
  which implements the identifiability gate (`ccbo/adjustment.py`,
  `ccbo/coarsening.py`) that decides prior tiers. They are otherwise
  unpinned in `pyproject.toml` — use the snapshot versions.
- External stacks are fetched by `scripts/fetch_{dcbo,mcbo,ceo}.sh` and
  SHA-pinned: DCBO `85a9bdf`, MCBO `0d0650e`, CEO `35dd277`.

## Reconstruction commands

```bash
# main env (Python 3.10)
python3.10 -m venv ~/venvs/ccbo
~/venvs/ccbo/bin/pip install -r envs/requirements-ccbo-py310.txt
~/venvs/ccbo/bin/pip install -e .          # this repo

# era MCBO env (Python 3.9)
python3.9 -m venv ~/venvs/mcbo
~/venvs/mcbo/bin/pip install -r envs/requirements-mcbo-py39.txt

# external stacks (SHA-pinned)
bash scripts/fetch_dcbo.sh && bash scripts/fetch_mcbo.sh && bash scripts/fetch_ceo.sh
```

`pygraphviz` requires system graphviz headers. Pin BLAS threads when
running suites (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`).

## Verification baseline (2026-08-10)

- `PYTHONPATH=. ~/venvs/ccbo/bin/python -m pytest` (default `-m 'not slow'`):
  **63 passed, 11 skipped, 11 deselected** as of 2026-08-11 (the
  pre-revision baseline of 52 passed plus the 11 CEO-protocol tests in
  `test_ceo_minimal_protocol.py`; skips depend on optional third-party
  stacks being fetched, so compare like-for-like).
- `latexmk -pdf paper/qcbo_aistats.tex`, post-revision state: clean
  build, zero undefined references, zero overfull boxes, 22 pages,
  references beginning on PDF p. 10, no rendered venue-branding strings.
  (Pre-revision: 17 pages with a 207-character near-empty p. 8 — since
  eliminated.)

## Open item

An untracked `uv.lock` (~835 KB) exists in a separate clone at
`/Users/fmfsa/Repositories/ccbo/uv.lock` (maintainer's machine). Its origin
is not established; it did not produce the archived traces (those came from
the venvs above). If it captures a relevant environment, add it and its
provenance here; otherwise record it as unrelated.
