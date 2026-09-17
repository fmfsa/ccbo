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

## Local checkout

The tracked `uv.lock` is retained for local environment resolution. Historical
result provenance comes from the two server snapshots above, not from that lock.
Run the current tests rather than relying on historical test counts. Exporting
results now uses `scripts/reproduce_results.sh` and does not require LaTeX.
