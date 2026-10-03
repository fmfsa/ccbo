# v3 static and ClusterChain results — MANIFEST

> **Status: ClusterChain audit NOT clean.** 209/210 protected checks pass; `ClusterChain-K3-QCBO-alt-seed2020`
> fails with `Protected executed trace differs` (see "Known issue" below). Because of this, `report.py` refuses to
> write ClusterChain figures/tables and `clusterchain.json` has `status: descriptive_only` (no paired effects).
> `paper_outputs/` contains the **static** outputs only, produced with `report.py --suites static`.
> The static suite is complete and passes all checks.

## Code
- Git commit: `a4f360ccce8117e2cb9b2bbdb42ca688c8b8d56d` (branch `exp/review-null-refine-clusterchain`)
- `code_identity` (all 1,470 `unit.json`): `61b4757aedf08761a1b409a464abc581e58d3447b51f0ac6921a8f0b97a8fc51`
- Protocol: `matched-controlled-noisy-v3`, seed namespace `matched-controlled-noisy-v2`

## Machine
- Date: 2026-10-03 (runs 00:51–02:33 CEST)
- CPU: AMD Ryzen 9 9950X 16-Core Processor (16 physical cores, 32 threads); 123 GB RAM
- OS: Ubuntu 24.04.2 LTS, kernel 6.17.0-35-generic
- Python 3.10.21 (conda-provided interpreter; no system python3.10 on this server), venv `.venv-static`
  built with `python3.10 -m venv .venv-static && .venv-static/bin/pip install -r requirements-static.txt`

## Tests
- `.venv-static/bin/python -m pytest -q`: 221 passed, 37 skipped, 9 deselected
- `MATCHED_INTEGRATION=1 ... test_matched_protocol.py test_matched_refinement.py test_matched_fallback.py`: 19 passed
- Smoke (`results/smoke_v3`, both suites): `"failures": []`, protected checks 2/2 and 7/7 equal

## Commands (`--jobs 14` = 16 physical cores − 2; suites run sequentially)
```
.venv-static/bin/python experiments/run.py --suite static --jobs 14 --outdir results/v3
.venv-static/bin/python experiments/run.py --suite clusterchain --jobs 14 --outdir results/v3
.venv-static/bin/python experiments/analyze.py --suite static --results results/v3 --out results/v3/static.json
.venv-static/bin/python experiments/analyze.py --suite clusterchain --results results/v3 --out results/v3/clusterchain.json
.venv-static/bin/python experiments/report.py --results results/v3 --out results/v3/paper_outputs --suites static clusterchain   # refused: clusterchain audit failed
.venv-static/bin/python experiments/report.py --results results/v3 --out results/v3/paper_outputs_static --suites static       # -> paper_outputs/ here
tar -czf results_v3_raw.tar.gz -C results/v3 paper
```

| Suite | Units | Wall clock | Total CPU (sum `wall_seconds`) |
|---|---|---|---|
| static | 630/630 complete | 343 s (5 min 43 s) | 4,073 s (1.13 h) |
| clusterchain | 840/840 complete | 5,716 s (1 h 35 min 16 s) | 78,666 s (21.85 h) |

## Audit
- static: `status: complete`, 630/630 valid, `failures: []`, protected checks 60/60 `exact_equal: true`
- clusterchain: `status: descriptive_only`, 840/840 valid, protected checks 209/210 `exact_equal: true`;
  failure `ClusterChain-K3-QCBO-alt-seed2020: Protected executed trace differs`

## Known issue
K0 and K3 traces for QCBO-alt seed 2020 are identical up to event 35; at event 36 the acquisition-optimized point
differs at ~1e-9 (A1 2.1474274773511906 vs 2.1474274741355357), after which the trajectories diverge (226 vs 222
purchases; final regret 1.2084 vs 1.1601). Four reruns of the backend (K0 ×2, K3 ×2, runner environment, outside
`results/v3`) all reproduce the stored K0 trace byte-for-byte. The stored K3 unit is therefore a non-reproducible
floating-point deviation; the source has not been identified. The unit is included as stored.

## v2/v3 consistency
Not performed: no `matched-controlled-noisy-v2` static results exist on this server (the only prior static run,
commit 83f1a09, uses `matched-controlled-population-v3`).

## Raw results
`results_v3_raw.tar.gz` (in this folder) = `results/v3/paper/` (1,470 unit folders), 78,292,322 bytes.

## pip freeze (`.venv-static`)
```
ananke-causal==0.5.0
annotated-types==0.8.0
-e git+ssh://git@github.com/fmfsa/ccbo.git@a4f360ccce8117e2cb9b2bbdb42ca688c8b8d56d#egg=ccbo&subdirectory=../../.claude/worktrees/pr9-experiment-runs-c22831
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.2
cloudpickle==3.1.2
contourpy==1.3.2
cryptography==50.0.2
cuda-bindings==13.4.3
cuda-pathfinder==1.8.3
cuda-toolkit==13.0.3.0
cycler==0.12.1
Cython==3.3.0
decorator==5.3.1
dill==0.4.1
emcee==3.1.6
emukit==0.5.1
exceptiongroup==1.3.1
filelock==4.0.9
fonttools==4.65.0
fsspec==2026.9.0
google-ai-generativelanguage==0.6.15
google-api-core==2.33.0
google-api-python-client==2.201.0
google-auth==2.59.1
google-auth-httplib2==0.4.4
google-generativeai==0.8.6
googleapis-common-protos==1.75.0
GPy==1.13.2
graphviz==0.21
grpcio==1.84.0
grpcio-status==1.71.2
httplib2==0.32.0
idna==3.20
iniconfig==2.3.0
jax==0.4.38
jaxlib==0.4.38
Jinja2==3.1.6
joblib==1.6.0
kiwisolver==1.5.1
klepto==0.2.8
MarkupSafe==3.0.3
matplotlib==3.10.9
ml_dtypes==0.5.4
mpmath==1.3.0
mystic==0.4.5
networkx==3.4.2
numpy==1.26.4
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.24.0.43
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu12==2.32.3
nvidia-nccl-cu13==2.30.7
nvidia-nvjitlink==13.4.92
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
opt_einsum==3.4.0
packaging==26.3
pandas==1.5.3
paramz==0.9.6
patsy==1.0.3
pgmpy==0.1.26
pillow==12.3.0
pluggy==1.6.0
pox==0.3.7
proto-plus==1.28.2
protobuf==5.29.6
pyasn1==0.6.4
pyasn1_modules==0.4.2
pycparser==3.0
pydantic==2.13.5
pydantic_core==2.46.5
Pygments==2.21.0
pygraphviz==2.0.3
pyparsing==3.3.3
pytest==9.1.1
python-dateutil==2.9.0.post0
pytz==2026.4
PyYAML==6.0.3
requests==2.34.2
scikit-learn==1.7.2
scipy==1.12.0
seaborn==0.13.2
six==1.17.0
statsmodels==0.13.5
sympy==1.14.0
threadpoolctl==3.7.0
tomli==2.4.1
torch==2.14.1
tqdm==4.70.1
triton==3.8.0
typing-inspection==0.4.4
typing_extensions==4.16.0
uritemplate==4.2.0
urllib3==2.8.0
xgboost==3.2.0
```

## SHA-256 (all files in `experiment_outputs/v3/` except this manifest)
```
ee31f10d4ac7797f795abd9fd20bdab78e680cf4e119448a99422a8c56f1ffa1  ./clusterchain.csv
1d78d18447e8c82886bac298075e99d54c7e5d67b4436a5a4775d19651f1fecd  ./clusterchain.json
07239bba95dbd3b67ef6273197bae2c17beb2581c24b0a2672f365b4b30e7709  ./paper_outputs/figures/cost_indexed.pdf
54716a607e6b5f07e39cb99455726554f329187b07310bf8fe346bb25d0f7893  ./paper_outputs/figures/minimal_refine.pdf
8457445ffebcdbb958a38f9614e6ee25f06d60c406c6b6ba222aa4172efc5c17  ./paper_outputs/tables/cost_accounting.tex
9075c71c737cd8d5dc24fed623146a9acee3ae6af6d797c844fbf64524569b77  ./paper_outputs/tables/minimal_ablations.tex
44af193e5ec624b04a1498666f89bb4b2b0ff8f70ceea1e3f4ee3b5aa0a60d4a  ./paper_outputs/tables/minimal_taxonomy.tex
c0b9c4b5df62989dd8649c7db9ca8374262922833c0681d467ec51b148966043  ./results_v3_raw.tar.gz
3eddd405ec508788efbce0dc1da856187d47f25e3944c79fb50fb881ad419f74  ./static.csv
e198dd208a86877ee816973934f9a97bfb8836e8804600d44fbd152aa018ff1d  ./static.json
```
